"""Protected Store allocation lifecycle. Build `_opaque_ltx` before running this file."""
from __future__ import annotations

import ctypes

import pytest
import torch

from jasna.ltx.kernels import supports
from jasna.ltx.model_files import PlaceholderHandle, PlaceholderTensors
from jasna.ltx import native
from jasna.ltx import transformer as T

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not supports(torch.device("cuda")), reason="needs a supported NVIDIA GPU",
)
CANVAS_SIDES = {8192: 16, 18432: 24}


class _Destination(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint64), ("process", ctypes.c_uint64), ("interpreter", ctypes.c_void_p),
        ("length", ctypes.c_size_t),
    ]


_Write = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)


class _Sink(ctypes.Structure):
    _fields_ = [("magic", ctypes.c_uint64), ("validate", ctypes.c_void_p),
                ("overlap", ctypes.c_void_p), ("write", _Write), ("wipe", ctypes.c_void_p)]


def _store(count: int = 1, resident: int = 1, quantization: str = "int8"):
    from jasna.protection.opaque_ltx import OpaqueStore

    handle = PlaceholderHandle(PlaceholderTensors(quantization, "distilled"))
    specs = {name.split(".", 2)[2]: spec for name, spec in handle.tensor_specs().items()
             if name.startswith("blocks.0.")}
    serialized = {name: (str(dtype).removeprefix("torch."), shape) for name, (dtype, shape) in specs.items()}
    store = OpaqueStore([serialized] * count, "cuda", resident)
    capsule_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    capsule_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    capsule_pointer.restype = ctypes.c_void_p
    capsule_context = ctypes.pythonapi.PyCapsule_GetContext
    capsule_context.argtypes = [ctypes.py_object]
    capsule_context.restype = ctypes.c_void_p
    for block in range(count):
        for name in serialized:
            value = handle.get_tensor("blocks.0." + name).contiguous()
            capsule = store.destination(block, name)
            pointer = capsule_pointer(capsule, b"jasna.protection.opaque-ltx-destination.v2")
            destination = _Destination.from_address(pointer)
            assert destination.length == value.nbytes
            assert _Destination.length.offset == 24
            sink = _Sink.from_address(capsule_context(capsule))
            assert sink.write(pointer, value.data_ptr(), value.nbytes) == 0
    return store


@pytest.mark.parametrize("quantization", ["int8", "nvfp4"])
def test_opaque_destination_capsules_expose_only_metadata(quantization):
    from jasna.protection import opaque_ltx

    store = _store(quantization=quantization)
    try:
        destination = store.destination(0, "scale_shift_table")
        sliced = opaque_ltx.slice_destination(destination, 1, 4)
        pointer = ctypes.pythonapi.PyCapsule_GetPointer
        pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        pointer.restype = ctypes.c_void_p
        header = _Destination.from_address(pointer(sliced, b"jasna.protection.opaque-ltx-destination.v2"))
        assert _Destination.length.offset == 24 and header.length == 4
        assert opaque_ltx.destination_size(sliced) == 4
        with pytest.raises(TypeError):
            memoryview(sliced)
        with pytest.raises(ValueError):
            opaque_ltx.slice_destination(destination, opaque_ltx.destination_size(destination), 1)
    finally:
        store.close()
    with pytest.raises(ValueError):
        opaque_ltx.destination_size(sliced)


def test_protected_weights_bypass_the_torch_caching_allocator_and_close_releases_them():
    from jasna.protection.opaque_ltx import _native

    store = _store()
    audits = _native()._wipe_audits() if hasattr(_native(), "_wipe_audits") else None
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    allocated_before = torch.cuda.memory_allocated()
    free_before = torch.cuda.mem_get_info()[0]
    store.finalize()
    torch.cuda.synchronize()
    allocated_after = torch.cuda.memory_allocated()
    free_after = torch.cuda.mem_get_info()[0]
    try:
        assert allocated_after == allocated_before
        assert free_before - free_after > 200 << 20
        assert free_before - free_after - (8 << 20) <= store.device_bytes() <= free_before - free_after
    finally:
        store.close()
        torch.cuda.synchronize()
    assert store.device_bytes() == 0
    assert torch.cuda.memory_allocated() == allocated_before
    assert torch.cuda.mem_get_info()[0] >= free_before - (8 << 20)
    if audits is not None:
        assert _native()._wipe_audits() >= audits + 2


def test_partial_load_close_is_idempotent_and_releases_no_cuda_allocation():
    store = _store()
    free_before = torch.cuda.mem_get_info()[0]
    store.close()
    store.close()
    torch.cuda.synchronize()
    assert torch.cuda.mem_get_info()[0] >= free_before - (1 << 20)


@pytest.mark.parametrize("quantization,tolerance", [("int8", 5e-3), ("nvfp4", 2.5e-2)])
def test_protected_store_uses_the_shared_block_semantics(quantization, tolerance):
    store = _store(quantization=quantization)
    handle = PlaceholderHandle(PlaceholderTensors(quantization, "distilled"))
    names = [name.split(".", 2)[2] for name in handle.tensor_specs() if name.startswith("blocks.0.")]
    weights = {name: handle.get_tensor("blocks.0." + name).cuda() for name in names}
    dim, rows = weights["scale_shift_table"].shape[1], 8192
    generator = torch.Generator(device="cuda").manual_seed(17)
    x = torch.randn(1, rows, dim, generator=generator, device="cuda").to(torch.bfloat16)
    condition = T.StepConditioning(
        block_rows=(torch.randn(1, 2, 9, dim, generator=generator, device="cuda") * .1).to(torch.bfloat16),
        output_rows=torch.zeros(1, 2, dim, dtype=torch.bfloat16, device="cuda"),
        prompt_rows=(torch.randn(1, 1, 2, dim, generator=generator, device="cuda") * .1).to(torch.bfloat16),
    )
    context = torch.randn(1, 1024, dim, generator=generator, device="cuda").to(torch.bfloat16)
    rope = T.build_rope(16, 16, 16, dim=dim, heads=T.HEADS, device=torch.device("cuda"))
    expected = T.block_forward(weights, x, condition, T.prompt_key_value(weights, condition, context), rope)
    try:
        store.finalize()
        store.begin_step(rows, 1, False)
        store.activate(0)
        actual = store.block_forward(
            x, condition.block_rows, condition.prompt_rows, context, rope.cos, rope.sin, False)
        relative = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
        assert relative < tolerance
    finally:
        store.close()


@pytest.mark.parametrize("quantization", ["int8", "nvfp4"])
@pytest.mark.parametrize("rows", list(CANVAS_SIDES))
def test_protected_block_is_bit_identical_to_public_native_block(quantization, rows):
    store = _store(quantization=quantization)
    handle = PlaceholderHandle(PlaceholderTensors(quantization, "distilled"))
    specs = {name.split(".", 2)[2]: spec for name, spec in handle.tensor_specs().items()
             if name.startswith("blocks.0.")}
    weights = {name: handle.get_tensor("blocks.0." + name).cuda() for name in specs}
    public = native.load(torch.device("cuda"), specs)
    assert public is not None
    dim = weights["scale_shift_table"].shape[1]
    generator = torch.Generator(device="cuda").manual_seed(rows + (1 if quantization == "nvfp4" else 0))
    x = torch.randn(1, rows, dim, generator=generator, device="cuda").to(torch.bfloat16)
    condition = T.StepConditioning(
        block_rows=(torch.randn(1, 2, 9, dim, generator=generator, device="cuda") * .1).to(torch.bfloat16),
        output_rows=torch.zeros(1, 2, dim, dtype=torch.bfloat16, device="cuda"),
        prompt_rows=(torch.randn(1, 1, 2, dim, generator=generator, device="cuda") * .1).to(torch.bfloat16),
    )
    context = torch.randn(1, 1024, dim, generator=generator, device="cuda").to(torch.bfloat16)
    side = CANVAS_SIDES[rows]
    rope = T.build_rope(16, side, side, dim=dim, heads=T.HEADS, device=torch.device("cuda"))
    linears = [tensor for name in T.LINEAR_NAMES for tensor in native._linear_tensors(weights, name)]
    prompt = public.block.prompt_key_value(
        linears, weights["attn2.k_norm.weight"], weights["prompt_scale_shift_table"],
        condition.prompt_rows, context,
    )
    expected = public(weights, x, condition, prompt, rope)
    try:
        store.finalize()
        store.begin_step(rows, 1, False)
        store.activate(0)
        actual = store.block_forward(
            x, condition.block_rows, condition.prompt_rows, context, rope.cos, rope.sin, False)
        torch.cuda.synchronize()
        if not torch.equal(actual, expected):
            pytest.fail("protected and public native block outputs differ")
    finally:
        store.close()


def test_streamed_blocks_prefetch_in_authenticated_order():
    from jasna.protection.opaque_ltx import _native

    store = _store(count=3, resident=1)
    audits = _native()._wipe_audits() if hasattr(_native(), "_wipe_audits") else None
    dim, rows = 4096, 8192
    generator = torch.Generator(device="cuda").manual_seed(23)
    x = torch.randn(1, rows, dim, generator=generator, device="cuda").to(torch.bfloat16)
    block_rows = (torch.randn(1, 2, 9, dim, generator=generator, device="cuda") * .1).to(torch.bfloat16)
    prompt_rows = (torch.randn(1, 1, 2, dim, generator=generator, device="cuda") * .1).to(torch.bfloat16)
    context = torch.randn(1, 1024, dim, generator=generator, device="cuda").to(torch.bfloat16)
    rope = T.build_rope(16, 16, 16, dim=dim, heads=T.HEADS, device=torch.device("cuda"))
    try:
        store.finalize()
        store.begin_step(rows, 1, False)
        for block in range(3):
            store.activate(block)
            x = store.block_forward(x, block_rows, prompt_rows, context, rope.cos, rope.sin, False)
        torch.cuda.synchronize()
        assert torch.isfinite(x.float()).all()
    finally:
        store.close()
    if audits is not None:
        assert _native()._wipe_audits() >= audits + 6


def test_injected_finalize_error_wipes_partial_allocations():
    from jasna.protection.opaque_ltx import _native

    if not hasattr(_native(), "_fail_next_stage"):
        pytest.skip("build the extension with test_observer=True for full-buffer wipe auditing")
    store = _store()
    audits = _native()._wipe_audits()
    _native()._fail_next_stage()
    with pytest.raises(RuntimeError, match="injected opaque LTX copy failure"):
        store.finalize()
    store.close()
    assert _native()._wipe_audits() >= audits + 2
