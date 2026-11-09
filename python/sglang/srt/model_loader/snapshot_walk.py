"""The walk over the state that a loaded model keeps, for weight snapshots."""

from __future__ import annotations

import enum
import functools
import numbers
import types
from typing import Any, Iterator

import torch
from torch import nn

# Instances of classes from these packages (quant methods, MoE runners and
# dispatchers, top-k configs) hold model state and are walked; any other object,
# such as a process group or a stream, is recorded by its class only.
MODEL_CODE = (
    "sglang.srt.layers.",
    "sglang.srt.models.",
    "sglang.srt.batch_overlap.",
    "sglang.srt.hardware_backend.",
    "sglang.srt.lora.",
)
TENSOR_KINDS = ("param", "buffer", "attr")
# Values whose repr is the value itself, with no address in it.
_PLAIN = (numbers.Number, str, bytes, range, slice, torch.dtype)
_MODULE_FIELDS = frozenset(vars(nn.Module()))  # training, _parameters, hooks, ...


def walk(model: nn.Module) -> Iterator[tuple[str, str, Any]]:
    """Yield (path, kind, value) for each module's parameters, buffers and attributes.

    kind is "param", "buffer" or "attr" for a tensor and "value" for anything else.
    A tensor sits at a dotted path ("...self_attn.w_kc"); anything else after "#" on
    its owner ("...self_attn#use_deep_gemm_bmm", "...weight_scale_inv#format_ue8m0").
    """
    modules = list(model.named_modules(remove_duplicate=False))
    # Objects opened already: every module (visited below), then shared ones, cycles.
    walked = {id(m) for _, m in modules}
    for path, module in modules:
        yield f"{path}#__class__", "value", type(module)
        yield f"{path}#training", "value", module.training
        for name, t in module._parameters.items():
            yield from _visit(path, name, "param", t, walked)
        for name, t in module._buffers.items():
            yield from _visit(path, name, "buffer", t, walked)
        for name, v in vars(module).items():
            if name not in _MODULE_FIELDS:
                yield from _visit(path, name, "attr", v, walked)


def canonical(v: Any, _open: frozenset[int] = frozenset()) -> str:
    """Text for a value that is the same in every correct process.

    No addresses (an object's repr carries one), no hash-seed order (the repr of a
    set of strings follows the per-process seed) and no checkpoint path (remote-code
    classes live in a module named after it).
    """
    if isinstance(v, enum.Enum):  # before int and str: IntEnum and str enums are both
        return f"{type(v).__qualname__}.{v.name}"
    if v is None or isinstance(v, _PLAIN):
        return repr(v)
    if isinstance(v, torch.device):
        return f"device {v.type}"  # the index is per process
    if isinstance(v, functools.partial):  # e.g. the FP8 GEMM kernel and its flags
        return "partial " + canonical((v.func, v.args, v.keywords), _open)
    if isinstance(v, types.MethodType):  # an inherited method acts on its receiver
        return f"{canonical(v.__func__)} of {canonical(type(v.__self__))}"
    if isinstance(v, (list, tuple, set, frozenset, dict)):
        if id(v) in _open:
            return "<cycle>"
        pairs = v.items() if isinstance(v, dict) else v
        items = [canonical(x, _open | {id(v)}) for x in pairs]
        if not isinstance(v, (list, tuple)):  # a set iterates in hash-seed order
            items.sort()
        return f"{type(v).__name__}[{', '.join(items)}]"
    if callable(v) and hasattr(v, "__qualname__"):  # a function, class or torch op
        module = getattr(v, "__module__", None) or ""
        if module.startswith("transformers_modules."):
            module = module.rsplit(".", 1)[-1]
        return f"{module}.{v.__qualname__}"
    return f"object {canonical(type(v))}"


def _visit(owner: str, name: str, kind: str, v: Any, walked: set[int]):
    path = _join(owner, name)
    if isinstance(v, torch.Tensor):
        yield path, kind, v
        parts = vars(v)  # flags kept on the tensor itself, e.g. format_ue8m0
    else:
        parts = None if id(v) in walked else _parts(v)
        if parts is None:  # a plain value, or an object opened at another path
            yield f"{owner}#{name}", "value", v
            return
        walked.add(id(v))
        yield f"{owner}#{name}", "value", type(v)
    for k, x in parts.items():
        yield from _visit(path, k, "attr", x, walked)


def _parts(v: Any) -> dict[str, Any] | None:
    # What walk() opens: a container of tensors or model-code objects, or such an object.
    if isinstance(v, (list, tuple, dict)):
        values = v.values() if isinstance(v, dict) else v
        if any(isinstance(x, torch.Tensor) or _is_model_object(x) for x in values):
            return _items(v)
    return vars(v) if _is_model_object(v) else None


def _items(v: list | tuple | dict) -> dict[str, Any]:
    # Items as "[name]", by a text that is the same in every process: index or key.
    if isinstance(v, dict):
        named = {f"[{canonical(k)}]": x for k, x in v.items()}
        if len(named) == len(v):
            return named
        v = list(v.values())  # keys that print alike (objects of one class): by order
    return {f"[{i}]": x for i, x in enumerate(v)}


def _join(path: str, name: str) -> str:
    if name.startswith("["):
        return path + name
    return f"{path}.{name}" if path else name


def _is_model_object(v: Any) -> bool:
    # An object of model code that keeps its state in an instance __dict__.
    return (
        type(v).__module__.startswith(MODEL_CODE)
        and hasattr(v, "__dict__")
        and not isinstance(v, enum.Enum)
    )
