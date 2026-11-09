"""The walk over the state that a loaded model keeps, for weight snapshots."""

from __future__ import annotations

import enum
import functools
import inspect
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
_MODULE_FIELDS = frozenset(vars(nn.Module()))  # training, _parameters, hooks, ...


def walk(model: nn.Module) -> Iterator[tuple[str, str, Any]]:
    """Yield (path, kind, value) for each module's parameters, buffers and attributes.

    kind is "param", "buffer" or "attr" for a tensor and "value" for anything else.
    A tensor sits at a dotted path ("...self_attn.w_kc"); anything else after "#" on
    its owner ("...self_attn#use_deep_gemm_bmm", "...weight_scale_inv#format_ue8m0").
    """
    walked: set[int] = set()  # objects already opened: shared ones, and cycles
    for path, module in model.named_modules(remove_duplicate=False):
        yield f"{path}#__class__", "value", type(module)
        yield f"{path}#training", "value", module.training
        for name, t in module._parameters.items():
            yield from _tensor(path, name, "param", t, walked)
        for name, t in module._buffers.items():
            yield from _tensor(path, name, "buffer", t, walked)
        for name, v in vars(module).items():
            if name not in _MODULE_FIELDS:
                yield from _attribute(path, name, v, walked)


def canonical(v: Any, _open: frozenset[int] = frozenset()) -> str:
    """Text for a value that is the same in every correct process.

    No addresses (an object's repr carries one) and no hash-seed order (the repr of
    a set of strings follows the per-process seed).
    """
    if isinstance(v, enum.Enum):  # before int and str: IntEnum and str enums are both
        return f"{type(v).__qualname__}.{v.name}"
    if v is None or isinstance(v, (bool, int, str, bytes)):
        return repr(v)
    if isinstance(v, float):
        return v.hex()
    if isinstance(v, torch.dtype):
        return str(v)
    if isinstance(v, torch.device):
        return f"device {v.type}"  # the index is per process
    if isinstance(v, (list, tuple, set, frozenset, dict)):
        if id(v) in _open:
            return "<cycle>"
        inner = _open | {id(v)}
        if isinstance(v, dict):
            items = [
                f"{canonical(k, inner)}: {canonical(x, inner)}" for k, x in v.items()
            ]
        else:
            items = [canonical(x, inner) for x in v]
        if isinstance(v, (set, frozenset, dict)):
            items.sort()
        return f"{type(v).__name__}[{', '.join(items)}]"
    if isinstance(v, functools.partial):
        return f"partial[{_name(v.func)}, {canonical(v.args)}, {canonical(v.keywords)}]"
    if inspect.ismethod(v):
        return f"method {_name(v.__func__)} of {_name(type(v.__self__))}"
    if inspect.isroutine(v) or inspect.isclass(v):
        return _name(v)
    return f"object {_name(type(v))}"


def _tensor(owner: str, name: str, kind: str, t, walked: set[int]):
    if t is None:  # a registered slot that holds None is state too
        yield f"{owner}#{name}", "value", None
        return
    path = f"{owner}.{name}" if owner else name
    yield path, kind, t
    for k, v in vars(t).items():  # flags kept on the tensor itself, e.g. format_ue8m0
        yield from _attribute(path, k, v, walked)


def _attribute(owner: str, name: str, v: Any, walked: set[int]):
    if isinstance(v, nn.Module):
        return  # named_modules visits every module
    if isinstance(v, torch.Tensor):
        yield from _tensor(owner, name, "attr", v, walked)
    elif id(v) in walked:  # shared, or a cycle back: opened at its first path
        yield f"{owner}#{name}", "value", v
    elif isinstance(v, (list, tuple, dict)) and any(map(_holds_state, _items(v))):
        walked.add(id(v))  # tensors or helper objects kept in a container
        pairs = v.items() if isinstance(v, dict) else enumerate(v)
        for key, x in pairs:
            yield from _attribute(owner, f"{name}[{key!r}]", x, walked)
    elif _is_helper(v):
        walked.add(id(v))  # a quant method, an MoE runner, a top-k config
        yield f"{owner}#{name}", "value", type(v)
        path = f"{owner}.{name}" if owner else name
        for k, x in vars(v).items():
            yield from _attribute(path, k, x, walked)
    else:
        yield f"{owner}#{name}", "value", v


def _is_helper(v: Any) -> bool:
    # An object of model code that keeps its state in an instance __dict__.
    return (
        type(v).__module__.startswith(MODEL_CODE)
        and not isinstance(v, enum.Enum)
        and not inspect.isclass(v)
        and hasattr(v, "__dict__")
    )


def _holds_state(x: Any) -> bool:
    return isinstance(x, torch.Tensor) or _is_helper(x)


def _items(v: list | tuple | dict) -> Iterator[Any]:
    return iter(v.values()) if isinstance(v, dict) else iter(v)


def _name(x: Any) -> str:
    # Reflection over arbitrary callables: a method-wrapper has no __module__.
    return f"{getattr(x, '__module__', None)}.{x.__qualname__}"
