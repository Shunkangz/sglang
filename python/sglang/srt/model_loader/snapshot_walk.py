"""Reflective walk over the state that a loaded model keeps, for weight snapshots.

The walk yields every tensor the model holds (parameters, buffers and plain tensor
attributes at any depth), every plain value that modules, tensors and helper
objects of model code carry, and every other object by identity only. It stops at
objects outside model code, so process-wide state such as process groups,
communicators and Python modules is never walked.
"""

from __future__ import annotations

import enum
import functools
import inspect
import types
from typing import Any, Iterator

import msgspec
import torch
from torch import nn

# Packages whose helper objects hold model state; any other object is a boundary.
MODEL_CODE = (
    "sglang.srt.layers.",
    "sglang.srt.models.",
    "sglang.srt.batch_overlap.",
    "sglang.srt.hardware_backend.",
    "sglang.srt.lora.",
)
TENSOR_KINDS = ("param", "buffer", "attr")
PLAIN = (type(None), bool, int, float, str, bytes, enum.Enum, torch.dtype)
_MODULE_FIELDS = frozenset(vars(nn.Module()))


class Ref(msgspec.Struct, frozen=True):
    """An object or container that the walk first reached at another path."""

    path: str


def walk(model: nn.Module) -> Iterator[tuple[str, str, Any]]:
    """Yield (path, kind, value) for the model's own state.

    kind is "param", "buffer" or "attr" for a tensor, "value" for a plain value or
    a container of plain values, and "object" for anything compared by identity:
    a class, a callable, an object outside model code, or a Ref. A tensor sits at
    a dotted path ("...self_attn.w_kc"); everything else after "#" on its owner
    ("...self_attn#use_deep_gemm_bmm", "...weight_scale_inv#format_ue8m0").
    """
    seen: dict[int, str] = {}
    for mod_path, module in model.named_modules(remove_duplicate=False):
        yield f"{mod_path}#__class__", "object", type(module)
        yield f"{mod_path}#training", "value", module.training
        prefix = f"{mod_path}." if mod_path else ""
        local = snapshot_local(module)
        for name, t in module._parameters.items():
            if name not in local:
                yield from _tensor(path=prefix + name, kind="param", t=t, seen=seen)
        for name, t in module._buffers.items():
            if name not in local:
                yield from _tensor(path=prefix + name, kind="buffer", t=t, seen=seen)
        for name, v in _fields(module):
            if name not in _MODULE_FIELDS:
                yield from _member(owner=mod_path, name=name, v=v, seen=seen)


def snapshot_local(obj: Any) -> frozenset[str]:
    """The fields that obj's class declares valid only in the process that set them.

    Such a field (an address, a handle, a cache keyed by one) is neither shipped
    nor fingerprinted; a reader keeps its own value.
    """
    return frozenset(
        name
        for cls in type(obj).__mro__
        for name in vars(cls).get("__snapshot_local__", ())
    )


def canonical(v: Any) -> str:
    """Encode a value or object identically in every correct process.

    No addresses (the repr of an object carries one) and no hash-seed order (the
    repr of a set of strings follows the per-process seed).
    """
    if isinstance(v, Ref):
        return f"ref {v.path}"
    # Before int and str: IntEnum and str-valued enums are both.
    if isinstance(v, enum.Enum):
        return f"enum {type(v).__qualname__}.{v.name}"
    if v is None or isinstance(v, (bool, int, str, bytes)):
        return repr(v)
    if isinstance(v, float):
        return f"float {v.hex()}"
    if isinstance(v, torch.dtype):
        return f"dtype {v}"
    if isinstance(v, (set, frozenset)):
        return f"{type(v).__name__}{{{', '.join(sorted(canonical(x) for x in v))}}}"
    if isinstance(v, dict):
        items = sorted(f"{canonical(k)}: {canonical(x)}" for k, x in v.items())
        return f"dict{{{', '.join(items)}}}"
    if isinstance(v, (list, tuple)):
        return f"{type(v).__name__}[{', '.join(canonical(x) for x in v)}]"
    return _identity(v)


def _identity(v: Any) -> str:
    if isinstance(v, type):
        return f"class {_qualname(v)}"
    if isinstance(v, functools.partial):
        args = ", ".join(canonical(a) for a in v.args)
        kwargs = ", ".join(f"{k}={canonical(x)}" for k, x in sorted(v.keywords.items()))
        return f"partial({_identity(v.func)}; {args}; {kwargs})"
    if isinstance(v, types.MethodType):
        return f"method {_qualname(v.__func__)} of {_qualname(type(v.__self__))}"
    if inspect.isroutine(v):
        return f"function {_qualname(v)}"
    if isinstance(v, torch.device):
        return f"device {v.type}"  # the index is per process
    return f"object {_qualname(type(v))}"


def _qualname(x: Any) -> str:
    return f"{x.__module__}.{x.__qualname__}"


def _tensor(*, path: str, kind: str, t: torch.Tensor | None, seen: dict[int, str]):
    if t is None:  # a registered slot holding None is state too
        owner, _, name = path.rpartition(".")
        yield f"{owner}#{name}", "value", None
        return
    if vars(t).get("snapshot_local"):  # a tensor that kernels write, e.g. GEMM locks
        return
    yield path, kind, t
    for name, v in _fields(t):  # state kept on the tensor itself
        yield from _member(owner=path, name=name, v=v, seen=seen)


def _member(*, owner: str, name: str, v: Any, seen: dict[int, str]):
    path = f"{owner}.{name}" if owner else name
    if isinstance(v, nn.Module):
        return  # named_modules visits every registered module
    if isinstance(v, torch.Tensor):
        yield from _tensor(path=path, kind="attr", t=v, seen=seen)
    elif _is_plain(v):
        yield f"{owner}#{name}", "value", v
    elif id(v) in seen:  # shared, or a cycle back
        yield f"{owner}#{name}", "object", Ref(seen[id(v)])
    elif isinstance(v, (list, tuple, dict)):
        seen[id(v)] = path
        yield f"{owner}#{name}", "object", type(v)
        items = v.items() if isinstance(v, dict) else enumerate(v)
        for key, x in items:
            key_name = f"{name}[{canonical(key)}]"
            yield from _member(owner=owner, name=key_name, v=x, seen=seen)
    elif _is_routine(v) or not type(v).__module__.startswith(MODEL_CODE):
        yield (
            f"{owner}#{name}",
            "object",
            v,
        )  # a kernel choice, a process group, a stream
    else:  # a helper object of model code: a quant method, an MoE runner, a top-k config
        seen[id(v)] = path
        yield f"{owner}#{name}", "object", type(v)
        for k, x in _fields(v):
            yield from _member(owner=path, name=k, v=x, seen=seen)


def _fields(obj: Any) -> list[tuple[str, Any]]:
    local = snapshot_local(obj)
    return [(k, v) for k, v in _instance_fields(obj) if k not in local]


def _instance_fields(obj: Any) -> list[tuple[str, Any]]:
    if isinstance(obj, msgspec.Struct):
        return list(msgspec.structs.asdict(obj).items())
    try:
        return list(vars(obj).items())
    except TypeError:  # a __slots__ class has no __dict__
        pass
    fields = []
    for cls in type(obj).__mro__:
        slots = vars(cls).get("__slots__", ())
        for name in (slots,) if isinstance(slots, str) else slots:
            try:
                fields.append((name, object.__getattribute__(obj, name)))
            except AttributeError:  # an unset slot
                pass
    return fields


def _is_plain(v: Any, _open: frozenset[int] = frozenset()) -> bool:
    if isinstance(v, PLAIN):
        return True
    if id(v) in _open:  # a container that contains itself is not a plain value
        return False
    inner = _open | {id(v)}
    if isinstance(v, (list, tuple, set, frozenset)):
        return all(_is_plain(x, inner) for x in v)
    if isinstance(v, dict):
        return all(_is_plain(k, inner) and _is_plain(x, inner) for k, x in v.items())
    return False


def _is_routine(v: Any) -> bool:
    return inspect.isroutine(v) or isinstance(v, (type, functools.partial))
