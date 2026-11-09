"""Validator of weight-snapshot loads: fingerprint, comparison, scans and vote.

A snapshot reader compares fingerprint(model) with the owner's, entry by entry,
and checks that no meta tensor survived install; then every rank votes, so no
rank serves a model that any rank refused. A raw load is the reference and only
votes.
"""

from __future__ import annotations

import datetime
import gc

import blake3
import torch
import torch.distributed as dist
from torch import nn

from sglang.kernels.ops.memory.gpu_tensor_hash_blake3 import gpu_tensors_blake3
from sglang.srt.model_loader.snapshot_walk import TENSOR_KINDS, canonical, walk

# Matches the engine's post-load barrier (UNBALANCED_MODEL_LOADING_TIMEOUT_S).
_VOTE_TIMEOUT_S = 480
# Arbitrary cap on the lines a refusal reports per rank.
_MAX_REPORTED = 50
# Arbitrary bound on the int64 tensors scanned for addresses; pointer tables are small.
_MAX_SCANNED_NUMEL = 1 << 20


def fingerprint(model: nn.Module, *, expert_map: torch.Tensor | None) -> dict[str, str]:
    """One entry per path of walk(model); a tensor entry ends in a BLAKE3 digest.

    A tensor entry is "kind class dtype shape strides device rg=0|1
    @storage+byte_offset digest". The digest covers the tensor's logical bytes
    only, so the fields before it frame what the digest cannot see. No field is
    an address or a device index, so two correct processes produce equal text.
    """
    entries: dict[str, str] = {}
    tensors: list[tuple[str, torch.Tensor]] = []
    storage_names: dict[tuple, str] = {}
    for path, kind, v in walk(model):
        if kind not in TENSOR_KINDS:
            entries[path] = f"{kind} {canonical(v)}"
        elif v.is_meta:
            entries[path] = f"{_layout(kind=kind, t=v)} meta"
        else:
            # The first path that reaches an allocation names it, so aliases agree.
            storage = storage_names.setdefault(_storage_key(t=v, path=path), path)
            offset = v.storage_offset() * v.element_size()
            entries[path] = (
                f"{_layout(kind=kind, t=v)} {v.device.type} "
                f"rg={int(v.requires_grad)} @{storage}+{offset}"
            )
            tensors.append((path, v))
    digests = _blake3_by_device([t for _, t in tensors])
    for (path, _), digest in zip(tensors, digests):
        entries[path] += f" {digest.hex()}"
    if expert_map is None:  # a dense model
        entries["__expert_map__"] = "none"
    else:
        entries["__expert_map__"] = _blake3_by_device([expert_map])[0].hex()
    return entries


def compare(*, owner: dict[str, str], here: dict[str, str]) -> list[str]:
    """Every path whose entry differs, with both sides; a missing side is absent."""
    return [
        f"{path}: owner {owner.get(path, 'absent')} | here {here.get(path, 'absent')}"
        for path in sorted(owner.keys() | here.keys())
        if owner.get(path) != here.get(path)
    ]


def live_meta_tensors() -> list[str]:
    """Every meta tensor still alive in this process.

    On a snapshot reader the only meta tensors are the skeleton's, so a survivor
    is a reference that install did not replace. Using it does not fail: a custom
    op given a meta input runs its Meta kernel and silently does nothing.
    """
    gc.collect()
    # type() rather than isinstance(): isinstance reads __class__, which deprecated
    # torch objects turn into a warning.
    return [
        f"{type(o).__qualname__}{tuple(o.shape)}"
        for o in gc.get_objects()
        if issubclass(type(o), torch.Tensor) and o.is_meta
    ]


def find_device_addresses(model: nn.Module) -> list[str]:
    """Paths whose int value, or int64 tensor element, is a live device address.

    Such a value is valid only in the process that holds it: shipped, it compares
    equal on a reader and points into the owner's memory. Export refuses any that
    the owning class did not declare in __snapshot_local__.
    """
    found = []
    for path, kind, v in walk(model):
        if kind == "value":
            values = _ints(v)
        elif kind in TENSOR_KINDS and _is_int_table(v):
            values = v.detach().reshape(-1).tolist()
        else:
            continue
        if any(_is_device_address(x) for x in values):
            found.append(path)
    return found


class Validator:
    """check_after_load of one snapshot load, on one rank.

    Every rank of the group that builds the model together calls it in the same
    order, so the votes line up.
    """

    def __init__(
        self, *, cpu_group: dist.ProcessGroup, monitored_barrier: bool = True
    ) -> None:
        self.cpu_group = cpu_group
        # Mooncake elastic EP has no monitored_barrier; it votes after a plain one.
        self.monitored_barrier = monitored_barrier

    def check_after_load(
        self,
        *,
        model: nn.Module,
        owner_fingerprint: dict[str, str] | None,
        expert_map: torch.Tensor | None,
    ) -> None:
        """On a snapshot, compare with the owner and scan for meta tensors; then vote.

        Raises on every rank if any rank refused. owner_fingerprint is None for a
        raw load, which is the reference and only votes.
        """
        problem = None
        if owner_fingerprint is not None:
            here = fingerprint(model, expert_map=expert_map)
            problem = _refusal(
                mismatches=compare(owner=owner_fingerprint, here=here),
                stale=live_meta_tensors(),
            )
        self._vote(problem=problem)

    def _vote(self, *, problem: str | None) -> None:
        if self.monitored_barrier:
            dist.monitored_barrier(
                group=self.cpu_group,
                timeout=datetime.timedelta(seconds=_VOTE_TIMEOUT_S),
                wait_all_ranks=True,
            )
        else:
            dist.barrier(group=self.cpu_group)
        ballots: list[str | None] = [None] * dist.get_world_size(self.cpu_group)
        dist.all_gather_object(ballots, problem, group=self.cpu_group)
        refusals = [(rank, p) for rank, p in enumerate(ballots) if p is not None]
        if refusals:
            report = "\n".join(f"[rank {rank}] {p}" for rank, p in refusals)
            raise RuntimeError(f"Snapshot load refused:\n{report}")


def _refusal(*, mismatches: list[str], stale: list[str]) -> str | None:
    if not mismatches and not stale:
        return None
    lines = []
    if mismatches:
        lines.append(f"{len(mismatches)} fingerprint entries differ from the owner's")
        lines.extend(mismatches[:_MAX_REPORTED])
    if stale:
        lines.append(f"{len(stale)} meta tensors are still alive after install")
        lines.extend(stale[:_MAX_REPORTED])
    return "\n".join(lines)


def _layout(*, kind: str, t: torch.Tensor) -> str:
    cls = type(t)
    return (
        f"{kind} {cls.__module__}.{cls.__qualname__} {t.dtype} "
        f"{tuple(t.shape)} {t.stride()}"
    )


def _storage_key(*, t: torch.Tensor, path: str) -> tuple:
    # Groups the paths that view one allocation; the address never enters an entry.
    storage = t.untyped_storage()
    if storage.nbytes() == 0:  # every empty CUDA tensor reports data_ptr() == 0
        return ("empty", path)
    return (t.device.type, t.device.index, storage.data_ptr())


def _blake3_by_device(tensors: list[torch.Tensor]) -> list[bytes]:
    # gpu_tensors_blake3 takes one CUDA device per call and rejects host tensors;
    # the blake3 package gives host tensors the same BLAKE3-256.
    digests: list[bytes] = [b""] * len(tensors)
    by_device: dict[torch.device, list[int]] = {}
    for i, t in enumerate(tensors):
        by_device.setdefault(t.device, []).append(i)
    for device, indices in by_device.items():
        batch = [tensors[i] for i in indices]
        if device.type == "cuda":
            batch_digests = gpu_tensors_blake3(batch)
        else:
            batch_digests = [_host_blake3(t) for t in batch]
        for i, digest in zip(indices, batch_digests):
            digests[i] = digest
    return digests


def _host_blake3(t: torch.Tensor) -> bytes:
    data = t.detach().resolve_conj().resolve_neg().contiguous().reshape(-1)
    raw = data.view(torch.uint8).cpu().numpy().tobytes()
    return blake3.blake3(raw).digest()


def _ints(v: object) -> list[int]:
    if isinstance(v, bool):
        return []
    if isinstance(v, int):
        return [v]
    if isinstance(v, (list, tuple, set, frozenset)):
        return [x for item in v for x in _ints(item)]
    if isinstance(v, dict):
        return [x for k, item in v.items() for x in _ints(k) + _ints(item)]
    return []


def _is_int_table(t: torch.Tensor) -> bool:
    return (
        t.dtype in (torch.int64, torch.uint64)
        and not t.is_meta
        and 0 < t.numel() <= _MAX_SCANNED_NUMEL
    )


def _is_device_address(x: int) -> bool:
    # Arbitrary cut that skips the driver call for counts, ids and fds; device and
    # pinned-host addresses sit far above it. A Python int may not fit a pointer.
    if x < 1 << 32 or x >= 1 << 64:
        return False
    from cuda.bindings import driver

    attribute = driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_RANGE_START_ADDR
    err, _ = driver.cuPointerGetAttribute(attribute, x)
    return err == driver.CUresult.CUDA_SUCCESS
