"""Validator of weight-snapshot loads: fingerprint, load changes, meta scan and vote.

A snapshot reader compares fingerprint(model) and the values its load changed with
the owner's, and checks that no meta tensor survived install; then every rank votes,
so no rank serves a model that any rank refused. A raw load is the reference and
only votes.
"""

from __future__ import annotations

import datetime
import gc

import blake3
import torch
import torch.distributed as dist
from torch import nn

from sglang.kernels.ops.memory.gpu_tensor_hash_blake3 import gpu_tensors_blake3
from sglang.srt.model_loader.snapshot_walk import (
    TENSOR_KINDS,
    all_attributes,
    canonical,
    walk,
)

# Matches the engine's post-load barrier (UNBALANCED_MODEL_LOADING_TIMEOUT_S).
_VOTE_TIMEOUT_S = 480
# Arbitrary cap on the lines one rank reports.
_MAX_REPORTED = 50


def fingerprint(model: nn.Module, *, expert_map: torch.Tensor | None) -> dict[str, str]:
    """One text entry per path of walk(model); equal entries mean equal state.

    A tensor entry is "kind class dtype shape strides device rg=0|1
    @storage+byte_offset digest". The digest covers the logical bytes only, so the
    fields before it record what it cannot see. No field is an address.
    """
    entries: dict[str, str] = {}
    tensors: list[tuple[str, torch.Tensor]] = []
    storages: dict[tuple, str] = {}
    for path, kind, v in walk(model):
        if kind not in TENSOR_KINDS:
            entries[path] = canonical(v)
            continue
        layout = (
            f"{kind} {type(v).__qualname__} {v.dtype} {tuple(v.shape)} {v.stride()}"
        )
        if v.is_meta:
            entries[path] = f"{layout} meta"
            continue
        # The first path that reaches an allocation names it, so aliases agree.
        storage = storages.setdefault(_storage_key(v, path), path)
        offset = v.storage_offset() * v.element_size()
        entries[path] = (
            f"{layout} {v.device.type} rg={int(v.requires_grad)} @{storage}+{offset}"
        )
        tensors.append((path, v))
    for (path, _), digest in zip(tensors, _blake3([t for _, t in tensors])):
        entries[path] += f" {digest.hex()}"
    if expert_map is None:  # a dense model
        entries["__expert_map__"] = "none"
    else:
        entries["__expert_map__"] = _blake3([expert_map])[0].hex()
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

    On a snapshot reader the only meta tensors are the skeleton's, so a survivor is
    a reference that install did not replace. Using it does not fail: a custom op
    given a meta input runs its Meta kernel and silently does nothing.
    """
    gc.collect()
    # type() rather than isinstance(): isinstance reads __class__, which deprecated
    # torch objects turn into a warning.
    return [
        f"{type(o).__qualname__}{tuple(o.shape)}"
        for o in gc.get_objects()
        if issubclass(type(o), torch.Tensor) and o.is_meta
    ]


class Validator:
    """The checks of one load, on one rank.

    Every rank of the group that builds the model together calls them in the same
    order, so the votes line up.
    """

    def __init__(self, *, cpu_group: dist.ProcessGroup) -> None:
        self.cpu_group = cpu_group
        self.constructed: dict[str, str] = {}

    def record_construction(self, model: nn.Module) -> None:
        """Called right after the model is built: on the device for a raw load, on
        meta for a snapshot. Paths, addresses and handles that differ per process are
        the same before and after a load, so load_changes() leaves them out."""
        self.constructed = all_attributes(model)

    def load_changes(self, model: nn.Module) -> dict[str, str]:
        """Every value the load changed since record_construction(), from
        all_attributes(): a rule independent of the walk that export records with.
        The owner exports it; a reader must make exactly the same changes."""
        now, before = all_attributes(model), self.constructed
        paths = before.keys() | now.keys()
        return {p: now.get(p, "absent") for p in paths if before.get(p) != now.get(p)}

    def check_after_load(
        self,
        *,
        model: nn.Module,
        owner_fingerprint: dict[str, str] | None,
        owner_changes: dict[str, str] | None,
        expert_map: torch.Tensor | None,
    ) -> None:
        """On a snapshot, compare with the owner and scan for meta tensors; then vote.

        The owner's fingerprint and load changes are None for a raw load, which only
        votes. Raises on every rank if any rank refused.
        """
        problem = None
        if owner_fingerprint is not None:
            try:
                here = fingerprint(model, expert_map=expert_map)
                lines = compare(owner=owner_fingerprint, here=here)
                lines += compare(owner=owner_changes, here=self.load_changes(model))
                lines += [
                    f"{t}: still on meta after install" for t in live_meta_tensors()
                ]
            except Exception as e:  # a check that fails refuses; every rank still votes
                lines = [f"check failed: {e!r}"]
            if lines:
                problem = f"{len(lines)} problems\n" + "\n".join(lines[:_MAX_REPORTED])
        dist.monitored_barrier(
            group=self.cpu_group,
            timeout=datetime.timedelta(seconds=_VOTE_TIMEOUT_S),
            wait_all_ranks=True,
        )
        ballots: list[str | None] = [None] * dist.get_world_size(self.cpu_group)
        dist.all_gather_object(ballots, problem, group=self.cpu_group)
        refusals = [f"[rank {r}] {p}" for r, p in enumerate(ballots) if p is not None]
        if refusals:
            raise RuntimeError("Snapshot load refused:\n" + "\n".join(refusals))


def _storage_key(t: torch.Tensor, path: str) -> tuple:
    # Groups the paths that view one allocation; the address never enters an entry.
    storage = t.untyped_storage()
    if storage.nbytes() == 0:  # every empty CUDA tensor reports data_ptr() == 0
        return ("empty", path)
    return (t.device.type, t.device.index, storage.data_ptr())


def _blake3(tensors: list[torch.Tensor]) -> list[bytes]:
    # gpu_tensors_blake3 takes one CUDA device per call and rejects host tensors;
    # the blake3 package gives host tensors the same digest.
    by_device: dict[torch.device, list[int]] = {}
    for i, t in enumerate(tensors):
        by_device.setdefault(t.device, []).append(i)
    digests = [b""] * len(tensors)
    for device, indices in by_device.items():
        batch = [tensors[i] for i in indices]
        if device.type == "cuda":
            hashed = gpu_tensors_blake3(batch)
        else:
            hashed = [_host_blake3(t) for t in batch]
        for i, digest in zip(indices, hashed):
            digests[i] = digest
    return digests


def _host_blake3(t: torch.Tensor) -> bytes:
    data = t.detach().resolve_conj().resolve_neg().contiguous().reshape(-1)
    return blake3.blake3(data.view(torch.uint8).cpu().numpy().tobytes()).digest()
