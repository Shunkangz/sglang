# SPDX-License-Identifier: Apache-2.0
"""Native NCCL collectives as chunks for ``gated_transfer.TransferChain``."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.distributed.device_communicators.gated_transfer import (
    ChunkOp,
    chunk_spans,
)

if TYPE_CHECKING:
    from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator


def _overlaps(a: torch.Tensor, b: torch.Tensor) -> bool:
    return bool(a.numel() and b.numel()) and max(a.data_ptr(), b.data_ptr()) < min(
        a.data_ptr() + a.nbytes, b.data_ptr() + b.nbytes
    )


def _all_gather_chunk(comm, output, input_, stage, copies, stream):
    # The feeder may invoke this after its caller's context has exited.
    with torch.cuda.stream(stream):
        if stage is not None:
            input_.copy_(stage, non_blocking=True)
        if comm.world_size == 1:
            output.copy_(input_, non_blocking=True)
        else:
            with comm.change_state(enable=True):
                comm.all_gather(output, input_)
        for dst, src in copies:
            dst.copy_(src, non_blocking=True)


def all_gather_chunks(
    comm: PyNcclCommunicator,
    output: torch.Tensor,
    input_: torch.Tensor,
    *,
    scratch: Optional[torch.Tensor] = None,
    chunk_bytes: Optional[int] = None,
) -> List[ChunkOp]:
    """Build chunks gathering equal contiguous inputs into rank-major output.

    Each chunk calls native NCCL, which owns remote access and synchronization.
    Pass the result to a ``TransferChain`` to choose whether it is gated. The
    communicator must be reserved for that chain, and every rank must submit
    matching inputs and chunk sizes in the same order. A paused rank can hold
    its peers inside NCCL; warm up the communicator before any gate can park.

    ``chunk_bytes`` bounds each rank's contribution per collective. A whole
    input that fits is gathered directly, without scratch. Otherwise provide
    contiguous ``scratch`` of the same dtype/device with at least
    ``comm.world_size * (chunk_bytes // input_.element_size())`` elements.
    Allocate/register scratch before submission; NCCL's copy-engine path needs
    a symmetric window, using the existing ``allocate_symmetric_buffer`` API.
    A single-rank communicator performs one local copy; ``chunk_bytes`` does
    not split that copy because no collective is issued.

    A chunk stages its input into compact scratch, gathers there in place,
    then copies each rank's result to the corresponding slice of ``output``.
    This preserves ordinary allgather layout, including a short final chunk.
    Input may be exactly this rank's block of output; other overlaps, and any
    overlap with scratch, are rejected. Scratch is reused serially by chunks
    on one chain and must not be shared by concurrently executing chains.

    Views and closures are created here, before execution. The returned chunks
    retain the tensors; the caller must also keep them alive until the run has
    landed, since the chain may release a callable after enqueueing its work.
    No allocation or host/device synchronization occurs inside a chunk.
    """
    if not input_.is_contiguous() or not output.is_contiguous():
        raise ValueError("allgather input and output must be contiguous")
    if input_.dtype != output.dtype or input_.device != output.device:
        raise ValueError("allgather input and output must have the same dtype/device")
    world, rank, count = comm.world_size, comm.rank, input_.numel()
    if world < 1 or not 0 <= rank < world:
        raise ValueError("invalid communicator rank/world size")
    if world > 1 and input_.device != torch.device(comm.device):
        raise ValueError("allgather tensors must be on the communicator's device")
    if output.numel() != world * count:
        raise ValueError("allgather output must contain world_size * input elements")
    element_bytes = input_.element_size()
    if chunk_bytes is not None and (chunk_bytes <= 0 or chunk_bytes % element_bytes):
        raise ValueError("chunk_bytes must be a positive multiple of element size")
    if _overlaps(input_, output) and (
        input_.data_ptr() != output.data_ptr() + rank * input_.nbytes
    ):
        raise ValueError("input may overlap only its own complete output rank block")
    if count == 0:
        return []

    src, dst = input_.view(-1), output.view(-1)
    chunk_count = count if chunk_bytes is None else chunk_bytes // element_bytes
    if count <= chunk_count or world == 1:
        return [functools.partial(_all_gather_chunk, comm, dst, src, None, ())]
    if scratch is None:
        raise ValueError("chunked allgather requires preallocated scratch")
    if not scratch.is_contiguous():
        raise ValueError("allgather scratch must be contiguous")
    if scratch.dtype != input_.dtype or scratch.device != input_.device:
        raise ValueError("allgather scratch must match the input dtype/device")
    if scratch.numel() < world * chunk_count:
        raise ValueError("allgather scratch is too small for one chunk on every rank")
    if _overlaps(scratch, input_) or _overlaps(scratch, output):
        raise ValueError("allgather scratch must not overlap input or output")

    scratch = scratch.view(-1)
    chunks = []
    for offset, size in chunk_spans(count, chunk_count):
        gathered = scratch[: world * size]
        own = gathered[rank * size : (rank + 1) * size]
        copies = tuple(
            (
                dst[peer * count + offset : peer * count + offset + size],
                gathered[peer * size : (peer + 1) * size],
            )
            for peer in range(world)
        )
        chunks.append(
            functools.partial(
                _all_gather_chunk,
                comm,
                gathered,
                own,
                src[offset : offset + size],
                copies,
            )
        )
    return chunks
