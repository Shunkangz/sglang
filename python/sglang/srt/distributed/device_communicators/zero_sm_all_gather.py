"""All-gather on copy engines (zero SMs) over a dedicated NCCL communicator.

NCCL runs an all-gather on copy engines when its receive buffer lies in a
symmetric window of a communicator created with ``NCCL_CTA_POLICY_ZERO`` and
every rank sits in one NVLink domain. Each rank pushes its rows into every
peer's window between an entry and an exit barrier (stream memory operations),
so the transfer takes no SMs from kernels on other streams.
"""

from __future__ import annotations

import logging
import math
import os
from typing import TYPE_CHECKING, Dict, Optional

import torch

from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    allocate_symmetric_buffer,
)
from sglang.srt.distributed.device_communicators.pynccl_wrapper import (
    NCCL_CTA_POLICY_ZERO,
    NCCLLibrary,
)
from sglang.srt.distributed.parallel_state import in_the_same_node_as
from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator

logger = logging.getLogger(__name__)

# NCCL 2.31.2 fixes internal CE synchronization buffer initialization and the
# two-rank multicast-team setup race (upstream commit 430afbc). Zeroing our
# buffers cannot initialize NCCL's internal ones.
ZERO_SM_ALL_GATHER_MIN_NCCL_VERSION = 23102


def zero_sm_all_gather_nccl_supported() -> bool:
    """Whether the NCCL library PyNccl loads can run zero-SM all-gathers."""
    return (
        is_cuda()
        and NCCLLibrary().ncclGetRawVersion() >= ZERO_SM_ALL_GATHER_MIN_NCCL_VERSION
    )


def zero_sm_all_gather_supported(group: GroupCoordinator) -> bool:
    """Collective over ``group``: every rank returns the same verdict."""
    if not zero_sm_all_gather_nccl_supported():
        return False
    # Across nodes NCCL would run the windowed allgather on its hierarchical
    # path (inter-node RMA proxy puts); only the in-node CE path is verified.
    return all(in_the_same_node_as(group.cpu_group, source_rank=0))


class ZeroSMAllGather:
    """In-place all-gathers into symmetric windows on a dedicated communicator.

    Each ``key`` owns one window. It grows to the largest gather of that key;
    every rank asks for the same size, so the ranks grow it in the same call.
    The communicator runs its gathers in issue order whatever their streams,
    so a caller needs one key per result that is read while another gathers.
    """

    def __init__(self, group: GroupCoordinator):
        self.comm = PyNcclCommunicator(
            group=group.cpu_group,
            device=group.device,
            cta_policy=NCCL_CTA_POLICY_ZERO,
        )
        self.rank = self.comm.rank
        self.world_size = self.comm.world_size
        self._windows: Dict[str, torch.Tensor] = {}

    @classmethod
    def create(cls, group: GroupCoordinator) -> Optional[ZeroSMAllGather]:
        """None unless ``group``'s all-gathers can run on copy engines.

        Collective over ``group``: every rank gets the same answer.
        """
        # Without cuMem, ncclMemAlloc is a cudaMalloc and NCCL returns no
        # window: the gathers would run on SM kernels of a second communicator.
        if os.environ.get("NCCL_CUMEM_ENABLE") == "0":
            return None
        if not zero_sm_all_gather_supported(group):
            return None
        gather = cls(group)
        probe, registered = allocate_symmetric_buffer(gather.comm, 1)
        del probe
        if not registered:
            logger.warning(
                "NCCL returned no symmetric window (no CUDA P2P between the "
                "ranks?); the all-gather keeps NCCL SM kernels."
            )
            gather.comm.nccl.ncclCommDestroy(gather.comm.comm)
            return None
        return gather

    def all_gather(self, x: torch.Tensor, rows_per_rank: int, key: str) -> torch.Tensor:
        """Every rank's ``x`` rank-major in ``[world_size * rows_per_rank, ...]``.

        ``x`` holds at most ``rows_per_rank`` rows; the rows after it in each
        rank's block are stale. The result is a view of ``key``'s window, which
        the next gather with ``key`` overwrites: read it in stream order first.
        """
        shape = (self.world_size * rows_per_rank, *x.shape[1:])
        nbytes = math.prod(shape) * x.element_size()
        out = self._window(key, nbytes)[:nbytes].view(x.dtype).view(shape)
        send = out[self.rank * rows_per_rank : (self.rank + 1) * rows_per_rank]
        send[: x.shape[0]].copy_(x)
        with self.comm.change_state(enable=True):
            self.comm.all_gather(out, send)
        return out

    def _window(self, key: str, nbytes: int) -> torch.Tensor:
        window = self._windows.get(key)
        if window is not None and window.numel() >= nbytes:
            return window
        # Dropping the last reference deregisters and frees the old window
        # before the new one is registered.
        self._windows.pop(key, None)
        del window
        window, registered = allocate_symmetric_buffer(
            self.comm, 1 << (nbytes - 1).bit_length()
        )
        assert registered, "the probe in create() got a window on this comm"
        self._windows[key] = window
        return window
