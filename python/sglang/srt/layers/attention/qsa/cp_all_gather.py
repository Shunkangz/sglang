"""Copy-engine all-gathers of QSA prefill CP."""

from __future__ import annotations

import logging
from functools import partial
from typing import Callable, Optional, Tuple

import torch

from sglang.srt.distributed.device_communicators.zero_sm_all_gather import (
    ZeroSMAllGather,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.runtime_context import get_buffer, get_exec, get_parallel

logger = logging.getLogger(__name__)

# With fewer new tokens per rank the QSA layers are CPU-launch-bound, and the
# copy-engine path's host cost (NCCL's CE enqueue takes 46 us at 4 ranks and
# 74 us at 8, the ring kernel 22 us) outweighs its overlap. Measured on B200,
# CP4 and CP8: 2048 rows per rank lose 2-4%, 3072 win 3-5%.
_MIN_ROWS_PER_RANK = 3072


def qsa_cp_zero_sm_all_gather(
    x: torch.Tensor, forward_batch, key: str
) -> Optional[Callable[[torch.Tensor, int], torch.Tensor]]:
    """The copy-engine all-gather over the CP group for ``key``, or None to
    keep the group's SM all-gather. Every CP rank gets the same answer for a
    batch; the first CP prefill that uses it creates it, in the same layer on
    every rank."""
    if (
        not envs.SGLANG_ENABLE_QSA_CP_ZERO_SM_GATHER.get()
        or not x.is_cuda
        or max(forward_batch.attn_cp_metadata.per_rank_actual_token)
        < _MIN_ROWS_PER_RANK
        or get_is_capture_mode()
    ):
        return None
    (gather,) = get_buffer("qsa_cp_zero_sm_all_gather", _create)
    return None if gather is None else partial(gather.all_gather, key=key)


def _create() -> Tuple[Optional[ZeroSMAllGather]]:
    # ncclMemAlloc memory sits outside the memory saver's region and a custom
    # mem pool; the engine leaves NCCL cuMem off for them too.
    if (
        get_exec().features.enable_memory_saver
        or envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get() is not None
    ):
        gather = None
    else:
        gather = ZeroSMAllGather.create(get_parallel().attn_cp_group)
    logger.info(
        "QSA prefill CP gathers on %s.",
        "copy engines" if gather is not None else "the CP group's SM kernels",
    )
    # A 1-tuple, so an unsupported verdict is kept as well.
    return (gather,)
