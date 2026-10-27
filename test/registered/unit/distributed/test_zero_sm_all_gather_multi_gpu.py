"""Multi-GPU tests for ZeroSMAllGather and its QSA prefill-CP K/V gather.

Run directly with:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 \
        test/registered/unit/distributed/test_zero_sm_all_gather_multi_gpu.py
"""

import gc
import tempfile
import unittest
import weakref
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp

from sglang.srt.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.distributed.device_communicators.pynccl_wrapper import NCCLLibrary
from sglang.srt.distributed.device_communicators.zero_sm_all_gather import (
    ZERO_SM_ALL_GATHER_MIN_NCCL_VERSION,
    ZeroSMAllGather,
)
from sglang.srt.layers.attention import qwen_sparse_attn_backend
from sglang.srt.layers.cp import utils as cp_utils
from sglang.srt.layers.cp.base import get_cp_strategy, init_cp_strategy
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel, publish
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import temp_set_env
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, publish_build_topology

register_cuda_ci(est_time=60, stage="base-b", runner_config="2-gpu-large")
register_cuda_ci(est_time=60, stage="base-c", runner_config="4-gpu-b200")


def _rows(rank, num_rows, width, dtype, generation):
    """Rows rank ``rank`` contributes, distinct per rank, row and generation."""
    values = torch.arange(num_rows * width, device="cuda", dtype=torch.int64)
    values = (values * 7 + rank * 1009 + generation * 31) % 251
    return values.reshape(num_rows, width).to(dtype)


def _check_gather(gather, world, rank, rows_per_rank, width, dtype, key, generation):
    """Rank r sends rows_per_rank - r rows; returns the gathered view."""
    out = gather.all_gather(
        _rows(rank, rows_per_rank - rank, width, dtype, generation),
        rows_per_rank,
        key,
    )
    assert out.shape == (world * rows_per_rank, width), out.shape
    for peer in range(world):
        start = peer * rows_per_rank
        got = out[start : start + rows_per_rank - peer]
        expected = _rows(peer, rows_per_rank - peer, width, dtype, generation)
        assert torch.equal(got, expected), (rank, peer, dtype, rows_per_rank, key)
    return out


def _run(rank, world, init_method):
    torch.cuda.set_device(rank)
    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=rank,
        distributed_init_method=init_method,
        backend="nccl",
    )
    publish(
        ServerArgs(model_path="dummy", tp_size=world, attn_cp_size=world),
        role="scheduler",
    )
    publish_build_topology(tp_size=world, attn_cp_size=world, world_rank=rank)
    initialize_model_parallel()
    group = get_parallel().attn_cp_group
    gather = ZeroSMAllGather.create(group)
    assert gather is not None, "NCCL returned no symmetric window"

    # Uneven contributions, dtypes and growing windows; generations catch stale
    # peer data in a reused window.
    for generation, (rows_per_rank, dtype) in enumerate(
        [
            (3, torch.bfloat16),
            (3, torch.bfloat16),
            (1000, torch.float32),
            (37, torch.uint8),
        ]
    ):
        _check_gather(gather, world, rank, rows_per_rank, 64, dtype, "a", generation)

    # Growing a key's window releases the old one.
    old = weakref.ref(gather._windows["a"])
    _check_gather(gather, world, rank, 1 << 15, 64, torch.bfloat16, "a", 9)
    gc.collect()
    assert old() is None, "the outgrown window is still alive"

    # Keys own separate windows: a later gather on "b" leaves "a" intact.
    a = _check_gather(gather, world, rank, 512, 32, torch.bfloat16, "a", 10)
    a_before = a.clone()
    _check_gather(gather, world, rank, 512, 32, torch.bfloat16, "b", 11)
    torch.cuda.synchronize()
    assert torch.equal(a, a_before)

    # Copy engines only: no NCCL kernel, at least one peer copy.
    x = _rows(rank, 4096, 64, torch.bfloat16, 12)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        gather.all_gather(x, 4096, "a")
        torch.cuda.synchronize()
    activity = [
        event.name
        for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    assert not any("nccl" in name.lower() for name in activity), activity
    assert any("PtoP" in name for name in activity), activity

    _check_qsa_kv_gather(world, rank)
    torch.cuda.synchronize()
    torch.distributed.barrier()


def _check_qsa_kv_gather(world, rank):
    """QSA's K/V gather returns the global K/V; batches with few new tokens per
    rank keep the CP group's SM gather."""
    init_cp_strategy(enable_prefill_cp=True, cp_size=world, cp_strategy="zigzag")
    min_rows = cp_utils._ZERO_SM_MIN_ROWS_PER_RANK
    for extend_lens, zero_sm in (
        ([37, 64 * world + 3], False),
        ([37, min_rows * world + 3], True),
    ):
        _compare_qsa_kv_gather(world, extend_lens, zero_sm)


def _compare_qsa_kv_gather(world, extend_lens, zero_sm):
    strategy = get_cp_strategy()
    num_tokens = sum(extend_lens)
    layer = SimpleNamespace(
        tp_k_head_num=2, qk_head_dim=16, tp_v_head_num=2, v_head_dim=8
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        input_ids=torch.zeros(num_tokens, dtype=torch.int64, device="cuda"),
        extend_seq_lens_cpu=extend_lens,
        attn_cp_metadata=strategy.build_metadata(num_tokens, extend_lens, extend_lens),
    )
    # The same seed gives every rank the same global K/V.
    torch.manual_seed(0)
    k = torch.randn(num_tokens, 2, 16, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(num_tokens, 2, 8, device="cuda", dtype=torch.bfloat16)
    local_k = strategy.shard_hidden_states(k, batch)
    local_v = strategy.shard_hidden_states(v, batch)
    # The first call may create the communicator (a warmup all-reduce) and window.
    qwen_sparse_attn_backend._gather_cp_kv(layer, local_k, local_v, batch)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        full_k, full_v = qwen_sparse_attn_backend._gather_cp_kv(
            layer, local_k, local_v, batch
        )
        torch.cuda.synchronize()
    activity = [
        event.name
        for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    on_copy_engines = not any("nccl" in name.lower() for name in activity)
    assert on_copy_engines == zero_sm, (extend_lens, activity)
    if zero_sm:
        assert any("PtoP" in name for name in activity), activity
    assert torch.equal(full_k, k) and torch.equal(full_v, v)


@unittest.skipIf(torch.cuda.device_count() < 2, "needs 2 CUDA devices")
class TestZeroSMAllGatherMultiGpu(CustomTestCase):
    def _spawn(self, world):
        if torch.cuda.device_count() < world:
            self.skipTest(f"needs {world} CUDA devices")
        # Gate on the library PyNccl loads, which may differ from torch's.
        if NCCLLibrary().ncclGetRawVersion() < ZERO_SM_ALL_GATHER_MIN_NCCL_VERSION:
            self.skipTest("the zero-SM all-gather needs NCCL >= 2.31.2")
        # A global NCCL_CTA_POLICY would override the communicator's policy.
        with (
            temp_set_env(NCCL_CUMEM_ENABLE="1", NCCL_CTA_POLICY=None),
            tempfile.TemporaryDirectory(prefix="sglang-zero-sm-ag-") as tmp,
        ):
            mp.spawn(
                _run,
                args=(world, f"file://{tmp}/rendezvous"),
                nprocs=world,
                join=True,
            )

    def test_two_ranks(self):
        self._spawn(2)

    def test_four_ranks(self):
        self._spawn(4)


if __name__ == "__main__":
    unittest.main()
