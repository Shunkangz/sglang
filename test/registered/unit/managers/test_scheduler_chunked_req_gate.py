"""Regression tests for the SWA chunked-req stash gate (#24252)."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import (
    NextBatchPlan,
    Req,
    ReqKvInfo,
    ScheduleBatch,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.allocator.page_interleave import PageInterleavePoolAllocator
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.page_interleave import PageShardSpec
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils.common import Range

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _make_req(
    *,
    req_pool_idx: int,
    fill_ids: list,
    prefix_indices: torch.Tensor,
    extend_input_len: int,
    fill_len: int,
) -> Req:
    req = Req.__new__(Req)
    req.rid = "test-req"
    req.origin_input_ids = array("q", fill_ids)
    req.output_ids = array("q")
    req.full_untruncated_fill_ids = array("q", fill_ids)
    req.prefix_indices = prefix_indices
    req.extend_range = Range(fill_len - extend_input_len, fill_len)
    req.inflight_middle_chunks = 0
    req.host_hit_length = 0
    req.kv = ReqKvInfo(req_pool_idx=req_pool_idx)
    req.skip_radix_cache_insert = False
    req.last_node = None
    req.lock_receipt = DecLockRefParams()
    req.session = None
    req.return_logprob = False
    req.logprob_start_len = -1
    req.positional_embed_overrides = None
    req.extra_key = None
    req.cache_salt = None
    req.kv.mamba_pool_idx = None
    req.sampling_params = SimpleNamespace(max_new_tokens=128, ignore_eos=False)
    return req


def _make_req_to_token_pool(num_slots: int, max_context: int) -> SimpleNamespace:
    # Slot s contains a recognizable fingerprint [s*1000, s*1000+1, ...]
    # so we can tell a corrupted prefix_indices from a healthy one by content.
    pool = SimpleNamespace()
    pool.req_to_token = (
        torch.arange(max_context, dtype=torch.int32).unsqueeze(0).repeat(num_slots, 1)
        + torch.arange(num_slots, dtype=torch.int32).unsqueeze(1) * 1000
    )
    return pool


def _make_chunk_cache(req_to_token_pool) -> ChunkCache:
    return ChunkCache(
        SimpleNamespace(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=None,
            page_size=1,
        )
    )


def _scheduler_for_get_next_batch(*, tree_cache, chunked_req) -> Scheduler:
    s = Scheduler.__new__(Scheduler)
    s.scheduler_stage_metrics = None
    s.dllm_config = None
    s.dllm_manager = None
    s.enable_hisparse = False
    s.enable_fpm = False
    # Exercise the unconditional scheduler-loop HiCache event-drain point.
    s.enable_hierarchical_cache = True
    s.enable_hicache_storage = False
    s.enable_unified_cache_external_linker = False
    s.last_batch = None
    s.require_mlp_sync = False
    s.spec_algorithm = MagicMock()
    s.server_args = MagicMock(speculative_skip_dp_mlp_sync=True)
    s.running_batch = MagicMock()
    s.running_batch.is_empty.return_value = True
    s.running_batch.is_prefill_only = False
    s.running_batch.batch_is_full = False
    s.running_batch.reqs = []
    s.prefill_decode_interval = 0
    s._prefill_decode_interval_remaining = 0
    s.get_new_batch_prefill = MagicMock(
        return_value=NextBatchPlan(batch_to_run=None, running_batch=s.running_batch)
    )
    s.dp_attn_adapter = MagicMock()
    s.dp_attn_adapter.maybe_prepare_mlp_sync_batch = MagicMock(
        side_effect=lambda batch, **_: batch
    )
    s.ngram_embedding_manager = MagicMock()
    s.ngram_embedding_manager.prepare_for_forward = MagicMock(
        side_effect=lambda batch, **_: batch
    )
    s.update_running_batch = MagicMock(side_effect=lambda batch: batch)
    tree_cache.check_hicache_events = MagicMock()
    s.tree_cache = tree_cache
    s.chunked_req = chunked_req
    s._pending_chunked_abort_req = None
    return s


class TestStashGatePreservesPrefixIndices(CustomTestCase):
    """Consumer side: real ChunkCache.cache_unfinished_req mutates
    req.prefix_indices iff stash actually runs, so prefix_indices content
    is the bug-detection signal. The stash gate is content-based:
    `fill_len > len(prefix_indices)` means there is freshly computed KV to
    cache; otherwise the chunk was parked and stashing must be skipped."""

    POOL_IDX = 4
    INITIAL_PREFIX_LEN = 8  # what was really cached last iter
    POST_RESET_FILL_LEN = 32  # length after init_next_round_input rebuilds
    NUM_SLOTS = 8
    MAX_CONTEXT = 64

    def _build(self, *, fill_len: int):
        pool = _make_req_to_token_pool(self.NUM_SLOTS, self.MAX_CONTEXT)
        cache = _make_chunk_cache(pool)
        initial_prefix = pool.req_to_token[self.POOL_IDX, : self.INITIAL_PREFIX_LEN].to(
            dtype=torch.int64, copy=True
        )
        req = _make_req(
            req_pool_idx=self.POOL_IDX,
            fill_ids=list(range(self.POST_RESET_FILL_LEN)),
            prefix_indices=initial_prefix,
            extend_input_len=fill_len - self.INITIAL_PREFIX_LEN,
            fill_len=fill_len,
        )
        s = _scheduler_for_get_next_batch(tree_cache=cache, chunked_req=req)
        return s, req, initial_prefix, pool

    def test_parked_chunked_req_keeps_real_prefix_indices(self):
        # A parked chunk has fill_len == len(prefix_indices): no new KV was
        # computed, so the gate must skip stash and leave prefix_indices intact.
        s, req, initial_prefix, _ = self._build(fill_len=self.INITIAL_PREFIX_LEN)

        Scheduler.get_next_batch_to_run(
            s, running_batch=s.running_batch, last_batch=s.last_batch
        )

        self.assertEqual(req.prefix_indices.shape[0], self.INITIAL_PREFIX_LEN)
        self.assertTrue(torch.equal(req.prefix_indices, initial_prefix))

    def test_scheduled_chunked_req_advances_prefix_indices_via_real_stash(self):
        # Symmetric guard against over-gating: when fill_len has advanced past
        # the cached prefix, stash must run and advance prefix_indices.
        s, req, _, pool = self._build(fill_len=self.POST_RESET_FILL_LEN)

        Scheduler.get_next_batch_to_run(
            s, running_batch=s.running_batch, last_batch=s.last_batch
        )

        expected = pool.req_to_token[self.POOL_IDX, : self.POST_RESET_FILL_LEN].to(
            dtype=torch.int64
        )
        self.assertEqual(req.prefix_indices.shape[0], self.POST_RESET_FILL_LEN)
        self.assertTrue(torch.equal(req.prefix_indices, expected))

    def test_no_chunked_req_never_mutates_state(self):
        # The outer `if chunked_req is not None` guard must hold on the retract
        # path that clears chunked_req.
        pool = _make_req_to_token_pool(self.NUM_SLOTS, self.MAX_CONTEXT)
        cache = _make_chunk_cache(pool)
        s = _scheduler_for_get_next_batch(tree_cache=cache, chunked_req=None)

        Scheduler.get_next_batch_to_run(
            s, running_batch=s.running_batch, last_batch=s.last_batch
        )
        self.assertIsNone(s.chunked_req)


class TestShardedParkedChunk(CustomTestCase):
    def test_other_class_request_runs_without_counting_parked_chunk(self):
        """Run real admission and batch construction; stop before forward setup."""
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        allocator = PageInterleavePoolAllocator(
            size=32,
            physical_page_size=16,
            shard_size=4,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=None,
            need_sort=False,
            shard_spec=PageShardSpec(
                shard_rank=0,
                shard_size=4,
                page_size=16,
                max_prefix_tokens=64,
                chunk_tokens=64,
            ),
        )

        def allocate_page(owner):
            return allocator.alloc_extend(
                prefix_lens=torch.tensor([0]),
                prefix_lens_cpu=torch.tensor([0]),
                seq_lens=torch.tensor([16]),
                seq_lens_cpu=torch.tensor([16]),
                last_loc=torch.tensor([-1]),
                extend_num_tokens=16,
                rotation_bases=[owner],
            )

        prefix = allocate_page(3)
        allocate_page(0)
        allocate_page(0)
        self.assertEqual(allocator.class_free_page_counts(), [0, 2, 2, 1])

        pool = _make_req_to_token_pool(8, 64)
        pool.device = torch.device("cpu")
        pool.available_size = lambda: 7
        pool.mamba_allocator = None
        pool.req_to_token[0, :16] = prefix.to(torch.int32)
        cache = ChunkCache(
            SimpleNamespace(
                req_to_token_pool=pool,
                token_to_kv_pool_allocator=allocator,
                page_size=16,
            )
        )
        parked = Req(
            "parked",
            "",
            array("q", range(32)),
            SamplingParams(max_new_tokens=1),
        )
        parked.kv.req_pool_idx = 0
        parked.prefix_indices = prefix
        parked.kv_rotation_base = 3
        parked.set_extend_range(0, 16)
        waiting = Req(
            "other-class",
            "",
            array("q", range(15)),
            SamplingParams(max_new_tokens=1),
        )

        scheduler = Scheduler.__new__(Scheduler)
        for flag in (
            "enable_hierarchical_cache",
            "enable_unified_cache_external_linker",
            "enable_hicache_storage",
            "enable_priority_preemption",
            "enable_priority_scheduling",
            "is_hybrid_swa",
            "enable_lora",
            "is_mixed_chunk",
            "enable_overlap",
        ):
            setattr(scheduler, flag, False)
        scheduler.grammar_manager = MagicMock()
        scheduler.grammar_manager.has_waiting_grammars.return_value = False
        scheduler.min_free_slots_delayer = None
        scheduler.dynamic_chunk_sizer = None
        scheduler.dllm_config = None
        scheduler.chunked_req = parked
        scheduler.waiting_queue = [waiting]
        scheduler.policy = MagicMock()
        scheduler.get_num_allocatable_reqs = MagicMock(return_value=7)
        scheduler.chunked_prefill_size = 64
        scheduler.page_size = 16
        scheduler.max_prefill_tokens = 64
        scheduler.max_prefill_bs = 8
        scheduler.max_running_requests = 8
        scheduler.priority_scheduling_preemption_threshold = 0
        scheduler.truncation_align_size = None
        scheduler.new_token_ratio_tracker = SimpleNamespace(current=1.0)
        scheduler.processed_tokens_counter = 0
        scheduler.tree_cache = cache
        scheduler.req_to_token_pool = pool
        scheduler.token_to_kv_pool_allocator = allocator
        scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        scheduler.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                attn_backend=SimpleNamespace(extend_attention_block_m=64),
                prefill_aware_swa=False,
            )
        )
        scheduler.model_config = SimpleNamespace(vocab_size=100)
        scheduler.spec_algorithm = MagicMock()
        scheduler.load_inquirer = MagicMock()
        scheduler.load_inquirer._get_num_pending_tokens.return_value = 16
        running_batch = ScheduleBatch(reqs=[], batch_is_full=False)

        # Admission, queue filtering, and ScheduleBatch.init_new all execute.
        # Tensor/sampling setup for the forward is outside this scheduler test.
        with patch.object(ScheduleBatch, "prepare_for_extend") as prepare:
            batch, _ = Scheduler._get_new_batch_prefill_raw(
                scheduler, None, running_batch
            )

        prepare.assert_called_once()
        self.assertEqual(batch.reqs, [waiting])
        self.assertEqual(scheduler.waiting_queue, [])
        self.assertIs(scheduler.chunked_req, parked)
        self.assertEqual(parked.extend_range, Range(16, 16))
        self.assertEqual(parked.inflight_middle_chunks, 0)
        self.assertIsNone(batch.chunked_req)
        self.assertIsNone(batch.chunked_req_next_prompt_token)
        self.assertTrue(batch.contains_last_prefill_chunk)
        self.assertEqual(waiting.kv_shard_admission_base, 1)
        self.assertEqual(allocator.class_free_page_counts(), [0, 2, 2, 1])


if __name__ == "__main__":
    unittest.main()
