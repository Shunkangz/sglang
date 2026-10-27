"""CPU checks for QSA's zigzag-only context parallel entrypoints."""

import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.layers.attention import qwen_sparse_attn_backend
from sglang.srt.layers.attention.qsa import qsa_indexer
from sglang.srt.layers.attention.qsa.mqa import _scoring_dtype, torch_qsa_mqa_prefill
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.layers.cp import base as cp_base
from sglang.srt.layers.cp import zigzag
from sglang.srt.layers.cp.base import ContextParallelStrategyKind
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_resources
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=13, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "strategy",
    [
        None,
        SimpleNamespace(name="interleave", kind=ContextParallelStrategyKind.INTERLEAVE),
    ],
)
@pytest.mark.parametrize("entrypoint", ["indexer", "attention"])
def test_qsa_cp_rejects_unsupported_strategy_before_metadata_or_collectives(
    monkeypatch, strategy, entrypoint
):
    module = qsa_indexer if entrypoint == "indexer" else qwen_sparse_attn_backend
    monkeypatch.setattr(module, "get_cp_strategy", lambda: strategy)
    gather = Mock(side_effect=AssertionError("unsupported CP must not communicate"))
    monkeypatch.setattr(module, "cp_materialize_global_token_order", gather)
    # None metadata and tensors ensure rejection precedes zigzag-specific access.
    with pytest.raises(
        NotImplementedError, match="QSA prefill CP only supports the zigzag strategy"
    ):
        if entrypoint == "indexer":
            QSAIndexer.forward_cuda_cp(None, None, None, None, None, None)
        else:
            QwenSparseAttnBackend._forward_extend_cp(
                None, None, None, None, None, None, None, False
            )
    gather.assert_not_called()


@pytest.mark.parametrize(
    "cp_enabled,hybrid_backend,breakable,capture_mode",
    [
        (True, False, False, False),
        (True, True, False, False),
        (False, True, False, False),
        (True, True, True, False),
        (False, True, True, False),
        (True, True, False, True),
        (False, True, False, True),
    ],
)
def test_qsa_model_forward_prefetches_exact_kv_before_indexer(
    monkeypatch, cp_enabled, hybrid_backend, breakable, capture_mode
):
    from sglang.srt.models import qwen4_exp

    decoder = qwen4_exp.Qwen4ExpAttentionDecoderLayer
    hidden = torch.randn(4, 8)
    positions = torch.arange(4)
    q, k, v = (torch.randn(4, 8) for _ in range(3))
    topk = torch.arange(8, dtype=torch.int32).reshape(4, 2)
    expected = torch.randn(4, 8)
    metadata = object()
    batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND, positions=positions)
    events = []
    prefetch_expected = cp_enabled and not (breakable or capture_mode)

    def prefetch(attn, actual_k, actual_v, actual_batch):
        assert attn is layer.attn
        assert actual_k is k and actual_v is v
        assert actual_batch is batch
        events.append("prefetch")

    sparse_backend = SimpleNamespace(
        set_mtp_shared_sparse_indices=Mock(),
        get_indexer_metadata=Mock(return_value=metadata),
        prefetch_cp_kv=Mock(side_effect=prefetch),
    )
    backend = (
        SimpleNamespace(full_attn_backend=sparse_backend)
        if hybrid_backend
        else sparse_backend
    )
    monkeypatch.setattr(qwen4_exp, "get_attn_backend", lambda: backend)
    monkeypatch.setattr(qwen4_exp, "_shards_rows_over_cp", lambda _: cp_enabled)
    monkeypatch.setattr(qwen4_exp, "is_in_breakable_cuda_graph", lambda: breakable)
    monkeypatch.setattr(
        qwen4_exp, "get_is_capture_mode", lambda: breakable or capture_mode
    )
    monkeypatch.setattr(
        qwen4_exp,
        "get_tc_piecewise_forward_context",
        lambda: SimpleNamespace(forward_batch=batch),
    )

    def prepare(**kwargs):
        events.append("prepare")
        return q, k, v, None

    def prefetch_keys(actual_hidden, actual_batch):
        assert actual_hidden is hidden and actual_batch is batch
        events.append("prefetch_keys")

    def run_indexer(
        actual_hidden, actual_positions, actual_batch, actual_meta, **kwargs
    ):
        assert actual_hidden is hidden and actual_positions is positions
        assert actual_batch is batch and actual_meta is metadata
        assert ("cp_global_rope_positions" in kwargs) == cp_enabled
        if cp_enabled:
            assert kwargs["cp_global_rope_positions"] is batch.positions
        assert set(kwargs) == ({"cp_global_rope_positions"} if cp_enabled else set())
        events.append("indexer")
        return topk

    def attention(actual_q, actual_k, actual_v, actual_batch, **kwargs):
        assert actual_q is q and actual_k is k and actual_v is v
        assert actual_batch is batch
        torch.testing.assert_close(kwargs["topk_indices"], topk)
        events.append("attention")
        return expected

    layer = SimpleNamespace(
        is_qsa=True,
        layer_id=7,
        alt_stream=None,
        _qsa_prefill_topk_bridges=[],
        _prepare_attn_stage=lambda value, *args: value,
        _prepare_qkv_gate=prepare,
        indexer=Mock(side_effect=run_indexer, prefetch_cp_keys=prefetch_keys),
        attn=attention,
        o_proj=lambda value: (value, None),
        attn_boundary=SimpleNamespace(finish=lambda value, _: value),
        _run_ffn_stage=lambda value, _: value,
    )
    for name in (
        "self_attention",
        "_compute_qsa_topk_indices",
        "_compute_qsa_topk_indices_eager",
    ):
        setattr(layer, name, getattr(decoder, name).__get__(layer))

    assert decoder.forward(layer, positions, hidden, batch) is expected
    # The key gather runs under the qkv projection, the K/V gather after it.
    assert events == [
        *(["prefetch_keys"] if prefetch_expected else []),
        "prepare",
        *(["prefetch"] if prefetch_expected else []),
        "indexer",
        "attention",
    ]
    assert sparse_backend.prefetch_cp_kv.call_count == int(prefetch_expected)


def test_cuda_indexer_forwards_global_rope_to_cp_path():
    hidden, local_rope, metadata, global_rope = (object() for _ in range(4))
    batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
    expected = object()
    indexer = SimpleNamespace(forward_cuda_cp=Mock(return_value=expected))
    indexer._forward_impl = lambda *args: QSAIndexer._forward_impl(indexer, *args)

    actual = QSAIndexer.forward_cuda(
        indexer,
        hidden,
        local_rope,
        batch,
        metadata,
        cp_global_rope_positions=global_rope,
    )

    assert actual is expected
    indexer.forward_cuda_cp.assert_called_once_with(
        hidden, local_rope, batch, metadata, global_rope
    )


@pytest.mark.parametrize("prefetch", ["none", "this_batch", "stale"])
@pytest.mark.parametrize("compressed_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_cp_indexer_matches_compressed_dtype_without_storing_local_keys(
    monkeypatch, compressed_dtype, prefetch
):
    local_rows = torch.tensor([0, 1, 6, 7])
    hidden = (torch.arange(32).reshape(8, 4) / 13).to(torch.bfloat16)
    positions = torch.arange(8)
    compressed_keys = hidden.reshape(4, 2, 4).mean(1).unsqueeze(1).to(compressed_dtype)
    row_starts = torch.zeros(4, dtype=torch.int32)
    row_ends = (positions[local_rows] + 1).to(torch.int32) // 2
    pool = SimpleNamespace(qsa_compressed_dtype=compressed_dtype)
    metadata = SimpleNamespace(
        token_to_kv_pool=pool,
        pending_ring_slots=torch.arange(8),
        get_token_to_batch_idx=lambda: torch.zeros(8, dtype=torch.int32),
        prefill_max_row_keys=lambda *_: None,
        get_prefill_mqa_inputs=lambda *args, **kwargs: (
            compressed_keys,
            row_starts,
            row_ends,
            torch.tensor([8], dtype=torch.int32),
        ),
    )
    batch = SimpleNamespace(
        positions=positions,
        attn_cp_metadata=SimpleNamespace(total_q_prev_tokens=2, total_q_next_tokens=2),
    )
    monkeypatch.setattr(
        qsa_indexer,
        "get_cp_strategy",
        lambda: SimpleNamespace(kind=ContextParallelStrategyKind.ZIGZAG),
    )
    for name in ("cp_shard_position_ids", "cp_shard_hidden_states"):
        monkeypatch.setattr(qsa_indexer, name, lambda value, batch: value[local_rows])
    events = []

    def gather_raw_keys(local, batch, all_gather):
        torch.testing.assert_close(local, hidden[local_rows])
        # Inline, the key gather shares the CP group with the K/V prefetch.
        assert all_gather is None
        events.append("key_gather")
        return hidden

    monkeypatch.setattr(
        qsa_indexer, "cp_materialize_global_token_order", gather_raw_keys
    )

    def score_queries(
        q, k, starts, ends, query_positions, sequence_lengths, max_row_keys=None
    ):
        # Use the same operand check as GPU scoring; the CPU reference otherwise
        # accepts mixed dtypes and would conceal the FP8 compatibility failure.
        assert _scoring_dtype(q, k) == compressed_dtype
        return torch_qsa_mqa_prefill(q, k, starts, ends)

    indexer = SimpleNamespace(
        layer_id=0,
        index_n_heads=1,
        index_kv_heads=1,
        index_head_dim=4,
        index_qk_proj=lambda value: (torch.cat([value, value], dim=-1), None),
        q_layernorm=lambda value: value,
        apply_rope=Mock(side_effect=lambda positions, value, **kwargs: value),
        _use_fused_prep=Mock(
            side_effect=AssertionError("local keys must not be stored")
        ),
        update_key_state_and_compress=Mock(
            side_effect=lambda *args, **kwargs: events.append("compress")
        ),
        select_prefill_tokens=score_queries,
    )
    indexer._cp_key_pending = None
    for name in ("_project_index_q", "_gather_cp_keys"):
        setattr(indexer, name, getattr(QSAIndexer, name).__get__(indexer))
    main, side, ready = Mock(), Mock(), Mock()
    main.wait_event.side_effect = lambda event: events.append("join")
    side.wait_stream.side_effect = lambda stream: events.append("release")
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: main)
    monkeypatch.setattr(qsa_indexer, "get_stream", lambda name: side)
    if prefetch != "none":
        # What prefetch_cp_keys leaves: the projection and the gathered keys.
        qk = torch.cat([hidden[local_rows], hidden[local_rows]], dim=-1)
        owner = batch if prefetch == "this_batch" else SimpleNamespace()
        indexer._cp_key_pending = (owner, qk, hidden.unsqueeze(1), ready)
    if prefetch == "this_batch":
        indexer.index_qk_proj = Mock(side_effect=AssertionError("projected twice"))
    actual = QSAIndexer.forward_cuda_cp(
        indexer,
        hidden[local_rows],
        positions[local_rows],
        batch,
        metadata,
        positions,
    )
    # Compression needs the globally ordered raw keys on every rank: the
    # prefetched gather is joined before it, and a stale one is joined too.
    # The side stream frees the prefetched keys only after compression reads.
    assert (
        events
        == {
            "none": ["key_gather", "compress"],
            "this_batch": ["join", "compress", "release"],
            "stale": ["key_gather", "join", "compress", "release"],
        }[prefetch]
    )
    assert indexer._cp_key_pending is None
    if prefetch != "none":
        main.wait_event.assert_called_once_with(ready)
    expected = torch_qsa_mqa_prefill(
        hidden[local_rows].unsqueeze(1).to(compressed_dtype),
        compressed_keys,
        row_starts,
        row_ends,
    )
    torch.testing.assert_close(actual, expected)
    indexer._use_fused_prep.assert_not_called()
    # Sizing the RoPE cache reads positions.max() on the host: a device sync per layer.
    assert indexer.apply_rope.call_args.kwargs["ensure_cos_sin_cache"] is False
    update = indexer.update_key_state_and_compress.call_args
    torch.testing.assert_close(update.args[0], hidden.unsqueeze(1))
    assert update.kwargs["state_stored"] is False


@pytest.mark.parametrize("leftover", [False, True])
@pytest.mark.parametrize("zero_sm", [False, True])
def test_cp_indexer_prefetch_gathers_keys_on_the_side_stream(
    monkeypatch, zero_sm, leftover
):
    hidden = torch.randn(6, 4)
    batch = SimpleNamespace(
        attn_cp_metadata=SimpleNamespace(total_q_prev_tokens=2, total_q_next_tokens=2)
    )
    calls = Mock()
    side, main, ready = calls.side, calls.main, calls.ready
    in_side_stream = []
    all_gather = Mock()
    monkeypatch.setattr(
        qsa_indexer,
        "cp_zero_sm_all_gather",
        lambda x, actual_batch, key: (
            all_gather
            if zero_sm and actual_batch is batch and key == "index_k"
            else None
        ),
    )
    monkeypatch.setattr(qsa_indexer, "get_stream", lambda name: side)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: main)
    monkeypatch.setattr(torch.cuda, "Event", lambda: ready)

    @contextmanager
    def use_stream(stream):
        in_side_stream.append(stream is side)
        yield
        in_side_stream.append(False)

    monkeypatch.setattr(torch.cuda, "stream", use_stream)

    def gather(local, actual_batch, all_gather):
        calls.gather(local, all_gather)
        assert in_side_stream[-1], "the key gather must run on the side stream"
        return torch.cat([local, local])

    monkeypatch.setattr(qsa_indexer, "cp_materialize_global_token_order", gather)

    def project(value):
        calls.project()
        return torch.cat([value, 2 * value], dim=-1), None

    stale = calls.stale
    indexer = SimpleNamespace(
        index_n_heads=1,
        index_kv_heads=1,
        index_head_dim=4,
        index_qk_proj=project,
        # Left by a forward that failed before this layer's indexer.
        _cp_key_pending=(object(), None, None, stale) if leftover else None,
        _cp_key_ready=None,
    )
    indexer._gather_cp_keys = QSAIndexer._gather_cp_keys.__get__(indexer)
    QSAIndexer.prefetch_cp_keys(indexer, hidden, batch)
    if not zero_sm:
        # On the CP group's SM kernels the gather stays inline, behind the K/V one.
        assert calls.mock_calls == []
        return
    owner, qk, keys, event = indexer._cp_key_pending
    assert owner is batch and event is ready
    torch.testing.assert_close(qk, torch.cat([hidden[:4], 2 * hidden[:4]], dim=-1))
    torch.testing.assert_close(keys, torch.cat([2 * hidden[:4]] * 2).unsqueeze(1))
    # Join a leftover, fork after the projection, gather, then mark it ready.
    order = [c[0] for c in calls.mock_calls]
    assert order == ["main.wait_event"] * leftover + [
        "project",
        "side.wait_stream",
        "gather",
        "ready.record",
    ]
    by_name = {c[0]: c for c in calls.mock_calls}
    if leftover:
        assert by_name["main.wait_event"].args == (stale,)
    assert by_name["side.wait_stream"].args == (main,)
    assert by_name["gather"].args[1] is all_gather
    assert by_name["ready.record"].args == (side,)


def _cpu_sparse_chunk_attention(q, k, v, indices, cu_q, cu_k, kv_lens, scale):
    """CPU oracle for the chunk kernel's packed K/V and causal conventions."""
    output = torch.empty_like(q)
    for entry in range(kv_lens.numel()):
        start, end = int(cu_q[entry]), int(cu_q[entry + 1])
        for row in range(start, end):
            visible = int(kv_lens[entry]) - (end - start) + row - start + 1
            selected = indices[row]
            selected = selected[selected >= 0].long()
            assert (selected < visible).all()
            selected = selected + int(cu_k[entry])
            keys = k.index_select(0, selected).repeat_interleave(
                q.shape[1] // k.shape[1], dim=1
            )
            values = v.index_select(0, selected).repeat_interleave(
                q.shape[1] // v.shape[1], dim=1
            )
            scores = torch.einsum("hd,nhd->hn", q[row], keys) * scale
            output[row] = torch.einsum("hn,nhd->hd", scores.softmax(-1), values)
    return output


@pytest.mark.parametrize("prefix_lens", [(4, 8), (0, 4), (0, 0)])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("save_kv_cache", [True, False])
@pytest.mark.parametrize(
    "query_dtype,cache_dtype",
    [(torch.float32, torch.float32), (torch.bfloat16, torch.float8_e4m3fn)],
)
@pytest.mark.parametrize("prefetched", [False, True])
def test_qsa_cp_cached_prefix_matches_full_context_attention(
    monkeypatch, prefix_lens, rank, save_kv_cache, query_dtype, cache_dtype, prefetched
):
    # Unequal request lengths exercise both zigzag halves and uneven rank sizes.
    extend_lens = [8, 5]
    seq_lens = [prefix + length for prefix, length in zip(prefix_lens, extend_lens)]
    monkeypatch.setattr(
        cp_base, "get_parallel", lambda: SimpleNamespace(attn_cp_rank=rank)
    )
    monkeypatch.setattr(zigzag, "get_device", lambda: SimpleNamespace(device="cpu"))
    strategy = zigzag.ZigzagCPStrategy(cp_size=2)
    meta = strategy.build_metadata(sum(extend_lens), seq_lens, extend_lens)
    meta.per_rank_logical_token = meta.per_rank_actual_token
    meta.per_rank_actual_token = [8, 8]
    meta.max_rank_len = [8, 8]
    monkeypatch.setattr(qwen_sparse_attn_backend, "get_cp_strategy", lambda: strategy)

    generator = torch.Generator().manual_seed(42)
    keys = [
        torch.randn(length, 1, 4, generator=generator).to(query_dtype)
        for length in seq_lens
    ]
    values = [
        torch.randn(length, 1, 4, generator=generator).to(query_dtype)
        for length in seq_lens
    ]
    # Cached prefixes have already incurred the KV pool's storage rounding.
    for key, value, prefix in zip(keys, values, prefix_lens):
        key[:prefix] = key[:prefix].to(cache_dtype).to(query_dtype)
        value[:prefix] = value[:prefix].to(cache_dtype).to(query_dtype)
    queries = torch.randn(sum(extend_lens), 2, 4, generator=generator).to(query_dtype)
    new_k = torch.cat([key[prefix:] for key, prefix in zip(keys, prefix_lens)])
    new_v = torch.cat([value[prefix:] for value, prefix in zip(values, prefix_lens)])

    # Request IDs and physical token slots deliberately differ from packed order.
    req_indices = [4, 1]
    req_to_token = torch.zeros(5, max(seq_lens), dtype=torch.int32)
    slots = torch.randperm(80, generator=generator)[: sum(seq_lens)] + 1
    k_cache = torch.full((81, 1, 4), float("nan"), dtype=cache_dtype)
    v_cache = torch.full_like(k_cache, float("nan"))
    out_cache_locs = []
    offset = 0
    for i, (length, prefix) in enumerate(zip(seq_lens, prefix_lens)):
        request_slots = slots[offset : offset + length]
        offset += length
        req_to_token[req_indices[i], :length] = request_slots.to(torch.int32)
        k_cache[request_slots[:prefix]] = keys[i][:prefix].to(cache_dtype)
        v_cache[request_slots[:prefix]] = values[i][:prefix].to(cache_dtype)
        out_cache_locs.append(request_slots[prefix:])
    out_cache_loc = torch.cat(out_cache_locs)
    initial_k, initial_v = k_cache.clone(), v_cache.clone()

    events = []

    def store_kv(layer, locations, k, v):
        events.append("store")
        k_cache[locations.long()] = k.to(cache_dtype)
        v_cache[locations.long()] = v.to(cache_dtype)

    pool = SimpleNamespace(
        set_kv_buffer=Mock(side_effect=store_kv),
        get_key_buffer=Mock(return_value=k_cache),
        get_value_buffer=Mock(return_value=v_cache),
    )
    if not any(prefix_lens):
        pool.get_key_buffer.side_effect = AssertionError(
            "zero prefix needs no cache read"
        )
        pool.get_value_buffer.side_effect = AssertionError(
            "zero prefix needs no cache read"
        )
    backend = SimpleNamespace(
        token_to_kv_pool=pool,
        _pad_extend_output=QwenSparseAttnBackend._pad_extend_output,
        # Per-forward metadata: row i of the slot table is request i's row.
        _resolve_metadata=lambda forward_batch: SimpleNamespace(
            token_slot_table=req_to_token[req_indices],
            sequence_lengths=torch.tensor(seq_lens, dtype=torch.int32),
        ),
        _cp_kv_pending=None,
    )
    # No req_pool_indices: reading them on the host would sync every layer.
    batch = SimpleNamespace(
        attn_cp_metadata=meta,
        seq_lens_cpu=seq_lens,
        extend_seq_lens_cpu=extend_lens,
        extend_prefix_lens_cpu=list(prefix_lens),
        out_cache_loc=out_cache_loc,
    )
    local_q = strategy.shard_hidden_states(queries, batch)
    local_k = strategy.shard_hidden_states(new_k, batch)
    local_v = strategy.shard_hidden_states(new_v, batch)
    num_local = meta.total_q_prev_tokens + meta.total_q_next_tokens
    local_rows = strategy.shard_hidden_states(torch.arange(sum(extend_lens)), batch)[
        :num_local
    ]
    logical_positions = torch.cat(
        [torch.arange(prefix, length) for prefix, length in zip(prefix_lens, seq_lens)]
    )
    local_positions = logical_positions.index_select(0, local_rows)

    def gather_new_kv(local_kv, forward_batch, all_gather):
        events.append("gather")
        assert forward_batch is batch
        assert all_gather is None, "CPU tensors keep the CP group's all-gather"
        torch.testing.assert_close(
            local_kv, torch.cat([local_k.flatten(1), local_v.flatten(1)], dim=-1)
        )
        return torch.cat([new_k.flatten(1), new_v.flatten(1)], dim=-1)

    monkeypatch.setattr(
        qwen_sparse_attn_backend, "cp_materialize_global_token_order", gather_new_kv
    )

    def sparse_chunk_attention(q, k, v, indices, cu_q, cu_k, kv_lens, scale, max_q):
        # The last local query in each block identifies its absolute causal end.
        expected_ends = local_positions[cu_q[1:].long() - 1] + 1
        torch.testing.assert_close(kv_lens.long(), expected_ends)
        assert k.dtype == v.dtype == q.dtype
        # The host max_q sizes the kernel grid; a short one would skip query rows.
        assert max_q == int((cu_q[1:] - cu_q[:-1]).max())
        events.append("attention")
        return _cpu_sparse_chunk_attention(q, k, v, indices, cu_q, cu_k, kv_lens, scale)

    monkeypatch.setattr(
        qwen_sparse_attn_backend,
        "sparse_gqa_fwd_interface_triton_ck",
        sparse_chunk_attention,
    )
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=2,
        tp_k_head_num=1,
        tp_v_head_num=1,
        head_dim=4,
        qk_head_dim=4,
        v_head_dim=4,
        scaling=0.5,
    )
    # The indexer supplies causal logical positions and marks other slots invalid.
    candidates = torch.arange(max(seq_lens), dtype=torch.int32).expand(num_local, -1)
    topk = torch.where(candidates <= local_positions[:, None], candidates, -1)
    side, main = Mock(), Mock()
    side.wait_stream.side_effect = lambda stream: events.append(("side_waits", stream))
    main.wait_stream.side_effect = lambda stream: events.append(("main_waits", stream))

    @contextmanager
    def on_stream(stream):
        events.append(("enter", stream))
        yield
        events.append(("exit", stream))

    monkeypatch.setattr(qwen_sparse_attn_backend, "get_stream", lambda name: side)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: main)
    monkeypatch.setattr(torch.cuda, "stream", on_stream)
    if prefetched:
        QwenSparseAttnBackend.prefetch_cp_kv(backend, layer, local_k, local_v, batch)
    actual = QwenSparseAttnBackend._forward_extend_cp(
        backend, local_q, local_k, local_v, layer, batch, topk, save_kv_cache
    )
    assert backend._cp_kv_pending is None
    store = ["store"] if save_kv_cache else []
    if prefetched:
        # The side stream starts after the main stream's prior work (the key
        # gather); the main stream joins before using the gathered K/V; the
        # side stream, which owns their memory, then waits for those reads.
        assert events == [
            ("side_waits", main),
            ("enter", side),
            "gather",
            ("exit", side),
            ("main_waits", side),
            *store,
            "attention",
            ("side_waits", main),
        ]
    else:
        assert events == ["gather", *store, "attention"]

    expected = []
    row = 0
    for i, (length, prefix) in enumerate(zip(extend_lens, prefix_lens)):
        for position in range(prefix, prefix + length):
            visible_keys = keys[i][: position + 1, 0]
            visible_values = values[i][: position + 1, 0]
            scores = queries[row] @ visible_keys.T * layer.scaling
            expected.append(scores.softmax(-1) @ visible_values)
            row += 1
    expected = torch.stack(expected).index_select(0, local_rows).flatten(1)
    torch.testing.assert_close(actual[:num_local], expected)
    assert actual.shape == (8, 8)
    torch.testing.assert_close(actual[num_local:], torch.zeros_like(actual[num_local:]))
    if save_kv_cache:
        pool.set_kv_buffer.assert_called_once()
        initial_k[out_cache_loc] = new_k.to(cache_dtype)
        initial_v[out_cache_loc] = new_v.to(cache_dtype)
    else:
        pool.set_kv_buffer.assert_not_called()
    torch.testing.assert_close(k_cache.float(), initial_k.float(), equal_nan=True)
    torch.testing.assert_close(v_cache.float(), initial_v.float(), equal_nan=True)


@pytest.mark.parametrize("ensure_cos_sin_cache", [False, True])
def test_qsa_apply_rope_cache_sizing_is_optional(ensure_cos_sin_cache):
    class Sized(Exception):
        pass

    rotary_emb = SimpleNamespace(
        _ensure_cos_sin_cache_length=Mock(),
        get_cos_sin_with_position=Mock(side_effect=Sized),
    )
    with pytest.raises(Sized):
        QSAIndexer.apply_rope(
            SimpleNamespace(rotary_emb=rotary_emb),
            torch.arange(4),
            torch.zeros(4, 1, 4),
            ensure_cos_sin_cache=ensure_cos_sin_cache,
        )
    # Sizing reads positions.max() on the host: the CP path must skip it.
    assert rotary_emb._ensure_cos_sin_cache_length.called is ensure_cos_sin_cache


def test_qsa_cp_forward_metadata_joins_leftover_prefetch(monkeypatch):
    side, main = Mock(), Mock()
    monkeypatch.setattr(qwen_sparse_attn_backend, "get_stream", lambda name: side)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: main)
    backend = SimpleNamespace(_cp_kv_pending=(0, None, None))
    QwenSparseAttnBackend.init_forward_metadata(
        backend, SimpleNamespace(forward_mode=ForwardMode.IDLE)
    )
    # Later CP collectives on the main stream must not overlap the gather.
    main.wait_stream.assert_called_once_with(side)
    assert backend._cp_kv_pending is None


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("physical_padding", [False, True])
def test_zigzag_kv_gather_uses_given_all_gather(monkeypatch, rank, physical_padding):
    """A replacement all-gather gets this rank's logical rows and returns
    rank-major blocks; rows past a rank's own never reach the token order."""
    extend_lens = [8, 5]
    current = SimpleNamespace(attn_cp_rank=0)
    monkeypatch.setattr(cp_base, "get_parallel", lambda: current)
    monkeypatch.setattr(zigzag, "get_device", lambda: SimpleNamespace(device="cpu"))
    strategy = zigzag.ZigzagCPStrategy(cp_size=2)
    tokens = torch.arange(sum(extend_lens) * 3, dtype=torch.float32).reshape(-1, 3)
    metas, inputs, local_rows = [], [], []
    for cp_rank in range(2):
        current.attn_cp_rank = cp_rank
        meta = strategy.build_metadata(sum(extend_lens), extend_lens, extend_lens)
        logical = list(meta.per_rank_actual_token)
        if physical_padding:
            meta.per_rank_logical_token = logical
            meta.per_rank_actual_token = [max(logical) + 2] * 2
        rows = strategy.shard_hidden_states(
            tokens, SimpleNamespace(attn_cp_metadata=meta)
        )
        metas.append(meta)
        inputs.append(rows)
        local_rows.append(rows[: logical[cp_rank]])
    calls = []

    def all_gather(local, rows_per_rank):
        calls.append(rows_per_rank)
        torch.testing.assert_close(local, local_rows[rank])
        out = torch.full((2 * rows_per_rank, 3), float("nan"))
        for peer, rows in enumerate(local_rows):
            out[peer * rows_per_rank : peer * rows_per_rank + rows.shape[0]] = rows
        return out

    current.attn_cp_rank = rank
    batch = SimpleNamespace(attn_cp_metadata=metas[rank])
    actual = strategy.gather_kv_cache(inputs[rank], batch, all_gather=all_gather)
    torch.testing.assert_close(actual, tokens)
    assert calls == [max(rows.shape[0] for rows in local_rows)]


def test_cp_zero_sm_all_gather_is_decided_once_per_process(monkeypatch):
    from sglang.srt.layers.cp import utils as cp_utils

    verdicts = []
    gather = SimpleNamespace(all_gather=Mock(return_value="gathered"))

    def create(verdict):
        def run():
            verdicts.append(verdict)
            return (verdict,)

        return run

    monkeypatch.setattr(cp_utils, "get_is_capture_mode", lambda: False)
    on_gpu = SimpleNamespace(is_cuda=True)
    rows = cp_utils._ZERO_SM_MIN_ROWS_PER_RANK

    def batch(*per_rank_rows):
        return SimpleNamespace(
            attn_cp_metadata=SimpleNamespace(per_rank_actual_token=list(per_rank_rows))
        )

    large = batch(rows, rows - 1)
    pick = cp_utils.cp_zero_sm_all_gather
    with get_resources().override(buffers={}):
        monkeypatch.setattr(cp_utils, "_create_zero_sm_all_gather", create(gather))
        # CPU tensors and CPU-launch-bound batches keep the CP group's gather,
        # with no communicator created.
        assert pick(torch.zeros(1), large, "kv") is None
        assert pick(on_gpu, batch(rows - 1, rows - 1), "kv") is None
        assert verdicts == []
        for key in ("kv", "kv", "idx"):
            assert pick(on_gpu, large, key)(on_gpu, 3) == "gathered"
            gather.all_gather.assert_called_with(on_gpu, 3, key=key)
        # Creation is collective: every later call must reuse the verdict.
        assert verdicts == [gather]
    with get_resources().override(buffers={}):
        monkeypatch.setattr(cp_utils, "_create_zero_sm_all_gather", create(None))
        assert pick(on_gpu, large, "kv") is None
        assert pick(on_gpu, large, "kv") is None
        assert verdicts == [gather, None]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
