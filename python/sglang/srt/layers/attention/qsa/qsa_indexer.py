"""QSA indexer for Qwen4-Exp checkpoints."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
from sglang.srt.layers.attention.qsa.kernel import (
    average_pool_qsa_keys,
    expand_qsa_block_indices,
    qsa_fast_topk,
)
from sglang.srt.layers.attention.qsa.metadata import (
    build_group_ring_slots,
    build_pending_ring_slots,
    build_rope_position_matrix,
)
from sglang.srt.layers.attention.qsa.mqa import (
    HAS_TILELANG,
    qsa_mqa_decode,
    qsa_mqa_prefill,
    qsa_mqa_prefill_relative,
)
from sglang.srt.layers.cp.utils import (
    ContextParallelStrategyKind,
    cp_materialize_global_token_order,
    cp_shard_hidden_states,
    cp_shard_position_ids,
    cp_zero_sm_all_gather,
    get_cp_strategy,
)
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb
from sglang.srt.layers.utils.multi_platform import MultiPlatformOp
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.runtime_context import get_stream

# Cap on the fp32 [query_rows, compressed_keys] prefill logits workspace;
# top-k is per row, so tiling rows does not change the selection.
_QSA_PREFILL_LOGITS_BUDGET_BYTES = 128 * 1024 * 1024


def _qsa_prefill_row_chunk_size(rows: int, keys: int, heads: int) -> int:
    if rows <= 0 or keys <= 0:
        return max(rows, 1)
    block_q = max(1, 128 // heads)
    bytes_per_row = keys * torch.float32.itemsize
    max_padded_rows = max(block_q, _QSA_PREFILL_LOGITS_BUDGET_BYTES // bytes_per_row)
    max_padded_rows = max(block_q, max_padded_rows // block_q * block_q)
    return min(rows, max_padded_rows)


class QSAIndexer(MultiPlatformOp):
    """Config-driven fused-QK, weight-free sparse-attention indexer."""

    def __init__(
        self,
        config,
        layer_id: int,
        quant_config=None,
        prefix: str = "",
        rotary_emb=None,
    ) -> None:
        self._validate_config(config)
        super().__init__()
        self.layer_id = int(layer_id)
        self.index_n_heads = int(config.indexer_n_heads)
        self.index_kv_heads = int(config.indexer_kv_heads)
        self.index_head_dim = int(config.indexer_head_dim)
        self.token_topk = int(config.indexer_budget)
        self.compress_ratio = int(config.indexer_compress_ratio)
        self.block_topk = self.token_topk // self.compress_ratio
        if rotary_emb is None:
            raise ValueError("QSAIndexer must reuse its Qwen4-Exp attention RoPE")
        self.rotary_emb = rotary_emb
        if not 0 < self.rotary_emb.rotary_dim <= self.index_head_dim:
            raise ValueError(
                "Qwen4-Exp attention RoPE rotary_dim must fit the QSA index head: "
                f"{self.rotary_emb.rotary_dim=} {self.index_head_dim=}"
            )
        self.index_qk_proj = ReplicatedLinear(
            config.hidden_size,
            (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.index_qk_proj" if prefix else "index_qk_proj",
        )
        self.q_layernorm = GemmaRMSNorm(
            self.index_head_dim, eps=getattr(config, "rms_norm_eps", 1e-6)
        )
        self.k_layernorm = GemmaRMSNorm(
            self.index_head_dim, eps=getattr(config, "rms_norm_eps", 1e-6)
        )
        self._rope_axis_map_cache = None
        # (forward_batch, index q/k rows, gathered raw keys, ready event) of
        # the prefill-CP key gather prefetch_cp_keys started.
        self._cp_key_pending = None
        self._cp_key_ready = None

    @staticmethod
    def _validate_config(config) -> None:
        names = (
            "indexer_n_heads",
            "indexer_kv_heads",
            "indexer_head_dim",
            "indexer_budget",
            "indexer_compress_ratio",
        )
        missing = [name for name in names if getattr(config, name, None) is None]
        if missing:
            raise ValueError(f"QSA config is missing required fields: {missing}")
        values = {name: int(getattr(config, name)) for name in names}
        if any(value <= 0 for value in values.values()):
            raise ValueError(f"QSA config values must be positive: {values}")
        if values["indexer_compress_ratio"] < 2:
            # DP token-padding rows carry logical length 1, which must never
            # reach a compression boundary; ratio >= 2 guarantees that.
            raise ValueError(
                "QSA requires indexer_compress_ratio >= 2, got "
                f"{values['indexer_compress_ratio']}"
            )
        if values["indexer_kv_heads"] != 1:
            raise ValueError("the QSA MQA operators require indexer_kv_heads=1")
        if values["indexer_budget"] % values["indexer_compress_ratio"] != 0:
            raise ValueError(
                "indexer_budget must be divisible by indexer_compress_ratio"
            )
        block_topk = values["indexer_budget"] // values["indexer_compress_ratio"]
        if block_topk not in (512, 2048):
            raise ValueError(
                "fast_topk_v2 requires indexer_budget / indexer_compress_ratio "
                f"to be 512 or 2048, got {block_topk}"
            )

    def _use_fused_prep(self, tensor: torch.Tensor) -> bool:
        """Whether the fused indexer-prep kernels support this configuration."""
        return (
            tensor.is_cuda
            and tensor.dtype in (torch.bfloat16, torch.float16)
            and self.index_head_dim in (64, 128, 256)
            and self.rotary_emb.rotary_dim % 2 == 0
            and not getattr(self.rotary_emb, "mrope_interleaved_glm", False)
            and len(getattr(self.rotary_emb, "mrope_section", None) or ()) in (0, 3)
            and getattr(self.rotary_emb, "cos_sin_cache", None) is not None
            and self.rotary_emb.cos_sin_cache.is_cuda
            and self.rotary_emb.cos_sin_cache.dtype == torch.float32
        )

    def _rope_axis_map(self, device) -> torch.Tensor:
        """axis_map[i] is the MRoPE position axis whose cos/sin rotary pair i reads."""
        cache = self._rope_axis_map_cache
        if cache is not None and cache.device == device:
            return cache
        half = self.rotary_emb.rotary_dim // 2
        section = getattr(self.rotary_emb, "mrope_section", None) or None
        axis_map = torch.zeros(half, dtype=torch.int32)
        if section is not None:
            s0, s1, s2 = (int(v) for v in section)
            if getattr(self.rotary_emb, "mrope_interleaved", False):
                pair = torch.arange(half, dtype=torch.int32)
                axis_map[((pair % 3) == 1) & (pair < s1 * 3)] = 1
                axis_map[((pair % 3) == 2) & (pair < s2 * 3)] = 2
            else:
                axis_map[s0 : s0 + s1] = 1
                axis_map[s0 + s1 :] = 2
        self._rope_axis_map_cache = axis_map.to(device)
        return self._rope_axis_map_cache

    def project_qk(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        pool=None,
        cache_loc: torch.Tensor | None = None,
        q_heads_padded: int | None = None,
        ensure_cos_sin_cache: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        qk, _ = self.index_qk_proj(hidden_states)
        token_k = qk[:, self.index_n_heads * self.index_head_dim :].reshape(
            -1, self.index_kv_heads, self.index_head_dim
        )
        if (
            pool is not None
            and cache_loc is not None
            and qk.shape[0] > 0
            and self._use_fused_prep(qk)
        ):
            from sglang.kernels.ops.attention.qsa_indexer import (
                qsa_index_q_norm_rope_store,
            )

            if (
                ensure_cos_sin_cache
                and not get_is_capture_mode()
                and hasattr(self.rotary_emb, "_ensure_cos_sin_cache_length")
            ):
                self.rotary_emb._ensure_cos_sin_cache_length(
                    int(positions.max().item())
                )
            key_state_buffer = pool.get_qsa_key_state_buffer(self.layer_id)
            q = qsa_index_q_norm_rope_store(
                qk,
                positions.long(),
                self.rotary_emb.cos_sin_cache,
                self._rope_axis_map(qk.device),
                self.q_layernorm.weight.data,
                cache_loc[: qk.shape[0]].long(),
                key_state_buffer.view(key_state_buffer.shape[0], -1),
                pool.qsa_rope_position_buffer,
                self.index_n_heads,
                self.rotary_emb.rotary_dim,
                self.q_layernorm.variance_epsilon,
                self.rotary_emb.is_neox_style,
                q_heads_padded=q_heads_padded,
                out_dtype=pool.qsa_compressed_dtype,
            )
            return q, token_k, True
        q = self._project_index_q(qk, positions, pool, ensure_cos_sin_cache)
        return q, token_k, False

    def _project_index_q(
        self,
        qk: torch.Tensor,
        positions: torch.Tensor,
        pool,
        ensure_cos_sin_cache: bool,
    ) -> torch.Tensor:
        q_raw = qk[:, : self.index_n_heads * self.index_head_dim]
        q = self.q_layernorm(q_raw.reshape(-1, self.index_head_dim)).reshape(
            -1, self.index_n_heads, self.index_head_dim
        )
        q = self.apply_rope(positions, q, ensure_cos_sin_cache=ensure_cos_sin_cache)
        if pool is not None:
            q = q.to(pool.qsa_compressed_dtype)
        return q

    def normalize_compressed_keys(
        self, compressed_keys: torch.Tensor, block_positions: torch.Tensor
    ) -> torch.Tensor:
        normalized = self.k_layernorm(
            compressed_keys.reshape(-1, self.index_head_dim)
        ).reshape(-1, self.index_kv_heads, self.index_head_dim)
        return self.apply_rope(block_positions, normalized)

    def _use_fused_compress(self, pool) -> bool:
        return getattr(
            pool, "qsa_rope_position_buffer", None
        ) is not None and self._use_fused_prep(
            pool.get_qsa_key_state_buffer(self.layer_id)
        )

    def _fused_compress_store(
        self,
        pool,
        group_locs: torch.Tensor,
        write_locs: torch.Tensor,
        source_keys: torch.Tensor | None = None,
        source_rope: torch.Tensor | None = None,
    ) -> None:
        """Fused mean -> gemma norm -> MRoPE -> compressed-cache store;
        a None source_keys/source_rope reads the members from the pending ring."""
        from sglang.kernels.ops.attention.qsa_indexer import (
            qsa_index_k_compress_store,
        )

        if source_keys is None:
            source_keys = pool.get_qsa_key_state_buffer(self.layer_id)
        if source_rope is None:
            source_rope = pool.qsa_rope_position_buffer
        compressed_buffer = pool.get_qsa_compressed_k_buffer(self.layer_id)
        qsa_index_k_compress_store(
            source_keys.reshape(source_keys.shape[0], -1)
            .contiguous()
            .to(pool.index_state_dtype),
            group_locs.to(torch.int32),
            source_rope,
            self.rotary_emb.cos_sin_cache,
            self._rope_axis_map(source_keys.device),
            self.k_layernorm.weight.data,
            write_locs.to(torch.int32),
            compressed_buffer.view(compressed_buffer.shape[0], -1),
            self.compress_ratio,
            self.rotary_emb.rotary_dim,
            self.k_layernorm.variance_epsilon,
            self.rotary_emb.is_neox_style,
        )

    def _pending_ring_slots(
        self, metadata, logical_positions: torch.Tensor, is_extend: bool
    ) -> torch.Tensor:
        return build_pending_ring_slots(
            token_to_batch_idx=metadata.token_to_batch_idx,
            req_pool_indices=metadata.req_pool_indices,
            sequence_lengths=metadata.sequence_lengths,
            logical_positions=logical_positions,
            compress_ratio=self.compress_ratio,
            is_extend=is_extend,
        )

    def _group_ring_slots(
        self, metadata, group_end_positions: torch.Tensor, sequence_ids: torch.Tensor
    ) -> torch.Tensor:
        return build_group_ring_slots(
            req_pool_indices=metadata.req_pool_indices,
            group_end_positions=group_end_positions,
            sequence_ids=sequence_ids,
            compress_ratio=self.compress_ratio,
        )

    def update_key_state_and_compress(
        self,
        token_k: torch.Tensor,
        logical_positions: torch.Tensor,
        rope_positions: torch.Tensor,
        metadata,
        state_slots: torch.Tensor | None = None,
        state_stored: bool = False,
    ) -> None:
        """Store the pending-group ring and compress each completed group."""

        pool = metadata.token_to_kv_pool
        is_extend = metadata.compress_member_rows is not None
        if not state_stored:
            if state_slots is None:
                state_slots = self._pending_ring_slots(
                    metadata, logical_positions, is_extend
                )
            pool.set_qsa_key_state_buffer(
                self.layer_id, state_slots[: token_k.shape[0]], token_k
            )
            pool.set_qsa_rope_position_buffer(
                state_slots[: token_k.shape[0]], rope_positions
            )

        if metadata.is_cuda_graph:
            self._compress_decode_cuda_graph(metadata)
            return

        if metadata.write_locs is None:
            raise RuntimeError(
                "QSA metadata is missing the precomputed write plan; the "
                "sparse-attention backend derives it from the batch lengths"
            )
        if metadata.write_locs.numel() == 0:
            return
        group_end_positions = metadata.compress_group_positions.long()
        compressed_locs = metadata.write_locs
        if is_extend:
            # Extend chunks are group-aligned; each planned group lies in this forward,
            # so read its members from the packed chunk tensors.
            member_rows = metadata.compress_member_rows.long()
            group_locs = member_rows[:, None] + torch.arange(
                self.compress_ratio, device=member_rows.device, dtype=torch.long
            )
            source_keys = token_k
            group_locs = group_locs.clamp_max(source_keys.shape[0] - 1)
            source_rope = metadata.extend_rope_matrix
            if source_rope is None:
                source_rope = build_rope_position_matrix(
                    rope_positions, token_k.shape[0]
                )
        else:
            # Paged eager rows (speculative fallback) complete at most one
            # group each; its members are exactly the pending ring window.
            group_locs = metadata.compress_group_ring_locs
            if group_locs is None:
                group_locs = self._group_ring_slots(
                    metadata,
                    group_end_positions,
                    metadata.compress_sequence_ids.long(),
                )
            source_keys = pool.get_qsa_key_state_buffer(self.layer_id)
            source_rope = pool.qsa_rope_position_buffer
        if self._use_fused_compress(pool):
            self._fused_compress_store(
                pool,
                group_locs,
                compressed_locs,
                source_keys=source_keys,
                source_rope=source_rope,
            )
            return
        key_groups = source_keys[group_locs]
        pooled = average_pool_qsa_keys(key_groups)
        compressed_rope_positions = self._rope_from_matrix(
            source_rope[group_locs[:, 0]]
        )
        normalized = self.normalize_compressed_keys(pooled, compressed_rope_positions)
        pool.set_qsa_compressed_k_buffer(self.layer_id, compressed_locs, normalized)

    def _compress_decode_cuda_graph(self, metadata) -> None:
        """Fixed-shape graph-replay compression; non-boundary rows write slot 0."""

        if metadata.graph_write_locs is None or metadata.graph_ring_group_locs is None:
            raise RuntimeError("QSA CUDA graph compression metadata is incomplete")
        pool = metadata.token_to_kv_pool
        group_locs = metadata.graph_ring_group_locs
        if self._use_fused_compress(pool):
            self._fused_compress_store(
                pool,
                group_locs,
                metadata.graph_write_locs,
            )
            return
        key_groups = pool.get_qsa_key_state_buffer(self.layer_id)[group_locs]
        compressed = average_pool_qsa_keys(key_groups)
        compressed_rope_positions = self._rope_from_matrix(
            pool.qsa_rope_position_buffer[group_locs[:, 0]]
        )
        compressed = self.normalize_compressed_keys(
            compressed, compressed_rope_positions
        )
        pool.set_qsa_compressed_k_buffer(
            self.layer_id, metadata.graph_write_locs, compressed.contiguous()
        )

    def _rope_from_matrix(self, positions: torch.Tensor) -> torch.Tensor:
        """[n, 3] slot coordinates -> the layout apply_rope expects."""
        positions = positions.transpose(0, 1)
        if not getattr(self.rotary_emb, "mrope_section", None):
            return positions[0]
        return positions

    def apply_rope(
        self,
        positions: torch.Tensor,
        tensor: torch.Tensor,
        ensure_cos_sin_cache: bool = True,
    ) -> torch.Tensor:
        if tensor.numel() == 0:
            return tensor
        positions = positions.long()
        num_positions = (
            positions.shape[-1] if positions.ndim == 2 else positions.numel()
        )
        if num_positions != tensor.shape[0]:
            raise ValueError("QSA RoPE positions must match the token dimension")
        if (
            ensure_cos_sin_cache
            and not get_is_capture_mode()
            and hasattr(self.rotary_emb, "_ensure_cos_sin_cache_length")
        ):
            self.rotary_emb._ensure_cos_sin_cache_length(int(positions.max().item()))

        # position_cos/position_sin repeat cos/sin to the full rotary width;
        # apply_rotary_emb consumes one half.
        self.rotary_emb.get_cos_sin_with_position(positions)
        rotary_dim = self.rotary_emb.rotary_dim
        half_rotary_dim = rotary_dim // 2
        cos = self.rotary_emb.position_cos.reshape(num_positions, -1)[
            :, :half_rotary_dim
        ]
        sin = self.rotary_emb.position_sin.reshape(num_positions, -1)[
            :, :half_rotary_dim
        ]
        rotated = apply_rotary_emb(
            tensor[..., :rotary_dim],
            cos,
            sin,
            self.rotary_emb.is_neox_style,
        )
        return torch.cat([rotated, tensor[..., rotary_dim:]], dim=-1)

    def select_prefill_tokens(
        self,
        q: torch.Tensor,
        compressed_keys: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        query_positions: torch.Tensor,
        sequence_lengths_for_rows: torch.Tensor,
        max_row_keys: Optional[int] = None,
    ) -> torch.Tensor:
        """Top-k token selection for packed prefill rows.

        ``max_row_keys`` (host int, the longest request's compressed-block count)
        switches scoring to request-relative logits: [rows, max_row_keys] instead
        of [rows, all compressed keys of the batch]. The latter caps a chunk at
        a few dozen rows once a batch holds many long prefixes, so the scoring
        kernel ran as thousands of one-CTA launches. Selection is unchanged: the
        top-k reads only each row's own section and returns indices relative to
        the row's start either way."""
        rows = q.shape[0]
        output = torch.empty(
            (rows, self.token_topk + self.compress_ratio - 1),
            dtype=torch.int32,
            device=q.device,
        )
        if rows == 0:
            return output

        relative = (
            max_row_keys is not None and max_row_keys > 0 and q.is_cuda and HAS_TILELANG
        )
        row_chunk_size = _qsa_prefill_row_chunk_size(
            rows,
            max_row_keys if relative else compressed_keys.shape[0],
            q.shape[1],
        )
        for row_start in range(0, rows, row_chunk_size):
            row_end = min(row_start + row_chunk_size, rows)
            chunk_slice = slice(row_start, row_end)
            if compressed_keys.shape[0] == 0:
                block_indices = torch.full(
                    (row_end - row_start, self.block_topk),
                    -1,
                    dtype=torch.int32,
                    device=q.device,
                )
                logits = None
            elif relative:
                chunk_starts = row_starts[chunk_slice]
                chunk_lengths = row_ends[chunk_slice] - chunk_starts
                logits = qsa_mqa_prefill_relative(
                    q[chunk_slice],
                    compressed_keys,
                    chunk_starts,
                    row_ends[chunk_slice],
                    max_row_keys,
                )
                block_indices = qsa_fast_topk(
                    logits,
                    torch.zeros_like(chunk_starts),
                    chunk_lengths,
                    topk=self.block_topk,
                )
            else:
                logits = qsa_mqa_prefill(
                    q[chunk_slice],
                    compressed_keys,
                    row_starts[chunk_slice],
                    row_ends[chunk_slice],
                )
                block_indices = qsa_fast_topk(
                    logits,
                    row_starts[chunk_slice],
                    row_ends[chunk_slice],
                    topk=self.block_topk,
                )
            selected = expand_qsa_block_indices(
                block_indices,
                query_positions[chunk_slice],
                sequence_lengths_for_rows[chunk_slice],
                compress_ratio=self.compress_ratio,
                token_topk=self.token_topk,
            )
            output[chunk_slice].copy_(selected)
            del logits, block_indices, selected
        return output

    def select_decode_tokens(
        self,
        q: torch.Tensor,
        compressed_cache: torch.Tensor,
        compressed_page_table: torch.Tensor,
        compressed_lengths: torch.Tensor,
        max_model_len: int,
        query_positions: torch.Tensor,
        sequence_lengths: torch.Tensor,
        defer_expansion: bool = False,
    ) -> torch.Tensor:
        # fast_topk (k = 512) writes 0..length-1 for rows that fit k without reading
        # their scores, and never reads past a row's length.
        use_fast_topk = (
            q.is_cuda
            and self.block_topk == 512
            and not is_batch_invariant_mode_enabled()
        )
        logits = qsa_mqa_decode(
            q,
            compressed_cache,
            compressed_page_table,
            compressed_lengths,
            max_model_len,
            min_scored_len=self.block_topk if use_fast_topk else 0,
        )
        if use_fast_topk:
            # Decode rows start at zero, so compressed lengths double as row lengths;
            # skip the generic zero-fill + subtract.
            from sglang.kernels.ops.attention.fast_topk import fast_topk

            block_indices = fast_topk(
                logits,
                compressed_lengths.to(torch.int32),
                topk=self.block_topk,
                row_starts=None,
            )
        else:
            row_starts = torch.zeros_like(compressed_lengths, dtype=torch.int32)
            block_indices = qsa_fast_topk(
                logits, row_starts, compressed_lengths, topk=self.block_topk
            )
        if defer_expansion:
            return block_indices
        return expand_qsa_block_indices(
            block_indices,
            query_positions,
            sequence_lengths,
            compress_ratio=self.compress_ratio,
            token_topk=self.token_topk,
        )

    def prefetch_cp_keys(self, hidden_states: torch.Tensor, forward_batch) -> None:
        """Prefill CP: project this rank's index q/k rows and start the raw-key
        gather on the qsa_cp_kv stream, under the attention's qkv projection;
        this layer's forward_cuda_cp joins it.

        Call it on every rank before the attention's K/V prefetch, which then
        queues behind it on the same stream and communicator. Without the
        copy-engine all-gather the gather stays inline in forward_cuda_cp."""
        meta = forward_batch.attn_cp_metadata
        num_local = int(meta.total_q_prev_tokens + meta.total_q_next_tokens)
        all_gather = cp_zero_sm_all_gather(hidden_states, forward_batch, "index_k")
        if all_gather is None:
            return
        if self._cp_key_pending is not None:
            # Left by a forward that failed before this layer's indexer.
            torch.cuda.current_stream().wait_event(self._cp_key_pending[-1])
        qk, _ = self.index_qk_proj(hidden_states[:num_local])
        stream = get_stream("qsa_cp_kv")
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            token_k_full = self._gather_cp_keys(qk, forward_batch, all_gather)
        if self._cp_key_ready is None:
            self._cp_key_ready = torch.cuda.Event()
        # Re-recording is safe: the join already waited on the last record.
        ready = self._cp_key_ready
        ready.record(stream)
        # qk is read on the side stream and token_k_full allocated there: the
        # join's wait_event and its release wait order both lifetimes.
        self._cp_key_pending = (forward_batch, qk, token_k_full, ready)

    def _gather_cp_keys(self, qk: torch.Tensor, forward_batch, all_gather):
        # Raw keys of all new tokens in global packed order: the ring store
        # and compression use the global write plan unchanged. Cached prefix
        # groups already reside in the compressed-K pool on every rank.
        token_k = qk[:, self.index_n_heads * self.index_head_dim :]
        if all_gather is None:
            # The CP group's all-gather sends a contiguous buffer.
            token_k = token_k.contiguous()
        return cp_materialize_global_token_order(
            token_k, forward_batch, all_gather=all_gather
        ).reshape(-1, self.index_kv_heads, self.index_head_dim)

    def forward_cuda_cp(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        indexer_metadata,
        global_rope_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill CP: score only this rank's zigzag query rows.

        ``hidden_states``/``positions`` are the local (padded) rows and their
        RoPE positions. Index q/k are projected for the local rows only; the raw
        keys are all-gathered into global token order so the pending-group ring
        store and the block compression run on the batch's global write plan
        exactly as without CP (every rank ends up with the complete compressed-K
        cache). Selection then scores the local rows against all compressed
        blocks of their request, position-driven, so the output rows line up
        with the local query rows the sparse attention consumes.
        """
        strategy = get_cp_strategy()
        if strategy is None or strategy.kind != ContextParallelStrategyKind.ZIGZAG:
            strategy_name = strategy.name if strategy is not None else "none"
            raise NotImplementedError(
                "QSA prefill CP only supports the zigzag strategy; "
                f"got {strategy_name}."
            )
        meta = forward_batch.attn_cp_metadata
        num_local = int(meta.total_q_prev_tokens + meta.total_q_next_tokens)
        global_logical = getattr(forward_batch, "positions", None)
        global_logical = (
            global_logical[0] if global_logical.ndim == 2 else global_logical
        ).flatten()
        num_global = indexer_metadata.get_token_to_batch_idx().numel()
        global_logical = global_logical[:num_global]
        local_logical = cp_shard_position_ids(global_logical, forward_batch)[:num_local]
        local_sequence_ids = cp_shard_hidden_states(
            indexer_metadata.get_token_to_batch_idx(), forward_batch
        )[:num_local]
        positions = (
            positions[:, :num_local] if positions.ndim == 2 else positions[:num_local]
        )
        prefetched, self._cp_key_pending = self._cp_key_pending, None
        token_k_full = None
        if prefetched is not None and prefetched[0] is forward_batch:
            _, qk, token_k_full, _ = prefetched
        else:
            qk, _ = self.index_qk_proj(hidden_states[:num_local])
        # Use the compressed-cache dtype for Q. The ring store waits for the
        # raw local K to be gathered into global order.
        # The load-time RoPE reservation covers these positions, as this layer's
        # attention already assumed; re-checking costs a device sync per layer.
        q = self._project_index_q(
            qk,
            positions,
            pool=indexer_metadata.token_to_kv_pool,
            ensure_cos_sin_cache=False,
        )
        if token_k_full is None:
            token_k_full = self._gather_cp_keys(qk, forward_batch, all_gather=None)
        if prefetched is not None:
            # Also joins a gather a failed forward left behind.
            torch.cuda.current_stream().wait_event(prefetched[-1])
        global_rope = (
            global_rope_positions[:, :num_global]
            if global_rope_positions.ndim == 2
            else global_rope_positions[:num_global]
        )
        self.update_key_state_and_compress(
            token_k_full,
            global_logical,
            global_rope,
            indexer_metadata,
            state_slots=indexer_metadata.pending_ring_slots,
            state_stored=False,
        )
        if prefetched is not None:
            # The prefetched keys live in the side stream's pool; order their
            # release there (unmap, cudaFreeAsync) after this stream's reads.
            get_stream("qsa_cp_kv").wait_stream(torch.cuda.current_stream())
        compressed_keys, row_starts, row_ends, sequence_lengths = (
            indexer_metadata.get_prefill_mqa_inputs(
                self.layer_id, local_logical, query_sequence_ids=local_sequence_ids
            )
        )
        row_sequence_lengths = sequence_lengths.index_select(
            0, local_sequence_ids.long()
        )
        return self.select_prefill_tokens(
            q,
            compressed_keys,
            row_starts,
            row_ends,
            local_logical,
            row_sequence_lengths,
            max_row_keys=indexer_metadata.prefill_max_row_keys(
                getattr(forward_batch, "seq_lens_cpu", None)
            ),
        )

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        indexer_metadata,
        cp_global_rope_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Portable orchestration shared by different platforms.
        Fast paths are gated per platforms inside kernel calls.
        """
        forward_mode = forward_batch.forward_mode
        is_target_verify = getattr(forward_mode, "is_target_verify", lambda: False)()
        is_draft_extend = getattr(forward_mode, "is_draft_extend_v2", lambda: False)()
        if cp_global_rope_positions is not None and not (
            forward_mode.is_decode() or is_target_verify or is_draft_extend
        ):
            return self.forward_cuda_cp(
                hidden_states,
                positions,
                forward_batch,
                indexer_metadata,
                cp_global_rope_positions,
            )
        if forward_mode.is_decode() or is_target_verify or is_draft_extend:
            # EAGLE/MTP may advance the model's RoPE coordinate independently
            # from the physical paged-KV position.  Compression and sparse
            # selection must use the latter; otherwise a draft step can index
            # token_slot_table[:, seq_len] one past the valid range.
            logical_positions = indexer_metadata.decode_logical_positions
            if logical_positions is None:
                logical_positions = indexer_metadata.get_seqlens_expanded() - 1
        else:
            logical_positions = getattr(forward_batch, "positions", None)
            if logical_positions is None:
                logical_positions = positions[0] if positions.ndim == 2 else positions
            logical_positions = logical_positions.flatten()
        # DP MAX_LEN padding adds token rows without assigning them to a
        # request. token_to_batch_idx is the source of truth for semantic rows.
        num_valid_tokens = indexer_metadata.get_token_to_batch_idx().numel()
        if logical_positions.numel() < num_valid_tokens:
            raise ValueError(
                "QSA logical positions are shorter than the request mapping: "
                f"positions={logical_positions.numel()}, mapping={num_valid_tokens}"
            )
        if hidden_states.shape[0] < num_valid_tokens:
            raise ValueError(
                "QSA hidden states are shorter than the request mapping: "
                f"hidden={hidden_states.shape[0]}, mapping={num_valid_tokens}"
            )
        position_tokens = (
            positions.shape[-1] if positions.ndim == 2 else positions.numel()
        )
        if position_tokens < num_valid_tokens:
            raise ValueError(
                "QSA RoPE positions are shorter than the request mapping: "
                f"positions={position_tokens}, mapping={num_valid_tokens}"
            )

        logical_positions = logical_positions[:num_valid_tokens]
        hidden_states = hidden_states[:num_valid_tokens]
        positions = (
            positions[:, :num_valid_tokens]
            if positions.ndim == 2
            else positions[:num_valid_tokens]
        )
        state_slots = indexer_metadata.pending_ring_slots
        if state_slots is None:
            state_slots = self._pending_ring_slots(
                indexer_metadata,
                logical_positions,
                indexer_metadata.compress_member_rows is not None,
            )
        q, token_k, state_stored = self.project_qk(
            hidden_states,
            positions,
            pool=indexer_metadata.token_to_kv_pool,
            cache_loc=state_slots,
            q_heads_padded=(
                # The tilelang decode MQA kernel needs query heads in multiples of 8.
                ((self.index_n_heads + 7) // 8) * 8
                if (forward_mode.is_decode() or is_target_verify or is_draft_extend)
                else None
            ),
        )
        self.update_key_state_and_compress(
            token_k,
            logical_positions,
            positions,
            indexer_metadata,
            state_slots=state_slots,
            state_stored=state_stored,
        )
        if forward_mode.is_decode() or is_target_verify or is_draft_extend:
            compressed_cache, page_table, compressed_lengths, max_model_len = (
                indexer_metadata.get_decode_mqa_inputs(self.layer_id)
            )
            return self.select_decode_tokens(
                q,
                compressed_cache,
                page_table,
                compressed_lengths,
                max_model_len,
                logical_positions,
                indexer_metadata.get_seqlens_int32(),
                defer_expansion=q.is_cuda and indexer_metadata.defer_block_expansion,
            )

        compressed_keys, row_starts, row_ends, sequence_lengths = (
            indexer_metadata.get_prefill_mqa_inputs(self.layer_id, logical_positions)
        )
        query_sequence_ids = indexer_metadata.get_token_to_batch_idx()
        row_sequence_lengths = sequence_lengths.index_select(
            0, query_sequence_ids.long()
        )
        return self.select_prefill_tokens(
            q,
            compressed_keys,
            row_starts,
            row_ends,
            logical_positions,
            row_sequence_lengths,
            max_row_keys=indexer_metadata.prefill_max_row_keys(
                getattr(forward_batch, "seq_lens_cpu", None)
            ),
        )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        indexer_metadata,
        cp_global_rope_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._forward_impl(
            hidden_states,
            positions,
            forward_batch,
            indexer_metadata,
            cp_global_rope_positions,
        )

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        indexer_metadata,
    ) -> torch.Tensor:
        return self._forward_impl(
            hidden_states, positions, forward_batch, indexer_metadata
        )


__all__ = [
    "QSAIndexer",
]
