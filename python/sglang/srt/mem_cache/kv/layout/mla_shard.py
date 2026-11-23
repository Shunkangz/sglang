"""The MLA view of a unified pool sharded by logical page.

The band holds this rank's logical pages only. Writes are `MLAView`'s; a cached
prefix is read from the sharding gather scratch (`PageInterleaveKVPoolMixin`),
addressed by logical token ids.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.kv.layout.mla import MLAView
from sglang.srt.mem_cache.page_interleave_pool import (
    PageInterleaveKVPoolMixin,
    PageInterleaveMLATokenToKVPool,
)

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.mem_cache.kv.alloc.kv_shard import ShardedOwner
    from sglang.srt.mem_cache.page_interleave import PageShardSpec


class ShardedMLAView(PageInterleaveKVPoolMixin, MLAView):
    # The latent scratch and its prefix read are the static sharded MLA pool's.
    _scratch_tensor_specs = PageInterleaveMLATokenToKVPool._scratch_tensor_specs
    _gather_pairs = PageInterleaveMLATokenToKVPool._gather_pairs
    get_mla_kv_buffer = PageInterleaveMLATokenToKVPool.get_mla_kv_buffer

    def bind_shard(
        self,
        shard_owner: ShardedOwner,
        shard_spec: PageShardSpec,
        shard_group: GroupCoordinator,
    ) -> None:
        # A rank stores its own pages only, where its band placed them.
        self.storage_rank_replicated = False
        self._owner_band = shard_owner.band
        self._send_rows_epoch = -1
        # The gather scratch lives beside the arena, under its memory saver.
        self.memory_saver_adapter = self._arena.memory_saver_adapter
        self._init_page_shard_state(shard_spec, shard_group)
        # The gather plan indexes its page table by logical page.
        assert shard_owner.token_id_space // self.page_size <= self._page_pos.numel()

    def _refuse_page_table_read(self) -> None:
        # A reader that indexes the band through a page table (absorbed MLA,
        # one-shot MHA) would see this rank's pages only.
        if self._shard_extend_active:
            raise NotImplementedError(
                "a sharded unified MLA view serves the chunked-prefix read only"
            )

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        self._refuse_page_table_read()
        return super().get_key_buffer(layer_id)

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        self._refuse_page_table_read()
        return super().get_value_buffer(layer_id)

    def _gather_send_rows(self, name: str) -> torch.Tensor:
        # The plan names this rank's pages as ``logical page // N`` (0 pads the
        # gather block); the band binds page k at band page k - 1 + its first.
        # Resolved once per plan: a page compaction moves later stays readable
        # where it was until the forward that planned over it retires.
        if self._send_rows_epoch != self._epoch:
            band = self._owner_band
            local_pages = self._send_pages
            band_pages = (local_pages - 1 + band.min_page_index).clamp_min(0)
            physical = torch.where(
                local_pages > 0, band.ids.virtual_to_physical[band_pages], 0
            ).clamp_min(0)
            offsets = torch.arange(
                self.page_size, dtype=torch.int64, device=self.device
            )
            self._owned_send_rows = (
                physical[:, None] * self.page_size + offsets
            ).reshape(-1)
            self._send_rows_epoch = self._epoch
        return self._owned_send_rows
