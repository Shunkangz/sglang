"""The KV-shard own-block pack (``pack_rows``) is byte-identical to
``torch.index_select`` on every gather pair a page-interleave pool produces.

    python -m pytest test/registered/unit/mem_cache/test_page_interleave_pack_rows.py -v
"""

import unittest
from contextlib import nullcontext
from unittest.mock import patch

import torch

from sglang.kernels.ops.kvcache import pack_rows as pack_rows_module
from sglang.kernels.ops.kvcache.pack_rows import pack_rows
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

_PAGE = 64
_SIZE = _PAGE * 64
_SHARD_SIZE = 4


def _real_gather_pairs():
    """(label, pool buffer, rows unit) of each pool the sharded subclasses
    inherit; the indexer's first dimension counts pages, the others tokens."""
    common = dict(
        size=_SIZE,
        page_size=_PAGE,
        layer_num=1,
        device="cuda",
        enable_memory_saver=False,
    )
    mha = MHATokenToKVPool(
        dtype=torch.bfloat16,
        head_num=2,
        head_dim=256,
        enable_alt_stream=False,
        **common,
    )
    mha_fp8 = MHATokenToKVPool(
        dtype=torch.float8_e4m3fn,
        head_num=1,
        head_dim=128,
        enable_alt_stream=False,
        **common,
    )
    mla_kwargs = dict(kv_lora_rank=512, qk_rope_head_dim=64, **common)
    mla = MLATokenToKVPool(dtype=torch.bfloat16, **mla_kwargs)
    dsa = DSATokenToKVPool(
        dtype=torch.float8_e4m3fn,
        index_head_dim=128,
        kv_cache_dim=656,
        **mla_kwargs,
    )
    return [
        ("mha k bf16", mha.k_buffer[0], _PAGE),
        ("mha v bf16", mha.v_buffer[0], _PAGE),
        ("mha k fp8", mha_fp8.k_buffer[0], _PAGE),
        ("mla kv bf16", mla.kv_buffer[0], _PAGE),
        ("dsa kv packed fp8", dsa.kv_buffer[0], _PAGE),
        ("dsa index_k", dsa.index_k_with_scale_buffer[0], 1),
    ]


def _send_rows(num_pages, owned_pages, pad_pages, rows_per_page, generator):
    """Owned pages in shuffled order, padded with reserved page 0, expanded to
    rows as ``begin_shard_extend`` does."""
    pages = torch.randperm(num_pages - 1, generator=generator)[:owned_pages] + 1
    pages = torch.cat([pages, torch.zeros(pad_pages, dtype=torch.int64)]).cuda()
    offsets = torch.arange(rows_per_page, device="cuda")
    return (pages[:, None] * rows_per_page + offsets).reshape(-1)


def _fill_random(buf, generator):
    raw = buf.view(torch.uint8)
    raw.copy_(torch.randint(0, 256, raw.shape, generator=generator).to(raw))


@unittest.skipUnless(torch.cuda.is_available(), "pack_rows needs CUDA")
class TestPackRowsMatchesIndexSelect(unittest.TestCase):
    def _check(self, src, rows, *, kernel_only=False, block_offset=1):
        """Pack into a non-first rank block of a scratch, like the gather does."""
        block = rows.numel()
        scratch = torch.zeros(
            (_SHARD_SIZE * block, *src.shape[1:]), dtype=src.dtype, device="cuda"
        )
        expected = scratch.clone()
        torch.index_select(
            src,
            0,
            rows,
            out=expected[block_offset * block : (block_offset + 1) * block],
        )
        with (
            patch.object(
                pack_rows_module.torch,
                "index_select",
                side_effect=AssertionError("fell back to index_select"),
            )
            if kernel_only
            else nullcontext()
        ):
            pack_rows(
                src=src,
                rows=rows,
                out=scratch[block_offset * block : (block_offset + 1) * block],
            )
        self.assertTrue(
            torch.equal(scratch.view(torch.uint8), expected.view(torch.uint8))
        )

    def test_real_pairs_take_the_kernel_and_match(self):
        generator = torch.Generator().manual_seed(0)
        with get_parallel().override(attn_dcp_size=1):
            pairs = _real_gather_pairs()
        for label, src, rows_per_page in pairs:
            _fill_random(src, generator)
            num_pages = src.shape[0] // rows_per_page
            for owned, pad in ((num_pages // _SHARD_SIZE, 2), (5, 0), (0, 3)):
                rows = _send_rows(num_pages, owned, pad, rows_per_page, generator)
                with self.subTest(pair=label, owned=owned, pad=pad):
                    self._check(src, rows, kernel_only=True)

    def test_odd_and_strided_rows_still_match(self):
        """Rows whose bytes are not a multiple of 4 take narrower words; rows
        that are not contiguous fall back to index_select."""
        generator = torch.Generator().manual_seed(1)
        odd = torch.empty((_PAGE * 4, 3, 5), dtype=torch.uint8, device="cuda")
        wide = torch.empty((_PAGE * 4, 2, 64), dtype=torch.bfloat16, device="cuda")
        for src in (odd, odd[:, :, 1:], wide[:, 1:], wide[:, :, ::2]):
            _fill_random(odd, generator)
            _fill_random(wide, generator)
            rows = _send_rows(4, 2, 1, _PAGE, generator)
            with self.subTest(shape=tuple(src.shape), stride=src.stride()):
                self._check(src, rows)


if __name__ == "__main__":
    unittest.main()
