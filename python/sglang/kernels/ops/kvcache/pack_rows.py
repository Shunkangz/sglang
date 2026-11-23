"""Row gather (``index_select`` along dim 0) on a capped persistent grid, so a
side-stream pack leaves most SMs to the compute stream."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Leaves 116 of 148 B200 SMs to the compute stream; measured 1.6 ms per 2 GiB
# (index_select: 1.1 ms on every SM). Compute slowdown measured flat over 16-64.
PACK_ROWS_NUM_CTAS = 32
# 64 KiB per CTA per iteration in registers (no shared memory, so compute
# kernels can still share the SM); measured fastest at this grid on B200.
_TILE_WORDS = 16384
_MAX_BLOCK_COLS = 1024


# The DSA indexer's row count (block_pages) changes every batch; specializing
# on it would only compile more variants, and those mid-serving.
@triton.jit(do_not_specialize=["num_rows"])
def _pack_rows_kernel(
    src,
    rows,
    out,
    num_rows,
    src_row_stride,
    ROW_WORDS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    num_tiles = tl.cdiv(num_rows, BLOCK_ROWS)
    for tile in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
        r = tile * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        row_mask = r < num_rows
        idx = tl.load(rows + r, mask=row_mask, other=0).to(tl.int64)
        src_base = idx[:, None] * src_row_stride
        out_base = r.to(tl.int64)[:, None] * ROW_WORDS
        for c0 in tl.static_range(0, ROW_WORDS, BLOCK_COLS):
            cols = c0 + tl.arange(0, BLOCK_COLS)
            mask = row_mask[:, None] & (cols[None, :] < ROW_WORDS)
            values = tl.load(src + src_base + cols[None, :], mask=mask)
            tl.store(out + out_base + cols[None, :], values, mask=mask)


def _word_dtype(src: torch.Tensor, out: torch.Tensor):
    """Widest of int32/int16/uint8 dividing every row offset and pointer."""
    size = src.element_size()
    spans = (
        src.shape[-1] * size,
        *(stride * size for stride in src.stride()[:-1]),
        src.data_ptr(),
        out.data_ptr(),
    )
    for dtype in (torch.int32, torch.int16, torch.uint8):
        if all(span % dtype.itemsize == 0 for span in spans):
            return dtype


def pack_rows(src: torch.Tensor, rows: torch.Tensor, out: torch.Tensor) -> None:
    """``torch.index_select(src, 0, rows, out=out)`` on the current stream,
    byte for byte, when each row of ``src`` and all of ``out`` are contiguous."""
    # The 32-warp CTA exceeds a ROCm block (64-lane wavefronts).
    if not (
        src.is_cuda
        and torch.version.hip is None
        and src.dim() >= 2
        and src.stride(-1) == 1
        and src[0].is_contiguous()
        and out.is_contiguous()
        and out.dtype == src.dtype
        and out.shape == (rows.numel(), *src.shape[1:])
        and rows.dtype in (torch.int32, torch.int64)
        and rows.is_contiguous()
    ):
        torch.index_select(src, 0, rows, out=out)
        return
    word = _word_dtype(src, out)
    src_words, out_words = src.view(word), out.view(word)
    row_words = src_words[0].numel()
    block_cols = min(triton.next_power_of_2(row_words), _MAX_BLOCK_COLS)
    block_rows = max(1, _TILE_WORDS // block_cols)
    grid = min(PACK_ROWS_NUM_CTAS, triton.cdiv(rows.numel(), block_rows))
    _pack_rows_kernel[(grid,)](
        src_words,
        rows,
        out_words,
        rows.numel(),
        src_words.stride(0),
        ROW_WORDS=row_words,
        BLOCK_ROWS=block_rows,
        BLOCK_COLS=block_cols,
        num_warps=32,
        num_stages=1,
    )
