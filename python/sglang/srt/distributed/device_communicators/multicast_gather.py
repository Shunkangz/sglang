# SPDX-License-Identifier: Apache-2.0
"""Zero-SM, pausable allgather through the multicast VA of an NCCL window.

Every rank copies its own block into its slot of the output through the
output window's multicast (NVLS) address with copy-engine copies, so NVSwitch
replicates the bytes into every rank's backing memory: a rank's NVLink egress
is 1x its block, not (N-1)x, and no SM runs. One call is bracketed by two
barriers over the single-NVLink-domain team:

- entry: every rank's output is free to be overwritten (each rank's previous
  readers are ordered before its signal by the caller's stream order);
- exit: every rank's copies have landed, so stream-ordered readers on any rank
  see the full output.

A barrier is one 4-byte copy-engine copy of the constant 1 into this rank's
lane through the control region's multicast VA, then one batch of stream
memops on the LOCAL VA: wait until every lane is 1, then clear every lane.
Stream memops are rejected on multicast VAs, so only a copy can signal. The
entry and exit barriers alternate between two lane arrays: reusing one array
back to back would let a fast rank's next signal land before a slow rank's
clear and be lost.

Pause: with ``gated=True`` every data chunk is preceded by a stream wait for
this rank's local pause word to be 0. ``pause()`` / ``resume()`` write that
word in the order of any stream (local scope: each rank stops only its own
copies), so a paused gather stops at the next chunk boundary and resumes
without being resubmitted. Gates never precede a barrier, so a paused rank
never blocks a peer's barrier; it only delays its own copies and, through
them, every rank's exit. A resume must never be ordered behind a wait on the
paused gather. A parked gate also holds the hardware queue of its stream, so
work of any stream that shares that queue (CUDA_DEVICE_MAX_CONNECTIONS) stalls
behind it; a resume queued there never runs.

A stream may evaluate a wait ahead of the copies queued before it, which would
let several chunks pass their gates before a pause lands. Gated chunks
therefore alternate between the caller's stream and a side stream, and each
waits for an event recorded after the previous chunk's copy: a gate is read
only once the copy before it has retired, so at most one chunk drains.

Even unpaused, a long unbroken multicast copy starves concurrent NVLink
collectives (measured on 8x B200: a 32 MiB fused all-reduce 110 -> 960 us),
which chunking alone reduces (-> ~250-350 us) and pausing removes (-> 114 us).

Layout contract: the output is ``window[off : off + N * block]`` with rank r's
slot at ``off + r * block``, identical offsets on every rank. The control
region (``control_bytes(world_size)``) is carved out of the same window at an
offset identical on every rank and must be zeroed on every rank before any
rank's first call (``init_control``, a device sync, then a host barrier).
"""

from __future__ import annotations

import ctypes
import functools
from contextlib import nullcontext
from typing import Callable, Optional, Sequence, Tuple

import torch

_U32 = 4
_LINE = 64
# cuda.h: CUstreamBatchMemOpType / CUstreamWaitValue_flags / CUstreamWriteValue_flags
_CU_STREAM_MEM_OP_WAIT_VALUE_32 = 1
_CU_STREAM_MEM_OP_WRITE_VALUE_32 = 2
_CU_STREAM_WAIT_VALUE_EQ = 0x1
# CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER is rejected (CUDA_ERROR_INVALID_VALUE)
# on B200 / driver 13.4, so every write keeps the default fence.
_CU_STREAM_WRITE_VALUE_DEFAULT = 0x0
_CU_EVENT_DISABLE_TIMING = 0x2
# cuStreamBatchMemOp requires count < 256.
_MAX_BATCH_MEM_OPS = 255


class _CUstreamMemOpValueParams(ctypes.Structure):
    """The waitValue / writeValue member of the 48-byte
    ``CUstreamBatchMemOpParams`` union; the u32 value is the low half of a
    u32/u64 union."""

    _fields_ = [
        ("operation", ctypes.c_uint32),
        ("address", ctypes.c_uint64),
        ("value", ctypes.c_uint64),
        ("flags", ctypes.c_uint32),
        ("alias", ctypes.c_uint64),
        ("_pad", ctypes.c_uint64),
    ]


assert ctypes.sizeof(_CUstreamMemOpValueParams) == 48


class _CudaDriver:
    """The few driver entry points the gather needs, bound once via ctypes
    (about 2 us per call, mostly driver time)."""

    _instance: Optional[_CudaDriver] = None

    def __init__(self):
        lib = ctypes.CDLL("libcuda.so.1")
        void_p, u64, u32, uint = (
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint,
        )
        self._get_error_name = lib.cuGetErrorName
        self._get_error_name.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self.wait32 = self._bind(
            lib, "cuStreamWaitValue32_v2", [void_p, u64, u32, uint]
        )
        self.write32 = self._bind(
            lib, "cuStreamWriteValue32_v2", [void_p, u64, u32, uint]
        )
        self.batch_memop = self._bind(
            lib,
            "cuStreamBatchMemOp_v2",
            [void_p, uint, ctypes.POINTER(_CUstreamMemOpValueParams), uint],
        )
        self.memcpy = self._bind(
            lib, "cuMemcpyAsync", [u64, u64, ctypes.c_size_t, void_p]
        )
        self.event_create = self._bind(
            lib, "cuEventCreate", [ctypes.POINTER(void_p), uint]
        )
        self.event_record = self._bind(lib, "cuEventRecord", [void_p, void_p])
        self.stream_wait_event = self._bind(
            lib, "cuStreamWaitEvent", [void_p, void_p, uint]
        )

    @staticmethod
    def _bind(lib, name, argtypes):
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = ctypes.c_int
        return fn

    @classmethod
    def get(cls) -> _CudaDriver:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def check(self, result: int, what: str) -> None:
        if result != 0:
            name = ctypes.c_char_p()
            self._get_error_name(result, ctypes.byref(name))
            raise RuntimeError(
                f"{what} failed: {(name.value or b'?').decode()} ({result})"
            )


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def control_bytes(world_size: int) -> int:
    """Two lane arrays, the pause word and the constant 1 that signals copy
    from, each on its own 64-byte line."""
    return 2 * _align_up(world_size * _U32, _LINE) + 2 * _LINE


class MulticastAllGather:
    """Chunked copy-engine allgather into a multicast-mapped symmetric window.

    Calls must be issued in the same order on every rank, and consecutive
    calls on one instance must be ordered by the caller's stream.
    """

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        device_index: int,
        local_base: int,
        multicast_base: int,
        control_offset: int,
        chunk_bytes: Optional[int],
        gated: bool,
    ):
        assert 0 <= rank < world_size
        assert multicast_base, "the window has no multicast VA (NVLS off?)"
        assert chunk_bytes is None or chunk_bytes > 0
        assert not gated or chunk_bytes is not None, "pause gates need chunks"
        self.rank = rank
        self.world_size = world_size
        self.device_index = device_index
        self.local_base = local_base
        self.multicast_base = multicast_base
        self.chunk_bytes = chunk_bytes
        self.gated = gated
        self._drv = _CudaDriver.get()

        lane_stride = _align_up(world_size * _U32, _LINE)
        sync_offsets = (control_offset, control_offset + lane_stride)
        self._control_offset = control_offset
        self._pause_offset = control_offset + 2 * lane_stride
        self._one_offset = self._pause_offset + _LINE
        self._pause_addr = local_base + self._pause_offset
        self._one_addr = local_base + self._one_offset
        self._signal_addrs = tuple(
            multicast_base + off + rank * _U32 for off in sync_offsets
        )
        # Per channel: wait for every lane (this rank's included) to read 1,
        # then clear them all; batch ops execute in array order.
        assert 2 * world_size <= _MAX_BATCH_MEM_OPS
        self._barrier_ops = tuple(
            self._wait_then_clear_ops(local_base + off) for off in sync_offsets
        )
        if gated:
            # Chunk k runs on (caller's stream, side stream)[k % 2] and records
            # chunk_done[k % 2]; chunk k + 1 waits on it.
            with torch.cuda.device(device_index):
                self._side_stream = torch.cuda.Stream(device_index)
                self._chunk_done = tuple(self._create_event() for _ in range(2))

    def _create_event(self) -> int:
        event = ctypes.c_void_p()
        self._drv.check(
            self._drv.event_create(ctypes.byref(event), _CU_EVENT_DISABLE_TIMING),
            "cuEventCreate",
        )
        return event.value

    def _wait_then_clear_ops(self, lanes_addr: int):
        n = self.world_size
        ops = (_CUstreamMemOpValueParams * (2 * n))()
        for peer in range(n):
            wait, clear = ops[peer], ops[n + peer]
            wait.operation = _CU_STREAM_MEM_OP_WAIT_VALUE_32
            wait.address = lanes_addr + peer * _U32
            wait.value = 1
            wait.flags = _CU_STREAM_WAIT_VALUE_EQ
            clear.operation = _CU_STREAM_MEM_OP_WRITE_VALUE_32
            clear.address = lanes_addr + peer * _U32
            clear.value = 0
            clear.flags = _CU_STREAM_WRITE_VALUE_DEFAULT
        return ops

    def init_control(self, window: torch.Tensor) -> None:
        """Zero this rank's control region and set its constant 1, given a
        uint8 view of the whole window."""
        start = self._control_offset
        window[start : start + control_bytes(self.world_size)].zero_()
        window[self._one_offset : self._one_offset + _U32].view(torch.int32).fill_(1)

    def all_gather(
        self, blocks: Sequence[Tuple[int, int, int]], stream: torch.cuda.Stream
    ) -> None:
        """Enqueue one allgather on ``stream``.

        ``blocks``: ``(src_ptr, dst_offset, nbytes)`` per contiguous piece of
        this rank's contribution, ``dst_offset`` being where the piece sits in
        this rank's slot of the window. ``src_ptr`` may be that slot's own
        local address: the copy then rewrites identical bytes locally while
        multicasting them to the peers (verified bit-exact).
        """
        handle = stream.cuda_stream
        self._barrier(0, handle)
        copies = self._chunk_copies(blocks)
        if self.gated:
            self._enqueue_gated(copies, handle)
        else:
            for dst, src, nbytes in copies:
                self._copy(dst, src, nbytes, handle)
        self._barrier(1, handle)

    def _chunk_copies(self, blocks: Sequence[Tuple[int, int, int]]):
        """(multicast dst, src, nbytes) of every chunk, in order."""
        copies = []
        for src, dst_offset, nbytes in blocks:
            dst = self.multicast_base + dst_offset
            step = nbytes if self.chunk_bytes is None else self.chunk_bytes
            for off in range(0, nbytes, step):
                copies.append((dst + off, src + off, min(step, nbytes - off)))
        return copies

    def _enqueue_gated(self, copies, caller_handle: int) -> None:
        drv = self._drv
        streams = (caller_handle, self._side_stream.cuda_stream)
        done = self._chunk_done
        for k, (dst, src, nbytes) in enumerate(copies):
            handle = streams[k % 2]
            if k:
                drv.check(
                    drv.stream_wait_event(handle, done[(k - 1) % 2], 0),
                    "chunk chain (cuStreamWaitEvent)",
                )
            drv.check(
                drv.wait32(handle, self._pause_addr, 0, _CU_STREAM_WAIT_VALUE_EQ),
                "pause gate (cuStreamWaitValue32)",
            )
            self._copy(dst, src, nbytes, handle)
            drv.check(
                drv.event_record(done[k % 2], handle), "chunk chain (cuEventRecord)"
            )
        # The chain orders every chunk before the last; join it if it ran on
        # the side stream.
        if copies and len(copies) % 2 == 0:
            drv.check(
                drv.stream_wait_event(caller_handle, done[1], 0),
                "chunk chain join (cuStreamWaitEvent)",
            )

    def _copy(self, dst: int, src: int, nbytes: int, handle: int) -> None:
        self._drv.check(
            self._drv.memcpy(dst, src, nbytes, handle),
            "gather chunk (cuMemcpyAsync to multicast VA)",
        )

    def _barrier(self, channel: int, stream_handle: int) -> None:
        drv = self._drv
        drv.check(
            drv.memcpy(
                self._signal_addrs[channel], self._one_addr, _U32, stream_handle
            ),
            "barrier signal (cuMemcpyAsync to multicast VA)",
        )
        ops = self._barrier_ops[channel]
        drv.check(
            drv.batch_memop(stream_handle, len(ops), ops, 0),
            "barrier wait+clear (cuStreamBatchMemOp)",
        )

    def pause_current_stream(self) -> torch.cuda.Stream:
        """``pause`` on the device's current stream, which is returned."""
        stream = torch.cuda.current_stream(self.device_index)
        self.pause(stream)
        return stream

    def pause(self, stream: torch.cuda.Stream) -> None:
        """Stop this rank's copies at the next chunk boundary, in ``stream``
        order."""
        self._set_pause(1, stream)

    def resume(self, stream: torch.cuda.Stream) -> None:
        self._set_pause(0, stream)

    def _set_pause(self, value: int, stream: torch.cuda.Stream) -> None:
        drv = self._drv
        drv.check(
            drv.write32(
                stream.cuda_stream,
                self._pause_addr,
                value,
                _CU_STREAM_WRITE_VALUE_DEFAULT,
            ),
            "pause word (cuStreamWriteValue32)",
        )


# The gather a background prefetch may have in flight, set by its owner while
# it is pausable; None disables every bracket below.
_active_gather: Optional[MulticastAllGather] = None
# Brackets nest (a GroupCoordinator collective may call another); only the
# outermost one pauses and resumes.
_pause_depth = 0
_NOT_PAUSED = nullcontext()


def set_active_pausable_gather(gather: Optional[MulticastAllGather]) -> None:
    global _active_gather
    _active_gather = gather


def get_active_pausable_gather() -> Optional[MulticastAllGather]:
    return _active_gather


class _PausedGather:
    __slots__ = ("_gather", "_stream")

    def __init__(self, gather: MulticastAllGather):
        self._gather = gather
        self._stream = None

    def __enter__(self):
        global _pause_depth
        _pause_depth += 1
        if _pause_depth == 1:
            self._stream = self._gather.pause_current_stream()

    def __exit__(self, *exc):
        global _pause_depth
        _pause_depth -= 1
        if self._stream is not None:
            self._gather.resume(self._stream)
        return False


def background_gather_paused():
    """Bracket a latency-critical collective issued on the current stream:
    this rank's background gather copies stop for its duration.

    Deadlock-free as long as nothing inside the bracket waits on the paused
    gather: the resume follows the collective on this rank's own stream, and
    the collective needs only its peers' compute streams, never their gathers.
    """
    gather = _active_gather
    if gather is None or torch.compiler.is_compiling():
        return _NOT_PAUSED
    return _PausedGather(gather)


def pauses_background_gather(collective: Callable) -> Callable:
    """Decorator form of ``background_gather_paused`` for collective entry
    points; a plain call when no pausable gather is active."""

    @functools.wraps(collective)
    def wrapper(*args, **kwargs):
        if _active_gather is None or torch.compiler.is_compiling():
            return collective(*args, **kwargs)
        with _PausedGather(_active_gather):
            return collective(*args, **kwargs)

    return wrapper
