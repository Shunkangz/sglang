"""Native allgather chunks: final layout, buffer safety, and real gated runs."""

import time
import unittest
from contextlib import contextmanager, nullcontext
from unittest.mock import patch

import torch
import torch.multiprocessing as mp

from sglang.srt.distributed.device_communicators.gated_transfer import (
    Link,
    TransferChain,
    paused_background_transfers,
)
from sglang.srt.distributed.device_communicators.nccl_transfer import all_gather_chunks
from sglang.srt.environ import envs
from sglang.srt.utils.common import temp_set_env
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=60, stage="base-c", runner_config="4-gpu-b200")


class _CpuCommunicator:
    """An independent reference collective, including the local contribution."""

    def __init__(self, values, rank=1):
        self.values, self.rank = values, rank
        self.world_size, self.count = values.shape
        self.device = torch.device("cpu")
        self.enabled, self.offset = False, 0
        self.calls = []

    @contextmanager
    def change_state(self, enable):
        assert enable and not self.enabled
        self.enabled = True
        try:
            yield
        finally:
            self.enabled = False

    def all_gather(self, output, input_):
        assert self.enabled
        size = input_.numel()
        expected = self.values[:, self.offset : self.offset + size]
        torch.testing.assert_close(input_, expected[self.rank])
        output.copy_(expected.reshape(-1))
        self.calls.append(size)
        self.offset = (self.offset + size) % self.count


class TestNativeAllGatherChunks(CustomTestCase):
    def _run(self, chunks):
        stream = object()
        with patch.object(torch.cuda, "stream", return_value=nullcontext()) as current:
            for chunk in chunks:
                chunk(stream)
        self.assertEqual(current.call_count, len(chunks))
        for call in current.call_args_list:
            self.assertIs(call.args[0], stream)

    def test_chunked_layout_tail_offsets_and_in_place_source(self):
        values = torch.arange(3 * 19, dtype=torch.float32).view(3, 19)
        for in_place in (False, True):
            with self.subTest(in_place=in_place):
                comm = _CpuCommunicator(values)
                storage = torch.full((3 * 19 + 10,), -1.0)
                output = storage[3:-7]
                own = output[19:38]
                own.copy_(values[comm.rank])
                input_ = own if in_place else values[comm.rank].clone()
                scratch_storage = torch.full((3 * 8 + 5,), -2.0)
                scratch = scratch_storage[2:-3]
                chunks = all_gather_chunks(
                    comm,
                    output.view(57, 1),
                    input_.view(19, 1),
                    scratch=scratch,
                    chunk_bytes=8 * 4,
                )
                # Reusing scratch across queued runs must not change view layout.
                for _ in range(2):
                    self._run(chunks)
                    torch.testing.assert_close(output.view(3, 19), values)
                self.assertEqual(comm.calls, [8, 8, 3, 8, 8, 3])
                self.assertTrue(bool((storage[:3] == -1).all()))
                self.assertTrue(bool((storage[-7:] == -1).all()))
                self.assertTrue(bool((scratch_storage[:2] == -2).all()))
                self.assertTrue(bool((scratch_storage[-3:] == -2).all()))

    def test_one_native_collective_needs_no_scratch(self):
        values = torch.arange(3 * 19, dtype=torch.float32).view(3, 19)
        for chunk_bytes in (None, 19 * 4, 20 * 4):
            with self.subTest(chunk_bytes=chunk_bytes):
                comm = _CpuCommunicator(values)
                output = torch.empty(57)
                chunks = all_gather_chunks(
                    comm, output, values[comm.rank], chunk_bytes=chunk_bytes
                )
                self._run(chunks)
                self.assertEqual(comm.calls, [19])
                torch.testing.assert_close(output.view(3, 19), values)

    def test_single_rank_and_empty_inputs(self):
        values = torch.arange(19, dtype=torch.float32).view(1, 19)
        comm = _CpuCommunicator(values, rank=0)
        output = torch.empty(19)
        chunks = all_gather_chunks(comm, output, values[0], chunk_bytes=4)
        self.assertEqual(len(chunks), 1)  # single-rank local-copy shortcut
        self._run(chunks)
        torch.testing.assert_close(output, values[0])
        self.assertEqual(comm.calls, [])
        self.assertEqual(all_gather_chunks(comm, output[:0], values[0, :0]), [])

    def test_rejects_invalid_buffers_before_enqueuing(self):
        values = torch.arange(3 * 19, dtype=torch.float32).view(3, 19)
        comm = _CpuCommunicator(values)
        output, input_, scratch = torch.empty(57), values[1].clone(), torch.empty(24)
        cases = [
            {"chunk_bytes": 0},
            {"chunk_bytes": -4},
            {"chunk_bytes": 3},
            {"output": output[:-1]},
            {"output": output.to(torch.float64)},
            {"input_": torch.empty(38)[::2]},
            {"output": torch.empty(114)[::2]},
            {"input_": output[:19]},
            {"input_": output[20:39]},
            {"scratch": None},
            {"scratch": scratch[:-1]},
            {"scratch": scratch.to(torch.float64)},
            {"scratch": torch.empty(48)[::2]},
            {"scratch": output[:24]},
            {"scratch": input_[:12], "chunk_bytes": 4 * 4},
        ]
        for change in cases:
            with self.subTest(change=change):
                args = dict(
                    output=output, input_=input_, scratch=scratch, chunk_bytes=32
                )
                args.update(change)
                with self.assertRaises(ValueError):
                    all_gather_chunks(comm, **args)
        comm.device = torch.device("cuda", 0)
        with self.assertRaisesRegex(ValueError, "communicator's device"):
            all_gather_chunks(comm, output, input_)
        self.assertEqual(comm.calls, [])


_WORLD = 4 if torch.cuda.device_count() >= 4 else 2


def _wait_until(predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "native NCCL transfer did not complete"
        time.sleep(0.005)


def _run_native_allgather(rank, port, symmetric):
    from cuda.bindings import driver as cu

    from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        allocate_symmetric_buffer,
    )
    from sglang.srt.distributed.device_communicators.pynccl_wrapper import (
        NCCL_CTA_POLICY_ZERO,
    )
    from sglang.srt.utils.cuda_vmm_utils import check_drv

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=_WORLD
    )
    comm = PyNcclCommunicator(
        group=torch.distributed.group.WORLD,
        device=rank,
        cta_policy=NCCL_CTA_POLICY_ZERO if symmetric else None,
    )
    # Foreground calls must not share the background communicator's ordering.
    foreground = PyNcclCommunicator(group=torch.distributed.group.WORLD, device=rank)
    count, chunk = (1 << 18) + 17, 1 << 16
    storage = torch.full((_WORLD * count + 10,), -1.0, device="cuda")
    output = storage[3:-7]
    own = output[rank * count : (rank + 1) * count]
    local = torch.arange(count, dtype=torch.float32, device="cuda") + rank * count
    expected = torch.arange(_WORLD * count, dtype=torch.float32, device="cuda")
    if symmetric:
        scratch_buffer, registered = allocate_symmetric_buffer(comm, _WORLD * chunk * 4)
        assert registered, "NCCL did not register the symmetric scratch window"
        scratch = scratch_buffer.view(torch.float32)
    else:
        scratch = torch.empty(_WORLD * chunk, device="cuda")
    after, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    chain = TransferChain(Link.NVLINK)
    x = torch.full((4096,), float(rank + 1), device="cuda")
    with foreground.change_state(enable=True):
        foreground.all_reduce(x)
    own.copy_(local)
    chunks = all_gather_chunks(
        comm, output, own, scratch=scratch, chunk_bytes=chunk * 4
    )
    # Warm every native collective shape and local copy before a gate can park.
    for op in chunks:
        op(torch.cuda.current_stream())
    torch.cuda.synchronize()
    torch.distributed.barrier()

    def consume(ticket):
        chain.wait(ticket, stream=consumer)
        done = torch.cuda.Event()
        done.record(consumer)
        return done

    def prepare(round_id, target_storage=storage):
        # Fresh values and poisoned remote slots make stale warmup/prior results fail.
        target_storage.fill_(-1)
        target = target_storage[3:-7]
        target_own = target[rank * count : (rank + 1) * count]
        target_own.copy_(local)
        target_own.add_(round_id * _WORLD * count)

    def check(round_id, target_storage=storage):
        torch.testing.assert_close(
            target_storage[3:-7], expected + round_id * _WORLD * count, rtol=0, atol=0
        )
        assert bool((target_storage[:3] == -1).all())
        assert bool((target_storage[-7:] == -1).all())

    # Two outstanding calls share scratch safely because a chain serializes runs.
    second_storage = torch.empty_like(storage)
    second_output = second_storage[3:-7]
    second_chunks = all_gather_chunks(
        comm,
        second_output,
        second_output[rank * count : (rank + 1) * count],
        scratch=scratch,
        chunk_bytes=chunk * 4,
    )
    prepare(1)
    prepare(2, second_storage)
    scratch.fill_(-2)
    torch.cuda.synchronize()
    torch.distributed.barrier()
    first = chain.run(chunks, after=after)
    second = chain.run(second_chunks, after=after)
    _wait_until(consume(first).query)
    check(1)
    _wait_until(consume(second).query)
    check(2, second_storage)

    # A rank paused before submission holds every rank's NCCL completion.
    prepare(3)
    scratch.fill_(-2)
    torch.cuda.synchronize()
    if rank == 0:
        chain._gate._word.value = 3
    torch.distributed.barrier()
    done = consume(chain.run(chunks, after=after))
    time.sleep(0.1)
    stalled = not done.query()
    torch.distributed.barrier()
    if rank == 0:
        chain._gate.force_open()
    _wait_until(done.query)
    assert stalled, "allgather completed while one participant was paused"
    check(3)

    # Public bracket around a real foreground collective, with no watchdog rescue.
    blocker = torch.zeros(1, dtype=torch.int32).pin_memory()

    def held(stream):
        check_drv(
            cu.cuStreamWaitValue32(stream.cuda_stream, blocker.data_ptr(), 1, 1),
            "test native allgather blocker",
        )
        chunks[0](stream)

    prepare(4)
    scratch.fill_(-2)
    x.fill_(rank + 1)
    torch.cuda.synchronize()
    torch.distributed.barrier()
    ticket = chain.run([held, *chunks[1:]], after=after)
    pauses, forced = chain._gate._seq, chain._gate.forced_opens
    with foreground.change_state(enable=True):
        with paused_background_transfers(Link.NVLINK):
            foreground.all_reduce(x)
    assert chain._gate._seq == pauses + 1
    blocker[0] = 1
    _wait_until(consume(ticket).query)
    torch.cuda.synchronize()
    check(4)
    assert bool((x == sum(range(1, _WORLD + 1))).all())
    assert chain._gate.forced_opens == forced
    torch.distributed.barrier()


@unittest.skipIf(torch.cuda.device_count() < 2, "needs 2 CUDA devices")
class TestNativeAllGatherMultiGpu(CustomTestCase):
    def _spawn(self, symmetric, port):
        with (
            envs.SGLANG_GATED_TRANSFER_WATCHDOG_MS.override(0),
            temp_set_env(
                NCCL_CUMEM_ENABLE="1",
                NCCL_NVLS_ENABLE="1",
                CUDA_DEVICE_MAX_CONNECTIONS="32",
            ),
        ):
            mp.spawn(
                _run_native_allgather, args=(port, symmetric), nprocs=_WORLD, join=True
            )

    def test_native_allgather_chunks(self):
        self._spawn(False, 29834)

    def test_symmetric_native_allgather_chunks(self):
        from sglang.srt.distributed.device_communicators.pynccl_wrapper import (
            NCCLLibrary,
        )

        if NCCLLibrary().ncclGetRawVersion() < 23102:
            self.skipTest("symmetric copy-engine scratch requires NCCL >= 2.31.2")
        self._spawn(True, 29835)


if __name__ == "__main__":
    unittest.main()
