import unittest
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.model_executor.model_runner_components.load_model_utils import (
    Ballot,
    vote,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=47, suite="base-a-test-cpu")


def _check_failed_vote(group, ballot, *expected):
    try:
        vote(group=group, ballot=ballot)
    except RuntimeError as exc:
        report = str(exc)
    else:
        report = None

    reports = group.all_gather_object(report)
    assert report is not None and all(peer == report for peer in reports), reports
    for fragment in expected:
        assert fragment in report, report


def _run_rank(rank, init_method):
    timeout = timedelta(seconds=30)
    dist.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=3, timeout=timeout
    )
    try:
        # A PP stage can vote without global rank zero. Its reports must still
        # identify failures by their global rank.
        ranks = [1, 2]
        cpu_group = dist.new_group(ranks=ranks, backend="gloo", timeout=timeout)
        if rank not in ranks:
            return
        group = GroupCoordinator.__new__(GroupCoordinator)
        group.cpu_group = cpu_group
        group.ranks = ranks
        group.world_size = len(ranks)

        ballots = [
            Ballot(agree=b"version-1", payload=f"shard-{r}".encode()) for r in ranks
        ]
        assert vote(group=group, ballot=ballots[ranks.index(rank)]) == ballots

        _check_failed_vote(
            group,
            Ballot(agree=b"snapshot" if rank == 1 else b"miss"),
            "rank 1",
            "snapshot",
            "rank 2",
            "miss",
        )
        _check_failed_vote(
            group,
            Ballot(error="checksum mismatch" if rank == 2 else None),
            "rank 2",
            "checksum mismatch",
        )
        _check_failed_vote(
            group,
            Ballot(error="disk full" if rank == 1 else "write timeout"),
            "rank 1",
            "disk full",
            "rank 2",
            "write timeout",
        )
    finally:
        dist.destroy_process_group()


class TestLoadModelVote(unittest.TestCase):
    def test_subgroup_vote(self):
        with TemporaryDirectory() as directory:
            init_method = (Path(directory) / "gloo-init").as_uri()
            mp.spawn(_run_rank, args=(init_method,), nprocs=3, join=True)

    def test_missing_rank_preserves_gloo_diagnostic(self):
        group = Mock()
        error = RuntimeError("Rank 3 failed to pass monitoredBarrier")
        with (
            patch.object(dist, "get_backend", return_value="gloo"),
            patch.object(dist, "monitored_barrier", side_effect=error) as barrier,
            self.assertRaises(RuntimeError) as raised,
        ):
            vote(group=group, ballot=Ballot())

        self.assertIs(raised.exception, error)
        barrier.assert_called_once_with(
            group=group.cpu_group, timeout=timedelta(seconds=480), wait_all_ranks=True
        )
        group.all_gather_object.assert_not_called()

    def test_mooncake_uses_plain_cpu_barrier(self):
        group = Mock(ranks=[0])
        ballot = Ballot()
        group.all_gather_object.return_value = [ballot]
        with (
            patch.object(dist, "get_backend", return_value="mooncake-cpu"),
            patch.object(dist, "monitored_barrier") as monitored,
        ):
            self.assertEqual(vote(group=group, ballot=ballot), [ballot])

        monitored.assert_not_called()
        group.barrier.assert_called_once_with()
        group.all_gather_object.assert_called_once_with(ballot)


if __name__ == "__main__":
    unittest.main()
