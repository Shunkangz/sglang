# SPDX-License-Identifier: Apache-2.0
"""Collective decisions during model loading and snapshot publication."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import msgspec
import torch.distributed as dist

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator


class Ballot(msgspec.Struct, frozen=True, kw_only=True):
    """One rank's error, or a shared agreement value and a rank-specific payload."""

    error: str | None = None
    agree: bytes = b""
    payload: bytes = b""


def vote(
    *,
    group: GroupCoordinator,
    ballot: Ballot,
    timeout: timedelta = timedelta(seconds=480),
) -> list[Ballot]:
    """Return every ballot in group rank order, or raise on every participating rank.

    Call before fetching weights with a common spec/version, after validating with
    a local result, or before announcing a snapshot with a digest per owner rank.
    Every participant must enter the same vote, including ranks reporting an error.
    The timeout applies to the monitored Gloo barrier.
    """
    if dist.get_backend(group.cpu_group) == "mooncake-cpu":
        group.barrier()
    else:
        dist.monitored_barrier(
            group=group.cpu_group, timeout=timeout, wait_all_ranks=True
        )

    ballots = group.all_gather_object(ballot)
    disagreement = len({peer.agree for peer in ballots if peer.error is None}) > 1
    problems = []
    for rank, peer in zip(group.ranks, ballots):
        if peer.error is not None:
            problems.append(f"[rank {rank}] {peer.error}")
        elif disagreement:
            problems.append(f"[rank {rank}] conflicting agreement value {peer.agree!r}")
    if problems:
        raise RuntimeError("Model load vote failed:\n" + "\n".join(problems))
    return ballots
