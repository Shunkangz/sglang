"""When the engine turns NCCL cuMem on for the zero-SM all-gathers."""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.entrypoints import engine
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _cfg(**fields):
    base = dict(
        nnodes=1,
        enable_memory_saver=False,
        enable_kv_cache_sharding=False,
        enable_cp_tp_group_sharing=True,
        cp_strategy="zigzag",
    )
    base.update(fields)
    return SimpleNamespace(**base)


class TestZeroSMGatherNeedsCuMem(CustomTestCase):
    def _needs(self, cfg, nccl_supported=True):
        with mock.patch.object(
            engine, "zero_sm_all_gather_nccl_supported", return_value=nccl_supported
        ):
            return engine._zero_sm_gather_needs_cumem(cfg)

    def test_qsa_prefill_cp_turns_cumem_on(self):
        self.assertTrue(self._needs(_cfg()))

    def test_kv_shard_turns_cumem_on(self):
        cfg = _cfg(enable_kv_cache_sharding=True, enable_cp_tp_group_sharing=False)
        self.assertTrue(self._needs(cfg))

    def test_paths_that_cannot_use_the_windows_keep_it_off(self):
        """cuMem changes every communicator's memory, so it stays off where no
        zero-SM all-gather can run."""
        cases = {
            "multi-node": _cfg(nnodes=2),
            "memory saver": _cfg(enable_memory_saver=True),
            "GLM CP-TP sharing (interleave)": _cfg(cp_strategy="interleave"),
            "no CP-TP sharing": _cfg(enable_cp_tp_group_sharing=False),
        }
        for name, cfg in cases.items():
            with self.subTest(name):
                self.assertFalse(self._needs(cfg))
        with self.subTest("NCCL < 2.31.2"):
            self.assertFalse(self._needs(_cfg(), nccl_supported=False))
        with self.subTest("custom mem pool"):
            with envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.override("NVLINK"):
                self.assertFalse(self._needs(_cfg()))


if __name__ == "__main__":
    unittest.main()
