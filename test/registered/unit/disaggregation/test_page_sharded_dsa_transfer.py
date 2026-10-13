"""Owner-aware PD transfer for the MLA cache and DSA indexer keys."""

import asyncio
import dataclasses
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    PrefillServerInfo,
)
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.nixl.conn import (
    NixlKVManager,
    NixlKVSender,
    TransferStatus,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestPageShardedDSATransfer(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)

    def _manager(self, cls=CommonKVManager, rank=0, size=4):
        mgr = cls.__new__(cls)
        mgr.kv_shard_rank, mgr.kv_shard_size = rank, size
        mgr.attn_tp_size, mgr.attn_cp_size, mgr.attn_cp_rank = 4, 1, 0
        mgr.pp_size, mgr.pp_rank = 1, 0
        mgr.is_mla_backend, mgr.is_hybrid_mla_backend = True, False
        mgr.enable_all_cp_ranks_for_transfer = False
        mgr.kv_args = SimpleNamespace(
            engine_rank=0,
            state_types=[StateType.DSA],
            state_data_ptrs=[[1000, 2000]],
            state_item_lens=[[132, 132]],
            state_dim_per_tensor=[[]],
            state_layer_ids=[[0, 4]],
        )
        return mgr

    def _prefill(self, tp=4, cp=1):
        return PrefillServerInfo(
            attn_tp_size=tp,
            attn_cp_size=cp,
            dp_size=1,
            pp_size=1,
            page_size=64,
            kv_cache_dtype="bfloat16",
            follow_bootstrap_room=True,
            enable_kv_cache_sharding=True,
        )

    def test_all_source_owners_reconstruct_reordered_pages(self):
        # Ownership follows allocated page ids, not their prompt positions.
        pages = np.array([7, 16, 5, 10, 11, 20], dtype=np.int32)
        destinations = np.array([31, 25, 27, 40, 38, 30], dtype=np.int32)
        copied = {}
        for rank in range(4):
            mgr = self._manager(rank=rank)
            local, dst = mgr._prepare_dsa_state_indices(pages, destinations)
            for src_page, dst_page in zip(local, dst):
                self.assertNotIn(dst_page, copied)
                copied[dst_page] = src_page * 4 + rank
        self.assertEqual(copied, dict(zip(destinations, pages)))

    def test_state_indices_reject_positional_length_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "state index length mismatch"):
            self._manager()._prepare_dsa_state_indices([4, 5], [7])

    def test_unsharded_state_indices_are_preserved(self):
        src, dst = [4, 5], [12, 13]
        actual = self._manager(size=1)._prepare_dsa_state_indices(src, dst)
        self.assertIs(actual[0], src)
        self.assertIs(actual[1], dst)

    def test_kv_chunks_skip_empty_owners_but_deliver_final_metadata(self):
        sender = NixlKVSender.__new__(NixlKVSender)
        sender.kv_mgr = self._manager(rank=1)
        sender.curr_idx, sender.num_kv_indices = 0, 6
        local, positions, final, skip = sender._prepare_send_indices(
            np.array([8, 10], dtype=np.int32)
        )
        self.assertTrue(skip)
        self.assertFalse(final)
        local, positions, final, skip = sender._prepare_send_indices(
            np.array([17, 20], dtype=np.int32)
        )
        np.testing.assert_array_equal(local, [4])
        np.testing.assert_array_equal(positions, [2])
        self.assertFalse(skip)
        local, positions, final, skip = sender._prepare_send_indices(
            np.array([14, 15], dtype=np.int32)
        )
        self.assertTrue(final)
        self.assertFalse(skip)
        self.assertEqual(len(local), 0)

    def test_tp_and_cp_owners_all_participate(self):
        mgr = self._manager()
        info = self._prefill()
        mgr._resolve_rank_mapping(info)
        self.assertEqual(info.target_tp_ranks, [0, 1, 2, 3])
        self.assertIsNone(info.target_tp_rank)
        self.assertEqual(info.required_prefill_response_num, 4)
        self.assertEqual(info.required_dst_info_num, 4)
        info = self._prefill(cp=4)
        mgr._resolve_rank_mapping(info)
        self.assertEqual(info.target_tp_ranks, [0])
        self.assertEqual(info.target_cp_ranks, [0, 1, 2, 3])
        self.assertEqual(info.required_prefill_response_num, 4)
        mgr.attn_cp_size, mgr.attn_cp_rank = 4, 3
        self.assertFalse(mgr._should_skip_cp_replicated_state_transfer())
        mgr.kv_shard_size = 1
        self.assertTrue(mgr._should_skip_cp_replicated_state_transfer())

    def test_bootstrap_roundtrip_and_legacy_default(self):
        payload = dataclasses.asdict(self._prefill())
        self.assertTrue(PrefillServerInfo(**payload).enable_kv_cache_sharding)
        payload.pop("enable_kv_cache_sharding")
        self.assertFalse(PrefillServerInfo(**payload).enable_kv_cache_sharding)

    def test_bootstrap_endpoint_advertises_sharded_owners(self):
        bootstrap = CommonKVBootstrapServer.__new__(CommonKVBootstrapServer)
        for field in (
            "attn_tp_size",
            "attn_cp_size",
            "dp_size",
            "pp_size",
            "page_size",
            "kv_cache_dtype",
            "prefill_http_port",
            "follow_bootstrap_room",
            "enable_dsa_cache_layer_split",
            "enable_kv_cache_sharding",
        ):
            setattr(bootstrap, field, None)
        bootstrap._registered_count = 0
        bootstrap.prefill_port_table = {}
        bootstrap.lock = asyncio.Lock()
        payload = dataclasses.asdict(self._prefill(tp=1))
        payload.update(
            attn_tp_rank=0,
            attn_cp_rank=0,
            attn_dp_size=1,
            attn_dp_rank=0,
            pp_rank=0,
            system_dp_size=1,
            system_dp_rank=0,
            rank_ip="127.0.0.1",
            rank_port=10001,
        )
        response = asyncio.run(
            bootstrap._handle_route_put(
                SimpleNamespace(json=AsyncMock(return_value=payload))
            )
        )
        self.assertEqual(response.status, 200)
        response = asyncio.run(
            bootstrap._handle_route_get(
                SimpleNamespace(
                    query={
                        "prefill_dp_rank": "-1",
                        "prefill_cp_rank": "-1",
                        "target_tp_rank": "-1",
                        "target_pp_rank": "-1",
                    }
                )
            )
        )
        self.assertEqual(response.status, 200)
        self.assertTrue(
            PrefillServerInfo(**json.loads(response.text)).enable_kv_cache_sharding
        )

    def test_nixl_empty_owner_notification_completes_state_fan_in(self):
        mgr = self._manager(NixlKVManager)
        mgr.agent = Mock()
        mgr.agent.get_new_notifs.return_value = {"prefill": [b"1_state_3_0"]}
        status = TransferStatus(
            expected_kvs_per_pp={rank: 0 for rank in range(4)},
            num_pp_ranks_expected=4,
            received_aux=True,
            received_state_per_pp={0, 1, 2},
            expects_state=True,
        )
        mgr.transfer_statuses = {1: status}
        self.assertFalse(status.is_done())
        mgr.update_transfer_status()
        self.assertTrue(status.is_done())

    def test_both_transports_pair_indexer_sources_and_destinations(self):
        for cls in (NixlKVManager, MooncakeKVManager):
            for rank in (0, 3):
                with self.subTest(transport=cls.__name__, rank=rank):
                    mgr = self._manager(cls, rank=rank)
                    mgr._send_kvcache_generic = Mock(return_value=0)
                    mgr.agent = Mock()
                    src, dst = [16, 5, 20], [101, 102, 103]
                    if cls is NixlKVManager:
                        mgr.maybe_send_extra(
                            peer_name="decode",
                            prefill_state_indices=[src],
                            dst_state_data_ptrs=[[3000, 4000]],
                            dst_state_indices=[dst],
                            dst_gpu_id=0,
                            notif="room",
                            decode_tp_size=4,
                        )
                    else:
                        mgr.request_status = {1: KVPoll.Transferring}
                        mgr.maybe_send_extra(
                            req=SimpleNamespace(
                                dst_state_indices=[dst],
                                mooncake_session_id="decode",
                                room=1,
                            ),
                            prefill_state_indices=[src],
                            executor=None,
                        )
                    if rank == 3:
                        mgr._send_kvcache_generic.assert_not_called()
                        if cls is NixlKVManager:
                            mgr.agent.send_notif.assert_called_once_with(
                                "decode", b"room_0"
                            )
                    else:
                        kwargs = mgr._send_kvcache_generic.call_args.kwargs
                        np.testing.assert_array_equal(
                            kwargs["prefill_data_indices"], [4, 5]
                        )
                        np.testing.assert_array_equal(
                            kwargs["dst_data_indices"], [101, 103]
                        )


if __name__ == "__main__":
    unittest.main()
