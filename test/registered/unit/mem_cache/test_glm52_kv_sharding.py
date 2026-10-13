# Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 page-sharded KV cache configuration and pool construction."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.kv_shard_hook import handle_kv_cache_sharding
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.environ import envs
from sglang.srt.mem_cache import kv_cache_configurator, page_interleave
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _model_config():
    model = SimpleNamespace(
        is_encoder_decoder=False,
        is_hybrid_swa=False,
        linear_attn_registry_result=None,
        attention_chunk_size=None,
        attention_arch=AttentionArch.MLA,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        context_len=1024,
        hf_config=SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"],
            index_topk=2048,
            index_head_dim=128,
            indexer_types=["full", "full", "shared", "shared", "full"],
        ),
    )
    model.hf_config.get_text_config = lambda: model.hf_config
    return model


class TestGLM52ShardConfiguration(unittest.TestCase):
    def _args(self, **overrides):
        args = ServerArgs(
            model_path="dummy",
            enable_kv_cache_sharding=True,
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="mooncake",
            attention_backend="dsa",
            dsa_prefill_backend="flashmla_sparse",
            tp_size=2,
            page_size=64,
            chunked_prefill_size=128,
            device="cuda",
            cuda_graph_config=CudaGraphConfig(
                prefill=PhaseConfig(backend=Backend.DISABLED)
            ),
        )
        args._model_config = _model_config()
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def _validate(self, args):
        with (
            envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("python"),
            envs.SGLANG_DISAGG_STAGING_BUFFER.override(False),
        ):
            handle_kv_cache_sharding(args, 80 * 1024)

    def test_glm52_accepts_supported_prefill_backends(self):
        for backend in ("flashmla_sparse", "flashmla_kv", "trtllm"):
            for cp in (1, 2):
                with self.subTest(backend=backend, cp=cp):
                    self._validate(
                        self._args(
                            dsa_prefill_backend=backend,
                            attn_cp_size=cp,
                            cp_strategy="interleave",
                        )
                    )

    def test_unsupported_layouts_fail_before_allocation(self):
        cases = (
            ({"enable_dsa_cache_layer_split": True}, "mutually exclusive"),
            (
                {"disaggregation_enable_kv_checksum": True},
                "disaggregation-enable-kv-checksum",
            ),
            ({"dsa_prefill_backend": "tilelang"}, "requires CUDA"),
            ({"disable_chunked_prefix_cache": True}, "chunked prefix"),
            ({"attn_cp_size": 2, "cp_strategy": "in-seq-split"}, "interleave"),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    self._validate(self._args(**overrides))
        args = self._args()
        args._model_config.hf_config.index_kpool = 2
        with self.assertRaisesRegex(ValueError, "index_kpool=1"):
            self._validate(args)

    def test_hybrid_pool_layouts_are_rejected(self):
        for attr, value in (
            ("is_hybrid_swa", True),
            ("linear_attn_registry_result", ("custom", object())),
        ):
            with self.subTest(attr=attr):
                args = self._args()
                setattr(args._model_config, attr, value)
                with self.assertRaisesRegex(ValueError, "hybrid KV pools"):
                    self._validate(args)

    def test_scratch_budget_includes_latent_scales_and_indexer(self):
        group = SimpleNamespace(world_size=2, rank_in_group=0)
        parallel = SimpleNamespace(
            enable_kv_cache_sharding=True,
            attn_cp_group=SimpleNamespace(world_size=1),
            attn_tp_group=group,
        )
        for dtype, latent_bytes in ((torch.bfloat16, 1152), (torch.float8_e4m3fn, 656)):
            with self.subTest(dtype=dtype):
                kvc = SimpleNamespace(
                    is_draft_worker=False,
                    use_mla_backend=True,
                    page_size=64,
                    kv_cache_dtype=dtype,
                    model_config=_model_config(),
                )
                with (
                    patch.object(
                        page_interleave, "get_parallel", return_value=parallel
                    ),
                    patch.object(
                        page_interleave,
                        "get_schedule",
                        return_value=SimpleNamespace(chunked_prefill_size=128),
                    ),
                    patch.object(
                        kv_cache_configurator,
                        "get_exec",
                        return_value=SimpleNamespace(
                            kernel=SimpleNamespace(
                                dsa_prefill_backend="flashmla_sparse",
                                dsa_decode_backend="flashmla_kv",
                            )
                        ),
                    ),
                    patch.object(
                        kv_cache_configurator,
                        "get_disagg",
                        return_value=SimpleNamespace(disaggregation_mode="prefill"),
                    ),
                ):
                    spec = page_interleave.make_page_shard_spec(kvc)
                    self.assertEqual(
                        page_interleave.compute_page_shard_scratch_bytes(kvc),
                        2 * spec.scratch_rows * (latent_bytes + 128 + 4),
                    )

    def test_pool_factory_preserves_shared_indexer_layers(self):
        kvc = SimpleNamespace(
            model_config=_model_config(),
            kv_shard_rank=0,
            kv_shard_spec=object(),
            kv_shard_group=object(),
            pool_page_size=64,
            kv_cache_dtype=torch.bfloat16,
            layer_info=SimpleNamespace(
                start_layer=1, end_layer=5, num_effective_layers=4
            ),
            device="cuda",
            is_draft_worker=False,
        )
        with (
            patch(
                "sglang.srt.layers.cp.utils.get_glm_dsa_cp_layer_shard_info",
                return_value=(None, 1),
            ),
            patch.object(
                kv_cache_configurator, "PageInterleaveDSATokenToKVPool"
            ) as pool,
            patch.object(
                kv_cache_configurator, "_should_elide_dsa_index_k", return_value=True
            ),
            patch.object(
                kv_cache_configurator, "calculate_mla_kv_cache_dim", return_value=576
            ),
            patch.object(
                kv_cache_configurator,
                "max_speculative_num_draft_tokens",
                return_value=0,
            ),
            patch.object(
                kv_cache_configurator,
                "get_memory",
                return_value=SimpleNamespace(enable_hisparse=False),
            ),
            patch.object(
                kv_cache_configurator,
                "get_exec",
                return_value=SimpleNamespace(
                    features=SimpleNamespace(enable_memory_saver=False)
                ),
            ),
        ):
            result = KVCacheConfigurator._build_dsa_kv_pool(
                kvc,
                max_total_num_tokens=4096,
                max_running_requests=4,
                dsa_pool_class=DSATokenToKVPool,
            )
        self.assertIs(result, pool.return_value)
        self.assertIs(pool.call_args.kwargs["shard_spec"], kvc.kv_shard_spec)
        self.assertIs(pool.call_args.kwargs["shard_group"], kvc.kv_shard_group)
        self.assertEqual(
            pool.call_args.kwargs["skip_topk_layers"], [False, True, True, False]
        )


if __name__ == "__main__":
    unittest.main()
