"""Unit tests for model_loader/snapshot_walk.py and model_loader/validator.py."""

import functools
import os
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn

from sglang.srt.model_loader import snapshot_walk
from sglang.srt.model_loader.validator import (
    Validator,
    compare,
    fingerprint,
    live_meta_tensors,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _gemm_deepgemm(x, *, act_scale_ue8m0):
    return x


def _gemm_triton(x, *, act_scale_ue8m0):
    return x


class _QuantMethod:
    """Stands in for a quant method, a helper object of model code."""

    def __init__(self):
        self.weight_block_size = [128, 128]
        self.gemm = functools.partial(_gemm_deepgemm, act_scale_ue8m0=False)
        self.apply = self._apply  # a bound method: its repr carries an address

    def _apply(self, x):
        return x


class _Layer(nn.Module):
    def __init__(self, *, group):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 4), requires_grad=False)
        self.weight.format_ue8m0 = False  # a flag kept on the tensor, as on FP8 scales
        self.register_parameter("bias", None)
        self.register_buffer("cache", torch.arange(16.0), persistent=False)
        # Same layout as MLA's w_kc: a transposed view of a contiguous tensor.
        self.w_kc = torch.randn(2, 4, 8).transpose(1, 2).contiguous().transpose(1, 2)
        # A second path to the buffer, as a top-k config holds the gate's bias.
        self.cache_view = self.cache
        self.quant_method = _QuantMethod()
        self.tp_group = group


class _SubParameter(nn.Parameter):
    pass


class _Runner:
    def run(self):
        return None


class _EPRunner(_Runner):
    pass


def _model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    # A process-wide group outside model code, holding a Python module and a host
    # tensor, as GroupCoordinator holds torch.cuda and active_ranks_cpu.
    group = types.SimpleNamespace(
        device_module=torch.cuda, active_ranks_cpu=torch.ones(1, dtype=torch.int32)
    )
    model = nn.Module()
    model.layers = nn.ModuleList([_Layer(group=group), _Layer(group=group)])
    model.pp_group = group
    return model


def _closure_over(t: torch.Tensor):
    return lambda: t


def _paths(diff: list[str]) -> list[str]:
    return [line.split(":")[0] for line in diff]


def _diff(owner: nn.Module, here: nn.Module) -> list[str]:
    return compare(
        owner=fingerprint(owner, expert_map=None),
        here=fingerprint(here, expert_map=None),
    )


class TestSnapshotFingerprint(CustomTestCase):
    def setUp(self):
        # The helper classes above live in this test module, not under sglang.srt.
        model_code = (_QuantMethod.__module__,)
        self.enterContext(patch.object(snapshot_walk, "MODEL_CODE", model_code))

    def test_walk_stops_at_objects_outside_model_code(self):
        """Every linear layer keeps its TP group, whose device_module is the torch.cuda
        module: expanding it walked sys.modules until RecursionError, and the group's
        host tensor made the batched GPU hash raise."""
        entries = fingerprint(_model(), expert_map=None)
        self.assertEqual(entries["layers.0#tp_group"], "object types.SimpleNamespace")
        self.assertEqual([p for p in entries if "group." in p or "group#" in p], [])

    def test_cycles_end_the_walk(self):
        """A helper that refers back to itself, or a container that holds itself, must
        end the walk instead of recursing until RecursionError."""
        model = _model()
        method = model.layers[0].quant_method
        method.me = method
        cycle = [torch.ones(1)]
        cycle.append(cycle)
        method.cycle = cycle
        entries = fingerprint(model, expert_map=None)
        self.assertEqual(
            entries["layers.0.quant_method#me"],
            f"object {_QuantMethod.__module__}._QuantMethod",
        )
        self.assertEqual(
            entries["layers.0.quant_method.cycle#[1]"],
            "list[object torch.Tensor, <cycle>]",
        )

    def test_tensor_entry_frames_what_the_digest_cannot_see(self):
        """BLAKE3 covers a tensor's logical bytes only. Each change below keeps those
        bytes and must still change exactly its own entry."""
        model = _model()
        layer = model.layers[0]
        owner = fingerprint(model, expert_map=None)
        weight, w_kc, cache = layer.weight, layer.w_kc, layer.cache
        other_class = _SubParameter(weight.data, requires_grad=False)
        other_class.format_ue8m0 = False
        cases = [
            (
                "other strides",
                "layers.0.w_kc",
                lambda: setattr(layer, "w_kc", w_kc.contiguous()),
                lambda: setattr(layer, "w_kc", w_kc),
            ),
            (
                "other Parameter class",
                "layers.0.weight",
                lambda: layer._parameters.__setitem__("weight", other_class),
                lambda: layer._parameters.__setitem__("weight", weight),
            ),
            (
                "requires_grad set",
                "layers.0.weight",
                lambda: weight.requires_grad_(True),
                lambda: weight.requires_grad_(False),
            ),
            (
                "flag on the tensor flipped",
                "layers.0.weight#format_ue8m0",
                lambda: setattr(weight, "format_ue8m0", True),
                lambda: setattr(weight, "format_ue8m0", False),
            ),
            (
                "alias copied into its own storage",
                "layers.0.cache_view",
                lambda: setattr(layer, "cache_view", cache.clone()),
                lambda: setattr(layer, "cache_view", cache),
            ),
        ]
        for change, path, apply, revert in cases:
            with self.subTest(change=change):
                apply()
                try:
                    here = fingerprint(model, expert_map=None)
                finally:
                    revert()
                self.assertEqual(_paths(compare(owner=owner, here=here)), [path])

    def test_independent_builds_fingerprint_equal(self):
        """Nothing process-local may enter an entry. Two builds with the same values have
        other addresses, object ids and bound methods, and must fingerprint equal."""
        first = _model()
        shift = torch.empty(4099)  # moves the second build's allocations
        second = _model()
        for gpu, model in enumerate((first, second)):  # keyed by the local GPU
            model.per_device = {torch.device("cuda", gpu): torch.zeros(2)}
        # Their helpers' reprs differ, because a repr carries addresses.
        self.assertNotEqual(
            repr(vars(first.layers[0].quant_method)),
            repr(vars(second.layers[0].quant_method)),
        )
        self.assertEqual(_diff(first, second), [])
        del shift

    def test_set_text_does_not_depend_on_the_hash_seed(self):
        """The repr of a set of strings follows the per-process hash seed, and an SGLang
        module global already differs this way between two correct processes."""
        # Load the module alone: the model_loader package imports every loader.
        code = (
            "import importlib.util\n"
            f"spec = importlib.util.spec_from_file_location('w', {snapshot_walk.__file__!r})\n"
            "w = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(w)\n"
            "print(w.canonical(frozenset(['w13', 'w2', 'w13_scale', 'w2_scale', 'bias'])))\n"
        )
        texts = {
            subprocess.run(
                [sys.executable, "-c", code],
                env={**os.environ, "PYTHONHASHSEED": seed},
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            for seed in ("1", "2", "3")
        }
        self.assertEqual(len(texts), 1)

    def test_kernel_choice_inside_a_partial_is_compared(self):
        """The FP8 block GEMM is partial(deepgemm...) or partial(triton...); recorded by
        type name, both read 'partial' and a reader with the other kernel passes."""
        owner, here = _model(), _model()
        here.layers[0].quant_method.gemm = functools.partial(
            _gemm_triton, act_scale_ue8m0=False
        )
        self.assertEqual(_paths(_diff(owner, here)), ["layers.0.quant_method#gemm"])

    def test_none_slot_differs_from_absent(self):
        """A registered slot holding None is state: code tests it with `is None`, while a
        missing slot raises AttributeError."""
        owner, here = _model(), _model()
        del here.layers[0]._parameters["bias"]
        self.assertEqual(_paths(_diff(owner, here)), ["layers.0#bias"])

    def test_canonical_keeps_values_and_drops_the_checkpoint_path(self):
        """A PP stage's range or a rank's slice is a value; an inherited method acts on
        its receiver, whose class counts; and transformers names a remote-code module
        after the checkpoint's path, which differs between correct processes."""
        first, second = types.new_class("Config"), types.new_class("Config")
        first.__module__ = "transformers_modules.fp8.45b5eb19.configuration_deepseek"
        second.__module__ = (
            "transformers_modules.org.repo.c0ffee.configuration_deepseek"
        )
        cases = [
            ("slice", slice(0, 4), slice(4, 8), False),
            ("range", range(0, 2), range(2, 4), False),
            ("inherited method", _Runner().run, _EPRunner().run, False),
            ("remote-code class", first, second, True),
        ]
        for name, a, b, same in cases:
            with self.subTest(name):
                texts = snapshot_walk.canonical(a), snapshot_walk.canonical(b)
                self.assertEqual(texts[0] == texts[1], same, texts)

    def test_live_meta_scan_finds_a_reference_no_walk_reaches(self):
        """A meta tensor that install did not replace makes custom ops do nothing (a stale
        meta top-k bias gave wrong logits and no error), so the scan must find one that
        only a closure still reaches."""
        hidden = _closure_over(torch.empty(3, 7, 11, device="meta"))
        self.assertIn("Tensor(3, 7, 11)", live_meta_tensors())
        del hidden
        self.assertNotIn("Tensor(3, 7, 11)", live_meta_tensors())


class TestValidatorVote(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        dist.init_process_group(
            backend="gloo", store=dist.HashStore(), rank=0, world_size=1
        )

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()
        super().tearDownClass()

    def test_a_snapshot_mismatch_refuses_and_a_raw_load_is_not_compared(self):
        """A snapshot whose entry differs from the owner's must fail the vote and name the
        path; a raw load is the reference and passes whatever it holds."""
        model_code = (_QuantMethod.__module__,)
        with patch.object(snapshot_walk, "MODEL_CODE", model_code):
            model = _model()
            owner = fingerprint(model, expert_map=None)
            validator = Validator(cpu_group=dist.group.WORLD)
            validator.check_after_load(
                model=model, owner_fingerprint=owner, expert_map=None
            )
            model.layers[0].quant_method.weight_block_size = [64, 64]
            with self.assertRaisesRegex(
                RuntimeError, r"layers\.0\.quant_method#weight_block_size"
            ):
                validator.check_after_load(
                    model=model, owner_fingerprint=owner, expert_map=None
                )
            validator.check_after_load(
                model=model, owner_fingerprint=None, expert_map=None
            )


if __name__ == "__main__":
    unittest.main()
