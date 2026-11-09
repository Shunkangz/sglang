"""CUDA unit test for model_loader/validator.py: hashing tensors on every device."""

import unittest

import torch
from torch import nn

from sglang.srt.model_loader.validator import fingerprint
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=40, stage="base-b", runner_config="1-gpu-small")


class TestValidatorOnCuda(CustomTestCase):
    def test_host_and_device_tensors_hash_alike(self):
        """gpu_tensors_blake3 rejects host tensors and mixed devices, and every process
        group holds a host tensor. Each device is hashed on its own, and a tensor's
        digest must not depend on where it lives."""
        data = torch.arange(1023, dtype=torch.int32).to(torch.uint8)  # odd byte count
        model = nn.Module()
        model.register_buffer("on_device", data.cuda())
        model.on_host = data
        entries = fingerprint(model, expert_map=None)
        self.assertEqual(
            entries["on_device"].split()[-1], entries["on_host"].split()[-1]
        )
        self.assertIn(" cuda ", entries["on_device"])
        self.assertIn(" cpu ", entries["on_host"])


if __name__ == "__main__":
    unittest.main()
