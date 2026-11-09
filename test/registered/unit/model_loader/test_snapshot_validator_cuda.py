"""CUDA unit tests for model_loader/validator.py: device hashing and address detection."""

import unittest

import torch
from torch import nn

from sglang.srt.model_loader.validator import find_device_addresses, fingerprint
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

    def test_device_addresses_are_found(self):
        """Shipped, a device address compares equal on a reader and points into the
        owner's memory, as the DeepSeek-V4 engram peer tables would. Export must find
        every undeclared one, and no large int that is not an address."""
        device_buffer = torch.empty(1024, device="cuda")
        pinned = torch.empty(64, pin_memory=True)
        model = nn.Module()
        model.ptrs = (device_buffer.data_ptr() + 128, 0)
        model.table = torch.tensor(
            [pinned.data_ptr(), 0], dtype=torch.int64, device="cuda"
        )
        model.count = (1 << 40) + 7
        model.huge = 1 << 70
        self.assertEqual(sorted(find_device_addresses(model)), ["#ptrs", "table"])


if __name__ == "__main__":
    unittest.main()
