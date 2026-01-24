"""
Block swapping utilities for memory-efficient inference.

This module provides the ModelOffloader class for async CPU/GPU block swapping,
compatible with torch.compile. Based on the implementation from WAN/musubi-tuner.
"""

from concurrent.futures import ThreadPoolExecutor
import gc
import time
from typing import Optional
import torch
import torch.nn as nn


def clean_memory_on_device(device: torch.device):
    """Clean memory on the specified device."""
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        torch.xpu.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def synchronize_device(device: torch.device):
    """Synchronize the specified device."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def swap_weight_devices_cuda(device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
    """
    Swap weights between two layers, moving layer_to_cpu's weights to CPU and layer_to_cuda's weights to GPU.
    Uses buffer reuse for large weight tensors to minimize GPU memory allocation.
    """
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    weight_swap_jobs = []
    other_param_jobs = []

    modules_to_cpu = {k: v for k, v in layer_to_cpu.named_modules()}
    for module_to_cuda_name, module_to_cuda in layer_to_cuda.named_modules():
        module_to_cpu = modules_to_cpu.get(module_to_cuda_name, None)
        if module_to_cpu is None:
            continue

        # Handle weight parameter with buffer reuse
        if hasattr(module_to_cuda, "weight") and module_to_cuda.weight is not None:
            if module_to_cpu.weight is not None and module_to_cpu.weight.shape == module_to_cuda.weight.shape:
                weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))
            elif module_to_cuda.weight.data.device.type != device.type:
                module_to_cuda.weight.data = module_to_cuda.weight.data.to(device)

        # Handle all other parameters (bias, etc.)
        for param_name, param in module_to_cuda.named_parameters(recurse=False):
            if param_name == "weight":
                continue
            if param is not None:
                cpu_param = getattr(module_to_cpu, param_name, None)
                if cpu_param is not None:
                    other_param_jobs.append((module_to_cpu, module_to_cuda, param_name, cpu_param.data, param.data))

    torch.cuda.current_stream().synchronize()

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # cuda to cpu - weights (with buffer reuse)
        for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
            cuda_data_view.record_stream(stream)
            module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)

        # cuda to cpu - other params
        for module_to_cpu, module_to_cuda, param_name, cpu_param_data, cuda_param_data in other_param_jobs:
            if cpu_param_data.device.type == device.type:
                setattr(module_to_cpu, param_name + "_data_backup", cpu_param_data)
                getattr(module_to_cpu, param_name).data = cpu_param_data.to("cpu", non_blocking=True)

        stream.synchronize()

        # cpu to cuda - weights (reuse GPU buffer)
        for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
            cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
            module_to_cuda.weight.data = cuda_data_view

        # cpu to cuda - other params
        for module_to_cpu, module_to_cuda, param_name, cpu_param_data, cuda_param_data in other_param_jobs:
            backup_key = param_name + "_data_backup"
            if hasattr(module_to_cpu, backup_key):
                gpu_buffer = getattr(module_to_cpu, backup_key)
                gpu_buffer.copy_(cuda_param_data, non_blocking=True)
                getattr(module_to_cuda, param_name).data = gpu_buffer
                delattr(module_to_cpu, backup_key)
            else:
                getattr(module_to_cuda, param_name).data = cuda_param_data.to(device, non_blocking=True)

    stream.synchronize()
    torch.cuda.current_stream().synchronize()


def swap_weight_devices_no_cuda(device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
    """Non-CUDA fallback for weight swapping."""
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    weight_swap_jobs = []
    for module_to_cpu, module_to_cuda in zip(layer_to_cpu.modules(), layer_to_cuda.modules()):
        if hasattr(module_to_cpu, "weight") and module_to_cpu.weight is not None:
            weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))

    # device to cpu
    for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
        module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)

    synchronize_device(device)

    # cpu to device
    for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
        cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
        module_to_cuda.weight.data = cuda_data_view

    synchronize_device(device)


def weights_to_device(layer: nn.Module, device):
    """Move all parameters to the specified device."""
    for module in layer.modules():
        for param_name, param in list(module.named_parameters(recurse=False)):
            if param is not None:
                param.data = param.data.to(device, non_blocking=True)


class Offloader:
    """Base offloading class with async block swapping."""

    def __init__(self, block_type: str, num_blocks: int, blocks_to_swap: int, device: torch.device, debug: bool = False):
        self.block_type = block_type
        self.num_blocks = num_blocks
        self.blocks_to_swap = blocks_to_swap
        self.device = device
        self.debug = debug

        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.futures = {}
        self.cuda_available = device.type == "cuda"

    def swap_weight_devices(self, block_to_cpu: nn.Module, block_to_cuda: nn.Module):
        if self.cuda_available:
            swap_weight_devices_cuda(self.device, block_to_cpu, block_to_cuda)
        else:
            swap_weight_devices_no_cuda(self.device, block_to_cpu, block_to_cuda)

    def _submit_move_blocks(self, blocks, block_idx_to_cpu, block_idx_to_cuda):
        def move_blocks(bidx_to_cpu, block_to_cpu, bidx_to_cuda, block_to_cuda):
            if self.debug:
                start_time = time.perf_counter()
                print(f"[{self.block_type}] Move block {bidx_to_cpu} to CPU and block {bidx_to_cuda} to GPU")

            self.swap_weight_devices(block_to_cpu, block_to_cuda)

            if self.debug:
                print(f"[{self.block_type}] Moved blocks {bidx_to_cpu} and {bidx_to_cuda} in {time.perf_counter()-start_time:.2f}s")
            return bidx_to_cpu, bidx_to_cuda

        block_to_cpu = blocks[block_idx_to_cpu]
        block_to_cuda = blocks[block_idx_to_cuda]

        self.futures[block_idx_to_cuda] = self.thread_pool.submit(
            move_blocks, block_idx_to_cpu, block_to_cpu, block_idx_to_cuda, block_to_cuda
        )

    def _wait_blocks_move(self, block_idx):
        if block_idx not in self.futures:
            return

        if self.debug:
            print(f"[{self.block_type}] Wait for block {block_idx}")
            start_time = time.perf_counter()

        future = self.futures.pop(block_idx)
        _, bidx_to_cuda = future.result()

        assert block_idx == bidx_to_cuda, f"Block index mismatch: {block_idx} != {bidx_to_cuda}"

        if self.cuda_available:
            torch.cuda.synchronize()
            if self.debug:
                print(f"[{self.block_type}] Swap complete for block {block_idx}: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        if self.debug:
            print(f"[{self.block_type}] Waited for block {block_idx}: {time.perf_counter()-start_time:.2f}s")


class ModelOffloader(Offloader):
    """
    Model offloader supporting forward-only block swapping.

    This offloader is compatible with torch.compile because it orchestrates
    block swapping from outside the layer forward calls, rather than wrapping
    individual layer forwards.
    """

    def __init__(
        self,
        block_type: str,
        blocks: list,
        num_blocks: int,
        blocks_to_swap: int,
        supports_backward: bool,
        device: torch.device,
        debug: bool = False,
    ):
        super().__init__(block_type, num_blocks, blocks_to_swap, device, debug)

        self.supports_backward = supports_backward
        self.forward_only = not supports_backward

        if self.supports_backward:
            self.remove_handles = []
            for i, block in enumerate(blocks):
                hook = self.create_backward_hook(blocks, i)
                if hook is not None:
                    handle = block.register_full_backward_hook(hook)
                    self.remove_handles.append(handle)

    def set_forward_only(self, forward_only: bool):
        self.forward_only = forward_only

    def __del__(self):
        if hasattr(self, 'supports_backward') and self.supports_backward:
            for handle in self.remove_handles:
                handle.remove()

    def create_backward_hook(self, blocks: list, block_index: int) -> Optional[callable]:
        num_blocks_propagated = self.num_blocks - block_index - 1
        swapping = num_blocks_propagated > 0 and num_blocks_propagated <= self.blocks_to_swap
        waiting = block_index > 0 and block_index <= self.blocks_to_swap

        if not swapping and not waiting:
            return None

        block_idx_to_cpu = self.num_blocks - num_blocks_propagated
        block_idx_to_cuda = self.blocks_to_swap - num_blocks_propagated
        block_idx_to_wait = block_index - 1

        def backward_hook(module, grad_input, grad_output):
            if self.debug:
                print(f"Backward hook for block {block_index}")

            if swapping:
                self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
            if waiting:
                self._wait_blocks_move(block_idx_to_wait)
            return None

        return backward_hook

    def prepare_block_devices_before_forward(self, blocks: list):
        """Prepare block devices before forward pass - move resident blocks to GPU, others to CPU."""
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        num_resident = self.num_blocks - self.blocks_to_swap
        if self.debug:
            print(f"[{self.block_type}] Prepare block devices: {num_resident} blocks on GPU, {self.blocks_to_swap} blocks on CPU")

        # Move first (num_blocks - blocks_to_swap) blocks to GPU
        for i, b in enumerate(blocks[0:num_resident]):
            b.to(self.device)
            weights_to_device(b, self.device)
            if self.debug and self.device.type == "cuda":
                print(f"  Block {i} moved to GPU. GPU memory: {torch.cuda.memory_allocated(self.device) / 1e9:.2f} GB")

        # Keep remaining blocks on CPU
        for i, b in enumerate(blocks[num_resident:]):
            weights_to_device(b, "cpu")

        synchronize_device(self.device)
        clean_memory_on_device(self.device)

        if self.debug and self.device.type == "cuda":
            print(f"[{self.block_type}] After prepare: GPU memory: {torch.cuda.memory_allocated(self.device) / 1e9:.2f} GB")

    def wait_for_block(self, block_idx: int):
        """Wait for a block to finish swapping to GPU."""
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self._wait_blocks_move(block_idx)

    def submit_move_blocks_forward(self, blocks: list, block_idx: int):
        """Submit async swap after processing a block in forward pass."""
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        if not self.forward_only and block_idx >= self.blocks_to_swap:
            return

        block_idx_to_cpu = block_idx
        block_idx_to_cuda = self.num_blocks - self.blocks_to_swap + block_idx
        block_idx_to_cuda = block_idx_to_cuda % self.num_blocks
        self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
