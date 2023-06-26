import torch
import torch.nn as nn
import time
from vllm import fuse_softmax_ops
import random


def apply_temp(logits, temperatures, dtype) -> torch.Tensor:
    logits.div_(temperatures.unsqueeze(dim=1))
    probs = torch.softmax(logits, dim=-1, dtype=dtype)
    return probs


def run_softmax_benchmark(batch_size, hidden_size, dtype) -> None:
    logits = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    fuse_probs = torch.empty_like(logits, dtype=torch.float32)
    # TODO Calculation logic is inconsistent when temperatures are too small
    temperatures = [random.uniform(0.01, 1) for _ in range(batch_size)]
    t = torch.tensor(temperatures, dtype=torch.float32, device="cuda")
    t_half = torch.tensor(temperatures, dtype=dtype, device="cuda")
    start_time = time.time()

    fuse_softmax_ops.fuse_softmax(fuse_probs, logits, t)
    for _ in range(10000):
        fuse_softmax_ops.fuse_softmax(fuse_probs, logits, t)
    fuse_softmax_time = (time.time() - start_time) * 1000

    start_time = time.time()
    probs = apply_temp(logits, t_half, torch.float32)
    for _ in range(10000):
        probs = apply_temp(logits, t_half, torch.float32)
    origin_softmax_time = (time.time() - start_time) * 1000
    torch.allclose(fuse_probs, probs, atol=1e-3, rtol=1e-5)

    start_time = time.time()
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    for _ in range(10000):
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    no_temperatures_time = (time.time() - start_time) * 1000

    temperatures = [1 for _ in range(batch_size)]
    t = torch.tensor(temperatures, dtype=torch.float32, device="cuda")
    start_time = time.time()
    for _ in range(10000):
        fuse_softmax_ops.fuse_softmax(fuse_probs, logits, t)
    no_temperatures_time_f = (time.time() - start_time) * 1000
    torch.allclose(fuse_probs, probs, atol=1e-3, rtol=1e-5)

    print(
        f"Testing batch_size={batch_size}, hidden_size={hidden_size},origin time={origin_softmax_time:.2f}ms,"
        f"fuse_softmax time={fuse_softmax_time:.2f}ms,"
        f"no_temperatures Speedup: {no_temperatures_time / no_temperatures_time_f:.2f}x,"
        f"apply temperatures Speedup: {origin_softmax_time / fuse_softmax_time:.2f}x"
    )


def test_fuse_softmax() -> None:
    dtype = torch.half
    # vocab_size: vicuna13b 32000,gpt_neox 50432
    for batch_size in [1, 10, 30]:
        for hidden_size in [32000, 50432]:
            run_softmax_benchmark(
                batch_size=batch_size, hidden_size=hidden_size, dtype=dtype
            )


test_fuse_softmax()
