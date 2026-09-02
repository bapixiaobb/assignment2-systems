import torch
import timeit
import statistics
import argparse
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

MODEL_CONFIGS = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
    },
    "medium": {
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16,
    },
    "large": {
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20,
    },
    "xl": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32,
    },
    "10b": {
        "d_model": 4608,
        "d_ff": 12288,
        "num_layers": 50,
        "num_heads": 36,
    },
}

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark script for Language Modeling")
    parser.add_argument("--mode", "-m", choices=["forward", "backward", "forward_backward", "optimizer", "training"], default="forward", help="Benchmark mode")
    parser.add_argument("--model-size", "-s", choices=tuple(MODEL_CONFIGS), default="small", help="Model size")
    parser.add_argument("--num-iterations", "-i", type=int, default=10, help="Number of iterations for benchmarking")
    parser.add_argument("--num-warmup", "-w", type=int, default=5, help="Number of warmup iterations before benchmarking")
    parser.add_argument("--device-id", "-d", type=int, default=0, help="NPU or CUDA device index")
    return parser.parse_args()

def get_device():
    try:
        import torch_npu  # noqa: F401
    except (ImportError, OSError):
        pass
    else:
        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"

def validate_device(device):
    if device not in ("cpu", "cuda", "npu"):
        raise ValueError(f"Unsupported device: {device}")

def configure_device(device_type, device_id):
    validate_device(device_type)

    if device_id < 0:
        raise ValueError(f"Device ID must be non-negative, got {device_id}")

    if device_type == "cpu":
        return torch.device("cpu")

    if device_type == "cuda":
        device_count = torch.cuda.device_count()
        if device_id >= device_count:
            raise ValueError(
                f"CUDA device ID {device_id} is unavailable; "
                f"this process can see {device_count} CUDA device(s)"
            )
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device)
        return device

    device_count = torch.npu.device_count()
    if device_id >= device_count:
        raise ValueError(
            f"NPU device ID {device_id} is unavailable; "
            f"this process can see {device_count} NPU device(s)"
        )
    device = torch.device(f"npu:{device_id}")
    torch.npu.set_device(device)
    return device

def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu":
        torch.npu.synchronize(device)

def benchmark_forward(model, input_ids, device, num_iterations=10, num_warmup=5):
    # warm up
    for _ in range(num_warmup):
        logits = model(input_ids)

    timing = []
    sync_device(device)
    for _ in range(num_iterations):
        start_time = timeit.default_timer()
        logits = model(input_ids)
        sync_device(device)
        end_time = timeit.default_timer()
        timing.append(end_time - start_time)

    return timing

def run_forward_backward(model, input_ids):
    logits = model(input_ids)
    loss = logits.mean()
    loss.backward()

def benchmark_forward_backward(model, input_ids, device, num_iterations=10, num_warmup=5):
    # warm up
    for _ in range(num_warmup):
        model.zero_grad(set_to_none=True)
        run_forward_backward(model, input_ids)

    timing = []
    sync_device(device)
    for _ in range(num_iterations):
        model.zero_grad(set_to_none=True)

        start_time = timeit.default_timer()
        run_forward_backward(model, input_ids)
        sync_device(device)
        end_time = timeit.default_timer()

        timing.append(end_time - start_time)

    return timing

def benchmark_backward(model, input_ids, device, num_iterations=10, num_warmup=5):
    # warm up
    for _ in range(num_warmup):
        model.zero_grad(set_to_none=True)
        run_forward_backward(model, input_ids)

    timing = []
    sync_device(device)
    for _ in range(num_iterations):
        model.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = logits.mean()
        sync_device(device)

        start_time = timeit.default_timer()
        loss.backward()
        sync_device(device)
        end_time = timeit.default_timer()

        timing.append(end_time - start_time)
    return timing

def benchmark_optimizer_step(model, input_ids, optimizer, device, num_iterations=10, num_warmup=5):
    # warm up
    for _ in range(num_warmup):
        model.zero_grad(set_to_none=True)
        run_forward_backward(model, input_ids)
        optimizer.step()

    timing = []
    sync_device(device)
    for _ in range(num_iterations):
        model.zero_grad(set_to_none=True)
        run_forward_backward(model, input_ids)
        sync_device(device)

        start_time = timeit.default_timer()
        optimizer.step()
        sync_device(device)
        end_time = timeit.default_timer()

        timing.append(end_time - start_time)

    return timing

def run_training_step(model, input_ids, optimizer):
    run_forward_backward(model, input_ids)
    optimizer.step()

def benchmark_training_step(model, input_ids, optimizer, device, num_iterations=10, num_warmup=5):
    # warm up
    for _ in range(num_warmup):
        model.zero_grad(set_to_none=True)
        run_training_step(model, input_ids, optimizer)

    timing = []
    sync_device(device)
    for _ in range(num_iterations):
        model.zero_grad(set_to_none=True)
        start_time = timeit.default_timer()
        run_training_step(model, input_ids, optimizer)
        sync_device(device)
        end_time = timeit.default_timer()
        timing.append(end_time - start_time)
    return timing

def print_timing_summary(label, timings):
    mean_time = statistics.mean(timings)
    std_time = statistics.stdev(timings)

    print(f"Mean time per {label}: {mean_time:.6f} seconds")
    print(f"Standard deviation per {label}: {std_time:.6f} seconds")

def main():
    args = parse_args()
    model_size = args.model_size
    config = MODEL_CONFIGS[model_size]
    device_type = get_device()
    device = configure_device(device_type, args.device_id)
    print(f"Running benchmark on device: {device}")
    print(f"Model size: {model_size}")
    print(f"Model configuration: {config}")

    vocab_size = 10000
    context_length = 512
    d_model = config["d_model"]
    d_ff = config["d_ff"]
    num_layers = config["num_layers"]
    num_heads = config["num_heads"]

    batch_size = 4
    input_ids = torch.randint(0, vocab_size, (batch_size, context_length))

    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
    )


    model = model.to(device)
    input_ids = input_ids.to(device)

    # optimizer = AdamW(model.parameters(), lr=1e-3)
    # ==========================
    # Benchmarking
    # ==========================
    mode = args.mode
    num_iterations = args.num_iterations
    num_warmup = args.num_warmup
    print(f"Benchmarking mode: {mode}")
    if mode == "forward":
        timings = benchmark_forward(
            model,
            input_ids,
            device=device,
            num_iterations=num_iterations,
            num_warmup=num_warmup,
        )
        label = "forward pass"
    elif mode == "forward_backward":
        timings = benchmark_forward_backward(
            model,
            input_ids,
            device=device,
            num_iterations=num_iterations,
            num_warmup=num_warmup,
        )
        label = "forward backward pass"
    elif mode == "backward":
        timings = benchmark_backward(
            model,
            input_ids,
            device=device,
            num_iterations=num_iterations,
            num_warmup=num_warmup,
        )
        label = "backward pass"
    elif mode == "optimizer":
        optimizer = AdamW(model.parameters(), lr=1e-3)
        timings = benchmark_optimizer_step(
            model,
            input_ids,
            optimizer,
            device=device,
            num_iterations=num_iterations,
            num_warmup=num_warmup,
        )
        label = "optimizer step"
    elif mode == "training":
        optimizer = AdamW(model.parameters(), lr=1e-3)
        timings = benchmark_training_step(
            model,
            input_ids,
            optimizer,
            device=device,
            num_iterations=num_iterations,
            num_warmup=num_warmup,
        )
        label = "training step"
    else:
        raise ValueError(f"Unsupported benchmarking mode: {mode}")
    print_timing_summary(label, timings)
if __name__ == "__main__":
    main()
