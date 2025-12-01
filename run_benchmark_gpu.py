from model import NanoTabPFNModel, NanoTabPFNDSAModel
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from torch import nn
import time

def generate_synthetic_tabular_data(
    batch_size: int, 
    num_rows: int, 
    num_cols: int, 
    device: torch.device
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], int]:
    """
    Generates synthetic tabular data for benchmarking.
    Returns: ((X, y), train_test_split_index)
    """
    # X: (Batch, Rows, Cols)
    x = torch.randn(batch_size, num_rows, num_cols, device=device, dtype=torch.float32)
    # y: (Batch, Rows, 1) - Binary classification targets
    y = torch.randint(0, 2, (batch_size, num_rows, 1), device=device, dtype=torch.float32)
    
    # Simulate a 50/50 train/test split
    split_idx = num_rows // 2
    return (x, y), split_idx

def measure_inference_latency(
    model: nn.Module, 
    input_data: Tuple[torch.Tensor, torch.Tensor], 
    split_idx: int, 
    num_warmup: int = 5, 
    num_runs: int = 20
) -> Tuple[float, float, float]:
    """
    Benchmarks inference latency with strict warm-up and CUDA synchronization.
    Returns: (Mean Latency (ms), Std Dev (ms), Peak Memory (MB))
    """
    device = input_data[0].device
    
    # 1. Warm-up Phase
    # Critical for: 
    #   a) Waking up the GPU clock
    #   b) Allowing torch.compile to trace and optimize kernels
    #   c) Filling the cache
    print(f"  > Warming up for {num_warmup} steps...", end="", flush=True)
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_data, split_idx)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    print(" Done.")

    # 2. Measurement Phase
    timings = []
    # Use CUDA Events for microsecond-precision timing on GPU
    if device.type == 'cuda':
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
    
    print(f"  > Benchmarking for {num_runs} runs...", end="", flush=True)
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                starter.record()
                _ = model(input_data, split_idx)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender)) # Returns milliseconds
            else:
                # CPU/MPS Fallback
                t0 = time.perf_counter()
                _ = model(input_data, split_idx)
                t1 = time.perf_counter()
                timings.append((t1 - t0) * 1000) # Convert to ms
    print(" Done.")
    
    # 3. Memory Measurement
    peak_mem = 0.0
    if device.type == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) # MB
        
    return np.mean(timings), np.std(timings), peak_mem

def run_suite():
    # Configuration
    EMBED_DIM = 64
    HEADS = 4
    LAYERS = 2
    HIDDEN_DIM = 128
    OUTPUTS = 2
    BATCH_SIZE = 1 # Tabular inference is often latency-sensitive (batch 1)
    
    # Context Lengths to benchmark (Rows)
    # We want to see the crossover point where DSA beats Dense
    SEQ_LENGTHS = [1024, 2048, 4096, 8192,11000, 16384, 32768] 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Benchmark Suite on {device} ===")
    
    results = {
        "lengths": SEQ_LENGTHS,
        "dense_latency": [], "dense_mem": [],
        "dsa_latency": [], "dsa_mem": []
    }

    # --- 1. Benchmarking DenseTFM ---
    print("\n--- Benchmarking Dense Attention (Baseline) ---")
    dense_model = NanoTabPFNModel(
        embedding_size=96,
        num_attention_heads=4,
        mlp_hidden_size=192,
        num_layers=3,
        num_outputs=2
    ).to(device)
    dense_model.eval()
    
    # Optional: Compile
    # dense_model = torch.compile(dense_model) 

    for seq_len in SEQ_LENGTHS:
        try:
            print(f"Sequence Length: {seq_len}")
            data, split = generate_synthetic_tabular_data(BATCH_SIZE, seq_len, 10, device)
            
            lat, std, mem = measure_inference_latency(dense_model, data, split)
            results["dense_latency"].append(lat)
            results["dense_mem"].append(mem)
            print(f"  Result: {lat:.2f}ms ± {std:.2f} | Mem: {mem:.2f}MB")
        except torch.cuda.OutOfMemoryError:
            print(f"  Result: OOM")
            results["dense_latency"].append(None)
            results["dense_mem"].append(None)
            torch.cuda.empty_cache()
    
    # --- 2. Benchmarking DeepSeek Sparse TFM ---
    print("\n--- Benchmarking DeepSeek Sparse Attention (DSA) ---")
    dsa_model = NanoTabPFNDSAModel(
        EMBED_DIM, HEADS, HIDDEN_DIM, LAYERS, OUTPUTS, 
        use_dsa=True, top_k=64 # Select Top-64 rows
    ).to(device)
    dsa_model.eval()
    
    # Optional: Compile
    # dsa_model = torch.compile(dsa_model)

    for seq_len in SEQ_LENGTHS:
        try:
            print(f"Sequence Length: {seq_len}")
            data, split = generate_synthetic_tabular_data(BATCH_SIZE, seq_len, 10, device)
            
            lat, std, mem = measure_inference_latency(dsa_model, data, split)
            results["dsa_latency"].append(lat)
            results["dsa_mem"].append(mem)
            print(f"  Result: {lat:.2f}ms ± {std:.2f} | Mem: {mem:.2f}MB")
        except torch.cuda.OutOfMemoryError:
            print(f"  Result: OOM")
            results["dsa_latency"].append(None)
            results["dsa_mem"].append(None)
            torch.cuda.empty_cache()

    # --- 3. Visualization ---
    plot_results(results)

def plot_results(results):
    lengths = results["lengths"]
    
    # Filter Nones (OOMs)
    def clean(data): return [x if x is not None else float('nan') for x in data]
    
    d_lat = clean(results["dense_latency"])
    s_lat = clean(results["dsa_latency"])
    
    plt.figure(figsize=(10, 5))
    
    # Latency Plot
    plt.subplot(1, 2, 1)
    plt.plot(lengths, d_lat, 'o-', label='Dense Attention')
    plt.plot(lengths, s_lat, 's--', label='DeepSeek DSA')
    plt.xlabel('Number of Rows (Context Length)')
    plt.ylabel('Inference Latency (ms)')
    plt.title('Latency vs Context Length')
    plt.legend()
    plt.grid(True)
    
    # Memory Plot
    plt.subplot(1, 2, 2)
    plt.plot(lengths, clean(results["dense_mem"]), 'o-', label='Dense')
    plt.plot(lengths, clean(results["dsa_mem"]), 's--', label='DSA')
    plt.xlabel('Number of Rows')
    plt.ylabel('Peak Memory (MB)')
    plt.title('Memory Usage vs Context Length')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("deepseek_benchmark.png")
    print("\nBenchmark complete. Results saved to 'deepseek_benchmark.png'")

if __name__ == "__main__":
    # Ensure this runs only if classes are defined
    try:
        run_suite()
    except NameError as e:
        print(f"Setup Error: {e}.")