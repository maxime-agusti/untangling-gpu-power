# GPU Sharing & Performance Experiments

This repository contains a suite of Python scripts designed to evaluate how various GPU sharing mechanisms impact performance and energy consumption. The experiments explore NVIDIA MIG configurations, GPU time-slicing (both container-based and Kubernetes-based), and measure system-level metrics like power, temperature, and CPU usage.

## Goals

- Assess performance isolation and energy impact of **MIG configurations** (GPU and Compute Instances).
- Study the effects of **time-slicing** by running concurrent workloads without MIG.
- Correlate **performance metrics** with **system-level indicators** (power, utilization, etc.).
- Identify worst-case energy consumption scenarios for data center planning and sustainability.

## Experiment Overview

| Script | Description |
|--------|-------------|
| `exp-mig.py` | Varies both GPU and Compute Instance sizes using GPU burn tests. Includes complementary workloads to assess neighbor interference. |
| `exp-perf-mig-gi.py` | Uses real-world benchmarks (Blender, HPCG, LLaMA, YOLO) to study performance vs. metrics while varying **GI size only** (CI size = GI size). |
| `exp-perf-mig-ci.py` | Fixes the largest GI and varies **CI size**, still using real-world benchmarks to analyze impact. |
| `exp-perf-timeslices.py` | Evaluates performance degradation from GPU time-slicing by running multiple containers (no MIG). |
| `exp-perf-timeslice-k8s.py` | Similar to the above but in a **Kubernetes** setting by increasing the number of GPU-using pod replicas. |
| `exp-timeslices.py.py` | Measures **maximum power consumption** during GPU contention using multiple GPU burn containers (no benchmarks). |
| `exp-passthrough.py` | Applies all combinations of GPU usage levels across all MIG-enabled GPUs to test if **temperature sensors (e.g., IPMI)** can reliably indicate **per-GPU usage**. Used for reverse-inference and thermal diagnostics. |

## Metrics Collected

Across the experiments, the following metrics are typically collected:

- **DCGM**: GPU power draw, utilization, temperature, memory usage
- **IPMI**: Platform power and temperature metrics
- **CPU**: Utilization and context-switch behavior
- **Timing & Performance**: Benchmark runtimes, throughput, or FLOPS (when applicable)

## Requirements

- Python
- NVIDIA GPU with MIG support (A100 or similar for MIG experiments)
- `dcgm-exporter`, `ipmitool`, and benchmarking tools (e.g., Blender CLI, HPCG, YOLO, LLaMA)
- Docker
- `nvidia-container-toolkit` installed and configured
- Kubernetes (for time-slicing experiments)

## Installation

```bash
git clone https://github.com/jacquetpi/cloud-gpu-manager
mkdir data ; mkdir bench-res
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

Each benchmark should also be installed.
Refer to individuals ```init/bench/{benchname}/README.md```

## Usage

Each script is standalone and can be launched with:

```bash
python <script_name>.py
```

For MIG-based experiments, the GPUs should have MIG enabled prior to launch

Data obtained from monitoring and benchmarks will respectively be placed in ```data/``` and ```bench-res/``` folders

