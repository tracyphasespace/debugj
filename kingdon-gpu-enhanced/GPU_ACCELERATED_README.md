# GPU-Accelerated Geometric Algebra Simulation

This program implements GPU acceleration and multi-core CPU parallelism for the Kingdon Geometric Algebra library, optimized for systems with powerful GPUs and multi-core processors.

## Features

- **CUDA kernel implementation** for State Multivector operations
- **Multi-threaded CPU processing** for batch operations
- **Memory-efficient streaming** for large datasets
- **Performance profiling** and optimization
- **Dynamic workload balancing** between CPU and GPU

## Requirements

- Python 3.8+
- NumPy
- CuPy (for GPU acceleration)
- Kingdon Geometric Algebra library

## Installation

1. Make sure you have the Kingdon library installed:
   ```bash
   python setup.py install
   ```

2. Install required dependencies:
   ```bash
   pip install numpy
   pip install cupy-cuda11x  # Replace with appropriate CUDA version
   ```

## Usage

### Basic Simulation

Run a simulation with 100 wavelengths:

```bash
python gpu_accelerated_simulation.py --wavelengths 100
```

### Performance Profiling

Profile performance with different batch sizes and CPU worker counts:

```bash
python gpu_accelerated_simulation.py --profile
```

### CPU-Only Mode

Run without GPU acceleration:

```bash
python gpu_accelerated_simulation.py --no-gpu
```

### Advanced Options

```bash
python gpu_accelerated_simulation.py --wavelengths 200 --batch-size 2048 --cpu-workers 4 --output results.npz
```

## Command Line Arguments

- `--wavelengths`: Number of wavelengths to simulate (default: 100)
- `--no-gpu`: Disable GPU acceleration
- `--batch-size`: GPU batch size (default: 1024)
- `--cpu-workers`: Number of CPU worker processes (default: all cores)
- `--profile`: Run performance profiling
- `--output`: Output file for results (default: simulation_results.npz)
- `--verbose`: Enable verbose logging

## Performance Optimization

The program automatically selects the best execution strategy based on your hardware:

1. For large simulations (>100 rays) with GPU available: Uses GPU acceleration
2. For medium simulations (>10 rays) without GPU: Uses parallel CPU processing
3. For small simulations: Uses single-threaded CPU processing

## Implementation Details

### GPU Acceleration

The program implements two CUDA kernels:

1. `optical_surface_interaction`: Processes a single optical surface interaction
2. `batch_propagation`: Processes multiple surfaces in sequence for a batch of rays

These kernels are designed to maximize register utilization and minimize memory transfers.

### Multi-core CPU Processing

For systems without a GPU or when GPU acceleration is disabled, the program uses Python's `concurrent.futures` to parallelize ray propagation across all available CPU cores.

### Memory Optimization

The program processes rays in batches to minimize memory usage, making it possible to simulate thousands of wavelengths even on systems with limited memory.

## Example Output

```
2025-07-23 10:15:30 - GA-GPU-Accelerator - INFO - CuPy found. Using GPU acceleration with CUDA.
2025-07-23 10:15:30 - GA-GPU-Accelerator - INFO - GPU: NVIDIA GeForce RTX 3080
2025-07-23 10:15:30 - GA-GPU-Accelerator - INFO - Compute capability: 8.6
2025-07-23 10:15:30 - GA-GPU-Accelerator - INFO - Total memory: 10.00 GB
2025-07-23 10:15:30 - GA-GPU-Accelerator - INFO - Multiprocessors: 68
2025-07-23 10:15:30 - GA-GPU-Accelerator - INFO - Successfully compiled optical surface interaction kernel
2025-07-23 10:15:30 - GA-GPU-Accelerator - INFO - Successfully compiled batch propagation kernel
2025-07-23 10:15:30 - GA-GPU-Accelerator - INFO - Created default achromatic doublet optical system
2025-07-23 10:15:30 - GA-GPU-Accelerator - INFO - Created wavelength spectrum with 100 values from 400nm to 700nm
2025-07-23 10:15:30 - GA-GPU-Accelerator - INFO - Created 100 rays
2025-07-23 10:15:30 - GA-GPU-Accelerator - INFO - Running simulation with 100 rays
2025-07-23 10:15:30 - GA-GPU-Accelerator - INFO - Starting GPU propagation of 100 rays
2025-07-23 10:15:30 - GA-GPU-Accelerator - INFO - GPU propagation completed in 0.0123 seconds
2025-07-23 10:15:30 - GA-GPU-Accelerator - INFO - Simulation completed in 0.0123 seconds
2025-07-23 10:15:30 - GA-GPU-Accelerator - INFO - Performance: 8130 rays/second
2025-07-23 10:15:30 - GA-GPU-Accelerator - INFO - Operations: 32520 ray-surface interactions/second
2025-07-23 10:15:30 - GA-GPU-Accelerator - INFO - Analysis complete:
2025-07-23 10:15:30 - GA-GPU-Accelerator - INFO -   Best transmission: 0.823 at 650.0nm
2025-07-23 10:15:30 - GA-GPU-Accelerator - INFO -   Worst transmission: 0.712 at 400.0nm
2025-07-23 10:15:30 - GA-GPU-Accelerator - INFO -   Blue band avg: 0.725
2025-07-23 10:15:30 - GA-GPU-Accelerator - INFO -   Green band avg: 0.781
2025-07-23 10:15:30 - GA-GPU-Accelerator - INFO -   Red band avg: 0.815
2025-07-23 10:15:30 - GA-GPU-Accelerator - INFO - Results saved to simulation_results.npz
2025-07-23 10:15:30 - GA-GPU-Accelerator - INFO - Simulation completed successfully
```

## Theoretical Performance

On a modern GPU with 10,000+ shader cores, the theoretical performance can reach trillions of operations per second when using the register-resident computing approach described in the Kingdon library's architecture documents.

## License

Same as the Kingdon Geometric Algebra library.