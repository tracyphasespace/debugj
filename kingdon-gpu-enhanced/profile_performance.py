#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Performance profiling script for GPU-accelerated Geometric Algebra simulation.

This script runs detailed performance profiling to find optimal parameters
for your specific hardware configuration.
"""

import time
import numpy as np
import logging
import sys
import argparse
import matplotlib.pyplot as plt
from gpu_accelerated_simulation import (
    OpticalSimulation, PerformanceProfiler, HAS_CUPY, CPU_CORES
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("GA-GPU-Profiler")

def profile_batch_sizes(wavelength_count=100, batch_sizes=None):
    """Profile performance with different GPU batch sizes."""
    if not HAS_CUPY:
        logger.warning("GPU not available. Skipping batch size profiling.")
        return {}
        
    if batch_sizes is None:
        batch_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
    
    logger.info(f"Profiling GPU batch sizes with {wavelength_count} wavelengths")
    
    # Create simulation
    simulation = OpticalSimulation()
    simulation.create_default_system()
    simulation.create_wavelength_spectrum(count=wavelength_count)
    simulation.create_rays()
    
    # Test each batch size
    results = {}
    for batch_size in batch_sizes:
        logger.info(f"Testing batch size: {batch_size}")
        start_time = time.time()
        simulation.run_simulation(use_gpu=True, batch_size=batch_size)
        elapsed_time = time.time() - start_time
        results[batch_size] = elapsed_time
        logger.info(f"  Time: {elapsed_time:.4f} seconds")
    
    # Find optimal batch size
    optimal_batch_size = min(results, key=results.get)
    logger.info(f"Optimal batch size: {optimal_batch_size}")
    logger.info(f"Optimal time: {results[optimal_batch_size]:.4f} seconds")
    
    return results

def profile_worker_counts(wavelength_count=100, worker_counts=None):
    """Profile performance with different CPU worker counts."""
    if worker_counts is None:
        # Test powers of 2 up to CPU_CORES
        worker_counts = [2**i for i in range(0, int(np.log2(CPU_CORES)) + 1)]
        if CPU_CORES not in worker_counts:
            worker_counts.append(CPU_CORES)
    
    logger.info(f"Profiling CPU worker counts with {wavelength_count} wavelengths")
    
    # Create simulation
    simulation = OpticalSimulation()
    simulation.create_default_system()
    simulation.create_wavelength_spectrum(count=wavelength_count)
    simulation.create_rays()
    
    # Test each worker count
    results = {}
    for workers in worker_counts:
        logger.info(f"Testing worker count: {workers}")
        start_time = time.time()
        simulation.run_simulation(use_gpu=False, num_cpu_workers=workers)
        elapsed_time = time.time() - start_time
        results[workers] = elapsed_time
        logger.info(f"  Time: {elapsed_time:.4f} seconds")
    
    # Find optimal worker count
    optimal_workers = min(results, key=results.get)
    logger.info(f"Optimal worker count: {optimal_workers}")
    logger.info(f"Optimal time: {results[optimal_workers]:.4f} seconds")
    
    return results

def profile_wavelength_scaling(max_wavelengths=1000, step=100):
    """Profile how performance scales with wavelength count."""
    wavelength_counts = list(range(step, max_wavelengths + 1, step))
    
    logger.info(f"Profiling wavelength scaling from {step} to {max_wavelengths}")
    
    # Results for GPU and CPU
    gpu_results = {}
    cpu_results = {}
    
    for count in wavelength_counts:
        logger.info(f"Testing with {count} wavelengths")
        
        # Create simulation
        simulation = OpticalSimulation()
        simulation.create_default_system()
        simulation.create_wavelength_spectrum(count=count)
        simulation.create_rays()
        
        # Test GPU if available
        if HAS_CUPY:
            start_time = time.time()
            simulation.run_simulation(use_gpu=True)
            elapsed_time = time.time() - start_time
            gpu_results[count] = elapsed_time
            logger.info(f"  GPU time: {elapsed_time:.4f} seconds")
        
        # Test CPU
        start_time = time.time()
        simulation.run_simulation(use_gpu=False, num_cpu_workers=CPU_CORES)
        elapsed_time = time.time() - start_time
        cpu_results[count] = elapsed_time
        logger.info(f"  CPU time: {elapsed_time:.4f} seconds")
    
    return {
        'gpu': gpu_results,
        'cpu': cpu_results
    }

def plot_results(batch_results=None, worker_results=None, scaling_results=None):
    """Plot profiling results."""
    try:
        plt.figure(figsize=(15, 10))
        
        # Plot batch size results
        if batch_results:
            plt.subplot(2, 2, 1)
            batch_sizes = list(batch_results.keys())
            times = list(batch_results.values())
            plt.plot(batch_sizes, times, 'o-', label='Execution Time')
            plt.xlabel('Batch Size')
            plt.ylabel('Time (seconds)')
            plt.title('GPU Batch Size Performance')
            plt.grid(True)
            plt.xscale('log', base=2)
        
        # Plot worker count results
        if worker_results:
            plt.subplot(2, 2, 2)
            workers = list(worker_results.keys())
            times = list(worker_results.values())
            plt.plot(workers, times, 'o-', label='Execution Time')
            plt.xlabel('Worker Count')
            plt.ylabel('Time (seconds)')
            plt.title('CPU Worker Count Performance')
            plt.grid(True)
            plt.xscale('log', base=2)
        
        # Plot wavelength scaling results
        if scaling_results:
            plt.subplot(2, 2, 3)
            wavelengths = list(scaling_results['cpu'].keys())
            cpu_times = list(scaling_results['cpu'].values())
            plt.plot(wavelengths, cpu_times, 'o-', label='CPU')
            
            if 'gpu' in scaling_results and scaling_results['gpu']:
                gpu_times = [scaling_results['gpu'].get(w, float('nan')) for w in wavelengths]
                plt.plot(wavelengths, gpu_times, 'o-', label='GPU')
            
            plt.xlabel('Wavelength Count')
            plt.ylabel('Time (seconds)')
            plt.title('Wavelength Scaling')
            plt.grid(True)
            plt.legend()
            
            # Plot speedup
            if 'gpu' in scaling_results and scaling_results['gpu']:
                plt.subplot(2, 2, 4)
                speedups = []
                for w in wavelengths:
                    if w in scaling_results['gpu'] and w in scaling_results['cpu']:
                        speedup = scaling_results['cpu'][w] / scaling_results['gpu'][w]
                        speedups.append(speedup)
                    else:
                        speedups.append(float('nan'))
                
                plt.plot(wavelengths, speedups, 'o-', label='GPU vs CPU')
                plt.xlabel('Wavelength Count')
                plt.ylabel('Speedup Factor')
                plt.title('GPU Speedup vs CPU')
                plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('performance_profile.png')
        logger.info("Performance profile plot saved to 'performance_profile.png'")
        
        # Try to display the plot
        try:
            plt.show()
        except:
            pass
            
    except ImportError:
        logger.warning("Matplotlib not available. Skipping plot generation.")
    except Exception as e:
        logger.error(f"Error generating plots: {e}")

def main():
    """Main profiling function."""
    parser = argparse.ArgumentParser(description='Performance profiling for GPU-accelerated simulation')
    parser.add_argument('--wavelengths', type=int, default=100,
                       help='Number of wavelengths to use for profiling')
    parser.add_argument('--max-wavelengths', type=int, default=1000,
                       help='Maximum wavelengths for scaling test')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plot generation')
    
    args = parser.parse_args()
    
    logger.info("GPU-Accelerated Geometric Algebra Performance Profiler")
    logger.info("====================================================")
    logger.info(f"CPU cores: {CPU_CORES}")
    logger.info(f"GPU available: {HAS_CUPY}")
    
    # Profile batch sizes
    batch_results = profile_batch_sizes(args.wavelengths)
    
    # Profile worker counts
    worker_results = profile_worker_counts(args.wavelengths)
    
    # Profile wavelength scaling
    scaling_results = profile_wavelength_scaling(args.max_wavelengths)
    
    # Estimate theoretical performance
    profiler = PerformanceProfiler()
    profiler.estimate_theoretical_performance()
    
    # Plot results
    if not args.no_plot:
        plot_results(batch_results, worker_results, scaling_results)
    
    logger.info("Profiling completed.")
    
    # Print recommendations
    logger.info("\nRecommended Parameters:")
    logger.info("======================")
    
    if batch_results:
        optimal_batch = min(batch_results, key=batch_results.get)
        logger.info(f"Optimal GPU batch size: {optimal_batch}")
    
    if worker_results:
        optimal_workers = min(worker_results, key=worker_results.get)
        logger.info(f"Optimal CPU worker count: {optimal_workers}")
    
    # Recommend best approach
    if HAS_CUPY and batch_results and worker_results:
        best_gpu_time = min(batch_results.values())
        best_cpu_time = min(worker_results.values())
        
        if best_gpu_time < best_cpu_time:
            speedup = best_cpu_time / best_gpu_time
            logger.info(f"Recommendation: Use GPU acceleration (provides {speedup:.2f}x speedup)")
            logger.info(f"Command: python gpu_accelerated_simulation.py --batch-size {optimal_batch}")
        else:
            logger.info(f"Recommendation: Use CPU parallelism with {optimal_workers} workers")
            logger.info(f"Command: python gpu_accelerated_simulation.py --no-gpu --cpu-workers {optimal_workers}")
    elif worker_results:
        logger.info(f"Recommendation: Use CPU parallelism with {optimal_workers} workers")
        logger.info(f"Command: python gpu_accelerated_simulation.py --no-gpu --cpu-workers {optimal_workers}")

if __name__ == "__main__":
    main()