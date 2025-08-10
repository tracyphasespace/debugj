#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for GPU-accelerated Geometric Algebra simulation.

This script runs a simple test to verify that the GPU acceleration
is working correctly and measures the performance improvement.
"""

import time
import numpy as np
import logging
import sys
from gpu_accelerated_simulation import (
    OpticalSimulation, PerformanceProfiler, HAS_CUPY, CPU_CORES
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("GA-GPU-Test")

def run_comparison_test(wavelength_count=100):
    """Run a comparison test between GPU and CPU implementations."""
    logger.info(f"Running comparison test with {wavelength_count} wavelengths")
    
    # Create simulation
    simulation = OpticalSimulation()
    simulation.create_default_system()
    simulation.create_wavelength_spectrum(count=wavelength_count)
    simulation.create_rays()
    
    # Run with GPU (if available)
    if HAS_CUPY:
        logger.info("Running GPU test...")
        start_time = time.time()
        gpu_results = simulation.run_simulation(use_gpu=True)
        gpu_time = time.time() - start_time
        logger.info(f"GPU time: {gpu_time:.4f} seconds")
    else:
        logger.warning("GPU not available, skipping GPU test")
        gpu_time = float('inf')
        gpu_results = None
    
    # Run with single-threaded CPU
    logger.info("Running single-threaded CPU test...")
    start_time = time.time()
    cpu_single_results = simulation.run_simulation(use_gpu=False, num_cpu_workers=1)
    cpu_single_time = time.time() - start_time
    logger.info(f"Single-threaded CPU time: {cpu_single_time:.4f} seconds")
    
    # Run with multi-threaded CPU
    logger.info(f"Running multi-threaded CPU test with {CPU_CORES} cores...")
    start_time = time.time()
    cpu_multi_results = simulation.run_simulation(use_gpu=False, num_cpu_workers=CPU_CORES)
    cpu_multi_time = time.time() - start_time
    logger.info(f"Multi-threaded CPU time: {cpu_multi_time:.4f} seconds")
    
    # Calculate speedups
    multi_vs_single = cpu_single_time / cpu_multi_time if cpu_multi_time > 0 else float('inf')
    logger.info(f"Multi-threaded CPU speedup vs single-threaded: {multi_vs_single:.2f}x")
    
    if HAS_CUPY and gpu_time > 0:
        gpu_vs_single = cpu_single_time / gpu_time
        gpu_vs_multi = cpu_multi_time / gpu_time
        logger.info(f"GPU speedup vs single-threaded CPU: {gpu_vs_single:.2f}x")
        logger.info(f"GPU speedup vs multi-threaded CPU: {gpu_vs_multi:.2f}x")
    
    # Verify results match
    if HAS_CUPY and gpu_results:
        # Compare intensities between GPU and CPU results
        gpu_intensities = np.array([ray.intensity for ray in gpu_results])
        cpu_intensities = np.array([ray.intensity for ray in cpu_single_results])
        
        # Check if results are close (within tolerance)
        max_diff = np.max(np.abs(gpu_intensities - cpu_intensities))
        logger.info(f"Maximum difference between GPU and CPU results: {max_diff:.6f}")
        
        if max_diff < 1e-5:
            logger.info("PASS: GPU and CPU results match within tolerance")
        else:
            logger.warning("FAIL: GPU and CPU results differ significantly")
    
    return {
        'gpu_time': gpu_time if HAS_CUPY else None,
        'cpu_single_time': cpu_single_time,
        'cpu_multi_time': cpu_multi_time,
        'multi_vs_single_speedup': multi_vs_single,
        'gpu_vs_single_speedup': cpu_single_time / gpu_time if HAS_CUPY and gpu_time > 0 else None,
        'gpu_vs_multi_speedup': cpu_multi_time / gpu_time if HAS_CUPY and gpu_time > 0 else None,
    }

def run_scaling_test():
    """Test how performance scales with wavelength count."""
    wavelength_counts = [10, 50, 100, 500, 1000]
    results = {}
    
    logger.info("Running scaling test with different wavelength counts")
    
    for count in wavelength_counts:
        logger.info(f"Testing with {count} wavelengths")
        results[count] = run_comparison_test(count)
    
    # Print scaling summary
    logger.info("\nScaling Summary:")
    logger.info("=================")
    logger.info("Wavelengths | GPU Time (s) | Multi-CPU Time (s) | Single-CPU Time (s) | GPU Speedup")
    logger.info("------------|-------------|-------------------|-------------------|------------")
    
    for count in wavelength_counts:
        r = results[count]
        gpu_time = r['gpu_time'] if r['gpu_time'] is not None else float('inf')
        gpu_speedup = r['gpu_vs_multi_speedup'] if r['gpu_vs_multi_speedup'] is not None else "N/A"
        logger.info(f"{count:11d} | {gpu_time:11.4f} | {r['cpu_multi_time']:19.4f} | {r['cpu_single_time']:19.4f} | {gpu_speedup}")
    
    return results

def estimate_theoretical_performance():
    """Estimate theoretical performance based on hardware specs."""
    profiler = PerformanceProfiler()
    return profiler.estimate_theoretical_performance()

def main():
    """Main test function."""
    logger.info("GPU-Accelerated Geometric Algebra Test")
    logger.info("======================================")
    logger.info(f"CPU cores: {CPU_CORES}")
    logger.info(f"GPU available: {HAS_CUPY}")
    
    # Run basic comparison test
    logger.info("\nRunning basic comparison test...")
    run_comparison_test(100)
    
    # Run scaling test
    logger.info("\nRunning scaling test...")
    run_scaling_test()
    
    # Estimate theoretical performance
    logger.info("\nEstimating theoretical performance...")
    estimate_theoretical_performance()
    
    logger.info("\nAll tests completed.")

if __name__ == "__main__":
    main()