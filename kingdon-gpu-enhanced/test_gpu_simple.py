#!/usr/bin/env python3
"""
Simple GPU acceleration test for the Kingdon GA library
"""

try:
    import cupy as cp
    print(f"CuPy version: {cp.__version__}")
    print(f"CUDA available: {cp.cuda.is_available()}")
    
    # Test basic GPU operations
    x_cpu = [1, 2, 3, 4, 5]
    x_gpu = cp.asarray(x_cpu)
    y_gpu = x_gpu * 2
    y_cpu = cp.asnumpy(y_gpu)
    
    print(f"CPU array: {x_cpu}")
    print(f"GPU result: {y_cpu}")
    print("[SUCCESS] Basic GPU operations working")
    
    # Test GPU memory info
    mempool = cp.get_default_memory_pool()
    print(f"GPU memory pool: {mempool.total_bytes()} bytes total")
    
    # Get device info
    device = cp.cuda.Device()
    print(f"GPU name: {device.name}")
    print(f"GPU memory: {device.mem_info[1] / 1e9:.2f} GB total")
    
except ImportError:
    print("CuPy not available - GPU acceleration disabled")
except Exception as e:
    print(f"GPU test failed: {e}")