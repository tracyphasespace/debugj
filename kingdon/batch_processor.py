"""
Batch Processor Module for High-Throughput Geometric Algebra
============================================================

This module provides high-level classes for processing large batches of
multivector operations, leveraging GPU acceleration and vectorization.
"""

from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING
import numpy as np

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from .algebra import Algebra
    from .multivector import MultiVector

try:
    import cupy as cp
    CUDA_AVAILABLE = cp.cuda.is_available()
except ImportError:
    cp = None
    CUDA_AVAILABLE = False

class BatchProcessor:
    """
    Orchestrates high-performance, batched operations on MultiVectors.

    This class formalizes the optimization patterns used in advanced benchmarks,
    providing a clean API for processing large arrays of GA objects efficiently.
    """
    def __init__(self, algebra: 'Algebra', use_gpu: bool = True):
        """
        Initialize the BatchProcessor.

        Args:
            algebra: The Algebra instance to work within.
            use_gpu: If True, attempt to use CuPy for GPU acceleration.
                     Falls back to NumPy if CUDA is not available.
        """
        self.algebra = algebra
        self.use_gpu = use_gpu and CUDA_AVAILABLE
        self.xp = cp if self.use_gpu else np
        print(f"✅ BatchProcessor Initialized. Mode: {'GPU' if self.use_gpu else 'CPU'}")

    def _prepare_batch_mv(self, items: List['MultiVector']) -> 'MultiVector':
        """
        Converts a list of multivectors into a single, batched MultiVector.
        This is a critical step for vectorization.
        """
        if not items:
            raise ValueError("Input list of multivectors cannot be empty.")

        # Assume all multivectors in the list share the same keys (structure)
        template = items[0]
        keys = template.keys()
        batch_size = len(items)

        # Create a single array containing all values
        values_array = self.xp.empty((batch_size, len(keys)), dtype=np.float32)
        for i, mv in enumerate(items):
            # A more robust version would check keys, but we assume consistency for now
            values_array[i, :] = self.xp.asarray(mv.values())

        # Create the single, batched MultiVector object
        return self.algebra.multivector(keys=keys, values=values_array)

    def sandwich_product_batch(self, rays: List['MultiVector'], rotors: List['MultiVector']) -> List['MultiVector']:
        """
        Performs the sandwich product (R * M * ~R) on entire batches of
        rays and rotors.

        Args:
            rays: A list of multivectors to be transformed.
            rotors: A list of rotors. Must be the same length as `rays`.

        Returns:
            A list of the resulting transformed multivectors.
        """
        if len(rays) != len(rotors):
            raise ValueError(f"Input lists must have the same length. Got {len(rays)} rays and {len(rotors)} rotors.")

        # 1. Convert the Python lists of objects into single, batched MultiVectors
        batched_rays = self._prepare_batch_mv(rays)
        batched_rotors = self._prepare_batch_mv(rotors)

        # 2. Perform the entire operation with three high-level, vectorized calls
        # The kingdon JIT will compile this for (batch * batch) operations.
        batched_rotors_rev = batched_rotors.reverse()
        temp = self.algebra.gp(batched_rotors, batched_rays)
        result_batch = self.algebra.gp(temp, batched_rotors_rev)

        # 3. Unpack the resulting batched MultiVector back into a Python list
        result_values = result_batch.values() # This is a (batch_size, num_keys) array
        result_keys = result_batch.keys()
        
        results = []
        for i in range(len(rays)):
            # Create a new, single MultiVector for each result in the batch
            mv = self.algebra.multivector(keys=result_keys, values=result_values[i])
            results.append(mv)
            
        return results

    # We can add more batched methods here in the future (e.g., gp_batch, add_batch)