#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU-Accelerated Geometric Algebra Simulation
===========================================

This program implements GPU acceleration and multi-core CPU parallelism
for the Kingdon Geometric Algebra library, optimized for systems with
powerful GPUs and multi-core processors.
"""

import os
import time
import math
import numpy as np
import concurrent.futures
from typing import List, Tuple, Dict, Optional, Union, Any
from dataclasses import dataclass
import multiprocessing
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GA-GPU-Accelerator")

# Try importing GPU libraries, fall back gracefully if not available
try:
    import cupy as cp
    HAS_CUPY = True
    logger.info(f"CuPy found. Using GPU acceleration with CUDA.")
    # Get GPU info
    gpu_info = cp.cuda.runtime.getDeviceProperties(0)
    logger.info(f"GPU: {gpu_info['name'].decode()}")
    logger.info(f"Compute capability: {gpu_info['major']}.{gpu_info['minor']}")
    logger.info(f"Total memory: {gpu_info['totalGlobalMem'] / (1024**3):.2f} GB")
    logger.info(f"Multiprocessors: {gpu_info['multiProcessorCount']}")
except ImportError:
    HAS_CUPY = False
    logger.warning("CuPy not found. GPU acceleration disabled.")
    cp = None

# Import from Kingdon library
try:
    from src.algebra import Algebra
    from src.multivector import MultiVector
    from src.state_multivectors import (
        StateMultivectorBase, OpticalState, UltrasoundState, ElectromagneticState,
        create_optical_ray, state_to_gpu_format, gpu_format_to_state
    )
    from src.propagator_transforms import (
        PropagatorTransform, OpticalSurfacePropagator, SurfaceGeometry, MaterialProperties
    )
    KINGDON_AVAILABLE = True
except ImportError:
    logger.warning("Kingdon library not found in path. Using mock implementations.")
    KINGDON_AVAILABLE = False

# Constants
SPEED_OF_LIGHT = 299792458.0  # m/s
CPU_CORES = multiprocessing.cpu_count()
DEFAULT_BATCH_SIZE = 1024  # Default batch size for GPU processing
#-----
-------------------------------------------------------------------------
# Mock implementations if Kingdon library is not available
#------------------------------------------------------------------------------

if not KINGDON_AVAILABLE:
    @dataclass
    class Algebra:
        """Mock Algebra class with PGA signature."""
        p: int = 3
        q: int = 0
        r: int = 1  # PGA signature to match OpticalSimulation default
        
        def scalar(self, value):
            return MultiVector(self, keys=(0,), values=[value])
            
        def vector(self, components):
            return MultiVector(self, keys=(1, 2, 3), values=components)
            
        def translator(self, position, scale=1.0):
            return MultiVector(self, keys=(0,), values=[1.0])
    
    class MultiVector:
        """Mock MultiVector class."""
        def __init__(self, algebra, keys=(0,), values=[0]):
            self.algebra = algebra
            self._keys = keys
            self._values = values
            
        def copy(self):
            return MultiVector(self.algebra, self._keys, self._values.copy())
    
    @dataclass
    class StateMultivectorBase:
        """Mock StateMultivectorBase class."""
        algebra: Algebra
        _pose_motor: Optional[MultiVector] = None
        _attributes: Optional[np.ndarray] = None
        name: Optional[str] = None
        
        @property
        def pose_motor(self):
            if self._pose_motor is None:
                self._pose_motor = self.algebra.scalar(1.0)
            return self._pose_motor
            
        @property
        def attributes(self):
            if self._attributes is None:
                self._attributes = np.zeros(4, dtype=np.float32)
            return self._attributes
            
        def copy(self):
            new_state = type(self)(self.algebra, name=self.name)
            if self._pose_motor is not None:
                new_state._pose_motor = self._pose_motor.copy()
            if self._attributes is not None:
                new_state._attributes = self._attributes.copy()
            return new_state
    
    @dataclass
    class OpticalState(StateMultivectorBase):
        """Mock OpticalState class."""
        def __post_init__(self):
            if self._attributes is None:
                self._attributes = np.array([550e-9, 0.0, 0.0, 1.0], dtype=np.float32)
                
        @property
        def wavelength(self):
            return float(self.attributes[0])
            
        @wavelength.setter
        def wavelength(self, value):
            self.attributes[0] = value
            
        @property
        def intensity(self):
            return float(self.attributes[3])
            
        @intensity.setter
        def intensity(self, value):
            self.attributes[3] = value
    
    @dataclass
    class SurfaceGeometry:
        """Mock SurfaceGeometry class."""
        position: np.ndarray
        normal: np.ndarray
        radius_of_curvature: float
        conic_constant: float
        aspheric_coeffs: np.ndarray
    
    @dataclass
    class MaterialProperties:
        """Mock MaterialProperties class."""
        refractive_index: float
        absorption_coeff: float
        scatter_coeff: float
        dispersion_coeff: float = 0.0
    
    class PropagatorTransform:
        """Mock PropagatorTransform class."""
        def propagate(self, state, *args, **kwargs):
            return state.copy()
    
    class OpticalSurfacePropagator(PropagatorTransform):
        """Mock OpticalSurfacePropagator class."""
        def __init__(self, surface, material):
            self.surface = surface
            self.material = material
            
        def propagate(self, ray, **kwargs):
            new_ray = ray.copy()
            # Simple transmission calculation
            new_ray.intensity *= 0.95  # 5% loss
            return new_ray
    
    def create_optical_ray(algebra, position=(0,0,0), direction=(0,0,1), 
                          wavelength=550e-9, intensity=1.0, name=None):
        """Mock create_optical_ray function."""
        state = OpticalState(algebra, name=name)
        state.wavelength = wavelength
        state.intensity = intensity
        return state
        
    def state_to_gpu_format(state):
        """Mock state_to_gpu_format function."""
        motor_data = np.zeros(8, dtype=np.float32)
        return {
            'pose_motor': motor_data,
            'attributes': state.attributes.astype(np.float32),
            'name': state.name or ""
        }
        
    def gpu_format_to_state(algebra, gpu_data, state_type='optical'):
        """Mock gpu_format_to_state function."""
        state = OpticalState(algebra)
        state.attributes = gpu_data['attributes']
        return state
#---
---------------------------------------------------------------------------
# GPU Kernel Implementations
#------------------------------------------------------------------------------

class GPUKernels:
    """
    GPU kernel implementations for Geometric Algebra operations.
    
    This class provides CUDA kernels for efficient GPU computation
    of State Multivector operations.
    """
    
    @staticmethod
    def compile_kernels():
        """Compile all CUDA kernels."""
        if not HAS_CUPY:
            logger.warning("CuPy not available. Cannot compile GPU kernels.")
            return {}
            
        kernels = {}
        
        # Batch propagation kernel with proper GA operations
        #
        # IMPLEMENTATION NOTES AND SIMPLIFICATIONS:
        # ==========================================
        # 1. Geometric Algebra: Uses simplified GA rotors for ray direction updates
        #    - Full GA implementation would require complete Clifford algebra operations
        #    - Current version handles basic reflection/refraction via simplified rotors
        #    - For production use, extend with full bivector/multivector operations
        #
        # 2. Surface Geometry: Assumes planar surfaces with normal vectors
        #    - Real optical systems have curved surfaces requiring ray-surface intersection
        #    - Aspheric and freeform surfaces need additional geometric calculations
        #    - Current normal-based approach is first-order approximation
        #
        # 3. Physics Approximations:
        #    - Normal incidence assumption for Snell's law calculations
        #    - Fixed 1mm thickness for absorption calculations
        #    - Simplified Fresnel coefficients (s-polarized only)
        #    - Linear dispersion model (real glasses have complex dispersion)
        #
        # 4. Performance Optimizations:
        #    - Register-resident operations for maximum GPU throughput
        #    - Batched processing to maintain memory coalescing
        #    - Early termination for weak rays (intensity < 1e-6)
        #
        # 5. Extensibility:
        #    - Add polarization vector tracking for complete Jones calculus
        #    - Implement arbitrary surface equations (Zernike polynomials, etc.)
        #    - Add coherence/interference effects for wave optics
        #    - Support for gradient-index media and metamaterials
        #
        # This kernel demonstrates the GPU-accelerated GA framework approach
        # while maintaining practical performance for real-time optical simulation.
        batch_propagation_kernel_code = r'''
        // GA utility functions for Cl(3,1) operations
        __device__ void ga_reflect_direction_in_plane(
            float* motor,          // The current motor
            float nx, float ny, float nz, // The actual surface normal
            float* result         // The output motor
        ) {
            // Extract direction components from motor (simplified)
            float dx = motor[1], dy = motor[2], dz = motor[3];
            
            // Standard vector reflection formula: v' = v - 2*dot(v,n)*n
            // This is equivalent to the GA sandwich product p*v*p but optimized
            float dot = 2.0f * (dx*nx + dy*ny + dz*nz);
            
            result[0] = motor[0];      // scalar part unchanged
            result[1] = dx - dot*nx;   // correctly reflected x direction
            result[2] = dy - dot*ny;   // correctly reflected y direction  
            result[3] = dz - dot*nz;   // correctly reflected z direction
            
            // Copy other motor components unchanged (simplified)
            for (int i = 4; i < 8; i++) {
                result[i] = motor[i];
            }
        }
        
        extern "C" __global__ void batch_propagation(
            float* pose_motors,        // [num_rays * 8] pose motor components
            float* attributes,         // [num_rays * 4] optical attributes
            float* surfaces,           // [num_surfaces * 5] surface parameters
            float* materials,          // [num_surfaces * 4] material parameters
            int num_rays,
            int num_surfaces
        ) {
            // Get ray index
            int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (ray_idx >= num_rays) return;
            
            // Get base indices for this ray's data
            int pose_offset = ray_idx * 8;
            int attr_offset = ray_idx * 4;
            
            // Load current pose motor components
            float current_motor[8];
            for (int i = 0; i < 8; i++) {
                current_motor[i] = pose_motors[pose_offset + i];
            }
            
            // Extract ray attributes
            float wavelength = attributes[attr_offset + 0];
            float polarization = attributes[attr_offset + 1];
            float phase = attributes[attr_offset + 2];
            float intensity = attributes[attr_offset + 3];
            
            // Process each surface with GA operations
            for (int surf_idx = 0; surf_idx < num_surfaces; surf_idx++) {
                // Skip if ray is too weak
                if (intensity < 1e-6f) break;
                
                // Get surface and material parameters
                int surf_offset = surf_idx * 5;
                int mat_offset = surf_idx * 4;
                
                // Surface geometry (simplified to normal vector)
                float surf_nx = surfaces[surf_offset + 0];
                float surf_ny = surfaces[surf_offset + 1]; 
                float surf_nz = surfaces[surf_offset + 2];
                float radius = surfaces[surf_offset + 3];
                float conic = surfaces[surf_offset + 4];
                
                // Material properties
                float n2 = materials[mat_offset + 0];
                float absorption = materials[mat_offset + 1];
                float scatter = materials[mat_offset + 2];
                float dispersion = materials[mat_offset + 3];
                
                // Apply GA reflection to update ray direction (pose motor)
                // Uses direct surface normal for correct reflection calculation
                float new_motor[8];
                ga_reflect_direction_in_plane(current_motor, surf_nx, surf_ny, surf_nz, new_motor);
                
                // Update current motor for next surface
                for (int i = 0; i < 8; i++) {
                    current_motor[i] = new_motor[i];
                }
                
                // Physics calculations with proper Snell's law
                float n1 = 1.0f;  // Assume from air
                float cos_theta_i = fabsf(surf_nz);  // Simplified normal incidence
                float sin_theta_i = sqrtf(1.0f - cos_theta_i*cos_theta_i);
                float sin_theta_t = (n1/n2) * sin_theta_i;
                
                // Check for total internal reflection
                if (sin_theta_t <= 1.0f) {
                    // Transmission case
                    float cos_theta_t = sqrtf(1.0f - sin_theta_t*sin_theta_t);
                    
                    // Fresnel coefficients for s-polarized light
                    float rs = (n1*cos_theta_i - n2*cos_theta_t) / (n1*cos_theta_i + n2*cos_theta_t);
                    float reflectance = rs * rs;
                    float transmittance = 1.0f - reflectance;
                    
                    // Apply transmission loss
                    intensity *= transmittance;
                    
                    // Update phase for optical path length change
                    phase += 2.0f * 3.14159f * n2 * 0.001f / wavelength;  // 1mm thickness
                } else {
                    // Total internal reflection - intensity unchanged but direction reflected
                    intensity *= 0.95f;  // Small loss due to surface imperfections
                }
                
                // Apply absorption
                float absorption_loss = expf(-absorption * 0.001f);
                intensity *= absorption_loss;
                
                // Apply chromatic dispersion
                if (dispersion != 0.0f) {
                    wavelength *= (1.0f + dispersion * (n2 - n1));
                }
            }
            
            // Write back updated pose motor (GA result)
            for (int i = 0; i < 8; i++) {
                pose_motors[pose_offset + i] = current_motor[i];
            }
            
            // Update ray attributes after all surfaces
            attributes[attr_offset + 0] = wavelength;
            attributes[attr_offset + 1] = polarization;
            attributes[attr_offset + 2] = phase;
            attributes[attr_offset + 3] = intensity;
        }
        '''
        
        try:
            kernels['batch_propagation'] = cp.RawKernel(
                batch_propagation_kernel_code, 
                'batch_propagation'
            )
            logger.info("Successfully compiled batch propagation kernel")
        except Exception as e:
            logger.error(f"Failed to compile batch propagation kernel: {e}")
            
        return kernels#--
----------------------------------------------------------------------------
# GPU-Accelerated Optical System
#------------------------------------------------------------------------------

class GPUAcceleratedOpticalSystem:
    """
    GPU-accelerated optical system simulation.
    
    This class implements optical ray propagation through multiple surfaces
    using GPU acceleration for maximum performance.
    """
    
    def __init__(self, surfaces=None, materials=None):
        """
        Initialize the GPU-accelerated optical system.
        
        Args:
            surfaces: List of SurfaceGeometry objects
            materials: List of MaterialProperties objects
        """
        self.surfaces = surfaces or []
        self.materials = materials or []
        self.kernels = {}
        
        # Initialize GPU if available
        self.gpu_available = HAS_CUPY
        if self.gpu_available:
            try:
                self.kernels = GPUKernels.compile_kernels()
                if not self.kernels:
                    logger.warning("Failed to compile GPU kernels. Falling back to CPU.")
                    self.gpu_available = False
            except Exception as e:
                logger.error(f"Error initializing GPU: {e}")
                self.gpu_available = False
        
        # Create CPU propagators as fallback
        self.cpu_propagators = []
        if surfaces and materials:
            if len(surfaces) != len(materials):
                raise ValueError("Number of surfaces must match number of materials")
                
            self.cpu_propagators = [
                OpticalSurfacePropagator(surf, mat)
                for surf, mat in zip(surfaces, materials)
            ]
    
    def add_surface(self, surface, material):
        """Add a surface and its material to the optical system."""
        self.surfaces.append(surface)
        self.materials.append(material)
        self.cpu_propagators.append(OpticalSurfacePropagator(surface, material))
    
    def prepare_gpu_data(self, surfaces, materials):
        """
        Prepare surface and material data for GPU processing.
        
        Args:
            surfaces: List of SurfaceGeometry objects
            materials: List of MaterialProperties objects
            
        Returns:
            Tuple of (surface_array, material_array) for GPU
        """
        if not self.gpu_available:
            return None, None
            
        # Extract surface parameters (position, normal, radius, conic)
        surface_params = np.zeros((len(surfaces), 5), dtype=np.float32)
        for i, surface in enumerate(surfaces):
            surface_params[i, 0:3] = surface.position[0:3]
            surface_params[i, 3] = surface.radius_of_curvature
            surface_params[i, 4] = surface.conic_constant
            
        # Extract material parameters (n, absorption, scatter, dispersion)
        material_params = np.zeros((len(materials), 4), dtype=np.float32)
        for i, material in enumerate(materials):
            material_params[i, 0] = material.refractive_index
            material_params[i, 1] = material.absorption_coeff
            material_params[i, 2] = material.scatter_coeff
            material_params[i, 3] = material.dispersion_coeff
            
        return surface_params, material_params
    
    def prepare_ray_batch(self, rays):
        """
        Prepare ray batch data for GPU processing.
        
        Args:
            rays: List of OpticalState objects
            
        Returns:
            Tuple of (pose_motors_array, attributes_array) for GPU
        """
        if not rays:
            return None, None
            
        # Extract pose motors and attributes
        pose_motors = np.zeros((len(rays), 8), dtype=np.float32)
        attributes = np.zeros((len(rays), 4), dtype=np.float32)
        
        for i, ray in enumerate(rays):
            # Convert to GPU format
            gpu_data = state_to_gpu_format(ray)
            pose_motors[i] = gpu_data['pose_motor']
            attributes[i] = gpu_data['attributes']
            
        return pose_motors, attributes
    
    def propagate_gpu(self, rays, batch_size=DEFAULT_BATCH_SIZE):
        """
        Propagate rays through the optical system using GPU acceleration.
        
        Args:
            rays: List of OpticalState objects
            batch_size: Batch size for GPU processing
            
        Returns:
            List of propagated OpticalState objects
        """
        if not self.gpu_available or not self.kernels or 'batch_propagation' not in self.kernels:
            logger.warning("GPU acceleration not available. Falling back to CPU.")
            return self.propagate_cpu(rays)
            
        start_time = time.time()
        logger.info(f"Starting GPU propagation of {len(rays)} rays")
        
        # Prepare surface and material data
        surface_params, material_params = self.prepare_gpu_data(self.surfaces, self.materials)
        d_surface_params = cp.asarray(surface_params, dtype=cp.float32)
        d_material_params = cp.asarray(material_params, dtype=cp.float32)
        
        # Process in batches
        results = []
        for batch_start in range(0, len(rays), batch_size):
            batch_end = min(batch_start + batch_size, len(rays))
            batch_rays = rays[batch_start:batch_end]
            batch_size_actual = len(batch_rays)
            
            # Prepare ray batch data
            pose_motors, attributes = self.prepare_ray_batch(batch_rays)
            d_pose_motors = cp.asarray(pose_motors, dtype=cp.float32)
            d_attributes = cp.asarray(attributes, dtype=cp.float32)
            
            # Launch kernel
            threads_per_block = 256
            blocks = (batch_size_actual + threads_per_block - 1) // threads_per_block
            
            self.kernels['batch_propagation']((blocks,), (threads_per_block,), (
                d_pose_motors, d_attributes, 
                d_surface_params, d_material_params,
                batch_size_actual, len(self.surfaces)
            ))
            
            # Copy results back to CPU
            result_pose_motors = cp.asnumpy(d_pose_motors).reshape(batch_size_actual, 8)
            result_attributes = cp.asnumpy(d_attributes).reshape(batch_size_actual, 4)
            
            # Convert back to OpticalState objects
            for i, ray in enumerate(batch_rays):
                # Create GPU data dictionary
                gpu_data = {
                    'pose_motor': result_pose_motors[i],
                    'attributes': result_attributes[i],
                    'name': ray.name
                }
                
                # Convert back to OpticalState
                result_ray = gpu_format_to_state(ray.algebra, gpu_data, 'optical')
                results.append(result_ray)
                
            logger.debug(f"Processed batch {batch_start//batch_size + 1}, "
                        f"{batch_size_actual} rays")
        
        end_time = time.time()
        logger.info(f"GPU propagation completed in {end_time - start_time:.4f} seconds")
        
        return results
    
    def propagate_cpu(self, rays):
        """
        Propagate rays through the optical system using CPU.
        
        Args:
            rays: List of OpticalState objects
            
        Returns:
            List of propagated OpticalState objects
        """
        start_time = time.time()
        logger.info(f"Starting CPU propagation of {len(rays)} rays")
        
        results = []
        for ray in rays:
            current_ray = ray.copy()
            
            # Propagate through each surface
            for propagator in self.cpu_propagators:
                current_ray = propagator.propagate(current_ray)
                
                # Check if ray was absorbed or lost
                if current_ray.intensity < 1e-6:
                    break
                    
            results.append(current_ray)
            
        end_time = time.time()
        logger.info(f"CPU propagation completed in {end_time - start_time:.4f} seconds")
        
        return results
    
    @staticmethod
    def _propagate_ray_chunk(args):
        """
        Static worker function for parallel processing.
        
        Args:
            args: Tuple of (propagators, chunk) where chunk is list of rays
            
        Returns:
            List of propagated rays
        """
        propagators, chunk = args
        results = []
        for ray in chunk:
            current_ray = ray.copy()
            # Propagate through each surface
            for propagator in propagators:
                current_ray = propagator.propagate(current_ray)
                # Check if ray was absorbed or lost
                if current_ray.intensity < 1e-6:
                    break
            results.append(current_ray)
        return results

    def propagate_cpu_parallel(self, rays, num_workers=None):
        """
        Propagate rays through the optical system using parallel CPU processing.
        
        Args:
            rays: List of OpticalState objects
            num_workers: Number of worker processes (default: CPU core count)
            
        Returns:
            List of propagated OpticalState objects
        """
        if num_workers is None:
            num_workers = CPU_CORES
            
        start_time = time.time()
        logger.info(f"Starting parallel CPU propagation with {num_workers} workers")
        
        # Split rays into chunks for parallel processing
        chunk_size = max(1, len(rays) // num_workers)
        chunks = [rays[i:i+chunk_size] for i in range(0, len(rays), chunk_size)]
        
        # Package propagators with each chunk of rays
        tasks = [(self.cpu_propagators, chunk) for chunk in chunks]
        
        # Process chunks in parallel using the static method
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            chunk_results = list(executor.map(GPUAcceleratedOpticalSystem._propagate_ray_chunk, tasks))
            
        # Flatten results
        results = [ray for chunk in chunk_results for ray in chunk]
        
        end_time = time.time()
        logger.info(f"Parallel CPU propagation completed in {end_time - start_time:.4f} seconds")
        
        return results
    
    def propagate_ray_cpu(self, ray):
        """Propagate a single ray through all surfaces using CPU."""
        current_ray = ray.copy()
        
        # Propagate through each surface
        for propagator in self.cpu_propagators:
            current_ray = propagator.propagate(current_ray)
            
            # Check if ray was absorbed or lost
            if current_ray.intensity < 1e-6:
                break
                
        return current_ray
    
    def propagate(self, rays, use_gpu=True, batch_size=DEFAULT_BATCH_SIZE, num_cpu_workers=None):
        """
        Propagate rays through the optical system using the best available method.
        
        Args:
            rays: List of OpticalState objects
            use_gpu: Whether to use GPU acceleration if available
            batch_size: Batch size for GPU processing
            num_cpu_workers: Number of CPU workers for parallel processing
            
        Returns:
            List of propagated OpticalState objects
        """
        if use_gpu and self.gpu_available and len(rays) > 100:
            return self.propagate_gpu(rays, batch_size)
        elif len(rays) > 10:
            return self.propagate_cpu_parallel(rays, num_cpu_workers)
        else:
            return self.propagate_cpu(rays)#----
--------------------------------------------------------------------------
# Simulation Framework
#------------------------------------------------------------------------------

class OpticalSimulation:
    """
    High-performance optical simulation framework.
    
    This class provides a complete framework for running optical simulations
    with automatic hardware optimization.
    """
    
    def __init__(self, algebra=None):
        """
        Initialize the optical simulation.
        
        Args:
            algebra: Geometric algebra instance (default: 3D PGA)
        """
        self.algebra = algebra or Algebra(p=3, q=0, r=1)
        self.optical_system = GPUAcceleratedOpticalSystem()
        self.wavelengths = []
        self.rays = []
        self.results = []
        
    def create_default_system(self):
        """Create a default optical system (achromatic doublet)."""
        # Crown glass lens
        surface1 = SurfaceGeometry(
            position=np.array([0, 0, 0]),
            normal=np.array([0, 0, 1]),
            radius_of_curvature=50e-3,  # 50mm radius
            conic_constant=0.0,
            aspheric_coeffs=np.zeros(4)
        )
        
        material1 = MaterialProperties(
            refractive_index=1.52,
            absorption_coeff=0.001,
            scatter_coeff=0.0001,
            dispersion_coeff=0.01
        )
        
        # Air gap
        surface2 = SurfaceGeometry(
            position=np.array([0, 0, 10e-3]),
            normal=np.array([0, 0, 1]),
            radius_of_curvature=-60e-3,  # -60mm radius (concave)
            conic_constant=0.0,
            aspheric_coeffs=np.zeros(4)
        )
        
        material2 = MaterialProperties(
            refractive_index=1.0,  # Air
            absorption_coeff=0.0,
            scatter_coeff=0.0,
            dispersion_coeff=0.0
        )
        
        # Flint glass lens
        surface3 = SurfaceGeometry(
            position=np.array([0, 0, 12e-3]),
            normal=np.array([0, 0, 1]),
            radius_of_curvature=40e-3,  # 40mm radius
            conic_constant=0.0,
            aspheric_coeffs=np.zeros(4)
        )
        
        material3 = MaterialProperties(
            refractive_index=1.62,
            absorption_coeff=0.002,
            scatter_coeff=0.0002,
            dispersion_coeff=0.02
        )
        
        # Final surface to air
        surface4 = SurfaceGeometry(
            position=np.array([0, 0, 20e-3]),
            normal=np.array([0, 0, 1]),
            radius_of_curvature=-45e-3,  # -45mm radius (concave)
            conic_constant=0.0,
            aspheric_coeffs=np.zeros(4)
        )
        
        material4 = MaterialProperties(
            refractive_index=1.0,  # Air
            absorption_coeff=0.0,
            scatter_coeff=0.0,
            dispersion_coeff=0.0
        )
        
        # Add surfaces to optical system
        self.optical_system = GPUAcceleratedOpticalSystem(
            surfaces=[surface1, surface2, surface3, surface4],
            materials=[material1, material2, material3, material4]
        )
        
        logger.info("Created default achromatic doublet optical system")
        
    def create_wavelength_spectrum(self, start_nm=400, end_nm=700, count=100):
        """
        Create a spectrum of wavelengths for simulation.
        
        Args:
            start_nm: Starting wavelength in nanometers
            end_nm: Ending wavelength in nanometers
            count: Number of wavelengths to generate
            
        Returns:
            List of wavelengths in meters
        """
        wavelengths_nm = np.linspace(start_nm, end_nm, count)
        self.wavelengths = [w * 1e-9 for w in wavelengths_nm]  # Convert to meters
        logger.info(f"Created wavelength spectrum with {count} values from "
                   f"{start_nm}nm to {end_nm}nm")
        return self.wavelengths
    
    def create_rays(self, position=(0, 0, -10e-3), direction=(0, 0, 1), intensity=1.0):
        """
        Create rays for each wavelength in the spectrum.
        
        Args:
            position: Starting position for rays
            direction: Direction vector for rays
            intensity: Initial intensity
            
        Returns:
            List of OpticalState objects
        """
        if not self.wavelengths:
            logger.warning("No wavelengths defined. Creating default spectrum.")
            self.create_wavelength_spectrum()
            
        self.rays = []
        for i, wavelength in enumerate(self.wavelengths):
            ray = create_optical_ray(
                self.algebra,
                position=position,
                direction=direction,
                wavelength=wavelength,
                intensity=intensity,
                name=f"ray_{i:03d}_{int(wavelength*1e9):d}nm"
            )
            self.rays.append(ray)
            
        logger.info(f"Created {len(self.rays)} rays")
        return self.rays
    
    def run_simulation(self, use_gpu=True, batch_size=DEFAULT_BATCH_SIZE, num_cpu_workers=None):
        """
        Run the optical simulation.
        
        Args:
            use_gpu: Whether to use GPU acceleration if available
            batch_size: Batch size for GPU processing
            num_cpu_workers: Number of CPU workers for parallel processing
            
        Returns:
            List of propagated OpticalState objects
        """
        if not self.rays:
            logger.warning("No rays defined. Creating default rays.")
            self.create_rays()
            
        logger.info(f"Running simulation with {len(self.rays)} rays")
        
        # Run simulation with performance measurement
        start_time = time.time()
        self.results = self.optical_system.propagate(
            self.rays, 
            use_gpu=use_gpu,
            batch_size=batch_size,
            num_cpu_workers=num_cpu_workers
        )
        end_time = time.time()
        
        # Calculate performance metrics
        total_time = end_time - start_time
        rays_per_second = len(self.rays) / total_time
        surfaces = len(self.optical_system.surfaces)
        operations = len(self.rays) * surfaces
        operations_per_second = operations / total_time
        
        logger.info(f"Simulation completed in {total_time:.4f} seconds")
        logger.info(f"Performance: {rays_per_second:.0f} rays/second")
        logger.info(f"Operations: {operations_per_second:.0f} ray-surface interactions/second")
        
        return self.results
    
    def analyze_results(self):
        """
        Analyze simulation results.
        
        Returns:
            Dictionary of analysis results
        """
        if not self.results:
            logger.warning("No simulation results to analyze.")
            return {}
            
        # Calculate transmission by wavelength
        transmission_data = []
        for initial, final in zip(self.rays, self.results):
            wavelength_nm = initial.wavelength * 1e9
            transmission = final.intensity / initial.intensity
            transmission_data.append((wavelength_nm, transmission))
            
        # Find best and worst transmission
        transmission_data.sort(key=lambda x: x[1])
        worst_wavelength, worst_transmission = transmission_data[0]
        best_wavelength, best_transmission = transmission_data[-1]
        
        # Calculate average transmission by spectral band
        blue_band = [t for w, t in transmission_data if 400 <= w <= 500]
        green_band = [t for w, t in transmission_data if 500 <= w <= 600]
        red_band = [t for w, t in transmission_data if 600 <= w <= 700]
        
        # Sort by wavelength for plotting
        transmission_data.sort(key=lambda x: x[0])
        
        analysis = {
            'transmission_data': transmission_data,
            'best_transmission': (best_wavelength, best_transmission),
            'worst_transmission': (worst_wavelength, worst_transmission),
            'blue_band_avg': np.mean(blue_band) if blue_band else 0,
            'green_band_avg': np.mean(green_band) if green_band else 0,
            'red_band_avg': np.mean(red_band) if red_band else 0
        }
        
        logger.info(f"Analysis complete:")
        logger.info(f"  Best transmission: {best_transmission:.3f} at {best_wavelength:.1f}nm")
        logger.info(f"  Worst transmission: {worst_transmission:.3f} at {worst_wavelength:.1f}nm")
        logger.info(f"  Blue band avg: {analysis['blue_band_avg']:.3f}")
        logger.info(f"  Green band avg: {analysis['green_band_avg']:.3f}")
        logger.info(f"  Red band avg: {analysis['red_band_avg']:.3f}")
        
        return analysis
    
    def save_results(self, filename='simulation_results.npz'):
        """
        Save simulation results to file.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        if not self.results:
            logger.warning("No simulation results to save.")
            return None
            
        # Extract data to save
        wavelengths = np.array([ray.wavelength for ray in self.rays])
        intensities = np.array([ray.intensity for ray in self.results])
        
        # Save to numpy compressed format
        np.savez(
            filename,
            wavelengths=wavelengths,
            intensities=intensities,
            timestamp=time.time()
        )
        
        logger.info(f"Results saved to {filename}")
        return filename#-
-----------------------------------------------------------------------------
# Performance Profiling
#------------------------------------------------------------------------------

class PerformanceProfiler:
    """
    Performance profiling tools for optical simulations.
    
    This class provides tools for measuring and optimizing performance
    of optical simulations on different hardware.
    """
    
    def __init__(self):
        """Initialize the performance profiler."""
        self.results = {}
        
    def profile_simulation(self, simulation, use_gpu=True, batch_sizes=None, 
                          num_cpu_workers=None):
        """
        Profile simulation performance with different parameters.
        
        Args:
            simulation: OpticalSimulation instance
            use_gpu: Whether to use GPU
            batch_sizes: List of batch sizes to test
            num_cpu_workers: List of CPU worker counts to test
            
        Returns:
            Dictionary of profiling results
        """
        if batch_sizes is None:
            batch_sizes = [128, 256, 512, 1024, 2048, 4096]
            
        if num_cpu_workers is None:
            num_cpu_workers = [1, 2, 4, CPU_CORES]
            
        results = {}
        
        # Profile GPU performance with different batch sizes
        if use_gpu and HAS_CUPY:
            gpu_results = {}
            for batch_size in batch_sizes:
                logger.info(f"Profiling GPU with batch size {batch_size}")
                start_time = time.time()
                simulation.run_simulation(use_gpu=True, batch_size=batch_size)
                end_time = time.time()
                gpu_results[batch_size] = end_time - start_time
                
            results['gpu'] = gpu_results
            
            # Find optimal batch size
            optimal_batch_size = min(gpu_results, key=gpu_results.get)
            results['optimal_batch_size'] = optimal_batch_size
            logger.info(f"Optimal GPU batch size: {optimal_batch_size}")
            
        # Profile CPU performance with different worker counts
        cpu_results = {}
        for workers in num_cpu_workers:
            logger.info(f"Profiling CPU with {workers} workers")
            start_time = time.time()
            simulation.run_simulation(use_gpu=False, num_cpu_workers=workers)
            end_time = time.time()
            cpu_results[workers] = end_time - start_time
            
        results['cpu'] = cpu_results
        
        # Find optimal worker count
        optimal_workers = min(cpu_results, key=cpu_results.get)
        results['optimal_workers'] = optimal_workers
        logger.info(f"Optimal CPU worker count: {optimal_workers}")
        
        # Compare GPU vs CPU
        if 'gpu' in results:
            best_gpu_time = min(results['gpu'].values())
            best_cpu_time = min(results['cpu'].values())
            speedup = best_cpu_time / best_gpu_time if best_gpu_time > 0 else 0
            results['gpu_vs_cpu_speedup'] = speedup
            logger.info(f"GPU vs CPU speedup: {speedup:.2f}x")
            
        self.results = results
        return results
    
    def estimate_theoretical_performance(self, gpu_info=None):
        """
        Estimate theoretical performance based on hardware specs.
        
        Args:
            gpu_info: GPU information dictionary
            
        Returns:
            Dictionary of theoretical performance estimates
        """
        if gpu_info is None and HAS_CUPY:
            try:
                gpu_info = {
                    'name': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                    'cores': cp.cuda.runtime.getDeviceProperties(0)['multiProcessorCount'],
                    'clock': cp.cuda.runtime.getDeviceProperties(0)['clockRate'] / 1000
                }
            except:
                gpu_info = {'name': 'Unknown', 'cores': 0, 'clock': 0}
                
        # Theoretical calculations
        state_size_bytes = 96  # Size of state multivector
        register_size_kb = 64  # Size of GPU register file
        states_per_register = (register_size_kb * 1024) // state_size_bytes
        
        # Operations per ray
        operations_per_ray = 5  # GA rotor operations per surface
        surfaces_per_system = 4  # Default system
        total_ops_per_ray = operations_per_ray * surfaces_per_system
        
        # GPU theoretical performance
        if gpu_info['cores'] > 0:
            batch_size = 4  # Conservative estimate
            rays_per_core_per_cycle = batch_size / total_ops_per_ray
            cycles_per_second = gpu_info['clock'] * 1e6
            rays_per_core_per_second = rays_per_core_per_cycle * cycles_per_second
            total_rays_per_second = rays_per_core_per_second * gpu_info['cores']
        else:
            total_rays_per_second = 0
            
        # CPU theoretical performance
        cpu_rays_per_second = 100000  # Estimated baseline
        cpu_total_rays_per_second = cpu_rays_per_second * CPU_CORES
        
        results = {
            'gpu_info': gpu_info,
            'cpu_cores': CPU_CORES,
            'state_size_bytes': state_size_bytes,
            'states_per_register': states_per_register,
            'operations_per_ray': total_ops_per_ray,
            'gpu_theoretical_rays_per_second': total_rays_per_second,
            'cpu_theoretical_rays_per_second': cpu_total_rays_per_second,
            'theoretical_speedup': total_rays_per_second / cpu_total_rays_per_second if cpu_total_rays_per_second > 0 else 0
        }
        
        logger.info(f"Theoretical performance estimates:")
        logger.info(f"  GPU: {results['gpu_theoretical_rays_per_second']:,.0f} rays/second")
        logger.info(f"  CPU: {results['cpu_theoretical_rays_per_second']:,.0f} rays/second")
        logger.info(f"  Theoretical speedup: {results['theoretical_speedup']:.2f}x")
        
        return results

#------------------------------------------------------------------------------
# Main Function
#------------------------------------------------------------------------------

def main():
    """Main function for the GPU-accelerated simulation."""
    parser = argparse.ArgumentParser(description='GPU-Accelerated Geometric Algebra Simulation')
    parser.add_argument('--wavelengths', type=int, default=100,
                       help='Number of wavelengths to simulate')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU acceleration')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                       help='GPU batch size')
    parser.add_argument('--cpu-workers', type=int, default=CPU_CORES,
                       help='Number of CPU worker processes')
    parser.add_argument('--profile', action='store_true',
                       help='Run performance profiling')
    parser.add_argument('--output', type=str, default='simulation_results.npz',
                       help='Output file for results')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Print system information
    logger.info(f"System information:")
    logger.info(f"  CPU cores: {CPU_CORES}")
    logger.info(f"  GPU available: {HAS_CUPY}")
    
    # Create and run simulation
    simulation = OpticalSimulation()
    simulation.create_default_system()
    simulation.create_wavelength_spectrum(count=args.wavelengths)
    simulation.create_rays()
    
    if args.profile:
        # Run performance profiling
        profiler = PerformanceProfiler()
        profiler.profile_simulation(
            simulation,
            use_gpu=not args.no_gpu,
            batch_sizes=[128, 256, 512, 1024, 2048, 4096],
            num_cpu_workers=[1, 2, 4, CPU_CORES]
        )
        profiler.estimate_theoretical_performance()
    else:
        # Run normal simulation
        simulation.run_simulation(
            use_gpu=not args.no_gpu,
            batch_size=args.batch_size,
            num_cpu_workers=args.cpu_workers
        )
        
        # Analyze and save results
        simulation.analyze_results()
        simulation.save_results(args.output)
    
    logger.info("Simulation completed successfully")

if __name__ == '__main__':
    main()