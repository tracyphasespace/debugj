#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geometric Algebra Projection Operations Module
==============================================

This module implements general GA projection operations for interactions between
arbitrary state multivectors and geometric objects. Based on the design outlined
in comments.txt and the architectural vision for register-resident computing.

Key References:
- Doran & Lasenby, "Geometric Algebra for Physicists", Sec 3.4
- GPU_GA_State_Propagators.md
- PropagatorTransform base class architecture
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from .algebra import Algebra
    from .multivector import MultiVector

def project_multivector(a: 'MultiVector', b: 'MultiVector') -> 'MultiVector':
    """
    Compute the projection of multivector a onto the subspace defined by multivector b.
    
    This is the fundamental GA operation for geometric interactions:
    - In optics: refraction/reflection of rays (a) at surfaces (b)
    - In acoustics: wave transmission at tissue boundaries
    - In general physics: decomposition of forces, fields, etc.
    
    Args:
        a: The multivector to be projected (e.g., ray, wave, field)
        b: The multivector defining the subspace (e.g., surface normal, plane)
        
    Returns:
        The projection of a onto b: project(a, b) = (a >> b) / |b|²
        
    Notes:
        - For vectors: project(v, n) = (v · n̂) n̂
        - For general blades: uses left contraction (>>)
        - Handles grade mixing automatically
    """
    # Get the algebra instance
    algebra = a.algebra
    
    # Compute the norm squared of b
    b_reverse = algebra.reverse(b)
    b_norm_sq_mv = algebra.gp(b, b_reverse)
    
    # Extract scalar part for norm squared
    b_norm_sq = float(b_norm_sq_mv.values()[0]) if b_norm_sq_mv.values() else 0.0
    
    if abs(b_norm_sq) < 1e-12:
        # b is null or near-null, return zero multivector
        return algebra.scalar(0.0)
    
    # Compute left contraction a >> b
    left_contraction = algebra.left_contraction(a, b)
    
    # Normalize: project(a, b) = (a >> b) / |b|²
    projection = algebra.scalar_multiply(left_contraction, 1.0 / b_norm_sq)
    
    return projection

def reject_multivector(a: 'MultiVector', b: 'MultiVector') -> 'MultiVector':
    """
    Compute the rejection of multivector a from the subspace defined by multivector b.
    
    The rejection is the component of a orthogonal to b:
    reject(a, b) = a - project(a, b)
    
    This is crucial for reflection calculations:
    reflected = a - 2 * project(a, n) = reject(a, n) - project(a, n)
    
    Args:
        a: The multivector to be rejected
        b: The multivector defining the subspace
        
    Returns:
        The rejection of a from b
    """
    projection = project_multivector(a, b)
    return a.algebra.subtract(a, projection)

def reflect_multivector(a: 'MultiVector', n: 'MultiVector') -> 'MultiVector':
    """
    Reflect multivector a across the hyperplane with normal n.
    
    This implements the fundamental reflection operation:
    reflect(a, n) = a - 2 * project(a, n)
    
    In GA terms, this is equivalent to the sandwich product n * a * n
    for a unit normal n, but computed more efficiently.
    
    Args:
        a: The multivector to be reflected (ray, vector, etc.)
        n: The normal to the reflection plane/hyperplane
        
    Returns:
        The reflected multivector
    """
    algebra = a.algebra
    
    # Compute projection
    projection = project_multivector(a, n)
    
    # Reflection = a - 2 * project(a, n)
    two_projection = algebra.scalar_multiply(projection, 2.0)
    reflection = algebra.subtract(a, two_projection)
    
    return reflection

def decompose_multivector(a: 'MultiVector', b: 'MultiVector') -> tuple['MultiVector', 'MultiVector']:
    """
    Decompose multivector a into components parallel and perpendicular to b.
    
    Returns (parallel_component, perpendicular_component) where:
    - parallel_component = project(a, b)
    - perpendicular_component = reject(a, b)
    - a = parallel_component + perpendicular_component
    
    This is essential for:
    - Fresnel coefficient calculations in optics
    - Mode conversion in acoustics
    - Field decomposition in electromagnetics
    
    Args:
        a: The multivector to decompose
        b: The reference direction/subspace
        
    Returns:
        Tuple of (parallel_component, perpendicular_component)
    """
    parallel = project_multivector(a, b)
    perpendicular = reject_multivector(a, b)
    
    return parallel, perpendicular

# Specialized projection operations for common physics scenarios

def project_vector_onto_plane(vector: 'MultiVector', plane_normal: 'MultiVector') -> 'MultiVector':
    """
    Project a vector onto a plane defined by its normal.
    
    This is the component of the vector that lies in the plane.
    Equivalent to: vector - project(vector, normal)
    
    Args:
        vector: The vector to project
        plane_normal: Normal to the plane
        
    Returns:
        The component of vector lying in the plane
    """
    return reject_multivector(vector, plane_normal)

def vector_angle_with_normal(vector: 'MultiVector', normal: 'MultiVector') -> float:
    """
    Compute the angle between a vector and a normal (for incidence angle calculations).
    
    Args:
        vector: The incident vector
        normal: The surface normal
        
    Returns:
        Angle in radians between vector and normal
    """
    algebra = vector.algebra
    
    # Compute dot product using geometric product
    # For vectors: a · b = (a * b + b * a) / 2
    ab = algebra.gp(vector, normal)
    ba = algebra.gp(normal, vector)
    dot_product_mv = algebra.scalar_multiply(algebra.add(ab, ba), 0.5)
    
    dot_product = float(dot_product_mv.values()[0]) if dot_product_mv.values() else 0.0
    
    # Compute magnitudes
    vector_mag_sq = algebra.gp(vector, algebra.reverse(vector))
    normal_mag_sq = algebra.gp(normal, algebra.reverse(normal))
    
    vector_mag = np.sqrt(float(vector_mag_sq.values()[0])) if vector_mag_sq.values() else 0.0
    normal_mag = np.sqrt(float(normal_mag_sq.values()[0])) if normal_mag_sq.values() else 0.0
    
    if vector_mag < 1e-12 or normal_mag < 1e-12:
        return 0.0
    
    cos_angle = dot_product / (vector_mag * normal_mag)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
    
    return np.arccos(abs(cos_angle))  # Return acute angle

# GPU-optimized kernel support functions

def prepare_projection_data_for_gpu(state_multivectors: list, object_multivectors: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare multivector data for GPU projection operations.
    
    Args:
        state_multivectors: List of state multivectors (rays, waves, etc.)
        object_multivectors: List of object multivectors (surfaces, boundaries, etc.)
        
    Returns:
        Tuple of (state_data, object_data) as numpy arrays ready for GPU transfer
    """
    # Convert multivectors to compact array representation
    # This would need integration with the state multivector GPU format
    
    state_data = []
    object_data = []
    
    for state_mv in state_multivectors:
        # Extract components in canonical order
        state_components = np.zeros(8, dtype=np.float32)  # Assuming 8-component representation
        keys = state_mv.keys()
        values = state_mv.values()
        
        for i, (key, value) in enumerate(zip(keys, values)):
            if i < 8:  # Limit to GPU register size
                state_components[key] = float(value)
        
        state_data.append(state_components)
    
    for obj_mv in object_multivectors:
        # Similar extraction for object multivectors
        obj_components = np.zeros(8, dtype=np.float32)
        keys = obj_mv.keys()
        values = obj_mv.values()
        
        for i, (key, value) in enumerate(zip(keys, values)):
            if i < 8:
                obj_components[key] = float(value)
        
        object_data.append(obj_components)
    
    return np.array(state_data, dtype=np.float32), np.array(object_data, dtype=np.float32)

# Integration hints for AI/developer guidance

GA_PROJECTION_DESIGN_NOTES = """
===========================================================
DESIGN NOTE: Enabling Geometric Algebra Projection Operations
===========================================================

GOAL: Extend the core engine to support general GA projection operations,
allowing interactions between arbitrary state multivectors (representing
rays, fields, or wave states) and surfaces, volumes, or other objects,
each also modeled as a multivector (e.g., planes, spheres, tissue domains).

WHY:
  - In optics, refraction/reflection at a surface is fundamentally a projection
    (and/or rejection) of a multivector (the ray) with respect to another
    (the surface normal or full GA object).
  - In acoustics (ultrasound), the projection of an incident wave on tissue boundaries
    determines transmission, reflection, and mode conversion.
  - In general physics, projection encodes the geometric interaction between any
    two entities represented in GA.

WHAT TO IMPLEMENT:

1. **Multivector Projection Operation**:
   - Implement a general method: `project(a, b)` where `a` and `b` are arbitrary multivectors.
   - This method should compute the projection of `a` onto the subspace defined by `b`.
     (See: Doran & Lasenby, *Geometric Algebra for Physicists*, Sec 3.4)
   - For vectors: project(v, n) = (v ⋅ n̂) n̂, where n̂ is the normalized direction of n.
   - For higher-grade blades: projection uses the left contraction or `a >> b`.

2. **Propagator Integration**:
   - Update all PropagatorTransform subclasses (e.g., OpticalSurfacePropagator,
     UltrasoundTissuePropagator) to:
       a. Represent surfaces/objects as GA multivectors/blades.
       b. Use `project()` or appropriate contraction operations in their propagate() methods
          to determine the geometric decomposition of the incident state (ray, wave, field).
       c. Use the projection and rejection to calculate reflection, transmission, and other
          physically relevant phenomena.

3. **GPU Kernel Enhancement**:
   - For GPU: Implement the projection operation as an atomic register-resident function.
   - Design CUDA/HLSL kernels to apply `project(a, b)` for batches of state/object pairs.
   - Ensure the kernel uses efficient register or shared memory utilization to maximize throughput.

4. **Validation**:
   - Add tests validating that the projection operation recovers known results:
       - Reflection = v - 2 * project(v, n) for optics.
       - Transmission through boundaries in acoustics, using projection to calculate
         transmitted and reflected field components.
   - Compare CPU and GPU outputs for consistency.

5. **Documentation**:
   - Update architecture documents to note that all geometric interactions now
     support full GA projections between arbitrary state and object multivectors.

6. **AI Integration Hints**:
   - When an AI is tasked with extending or modifying the propagator logic,
     it should always check whether the operation can (or should) be implemented
     using the general project() interface rather than ad hoc vector algebra.
   - This enables easy extension to new physics domains (EM, elastodynamics, etc).

References:
  - Doran & Lasenby, "Geometric Algebra for Physicists" (projection: Eq. 3.57+)
  - Section 2, GPU_GA_State_Propagators.md (see domain-specific state extensions)
  - Kingdon GA implementation summary docs, PropagatorTransform base class

Example usage:
  projected_ray = project(ray_multivector, surface_blade)
  reflected_ray = ray_multivector - 2 * project(ray_multivector, surface_normal_blade)

===========================================================
"""

if __name__ == "__main__":
    print("GA Projections Module - Design Framework")
    print("="*50)
    print(GA_PROJECTION_DESIGN_NOTES)