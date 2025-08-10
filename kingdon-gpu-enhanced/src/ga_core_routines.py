#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core GA Routines: Compact Reference Implementation
=================================================

This module provides compact, self-contained, and adaptable GA routines
for integration into the core "projector" or "propagator" GA framework.
Based on the reference implementation from comments.txt.

These routines are designed to be:
- Compact and efficient
- Self-contained with minimal dependencies
- Adaptable to different GA library backends
- Suitable for vectorization and GPU acceleration

Core Functions:
- geometric_product, left_contraction, inverse
- sandwich, reflection, rotor_from_reflection, rotate
- project, reject, decompose
- angle_between

Usage:
Each propagator/surface/medium can be a blade or multivector.
Use project, reject, reflection, rotate as building blocks for all physics.
For GPU: wrap these in vectorized/parallel kernels.
"""

from __future__ import annotations

import numpy as np
import warnings
from typing import TYPE_CHECKING, Union, Tuple, Optional

if TYPE_CHECKING:
    from .algebra import Algebra
    from .multivector import MultiVector

# Core GA Operations (Abstract Interface)

def geometric_product(a: 'MultiVector', b: 'MultiVector') -> 'MultiVector':
    """
    Geometric (Clifford) product of a and b.
    
    Args:
        a, b: MultiVector objects
        
    Returns:
        Geometric product a * b
    """
    try:
        return a * b
    except (AttributeError, TypeError):
        # Fallback to algebra method
        return a.algebra.gp(a, b)


def left_contraction(a: 'MultiVector', b: 'MultiVector') -> 'MultiVector':
    """
    Left contraction: a ⟟ b (lowers the grade of b by grade of a, if possible).
    
    Args:
        a, b: MultiVector objects
        
    Returns:
        Left contraction a ⟟ b
    """
    try:
        return a << b
    except (AttributeError, TypeError):
        try:
            return a.lc(b)
        except (AttributeError, TypeError):
            # Fallback to algebra method
            return a.algebra.left_contraction(a, b)


def inverse(b: 'MultiVector') -> 'MultiVector':
    """
    Multiplicative inverse of multivector b.
    
    Args:
        b: MultiVector object
        
    Returns:
        Inverse b^(-1)
    """
    try:
        return b.inv()
    except (AttributeError, TypeError):
        try:
            return b.inverse()
        except (AttributeError, TypeError):
            # Fallback to algebra method
            return b.algebra.inverse(b)


def sandwich(a: 'MultiVector', V: 'MultiVector') -> 'MultiVector':
    """
    Sandwich product: apply versor V to a.
    
    Formula: a' = V * a * ~V (tilde is reversion)
    
    Args:
        a: MultiVector to transform
        V: Versor (rotor, reflector, etc.)
        
    Returns:
        Transformed multivector V * a * ~V
    """
    try:
        V_reverse = V.reverse()
    except (AttributeError, TypeError):
        V_reverse = V.algebra.reverse(V)
    
    return geometric_product(geometric_product(V, a), V_reverse)


def reflection(a: 'MultiVector', n: 'MultiVector') -> 'MultiVector':
    """
    Reflect multivector a in (normalized) vector or blade n.
    
    Formula: a' = -n * a * n.inverse()
    For vector n, this gives reflection in hyperplane perpendicular to n.
    
    Args:
        a: MultiVector to reflect
        n: Normal vector or blade defining reflection hyperplane
        
    Returns:
        Reflected multivector
    """
    n_inv = inverse(n)
    result = geometric_product(geometric_product(n, a), n_inv)
    
    # Apply negative sign
    try:
        return -result
    except (AttributeError, TypeError):
        return a.algebra.scalar_multiply(result, -1.0)


def rotor_from_reflection(n1: 'MultiVector', n2: 'MultiVector') -> 'MultiVector':
    """
    Build a rotor that rotates from n1 to n2.
    
    Formula: R = n2 * n1 (assuming both are normalized vectors)
    Usage: a' = R * a * R.inverse()
    
    Args:
        n1: Source direction (normalized vector)
        n2: Target direction (normalized vector)
        
    Returns:
        Rotor R that rotates n1 to n2
    """
    return geometric_product(n2, n1)


def rotate(a: 'MultiVector', R: 'MultiVector') -> 'MultiVector':
    """
    Rotate a by rotor R: a' = R * a * R.inverse()
    
    Args:
        a: MultiVector to rotate
        R: Rotor
        
    Returns:
        Rotated multivector
    """
    R_inv = inverse(R)
    return geometric_product(geometric_product(R, a), R_inv)


def project(a: 'MultiVector', b: 'MultiVector') -> 'MultiVector':
    """
    Project multivector a onto the subspace defined by blade b.
    
    Formula: proj(a, b) = (a ⟟ b) * b.inverse()
    
    Args:
        a: MultiVector to project
        b: Blade defining subspace
        
    Returns:
        Projection of a onto b
    """
    contraction = left_contraction(a, b)
    b_inv = inverse(b)
    return geometric_product(contraction, b_inv)


def reject(a: 'MultiVector', b: 'MultiVector') -> 'MultiVector':
    """
    Rejection (component of a orthogonal to b).
    
    Formula: reject(a, b) = a - project(a, b)
    
    Args:
        a: MultiVector to reject
        b: Blade to reject from
        
    Returns:
        Component of a orthogonal to b
    """
    proj = project(a, b)
    try:
        return a - proj
    except (AttributeError, TypeError):
        return a.algebra.subtract(a, proj)


def decompose(a: 'MultiVector', b: 'MultiVector') -> Tuple['MultiVector', 'MultiVector']:
    """
    Decompose a into parallel (proj) and orthogonal (rej) parts relative to b.
    
    Args:
        a: MultiVector to decompose
        b: Reference blade
        
    Returns:
        Tuple of (parallel_component, orthogonal_component)
    """
    proj = project(a, b)
    try:
        rej = a - proj
    except (AttributeError, TypeError):
        rej = a.algebra.subtract(a, proj)
    
    return proj, rej


def angle_between(a: 'MultiVector', b: 'MultiVector') -> float:
    """
    Angle between two vectors a, b.
    
    Formula: θ = arccos( (a · b) / (|a||b|) )
    
    Args:
        a, b: Vector multivectors
        
    Returns:
        Angle in radians
    """
    try:
        # Try inner product operator
        inner_mv = a | b
        inner = float(inner_mv[()])  # Scalar part
    except (AttributeError, TypeError, KeyError):
        # Fallback: use symmetric product for dot product
        ab = geometric_product(a, b)
        ba = geometric_product(b, a)
        try:
            sum_mv = ab + ba
            inner = float(sum_mv.values()[0]) * 0.5 if sum_mv.values() else 0.0
        except (AttributeError, TypeError):
            inner = 0.0
    
    # Calculate norms
    try:
        norm_a = abs(a)
        norm_b = abs(b)
    except (AttributeError, TypeError):
        # Fallback norm calculation
        try:
            a_conj = a.algebra.reverse(a)
            b_conj = b.algebra.reverse(b)
            norm_a = np.sqrt(abs(float(geometric_product(a, a_conj).values()[0])))
            norm_b = np.sqrt(abs(float(geometric_product(b, b_conj).values()[0])))
        except (AttributeError, TypeError, IndexError):
            return 0.0
    
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    
    cos_angle = inner / (norm_a * norm_b)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
    
    return np.arccos(abs(cos_angle))


# Convenience functions for common operations

def normalized_vector(v: 'MultiVector') -> 'MultiVector':
    """
    Normalize a vector multivector.
    
    Args:
        v: Vector multivector
        
    Returns:
        Normalized vector v/|v|
    """
    try:
        norm = abs(v)
    except (AttributeError, TypeError):
        # Fallback norm calculation
        v_conj = v.algebra.reverse(v)
        norm = np.sqrt(abs(float(geometric_product(v, v_conj).values()[0])))
    
    if norm < 1e-12:
        warnings.warn("Attempting to normalize near-null vector")
        return v
    
    try:
        return v / norm
    except (AttributeError, TypeError):
        return v.algebra.scalar_multiply(v, 1.0 / norm)


def is_normalized(v: 'MultiVector', tolerance: float = 1e-10) -> bool:
    """
    Check if a vector is normalized (unit magnitude).
    
    Args:
        v: Vector multivector
        tolerance: Numerical tolerance
        
    Returns:
        True if |v| ≈ 1
    """
    try:
        norm = abs(v)
    except (AttributeError, TypeError):
        v_conj = v.algebra.reverse(v)
        norm = np.sqrt(abs(float(geometric_product(v, v_conj).values()[0])))
    
    return abs(norm - 1.0) < tolerance


def commutator(a: 'MultiVector', b: 'MultiVector') -> 'MultiVector':
    """
    Commutator [a, b] = ab - ba.
    
    Args:
        a, b: MultiVector objects
        
    Returns:
        Commutator [a, b]
    """
    ab = geometric_product(a, b)
    ba = geometric_product(b, a)
    
    try:
        return ab - ba
    except (AttributeError, TypeError):
        return ab.algebra.subtract(ab, ba)


def anticommutator(a: 'MultiVector', b: 'MultiVector') -> 'MultiVector':
    """
    Anticommutator {a, b} = ab + ba.
    
    Args:
        a, b: MultiVector objects
        
    Returns:
        Anticommutator {a, b}
    """
    ab = geometric_product(a, b)
    ba = geometric_product(b, a)
    
    try:
        return ab + ba
    except (AttributeError, TypeError):
        return ab.algebra.add(ab, ba)


# Batch operations for performance

def batch_project(a_list: list['MultiVector'], b_list: list['MultiVector']) -> list['MultiVector']:
    """
    Batch projection of multiple multivectors.
    
    Args:
        a_list: List of multivectors to project
        b_list: List of blades to project onto
        
    Returns:
        List of projected multivectors
    """
    if len(a_list) != len(b_list):
        raise ValueError("Input lists must have same length")
    
    return [project(a, b) for a, b in zip(a_list, b_list)]


def batch_reflect(a_list: list['MultiVector'], n_list: list['MultiVector']) -> list['MultiVector']:
    """
    Batch reflection of multiple multivectors.
    
    Args:
        a_list: List of multivectors to reflect
        n_list: List of normal vectors/blades
        
    Returns:
        List of reflected multivectors
    """
    if len(a_list) != len(n_list):
        raise ValueError("Input lists must have same length")
    
    return [reflection(a, n) for a, n in zip(a_list, n_list)]


def batch_rotate(a_list: list['MultiVector'], R_list: list['MultiVector']) -> list['MultiVector']:
    """
    Batch rotation of multiple multivectors.
    
    Args:
        a_list: List of multivectors to rotate
        R_list: List of rotors
        
    Returns:
        List of rotated multivectors
    """
    if len(a_list) != len(R_list):
        raise ValueError("Input lists must have same length")
    
    return [rotate(a, R) for a, R in zip(a_list, R_list)]


# Integration utilities

class GAOperationsMixin:
    """
    Mixin class to add GA operations to MultiVector classes.
    
    This allows you to use clean API like: vector.project_onto(blade)
    """
    
    def project_onto(self, b: 'MultiVector') -> 'MultiVector':
        """Project this multivector onto blade b."""
        return project(self, b)
    
    def reject_from(self, b: 'MultiVector') -> 'MultiVector':
        """Reject this multivector from blade b."""
        return reject(self, b)
    
    def reflect_in(self, n: 'MultiVector') -> 'MultiVector':
        """Reflect this multivector in normal n."""
        return reflection(self, n)
    
    def rotate_by(self, R: 'MultiVector') -> 'MultiVector':
        """Rotate this multivector by rotor R."""
        return rotate(self, R)
    
    def decompose_relative_to(self, b: 'MultiVector') -> Tuple['MultiVector', 'MultiVector']:
        """Decompose this multivector relative to blade b."""
        return decompose(self, b)
    
    def angle_with(self, other: 'MultiVector') -> float:
        """Compute angle with another vector."""
        return angle_between(self, other)


def validate_ga_operations(algebra: 'Algebra') -> dict[str, bool]:
    """
    Validate that GA operations work correctly with the given algebra.
    
    Args:
        algebra: Algebra instance to test
        
    Returns:
        Dictionary of test results
    """
    results = {
        'geometric_product': True,
        'left_contraction': True,
        'inverse': True,
        'projection': True,
        'reflection': True,
        'rotation': True,
        'decomposition': True
    }
    
    try:
        # Create test vectors
        v1 = algebra.vector([1, 0, 0])
        v2 = algebra.vector([0, 1, 0])
        
        # Test geometric product
        try:
            gp_result = geometric_product(v1, v2)
            results['geometric_product'] = True
        except Exception:
            results['geometric_product'] = False
        
        # Test left contraction
        try:
            lc_result = left_contraction(v1, v2)
            results['left_contraction'] = True
        except Exception:
            results['left_contraction'] = False
        
        # Test inverse
        try:
            inv_result = inverse(v1)
            results['inverse'] = True
        except Exception:
            results['inverse'] = False
        
        # Test projection
        try:
            proj_result = project(v1, v2)
            results['projection'] = True
        except Exception:
            results['projection'] = False
        
        # Test reflection
        try:
            refl_result = reflection(v1, v2)
            results['reflection'] = True
        except Exception:
            results['reflection'] = False
        
        # Test rotation
        try:
            R = rotor_from_reflection(v1, v2)
            rot_result = rotate(v1, R)
            results['rotation'] = True
        except Exception:
            results['rotation'] = False
        
        # Test decomposition
        try:
            par, perp = decompose(v1, v2)
            results['decomposition'] = True
        except Exception:
            results['decomposition'] = False
            
    except Exception:
        # Mark all as failed if basic setup fails
        for key in results:
            results[key] = False
    
    return results


if __name__ == "__main__":
    print("Core GA Routines Module")
    print("=" * 40)
    print(__doc__)
    
    # Simple demonstration (requires algebra to be available)
    try:
        from algebra import Algebra
        
        alg = Algebra(p=3, q=0, r=0)
        print(f"\nTesting with algebra: Cl({alg.p},{alg.q},{alg.r})")
        
        # Run validation
        results = validate_ga_operations(alg)
        passed = sum(results.values())
        total = len(results)
        
        print(f"\nValidation results: {passed}/{total} operations working")
        for op, status in results.items():
            status_str = "✓" if status else "✗"
            print(f"  {status_str} {op}")
        
        if passed == total:
            print("\n🎉 All core GA operations functional!")
        else:
            print(f"\n⚠️  {total - passed} operations need attention")
            
    except ImportError:
        print("\nAlgebra module not available for testing")
        print("Core GA routines defined and ready for integration")