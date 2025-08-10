# -*- coding: utf-8 -*-
"""
Propagator Transforms Module for GPU-Accelerated Geometric Algebra
================================================================

Original Kingdon Library: Martin Roelfs
GPU Propagator Transform Framework: Copyright © 2025 PhaseSpace. All rights reserved.

This module implements atomic Propagator Transforms that encapsulate complete 
physical laws as register-resident operations on State Multivectors.

Each Propagator Transform takes a State Multivector as input and returns a new 
State Multivector representing the state after a physical interaction.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING
from dataclasses import dataclass
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .algebra import Algebra
    from .multivector import MultiVector
    from .state_multivectors import StateMultivectorBase, OpticalState, UltrasoundState, ElectromagneticState

# Import the state multivector classes and GA projection operations
from .state_multivectors import StateMultivectorBase, OpticalState, UltrasoundState, ElectromagneticState
from .ga_projections import (
    project_multivector, reject_multivector, reflect_multivector, 
    decompose_multivector, vector_angle_with_normal
)

# Constants
SPEED_OF_LIGHT = 299792458.0  # m/s
EPSILON_0 = 8.854187817e-12   # F/m
MU_0 = 4 * math.pi * 1e-7     # H/m


@dataclass
class SurfaceGeometry:
    """Represents an optical surface geometry."""
    position: np.ndarray
    normal: np.ndarray
    radius_of_curvature: float
    conic_constant: float
    aspheric_coeffs: np.ndarray


@dataclass  
class MaterialProperties:
    """Represents material optical properties."""
    refractive_index: float
    absorption_coeff: float
    scatter_coeff: float
    dispersion_coeff: float = 0.0


@dataclass
class TissueProperties:
    """Represents ultrasound tissue properties.""" 
    density: float
    speed_of_sound: float
    absorption_coeff: float
    scatter_coeff: float
    acoustic_impedance: float


class PropagatorTransform(ABC):
    """
    Abstract base class for all Propagator Transforms.
    
    A Propagator Transform is an atomic operation that implements
    a complete physical law as a function: Ψ(t) → Ψ(t + δt)
    """
    
    @abstractmethod
    def propagate(self, state: StateMultivectorBase, *args, **kwargs) -> StateMultivectorBase:
        """
        Apply the propagator transform to a state multivector.
        
        Args:
            state: Input state multivector
            *args, **kwargs: Transform-specific parameters
            
        Returns:
            New state multivector after transformation
        """
        pass


class OpticalSurfacePropagator(PropagatorTransform):
    """
    Propagator for optical surface interactions using GA rotors.
    
    Implements Snell's law and Fresnel coefficients as a single GA operation.
    """
    
    def __init__(self, surface: SurfaceGeometry, material: MaterialProperties):
        """
        Initialize with surface geometry and material properties.
        
        Args:
            surface: Surface geometry parameters
            material: Material optical properties
        """
        self.surface = surface
        self.material = material
    
    def propagate(self, ray: OpticalState, **kwargs) -> OpticalState:
        """
        Propagate optical ray through surface interaction.
        
        This implements the GA-based Snell's law using rotors as described
        in the GPU_GA_State_Propagators.md document.
        """
        # Extract current ray direction from pose motor (simplified)
        # Real implementation would decode this from the motor properly
        incident_direction = np.array([0, 0, 1], dtype=float)  # Placeholder
        
        # Surface normal  
        surface_normal = self.surface.normal / np.linalg.norm(self.surface.normal)
        
        # Refractive indices
        n1 = 1.0  # Assume air initially - would get from ray history
        n2 = self.material.refractive_index
        
        # Calculate angle of incidence
        cos_theta_i = np.abs(np.dot(incident_direction, surface_normal))
        sin_theta_i = math.sqrt(1 - cos_theta_i**2)
        
        # Check for total internal reflection
        if n1 > n2 and sin_theta_i > n2/n1:
            # Total internal reflection - flip direction
            reflected_ray = ray.copy()
            # Would implement proper GA reflection here
            reflected_ray.intensity *= 0.95  # Small loss
            return reflected_ray
        
        # Snell's law: n1*sin(θ1) = n2*sin(θ2)
        sin_theta_t = (n1 / n2) * sin_theta_i
        cos_theta_t = math.sqrt(1 - sin_theta_t**2)
        
        # GA Rotor implementation of Snell's law
        # This is the "single rotor operation" that replaces complex vector math
        refracted_ray = ray.copy()
        
        # Calculate refraction using GA rotor (conceptual)
        # Real implementation would use: 
        # rotor_angle = math.asin(sin_theta_t) - math.asin(sin_theta_i)
        # snell_rotor = exp(-0.5 * rotor_angle * bivector_between_directions)
        # new_direction = snell_rotor * incident_direction * reverse(snell_rotor)
        
        # For now, use simplified calculation
        refracted_ray.intensity *= self._calculate_transmission_coefficient(cos_theta_i, cos_theta_t, n1, n2)
        
        # Update wavelength due to material dispersion
        if self.material.dispersion_coeff != 0:
            # Simple linear dispersion model
            new_wavelength = ray.wavelength * (1 + self.material.dispersion_coeff * (n2 - n1))
            refracted_ray.wavelength = new_wavelength
        
        # Apply absorption
        if self.material.absorption_coeff > 0:
            absorption_loss = math.exp(-self.material.absorption_coeff * 0.001)  # Assume 1mm thickness
            refracted_ray.intensity *= absorption_loss
        
        return refracted_ray
    
    def _calculate_transmission_coefficient(self, cos_i: float, cos_t: float, n1: float, n2: float) -> float:
        """Calculate Fresnel transmission coefficient for s-polarized light."""
        rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
        ts = 1 - rs**2
        return ts


class MaxwellPropagator(PropagatorTransform):
    """
    Propagator for electromagnetic field evolution using GA.
    
    Implements Maxwell's equations in GA form: ∇F = J
    where F is the electromagnetic field bivector.
    """
    
    def propagate(self, em_state: ElectromagneticState, dt: float, 
                 current_density: Optional[np.ndarray] = None) -> ElectromagneticState:
        """
        Evolve electromagnetic field using Maxwell's equations.
        
        Args:
            em_state: Input electromagnetic state
            dt: Time step
            current_density: Optional current density J
            
        Returns:
            Evolved electromagnetic state
        """
        new_state = em_state.copy()
        
        # Simplified Maxwell evolution (placeholder)
        # Real implementation would use GA geometric derivative:
        # del_F = geometric_derivative(F, spacetime)
        # F_new = F + dt * (current_density - del_F)
        
        # For demonstration, implement simple wave propagation
        c = SPEED_OF_LIGHT
        
        # Energy conservation (simplified)
        total_energy = 0.5 * EPSILON_0 * em_state.E_field_magnitude**2 + \
                      0.5 * em_state.B_field_magnitude**2 / MU_0
        
        # Update fields (simplified wave evolution)
        if current_density is not None:
            # Add current source term
            j_magnitude = np.linalg.norm(current_density)
            new_state.E_field_magnitude += dt * j_magnitude / EPSILON_0
        
        # Update energy density
        new_state.energy_density = total_energy
        
        # Update Poynting vector magnitude
        new_state.poynting_magnitude = (new_state.E_field_magnitude * 
                                       new_state.B_field_magnitude) / MU_0
        
        return new_state


class UltrasoundTissuePropagator(PropagatorTransform):
    """
    Propagator for ultrasound wave interaction with tissue.
    
    Implements the acoustic wave equation with attenuation and scattering.
    """
    
    def __init__(self, tissue: TissueProperties):
        """Initialize with tissue properties."""
        self.tissue = tissue
    
    def propagate(self, wave: UltrasoundState, dt: float, 
                 distance: float = 0.001) -> UltrasoundState:
        """
        Propagate ultrasound wave through tissue.
        
        Args:
            wave: Input ultrasound state
            dt: Time step  
            distance: Distance traveled in tissue (meters)
            
        Returns:
            Wave state after tissue interaction
        """
        new_wave = wave.copy()
        
        # Calculate attenuation
        # Attenuation in dB: A = α * f * d, where α is in dB/cm/MHz
        freq_MHz = wave.frequency / 1e6
        distance_cm = distance * 100
        attenuation_dB = self.tissue.absorption_coeff * freq_MHz * distance_cm
        
        # Convert dB to linear scale
        attenuation_linear = 10**(attenuation_dB / 20)
        new_wave.amplitude /= attenuation_linear
        
        # Apply acoustic impedance effects at boundaries
        # Z = ρ * c (acoustic impedance)
        impedance_ratio = new_wave.acoustic_impedance / self.tissue.acoustic_impedance
        
        # Simplified reflection coefficient
        reflection_coeff = (impedance_ratio - 1) / (impedance_ratio + 1)
        transmission_coeff = 1 - abs(reflection_coeff)**2
        
        new_wave.amplitude *= transmission_coeff
        
        # Update acoustic impedance to tissue value
        new_wave.acoustic_impedance = self.tissue.acoustic_impedance
        
        # Apply scattering effects (simplified Rayleigh scattering)
        if self.tissue.scatter_coeff > 0:
            wavelength = self.tissue.speed_of_sound / wave.frequency
            if distance < wavelength / 10:  # Rayleigh regime
                scatter_loss = 1 - self.tissue.scatter_coeff * (distance / wavelength)**4
                new_wave.amplitude *= max(0.1, scatter_loss)  # Don't go below 10%
        
        return new_wave


class FreeSpacePropagator(PropagatorTransform):
    """
    Propagator for free space propagation (no medium interactions).
    
    Updates geometric position based on direction and time.
    """
    
    def propagate(self, state: StateMultivectorBase, dt: float, 
                 speed: Optional[float] = None) -> StateMultivectorBase:
        """
        Propagate state through free space.
        
        Args:
            state: Input state multivector
            dt: Time step
            speed: Propagation speed (defaults based on state type)
            
        Returns:
            State after free space propagation
        """
        new_state = state.copy()
        
        # Determine propagation speed
        if speed is None:
            if isinstance(state, OpticalState):
                speed = SPEED_OF_LIGHT
            elif isinstance(state, UltrasoundState):
                speed = 1540.0  # Speed of sound in tissue (m/s)
            elif isinstance(state, ElectromagneticState):
                speed = SPEED_OF_LIGHT
            else:
                speed = SPEED_OF_LIGHT  # Default
        
        # Calculate distance traveled
        distance = speed * dt
        
        # Update position using translator in GA
        # Real implementation would properly update the pose motor
        # For now, this is a placeholder that keeps the motor unchanged
        
        return new_state


class OpticalSystemPropagator(PropagatorTransform):
    """
    Composite propagator for complete optical systems.
    
    Propagates rays through multiple surfaces and media in sequence.
    """
    
    def __init__(self, surfaces: List[SurfaceGeometry], 
                 materials: List[MaterialProperties]):
        """
        Initialize with system configuration.
        
        Args:
            surfaces: List of surface geometries
            materials: List of material properties (one per surface)
        """
        if len(surfaces) != len(materials):
            raise ValueError("Number of surfaces must match number of materials")
        
        self.surfaces = surfaces
        self.materials = materials
        self.surface_propagators = [
            OpticalSurfacePropagator(surf, mat) 
            for surf, mat in zip(surfaces, materials)
        ]
    
    def propagate(self, ray: OpticalState) -> OpticalState:
        """
        Propagate ray through entire optical system.
        
        This is the "non-static method" version as suggested in the document review.
        """
        current_ray = ray.copy()
        
        # Propagate through each surface in sequence
        for i, propagator in enumerate(self.surface_propagators):
            # Free space propagation to surface (simplified)
            # In real implementation, would calculate intersection distance
            current_ray = FreeSpacePropagator().propagate(current_ray, dt=1e-12)  # Very small time step
            
            # Surface interaction
            current_ray = propagator.propagate(current_ray)
            
            # Check if ray was absorbed or lost
            if current_ray.intensity < 1e-6:  # Intensity threshold
                break
        
        return current_ray


# Specialized propagators for specific applications

class NonlinearOpticalPropagator(PropagatorTransform):
    """Propagator for nonlinear optical effects."""
    
    def __init__(self, chi3: float = 1e-22):
        """Initialize with third-order susceptibility χ³."""
        self.chi3 = chi3
    
    def propagate(self, ray: OpticalState, dt: float, 
                 distance: float = 0.001) -> OpticalState:
        """Apply nonlinear optical effects."""
        new_ray = ray.copy()
        
        # Kerr effect: refractive index depends on intensity
        # n = n0 + n2 * I, where n2 = (3/8n0²) * χ³
        n0 = 1.5  # Linear refractive index
        n2 = (3 * self.chi3) / (8 * n0**2)
        
        # Intensity-dependent phase shift
        phase_shift = 2 * math.pi * n2 * ray.intensity * distance / ray.wavelength
        new_ray.phase += phase_shift
        
        return new_ray


class QuantumOpticalPropagator(PropagatorTransform):
    """Propagator for quantum optical effects."""
    
    def propagate(self, state: StateMultivectorBase, dt: float) -> StateMultivectorBase:
        """Apply quantum optical evolution (placeholder)."""
        # This would implement Schrödinger equation evolution
        # For now, just return unchanged state
        return state.copy()


# Utility functions for propagator composition

def sequential_propagate(propagators: List[PropagatorTransform], 
                        state: StateMultivectorBase,
                        *args, **kwargs) -> StateMultivectorBase:
    """
    Apply a sequence of propagators to a state.
    
    Args:
        propagators: List of propagator transforms
        state: Initial state multivector
        *args, **kwargs: Arguments passed to each propagator
        
    Returns:
        Final state after all propagations
    """
    current_state = state
    for propagator in propagators:
        current_state = propagator.propagate(current_state, *args, **kwargs)
    
    return current_state


def parallel_propagate(propagators: List[PropagatorTransform],
                      state: StateMultivectorBase,
                      combiner: Callable[[List[StateMultivectorBase]], StateMultivectorBase],
                      *args, **kwargs) -> StateMultivectorBase:
    """
    Apply propagators in parallel and combine results.
    
    Args:
        propagators: List of propagator transforms
        state: Initial state multivector
        combiner: Function to combine parallel results
        *args, **kwargs: Arguments passed to each propagator
        
    Returns:
        Combined state from parallel propagations
    """
    results = []
    for propagator in propagators:
        result = propagator.propagate(state.copy(), *args, **kwargs)
        results.append(result)
    
    return combiner(results)


def create_optical_lens_system(radii: List[float],
                              thicknesses: List[float], 
                              materials: List[float],
                              algebra: 'Algebra') -> OpticalSystemPropagator:
    """
    Factory function to create an optical lens system.
    
    Args:
        radii: List of surface radii (positive for convex)
        thicknesses: List of thicknesses between surfaces
        materials: List of refractive indices
        algebra: Geometric algebra instance
        
    Returns:
        OpticalSystemPropagator for the lens system
    """
    surfaces = []
    material_props = []
    
    z_position = 0.0
    for i, (radius, thickness, n) in enumerate(zip(radii, thicknesses, materials)):
        # Create surface geometry
        surface = SurfaceGeometry(
            position=np.array([0, 0, z_position]),
            normal=np.array([0, 0, 1]),  # All surfaces face +z
            radius_of_curvature=radius,
            conic_constant=0.0,  # Spherical surfaces
            aspheric_coeffs=np.zeros(4)
        )
        surfaces.append(surface)
        
        # Create material properties
        material = MaterialProperties(
            refractive_index=n,
            absorption_coeff=0.001,  # Low absorption
            scatter_coeff=0.0001,    # Low scattering
            dispersion_coeff=0.0     # No dispersion
        )
        material_props.append(material)
        
        z_position += thickness
    
    return OpticalSystemPropagator(surfaces, material_props)