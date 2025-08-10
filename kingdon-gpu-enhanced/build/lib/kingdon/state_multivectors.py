# -*- coding: utf-8 -*-
"""
State Multivector Module for GPU-Accelerated Geometric Algebra
=============================================================

Original Kingdon Library: Martin Roelfs
GPU State Multivector Framework: Copyright © 2025 PhaseSpace. All rights reserved.

This module provides State Multivector classes that extend the base kingdon 
MultiVector framework with specialized physical state encoding and GPU 
optimization capabilities.

State Multivectors are ultra-compact GA structures that holistically encode 
all physical attributes of an entity into a single algebraic object, designed 
for register-resident GPU operations.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .algebra import Algebra
    from .multivector import MultiVector

# Import base classes
from .multivector import MultiVector

# GPU-optimized data structures
@dataclass
class StateMultivectorBase:
    """
    Base class for all State Multivectors.
    
    Provides the foundation for ultra-compact state representation
    optimized for GPU register-resident operations.
    """
    algebra: 'Algebra'
    _pose_motor: Optional['MultiVector'] = None
    _attributes: Optional[np.ndarray] = None
    name: Optional[str] = None
    
    @property 
    def pose_motor(self) -> 'MultiVector':
        """Get the geometric pose motor (position + orientation)."""
        if self._pose_motor is None:
            # Initialize as identity motor
            self._pose_motor = self.algebra.scalar(1.0)
        return self._pose_motor
    
    @pose_motor.setter
    def pose_motor(self, value: 'MultiVector') -> None:
        """Set the geometric pose motor."""
        self._pose_motor = value
    
    @property
    def attributes(self) -> np.ndarray:
        """Get the physical attributes array."""
        if self._attributes is None:
            # Initialize with zeros - subclasses should override size
            self._attributes = np.zeros(4, dtype=np.float32)
        return self._attributes
    
    @attributes.setter  
    def attributes(self, value: Union[np.ndarray, Sequence[float]]) -> None:
        """Set the physical attributes array."""
        self._attributes = np.asarray(value, dtype=np.float32)
    
    def copy(self) -> 'StateMultivectorBase':
        """Create a deep copy of this state multivector."""
        new_state = type(self)(self.algebra, name=self.name)
        if self._pose_motor is not None:
            new_state._pose_motor = self._pose_motor.copy()
        if self._attributes is not None:
            new_state._attributes = self._attributes.copy()
        return new_state
    
    def __repr__(self) -> str:
        """String representation of the state multivector."""
        name_str = f"{self.name} = " if self.name else ""
        return f"{name_str}{type(self).__name__}(pose={self.pose_motor}, attrs={list(self.attributes)})"


@dataclass  
class OpticalState(StateMultivectorBase):
    """
    State Multivector for optical ray simulation.
    
    Encodes complete photon state including geometric pose,
    wave properties, and material interaction data in 96 bytes.
    """
    
    def __post_init__(self):
        """Initialize optical-specific attributes."""
        if self._attributes is None:
            # wavelength, polarization_angle, phase, intensity
            self._attributes = np.array([550e-9, 0.0, 0.0, 1.0], dtype=np.float32)
    
    @property
    def wavelength(self) -> float:
        """Get the wavelength in meters."""
        return float(self.attributes[0])
    
    @wavelength.setter
    def wavelength(self, value: float) -> None:
        """Set the wavelength in meters."""
        self.attributes[0] = value
    
    @property
    def polarization_angle(self) -> float:
        """Get the polarization angle in radians."""
        return float(self.attributes[1])
    
    @polarization_angle.setter
    def polarization_angle(self, value: float) -> None:
        """Set the polarization angle in radians."""
        self.attributes[1] = value
    
    @property
    def phase(self) -> float:
        """Get the optical phase in radians."""
        return float(self.attributes[2])
    
    @phase.setter
    def phase(self, value: float) -> None:
        """Set the optical phase in radians."""
        self.attributes[2] = value
    
    @property
    def intensity(self) -> float:
        """Get the optical intensity."""
        return float(self.attributes[3])
    
    @intensity.setter
    def intensity(self, value: float) -> None:
        """Set the optical intensity."""
        self.attributes[3] = value
    
    @classmethod
    def from_ray_direction(cls, 
                          algebra: 'Algebra',
                          position: Sequence[float],
                          direction: Sequence[float], 
                          wavelength: float = 550e-9,
                          intensity: float = 1.0,
                          name: Optional[str] = None) -> 'OpticalState':
        """
        Create an OpticalState from position and direction vectors.
        
        Args:
            algebra: The geometric algebra instance
            position: Ray starting position [x, y, z]
            direction: Ray direction (will be normalized) [dx, dy, dz]  
            wavelength: Wavelength in meters (default 550nm)
            intensity: Optical intensity (default 1.0)
            name: Optional name for the state
            
        Returns:
            New OpticalState instance
        """
        # Normalize direction
        dir_array = np.array(direction, dtype=float)
        dir_norm = np.linalg.norm(dir_array)
        if dir_norm == 0:
            raise ValueError("Direction vector cannot be zero")
        dir_normalized = dir_array / dir_norm
        
        # Create pose motor from position and direction
        # In PGA, this combines translation and rotation
        state = cls(algebra, name=name)
        
        # Set position using translator
        if hasattr(algebra, 'r') and algebra.r >= 1:
            # PGA: Use translator for position
            translator = algebra.translator(position, 1.0)
            state.pose_motor = translator
        else:
            # Euclidean GA: Encode position in motor somehow
            # For now, use the direction as a rotor
            state.pose_motor = algebra.scalar(1.0)
        
        # Set optical properties
        state.wavelength = wavelength
        state.intensity = intensity
        
        return state


@dataclass
class UltrasoundState(StateMultivectorBase):
    """
    State Multivector for ultrasound wave simulation.
    
    Encodes acoustic wave state including geometric pose,
    wave properties, and tissue interaction data.
    """
    
    def __post_init__(self):
        """Initialize ultrasound-specific attributes.""" 
        if self._attributes is None:
            # frequency, amplitude, acoustic_impedance, attenuation_coeff
            self._attributes = np.array([5e6, 1.0, 1.5e6, 0.5], dtype=np.float32)
    
    @property
    def frequency(self) -> float:
        """Get the acoustic frequency in Hz."""
        return float(self.attributes[0])
    
    @frequency.setter
    def frequency(self, value: float) -> None:
        """Set the acoustic frequency in Hz."""
        self.attributes[0] = value
    
    @property
    def amplitude(self) -> float:
        """Get the acoustic amplitude."""
        return float(self.attributes[1])
    
    @amplitude.setter 
    def amplitude(self, value: float) -> None:
        """Set the acoustic amplitude."""
        self.attributes[1] = value
    
    @property
    def acoustic_impedance(self) -> float:
        """Get the acoustic impedance in Pa⋅s/m."""
        return float(self.attributes[2])
    
    @acoustic_impedance.setter
    def acoustic_impedance(self, value: float) -> None:
        """Set the acoustic impedance in Pa⋅s/m."""
        self.attributes[2] = value
    
    @property  
    def attenuation_coeff(self) -> float:
        """Get the attenuation coefficient in dB/cm/MHz."""
        return float(self.attributes[3])
    
    @attenuation_coeff.setter
    def attenuation_coeff(self, value: float) -> None:
        """Set the attenuation coefficient in dB/cm/MHz."""
        self.attributes[3] = value


@dataclass 
class ElectromagneticState(StateMultivectorBase):
    """
    State Multivector for electromagnetic field simulation.
    
    Encodes EM field state using GA bivector representation
    where F = E + IcB (electromagnetic field bivector).
    """
    
    def __post_init__(self):
        """Initialize EM-specific attributes."""
        if self._attributes is None:
            # E_field_magnitude, B_field_magnitude, energy_density, poynting_magnitude  
            self._attributes = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    
    @property
    def E_field_magnitude(self) -> float:
        """Get the electric field magnitude in V/m."""
        return float(self.attributes[0])
    
    @E_field_magnitude.setter
    def E_field_magnitude(self, value: float) -> None:
        """Set the electric field magnitude in V/m.""" 
        self.attributes[0] = value
    
    @property
    def B_field_magnitude(self) -> float:
        """Get the magnetic field magnitude in T."""
        return float(self.attributes[1])
    
    @B_field_magnitude.setter
    def B_field_magnitude(self, value: float) -> None:
        """Set the magnetic field magnitude in T."""
        self.attributes[1] = value
    
    @property
    def energy_density(self) -> float:
        """Get the electromagnetic energy density in J/m³."""
        return float(self.attributes[2])
    
    @energy_density.setter
    def energy_density(self, value: float) -> None:
        """Set the electromagnetic energy density in J/m³."""
        self.attributes[2] = value
    
    @property
    def poynting_magnitude(self) -> float:
        """Get the Poynting vector magnitude in W/m²."""
        return float(self.attributes[3])
    
    @poynting_magnitude.setter
    def poynting_magnitude(self, value: float) -> None:
        """Set the Poynting vector magnitude in W/m²."""
        self.attributes[3] = value


# Factory functions for creating state multivectors
def create_optical_ray(algebra: 'Algebra',
                      position: Sequence[float] = (0, 0, 0),
                      direction: Sequence[float] = (0, 0, 1), 
                      wavelength: float = 550e-9,
                      intensity: float = 1.0,
                      name: Optional[str] = None) -> OpticalState:
    """
    Factory function to create an optical ray state.
    
    Args:
        algebra: Geometric algebra instance
        position: Ray starting position [x, y, z] in meters
        direction: Ray direction [dx, dy, dz] (will be normalized)
        wavelength: Wavelength in meters (default 550nm green light)
        intensity: Optical intensity (default 1.0)
        name: Optional name for the ray
        
    Returns:
        OpticalState representing the ray
        
    Example:
        >>> from kingdon.algebra import Algebra
        >>> from kingdon.state_multivectors import create_optical_ray
        >>> alg = Algebra(p=3, q=0, r=1)  # 3D PGA for optics
        >>> ray = create_optical_ray(alg, position=[0, 0, 0], direction=[1, 0, 0], 
        ...                         wavelength=632.8e-9, name="HeNe_ray")
    """
    return OpticalState.from_ray_direction(algebra, position, direction, 
                                          wavelength, intensity, name)


def create_ultrasound_wave(algebra: 'Algebra',
                          position: Sequence[float] = (0, 0, 0),
                          direction: Sequence[float] = (0, 0, 1),
                          frequency: float = 5e6,
                          amplitude: float = 1.0,
                          name: Optional[str] = None) -> UltrasoundState:
    """
    Factory function to create an ultrasound wave state.
    
    Args:
        algebra: Geometric algebra instance  
        position: Wave starting position [x, y, z] in meters
        direction: Wave propagation direction [dx, dy, dz]
        frequency: Acoustic frequency in Hz (default 5MHz)
        amplitude: Wave amplitude (default 1.0)
        name: Optional name for the wave
        
    Returns:
        UltrasoundState representing the acoustic wave
    """
    state = UltrasoundState(algebra, name=name)
    
    # Set wave properties
    state.frequency = frequency
    state.amplitude = amplitude
    
    # Create pose motor for position and direction
    # This is simplified - real implementation would encode direction properly
    if hasattr(algebra, 'r') and algebra.r >= 1:
        translator = algebra.translator(position, 1.0)
        state.pose_motor = translator
    else:
        state.pose_motor = algebra.scalar(1.0)
    
    return state


def create_em_field(algebra: 'Algebra',
                   position: Sequence[float] = (0, 0, 0),
                   E_field: float = 1.0,
                   B_field: float = 1.0,
                   name: Optional[str] = None) -> ElectromagneticState:
    """
    Factory function to create an electromagnetic field state.
    
    Args:
        algebra: Geometric algebra instance
        position: Field sampling position [x, y, z] in meters  
        E_field: Electric field magnitude in V/m
        B_field: Magnetic field magnitude in T
        name: Optional name for the field
        
    Returns:
        ElectromagneticState representing the EM field
    """
    state = ElectromagneticState(algebra, name=name)
    
    # Set field properties  
    state.E_field_magnitude = E_field
    state.B_field_magnitude = B_field
    
    # Create pose motor for position
    if hasattr(algebra, 'r') and algebra.r >= 1:
        translator = algebra.translator(position, 1.0)
        state.pose_motor = translator
    else:
        state.pose_motor = algebra.scalar(1.0)
    
    return state


# Utility functions for state manipulation
def extract_position(state: StateMultivectorBase) -> np.ndarray:
    """Extract position from a state multivector's pose motor."""
    # This is a simplified extraction - real implementation would
    # properly decode the position from the motor/translator
    return np.array([0.0, 0.0, 0.0], dtype=np.float32)


def extract_direction(state: StateMultivectorBase) -> np.ndarray:
    """Extract direction from a state multivector's pose motor.""" 
    # This is a simplified extraction - real implementation would
    # properly decode the direction from the motor/rotor
    return np.array([0.0, 0.0, 1.0], dtype=np.float32)


def state_to_gpu_format(state: StateMultivectorBase) -> Dict[str, np.ndarray]:
    """
    Convert a state multivector to GPU-friendly format.
    
    Returns a dictionary with numpy arrays suitable for GPU transfer.
    """
    # Extract motor components (simplified)
    motor = state.pose_motor
    motor_data = np.zeros(8, dtype=np.float32)  # 8 components for motor
    
    if hasattr(motor, '_values') and hasattr(motor, '_keys'):
        for i, (key, value) in enumerate(zip(motor._keys, motor._values)):
            if i < 8:  # Limit to 8 components
                motor_data[i] = float(value)
    
    return {
        'pose_motor': motor_data,
        'attributes': state.attributes.astype(np.float32),
        'name': state.name or ""
    }


def gpu_format_to_state(algebra: 'Algebra', 
                       gpu_data: Dict[str, np.ndarray],
                       state_type: str = 'optical') -> StateMultivectorBase:
    """
    Convert GPU format data back to a state multivector.
    
    Args:
        algebra: Geometric algebra instance
        gpu_data: Dictionary with 'pose_motor' and 'attributes' arrays
        state_type: Type of state to create ('optical', 'ultrasound', 'em')
        
    Returns:
        Reconstructed state multivector
    """
    # Create appropriate state type
    if state_type == 'optical':
        state = OpticalState(algebra)
    elif state_type == 'ultrasound':
        state = UltrasoundState(algebra)
    elif state_type == 'em':
        state = ElectromagneticState(algebra) 
    else:
        raise ValueError(f"Unknown state type: {state_type}")
    
    # Set attributes
    state.attributes = gpu_data['attributes']
    
    # Reconstruct motor (simplified)
    # Real implementation would properly reconstruct from motor components
    state.pose_motor = algebra.scalar(1.0)
    
    return state