#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for Kingdon GPU-Accelerated Geometric Algebra Library
================================================================

Enhanced version with State Multivectors and Propagator Transforms
for GPU-accelerated physics simulation.
"""

from setuptools import setup, find_packages
import os

# Read version from __init__.py
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'src', '__init__.py')
    with open(version_file, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"\'')
    return '1.3.1'

# Read long description from README
def get_long_description():
    readme_file = os.path.join(os.path.dirname(__file__), 'INSTALLATION_README.md')
    if os.path.exists(readme_file):
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return "GPU-Accelerated Geometric Algebra Library with State Multivectors"

setup(
    name="kingdon",
    version=get_version(),
    author="Martin Roelfs (Original Kingdon), PhaseSpace (GPU Enhancements)",
    author_email="contact@phasespace.ai",
    description="GPU-Accelerated Geometric Algebra Library with State Multivectors and Propagator Transforms",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/tracyphasespace/kingdon",
    packages=['kingdon'],
    package_dir={'kingdon': 'src'},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Hardware :: Symmetric Multi-processing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "sympy>=1.8.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
        ],
        "visualization": [
            "matplotlib>=3.0.0",
            "plotly>=4.0.0",
        ],
        "gpu": [
            "cupy-cuda11x>=9.0.0",  # For GPU acceleration when available
            "numba>=0.50.0",        # For JIT compilation
        ],
    },
    keywords=[
        "geometric algebra",
        "clifford algebra", 
        "GPU acceleration",
        "physics simulation",
        "optics",
        "electromagnetics",
        "ultrasound",
        "state multivectors",
        "propagator transforms",
        "register-resident computing",
        "parallel processing",
        "quantum field dynamics",
        "multispectral imaging",
    ],
    project_urls={
        "Bug Reports": "https://github.com/tracyphasespace/kingdon/issues",
        "Source": "https://github.com/tracyphasespace/kingdon",
        "Documentation": "https://github.com/tracyphasespace/kingdon/blob/master/README.md",
    },
    include_package_data=True,
    package_data={
        'kingdon': [
            '*.md',
            '*.txt', 
            '*.js',
            'tests/*.py',
            'examples/*.py',
        ],
    },
    entry_points={
        'console_scripts': [
            'kingdon-validate=kingdon.validate_ga_library:main',
            'kingdon-demo=kingdon.example_100_wavelength_simulation:main',
            'kingdon-test=kingdon.simple_test:run_all_tests',
        ],
    },
    zip_safe=False,  # Allow access to package data files
)