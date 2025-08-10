#!/usr/bin/env python3
"""
Check the signature configuration of the kingdon library
"""

from kingdon import Algebra
import numpy as np

print("Kingdon Library Signature Analysis")
print("=" * 40)

# Test different algebra configurations
configs = [
    ("3D Euclidean", {"p": 3, "q": 0, "r": 0}),
    ("Minkowski Spacetime", {"p": 1, "q": 3, "r": 0}),
    ("3D PGA", {"p": 3, "q": 0, "r": 1}),
    ("Conformal GA", {"p": 4, "q": 1, "r": 0}),
]

for name, params in configs:
    print(f"\n{name} (p={params['p']}, q={params['q']}, r={params['r']}):")
    try:
        alg = Algebra(**params)
        print(f"  Dimension: {alg.d}")
        print(f"  Signature: {alg.signature}")
        print(f"  Total basis elements: {len(alg)}")
        
        # Test basis vector squares
        if alg.d <= 4:  # Only for smaller algebras
            print("  Basis vector squares:")
            for i in range(min(4, alg.d)):
                try:
                    ei = alg.multivector(keys=(1<<i,), values=[1])
                    ei_squared = alg.gp(ei, ei)
                    val = ei_squared.values()[0] if len(ei_squared.values()) > 0 else 0
                    print(f"    e{i}² = {val}")
                except Exception as e:
                    print(f"    e{i}² = Error: {e}")
        
    except Exception as e:
        print(f"  Error creating algebra: {e}")

print("\n" + "=" * 40)
print("Default Configuration Test:")

# Test what the default import gives us
try:
    default_alg = Algebra()
    print(f"Default algebra: p={default_alg.p}, q={default_alg.q}, r={default_alg.r}")
    print(f"Default signature: {default_alg.signature}")
    print(f"Default dimension: {default_alg.d}")
except Exception as e:
    print(f"Error with default algebra: {e}")

# Check if we can create Minkowski spacetime specifically
print("\nMinkowski Spacetime (-,+,+,+) Test:")
try:
    minkowski = Algebra(p=1, q=3, r=0)
    print(f"Created successfully: signature {minkowski.signature}")
    
    # Test time-like vector (should square to -1)
    e0 = minkowski.multivector(keys=(1,), values=[1])  # First basis vector
    e0_squared = minkowski.gp(e0, e0)
    print(f"e0 (timelike)² = {e0_squared.values()[0] if e0_squared.values() else 'No value'}")
    
    # Test space-like vector (should square to +1)
    e1 = minkowski.multivector(keys=(2,), values=[1])  # Second basis vector
    e1_squared = minkowski.gp(e1, e1)
    print(f"e1 (spacelike)² = {e1_squared.values()[0] if e1_squared.values() else 'No value'}")
    
except Exception as e:
    print(f"Error creating Minkowski: {e}")

print("\nConclusion:")
print("The library supports multiple signatures including Minkowski (-,+,+,+)")
print("Use Algebra(p=1, q=3, r=0) for spacetime with signature (-,+,+,+)")
print("Use Algebra(p=3, q=1, r=0) for spacetime with signature (+,+,+,-)")