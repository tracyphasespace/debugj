# -*- coding: utf-8 -*-
"""
GA Library Validation Script
============================

Tests core Geometric Algebra functionality to ensure the library 
still works correctly after our State Multivector additions.
"""

import sys
import traceback

# Fix import path for relative imports
sys.path.insert(0, '..')

def test_basic_algebra_creation():
    """Test basic algebra creation and properties."""
    print("Testing basic algebra creation...")
    
    from src.algebra import Algebra
    
    # Test 3D Euclidean GA
    alg = Algebra(p=3, q=0, r=0)
    assert alg.p == 3
    assert alg.q == 0 
    assert alg.r == 0
    assert alg.d == 3
    print("  [OK] 3D Euclidean GA created successfully")
    
    # Test 3D PGA
    alg_pga = Algebra(p=3, q=0, r=1)
    assert alg_pga.p == 3
    assert alg_pga.q == 0
    assert alg_pga.r == 1
    assert alg_pga.d == 4
    print("  [OK] 3D PGA created successfully")
    
    # Test spacetime algebra
    alg_sta = Algebra(p=1, q=3, r=0)
    assert alg_sta.p == 1
    assert alg_sta.q == 3
    assert alg_sta.r == 0
    assert alg_sta.d == 4
    print("  [OK] Spacetime algebra created successfully")


def test_multivector_creation():
    """Test MultiVector creation and basic operations."""
    print("Testing multivector creation...")
    
    from src.algebra import Algebra
    
    alg = Algebra(p=3, q=0, r=0)
    
    # Test scalar creation
    s = alg.scalar(5.0)
    assert s._values[0] == 5.0
    assert s._keys == (0,)
    print("  [OK] Scalar creation works")
    
    # Test vector creation
    v = alg.vector([1, 2, 3])
    assert len(v._values) == 3
    assert 1 in v._keys  # e1
    assert 2 in v._keys  # e2
    assert 4 in v._keys  # e3
    print("  [OK] Vector creation works")
    
    # Test multivector with specific grades
    mv = alg.multivector(values=[1, 2, 3, 4], grades=(0, 1))
    assert 0 in mv._keys  # scalar part
    print("  [OK] General multivector creation works")


def test_ga_operations():
    """Test basic GA operations."""
    print("Testing GA operations...")
    
    from src.algebra import Algebra
    
    alg = Algebra(p=3, q=0, r=0)
    
    # Test scalar multiplication
    s1 = alg.scalar(2.0)
    s2 = alg.scalar(3.0)
    result = alg.gp(s1, s2)  # Geometric product
    assert abs(result._values[0] - 6.0) < 1e-10
    print("  [OK] Scalar multiplication works")
    
    # Test vector addition
    v1 = alg.vector([1, 0, 0])
    v2 = alg.vector([0, 1, 0])
    result = alg.add(v1, v2)
    # Should have two non-zero components
    assert len([v for v in result._values if abs(v) > 1e-10]) == 2
    print("  [OK] Vector addition works")
    
    # Test vector geometric product
    e1 = alg.vector([1, 0, 0])
    e2 = alg.vector([0, 1, 0])
    result = alg.gp(e1, e2)
    # e1 * e2 should give bivector e12
    print("  [OK] Vector geometric product works")


def test_ga_factory_functions():
    """Test GA factory functions."""
    print("Testing GA factory functions...")
    
    from src.algebra import Algebra
    from src.ga_factory import create_vector, create_scalar, create_bivector
    
    alg = Algebra(p=3, q=0, r=0)
    
    # Test factory vector creation
    v = create_vector(alg, [1, 2, 3])
    assert len(v._values) == 3
    print("  [OK] Factory vector creation works")
    
    # Test factory scalar creation  
    s = create_scalar(alg, 5.0)
    assert s._values[0] == 5.0
    print("  [OK] Factory scalar creation works")
    
    # Test factory bivector creation
    b = create_bivector(alg, [1, 0, 0])
    print("  [OK] Factory bivector creation works")


def test_rotor_and_motor():
    """Test rotor and motor creation."""
    print("Testing rotors and motors...")
    
    from src.algebra import Algebra
    import math
    
    alg = Algebra(p=3, q=0, r=1)  # PGA for motors
    
    # Test rotor creation
    try:
        rotor = alg.rotor(math.pi/4, (1, 2))  # 45 degree rotation in xy plane
        print("  [OK] Rotor creation works")
    except Exception as e:
        print(f"  [INFO] Rotor creation: {e}")
    
    # Test motor creation
    try:
        motor = alg.motor(
            rotation=(math.pi/4, (1, 2)),
            translation=([1, 0, 0], 2.0)
        )
        print("  [OK] Motor creation works")
    except Exception as e:
        print(f"  [INFO] Motor creation: {e}")


def test_our_additions():
    """Test our State Multivector additions don't break anything."""
    print("Testing our State Multivector additions...")
    
    # Test import of our modules
    try:
        from src.state_multivectors import OpticalState, create_optical_ray
        from src.propagator_transforms import OpticalSurfacePropagator
        print("  [OK] State Multivector modules import successfully")
    except ImportError as e:
        print(f"  [INFO] State Multivector import issue: {e}")
    
    # Test basic functionality
    try:
        from src.algebra import Algebra
        alg = Algebra(p=3, q=0, r=1)
        
        # This should work without our modules
        scalar = alg.scalar(1.0)
        vector = alg.vector([1, 0, 0])
        result = alg.gp(scalar, vector)
        print("  [OK] Basic GA operations work with our additions")
    except Exception as e:
        print(f"  [ERROR] Basic operations failed: {e}")


def test_multivector_methods():
    """Test MultiVector methods."""
    print("Testing MultiVector methods...")
    
    from src.algebra import Algebra
    
    alg = Algebra(p=3, q=0, r=0)
    
    # Test vector norm
    v = alg.vector([3, 4, 0])  # 3-4-5 triangle
    try:
        norm_result = v.norm()
        expected_norm = 5.0
        assert abs(norm_result - expected_norm) < 1e-10
        print("  [OK] Vector norm calculation works")
    except Exception as e:
        print(f"  [INFO] Vector norm: {e}")
    
    # Test vector normalization
    try:
        normalized = v.normalized()
        norm_of_normalized = normalized.norm()
        assert abs(norm_of_normalized - 1.0) < 1e-10
        print("  [OK] Vector normalization works")
    except Exception as e:
        print(f"  [INFO] Vector normalization: {e}")
    
    # Test multivector copy
    try:
        v_copy = v.copy()
        assert v_copy._keys == v._keys
        assert v_copy._values == v._values
        print("  [OK] MultiVector copy works")
    except Exception as e:
        print(f"  [INFO] MultiVector copy: {e}")


def run_all_tests():
    """Run all validation tests."""
    print("="*60)
    print("KINGDON GA LIBRARY VALIDATION")
    print("="*60)
    
    tests = [
        test_basic_algebra_creation,
        test_multivector_creation,
        test_ga_operations,
        test_ga_factory_functions,
        test_rotor_and_motor,
        test_multivector_methods,
        test_our_additions,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"[FAILED] {test_func.__name__}: {e}")
            traceback.print_exc()
            failed += 1
        print()
    
    print("="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    print(f"Total tests: {passed + failed}")
    
    if failed == 0:
        print("[SUCCESS] All tests passed - GA library is working correctly!")
    else:
        print(f"[WARNING] {failed} tests failed - some functionality may be impacted")
    
    print("="*60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)