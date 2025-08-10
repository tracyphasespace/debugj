#!/usr/bin/env python3
"""
Fix script to correct MultiVector data structure issues with Python 3.13
"""

import sys
import os
import re

def fix_multivector_values_method():
    """Fix the values() method to always return a list/array"""
    filepath = "src/multivector.py"
    
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the values() method to ensure it always returns a list
    old_values_method = r'def values\(self\) -> Union\[List\[Any\], np\.ndarray\]:\s*"""[^"]*"""\s*return self\._values if hasattr\(self, \'_values\'\) else \[\]'
    
    new_values_method = '''def values(self) -> Union[List[Any], np.ndarray]:
        """
        Get the coefficients corresponding to the basis blades keys.

        Returns:
            List or NumPy array of coefficients.
        """
        if not hasattr(self, '_values'):
            return []
        
        # Ensure we always return a list or array, never a single value
        if isinstance(self._values, (list, np.ndarray)):
            return self._values
        else:
            # If _values is a single value, wrap it in a list
            return [self._values]'''
    
    # Use a more flexible pattern
    pattern = r'def values\(self\)[^:]*:[^"]*"""[^"]*"""[^}]*return[^}]*'
    
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, new_values_method, content, flags=re.DOTALL)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Fixed values() method in multivector.py")
        return True
    else:
        print("Could not find values() method pattern to fix")
        return False

def fix_operator_dict_extend():
    """Fix the operator_dict.py extend issue"""
    filepath = "src/operator_dict.py"
    
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the extend operation to handle single values
    old_extend = r'values_in\.extend\(mv\.values\(\)\)'
    new_extend = '''mv_values = mv.values()
            if isinstance(mv_values, (list, tuple, np.ndarray)):
                values_in.extend(mv_values)
            else:
                values_in.append(mv_values)'''
    
    if re.search(old_extend, content):
        content = re.sub(old_extend, new_extend, content)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Fixed extend operation in operator_dict.py")
        return True
    else:
        print("Could not find extend pattern to fix")
        return False

def ensure_values_is_list():
    """Fix the fromkeysvalues method to ensure _values is always a list"""
    filepath = "src/multivector.py"
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the fromkeysvalues method and ensure values is wrapped properly
    pattern = r'(obj\._values = values)'
    replacement = '''# Ensure _values is always a list or array
        if isinstance(values, (list, np.ndarray)):
            obj._values = values
        else:
            obj._values = [values]'''
    
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Fixed _values assignment in fromkeysvalues method")
        return True
    else:
        print("Could not find _values assignment pattern")
        return False

if __name__ == "__main__":
    print("Fixing MultiVector compatibility issues...")
    
    success = True
    success &= fix_multivector_values_method()
    success &= fix_operator_dict_extend()
    success &= ensure_values_is_list()
    
    if success:
        print("\nAll fixes applied successfully!")
    else:
        print("\nSome fixes failed. Manual intervention may be required.")