"""
Array Utilities — GPU-Native Array Handling

Provides utilities for safely handling arrays in GPU-native mode.

DESIGN PRINCIPLE (GPU-NATIVE):
    - All arrays should be on the same device as xp
    - When xp=cupy, all arrays should be CuPy arrays
    - When xp=numpy, all arrays should be NumPy arrays
    - No mixing of NumPy and CuPy in operations
    
This module provides the `stack_matrices` function that ensures device consistency.
"""

import numpy as np


def to_numpy(arr):
    """
    Convert array to NumPy if it's CuPy.
    
    Args:
        arr: NumPy or CuPy array
        
    Returns:
        NumPy array
    """
    if hasattr(arr, 'get'):
        return arr.get()
    return arr


def to_xp(arr, xp):
    """
    Convert array to target array module (NumPy or CuPy).
    
    GPU-NATIVE: Ensures array is on the correct device.
    
    Args:
        arr: NumPy or CuPy array
        xp: Target array module (numpy or cupy)
        
    Returns:
        Array in target format
    """
    # Check if xp is CuPy
    is_cupy = hasattr(xp, 'cuda')
    
    if is_cupy:
        # Target is CuPy - ensure array is CuPy
        if hasattr(arr, '__cuda_array_interface__'):
            return arr  # Already CuPy
        return xp.asarray(arr)  # Convert NumPy to CuPy
    else:
        # Target is NumPy - ensure array is NumPy
        if hasattr(arr, 'get'):
            return arr.get()  # Convert CuPy to NumPy
        return arr  # Already NumPy


def stack_matrices(matrices_list, xp):
    """
    Stack a list of matrices into a batch array, ensuring device consistency.
    
    GPU-NATIVE: All matrices are converted to xp's device before stacking.
    
    Args:
        matrices_list: List of [4, 4] matrices (NumPy or CuPy)
        xp: Target array module (numpy or cupy)
        
    Returns:
        [N, 4, 4] stacked array on xp's device
    """
    if not matrices_list:
        return xp.zeros((0, 4, 4), dtype=np.float32)
    
    # GPU-NATIVE: Convert all matrices to target device, then stack
    converted = [xp.asarray(m) for m in matrices_list]
    return xp.stack(converted)
