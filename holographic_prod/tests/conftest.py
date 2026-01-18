"""
Pytest configuration for holographic_prod tests.

Ensures caches are cleared between tests for reproducibility.
"""

import pytest
import sys


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear all module-level caches before each test."""
    # Import and clear tower memory caches
    try:
        from holographic_prod.memory import tower_memory as tm
        if hasattr(tm, '_ROTATION_CACHE'):
            tm._ROTATION_CACHE.clear()
    except ImportError:
        pass
    
    # Import and clear algebra caches
    try:
        from holographic_prod.core import algebra as alg
        if hasattr(alg, '_BASIS_CACHE'):
            alg._BASIS_CACHE.clear()
        if hasattr(alg, '_STRUCTURE_TENSOR_CACHE'):
            alg._STRUCTURE_TENSOR_CACHE.clear()
    except ImportError:
        pass
    
    # IMPORTANT: Also reload modules to reset any module-level state
    # This ensures tests are isolated
    modules_to_reload = [k for k in list(sys.modules.keys()) if k.startswith('holographic_prod.memory')]
    for mod_name in modules_to_reload:
        del sys.modules[mod_name]
    
    yield  # Test runs here
    
    # No cleanup needed after
