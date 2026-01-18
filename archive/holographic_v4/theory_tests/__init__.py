"""
Theory Validation Test Suite — Systematic Validation of High-Priority Research Directions
==========================================================================================

This test suite validates theory-derived predictions with real Gutenberg text data.

TEST MODULES (Run in order due to dependencies):
    test_00_foundations - Verify claimed implementations work
    test_01_dynamics    - Chaos, criticality, phase transitions
    test_02_information - Information bottleneck, capacity, geometry
    test_03_memory      - Encoding, retrieval, consolidation
    test_04_language    - Semantics, grammar, discourse
    test_05_creativity  - Bisociation, metaphor, compression

PRINCIPLES:
    1. Theory-First: Derive prediction from theory BEFORE measuring
    2. No Early Conclusions: Require statistical significance with Bonferroni correction
    3. Parsimonious: Reuse fixtures, minimize setup code
    4. Real Data: Use Gutenberg text for language tests
    5. Dependency Order: Run 00 → 01 → 02 → 03 → 04 → 05
    6. Fail Fast: If foundations fail, skip dependent tests

USAGE:
    # Run all tests
    pytest holographic_v4/theory_tests/ -v --tb=short
    
    # Run specific module
    pytest holographic_v4/theory_tests/test_00_foundations.py -v
    
    # Run with timeout (recommended)
    pytest holographic_v4/theory_tests/ -v --timeout=300
"""

__version__ = "1.0.0"
