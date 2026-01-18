#!/usr/bin/env python3
"""
DMN/Genius Test Suite Runner

Run all tests from the DMN/Genius configuration test plan.
Each test is run in sequence and results are aggregated.

Usage:
    python holographic_prod/tests/run_genius_tests.py [--test TEST_NAME] [--all]
    
    Options:
        --test TEST_NAME: Run only the specified test
        --all: Run all tests in sequence
        --summary: Print summary of completed tests
"""

import subprocess
import sys
import json
import os
from pathlib import Path

# Test suite in dependency order
TEST_SUITE = [
    # Baseline (must run first)
    ("baseline", "test_baseline_characterization.py", "Establish baseline metrics"),
    
    # Phase space tests (depend on baseline)
    ("delayed-collapse", "test_delayed_collapse.py", "Delayed scalar collapse hypothesis"),
    
    # DMN component tests (depend on baseline)
    ("parallel-proposals", "test_parallel_proposals.py", "DMN-like parallel proposal generation"),
    ("sandboxing", "test_sandboxed_simulation.py", "Sandboxed vs direct writes"),
    
    # DMN integration (depends on proposals + sandboxing)
    ("dmn-interleaved", "test_dmn_interleaved.py", "DMN during active learning"),
    
    # Stability tests (depend on baseline)
    ("witness-stability", "test_witness_stability.py", "Witness invariance and churn"),
    ("satellite-holonomy", "test_satellite_holonomy.py", "Satellite-level holonomy tracking"),
    
    # Response pattern tests (depend on baseline)
    ("conflict-curiosity", "test_conflict_curiosity.py", "Conflict-as-curiosity vs avoidance"),
    ("coherence-reward", "test_coherence_reward.py", "Coherence vs novelty reward"),
    
    # Initial integration (depends on all above)
    ("integration", "test_genius_configuration.py", "Full genius configuration"),
    
    # === BREAKTHROUGH TESTS ===
    # Adaptive gating v1 (simple)
    ("adaptive-gate-v1", "test_adaptive_dmn_gate.py", "State-dependent DMN gating"),
    
    # Collapse detection (psychosis prevention)
    ("collapse-detection", "test_overdreaming_collapse.py", "Over-dreaming collapse detection"),
    
    # Adaptive gating v2 (theory-true, distance-aware)
    ("adaptive-gate-v2", "test_adaptive_dmn_gate_v2.py", "Theory-true adaptive gate (BREAKTHROUGH)"),
]


def run_test(test_file: str, timeout: int = 1800) -> dict:
    """Run a single test on Modal."""
    print(f"\n{'='*60}")
    print(f"Running: {test_file}")
    print(f"{'='*60}")
    
    cmd = f"timeout {timeout} modal run holographic_prod/tests/{test_file}"
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout + 60  # Extra buffer for Modal startup
        )
        
        if result.returncode == 0:
            print(f"✓ {test_file} completed successfully")
            return {"status": "success", "output": result.stdout}
        else:
            print(f"✗ {test_file} failed with code {result.returncode}")
            print(f"  Error: {result.stderr[:500]}")
            return {"status": "failed", "error": result.stderr}
            
    except subprocess.TimeoutExpired:
        print(f"✗ {test_file} timed out after {timeout}s")
        return {"status": "timeout"}
    except Exception as e:
        print(f"✗ {test_file} exception: {e}")
        return {"status": "error", "error": str(e)}


def print_summary():
    """Print summary of all test results from checkpoint files."""
    checkpoint_dir = Path("/checkpoints")
    
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    result_files = [
        ("baseline", "baseline_metrics.json"),
        ("delayed-collapse", "delayed_collapse_results.json"),
        ("parallel-proposals", "parallel_proposals_results.json"),
        ("sandboxing", "sandboxed_simulation_results.json"),
        ("dmn-interleaved", "dmn_interleaved_results.json"),
        ("witness-stability", "witness_stability_results.json"),
        ("satellite-holonomy", "satellite_holonomy_results.json"),
        ("conflict-curiosity", "conflict_curiosity_results.json"),
        ("coherence-reward", "coherence_reward_results.json"),
        ("integration", "genius_configuration_results.json"),
    ]
    
    print("\nNote: Results are stored in Modal volume /checkpoints/")
    print("To view, run: modal volume get holographic-checkpoints <filename>")
    
    for test_name, filename in result_files:
        print(f"\n  {test_name}:")
        print(f"    → /checkpoints/{filename}")


def main():
    if len(sys.argv) < 2 or "--help" in sys.argv:
        print(__doc__)
        print("\nAvailable tests:")
        for name, file, desc in TEST_SUITE:
            print(f"  {name:20s} {file:40s} {desc}")
        return
    
    if "--summary" in sys.argv:
        print_summary()
        return
    
    if "--all" in sys.argv:
        print("Running ALL tests in sequence...")
        print("This will take approximately 5-6 hours on Modal H100.")
        print("\nTests to run:")
        for name, file, desc in TEST_SUITE:
            print(f"  [{name}] {desc}")
        
        results = {}
        for name, file, desc in TEST_SUITE:
            results[name] = run_test(file)
            if results[name]["status"] != "success":
                print(f"\n⚠️  Test {name} did not succeed. Continuing...")
        
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        for name, result in results.items():
            status = result["status"]
            icon = "✓" if status == "success" else "✗"
            print(f"  {icon} {name}: {status}")
        
        return
    
    if "--test" in sys.argv:
        idx = sys.argv.index("--test")
        if idx + 1 >= len(sys.argv):
            print("Error: --test requires a test name")
            return
        
        test_name = sys.argv[idx + 1]
        
        for name, file, desc in TEST_SUITE:
            if name == test_name or file == test_name:
                run_test(file)
                return
        
        print(f"Error: Unknown test '{test_name}'")
        print("\nAvailable tests:")
        for name, file, desc in TEST_SUITE:
            print(f"  {name}")
        return
    
    # Default: show help
    print(__doc__)


if __name__ == "__main__":
    main()
