"""
CynthAI_v2 — Universal test runner.

Runs all test scripts in tests/ and reports a summary.
Tests requiring the Rust simulator (PyBattle) are skipped if the simulator
is not available (e.g. not compiled yet).

Usage:
    .venv\\Scripts\\python.exe tests/run_all_tests.py
"""

import sys
import io
import subprocess
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR    = PROJECT_ROOT / "tests"
PYTHON       = sys.executable

# Tests that do NOT require the Rust simulator
TESTS_NO_SIM = [
    "test_full_pipeline.py",
]

# Tests that DO require the Rust simulator (PyBattle)
TESTS_WITH_SIM = [
    "test_simulator.py",
    "test_encoder.py",
    "test_attention_maps.py",
]

# Check if simulator is available
def sim_available() -> bool:
    try:
        result = subprocess.run(
            [PYTHON, "-c", "from simulator import PyBattle"],
            capture_output=True, cwd=PROJECT_ROOT
        )
        return result.returncode == 0
    except Exception:
        return False


def run_test(script: str) -> tuple[bool, float, str]:
    """Run a test script, return (passed, duration_s, output)."""
    path = TESTS_DIR / script
    start = time.time()
    result = subprocess.run(
        [PYTHON, str(path)],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
    )
    duration = time.time() - start
    output = result.stdout + result.stderr
    passed = result.returncode == 0
    return passed, duration, output


def main():
    print("=" * 60)
    print("CynthAI_v2 Test Suite")
    print("=" * 60)

    has_sim = sim_available()
    if has_sim:
        print("Rust simulator: available")
        all_tests = TESTS_NO_SIM + TESTS_WITH_SIM
    else:
        print("Rust simulator: NOT available -- skipping simulator tests")
        all_tests = TESTS_NO_SIM

    results = []
    for script in all_tests:
        print(f"\n{'─' * 60}")
        print(f"Running: {script}")
        print(f"{'─' * 60}")

        passed, duration, output = run_test(script)
        print(output.rstrip())
        results.append((script, passed, duration))

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = sum(1 for _, ok, _ in results if not ok)
    for script, passed, duration in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]  {script:<35}  {duration:.1f}s")

    if not has_sim:
        for script in TESTS_WITH_SIM:
            print(f"  [SKIP]  {script:<35}  (no simulator)")

    print(f"\n  {n_pass} passed, {n_fail} failed", end="")
    if not has_sim:
        print(f", {len(TESTS_WITH_SIM)} skipped", end="")
    print()

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
