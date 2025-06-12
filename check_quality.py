#!/usr/bin/env python3
"""
Code quality checker script for ASCII Path Solver project.

This script runs all the code quality tools in sequence and reports results.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n🔍 {description}")
    print("-" * 50)

    try:
        result = subprocess.run(
            command.split(), capture_output=True, text=True, check=True
        )
        print(f"✅ {description} passed")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False
    except FileNotFoundError:
        print(f"⚠️  Tool not found. Install with: pip install {command.split()[0]}")
        return False


def main():
    """Run all quality checks."""
    print("🚀 ASCII Path Solver - Code Quality Checker")
    print("=" * 60)

    # Check if we're in the right directory
    if not Path("ascii_path_solver.py").exists():
        print(
            "❌ ascii_path_solver.py not found. Run this script from the project root."
        )
        sys.exit(1)

    # Define main project files to check
    main_files = [
        "ascii_path_solver.py",
        "test_ascii_path_solver.py",
        "test_integration.py",
        "check_quality.py",
    ]
    files_str = " ".join(main_files)

    checks = [
        (f"black --check {files_str}", "Code formatting (black)"),
        (f"isort --check-only --profile black {files_str}", "Import sorting (isort)"),
        ("mypy ascii_path_solver.py", "Type checking (mypy)"),
        (f"flake8 {files_str}", "Code linting (flake8)"),
        ("python -m pytest test_ascii_path_solver.py -v", "Unit tests (pytest)"),
        ("python test_integration.py", "Integration tests"),
    ]

    results = []
    for command, description in checks:
        success = run_command(command, description)
        results.append((description, success))

    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for description, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {description}")

    print(f"\nOverall: {passed}/{total} checks passed")

    if passed == total:
        print("🎉 All quality checks passed! Ready to commit.")
        sys.exit(0)
    else:
        print("🔧 Some checks failed. Please fix the issues before committing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
