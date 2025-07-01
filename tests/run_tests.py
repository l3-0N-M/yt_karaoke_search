#!/usr/bin/env python3
"""Test runner script for the karaoke collector unit tests."""

import sys
from pathlib import Path

import pytest

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_all_tests():
    """Run all unit tests with coverage report."""
    args = [
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--cov=collector",  # Coverage for collector module
        "--cov-report=term-missing",  # Show missing lines
        "--cov-report=html:tests/htmlcov",  # HTML coverage report
        "tests/unit",  # Test directory
    ]

    return pytest.main(args)


def run_specific_tests(test_module):
    """Run tests for a specific module."""
    test_map = {
        "db": "tests/unit/test_db_optimized.py",
        "web": "tests/unit/test_web_search_pass.py",
        "utils": "tests/unit/test_utils.py",
        "processor": "tests/unit/test_processor.py",
    }

    if test_module in test_map:
        args = ["-v", "--tb=short", test_map[test_module]]
        return pytest.main(args)
    else:
        print(f"Unknown test module: {test_module}")
        print(f"Available modules: {', '.join(test_map.keys())}")
        return 1


def run_failed_tests():
    """Run only previously failed tests."""
    args = [
        "-v",
        "--tb=short",
        "--lf",  # Run last failed tests
        "tests/unit",
    ]

    return pytest.main(args)


def main():
    """Main entry point for test runner."""
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "all":
            return run_all_tests()
        elif command == "failed":
            return run_failed_tests()
        elif command in ["db", "web", "utils", "processor"]:
            return run_specific_tests(command)
        else:
            print(f"Unknown command: {command}")
            print("\nUsage:")
            print("  python tests/run_tests.py all        # Run all tests with coverage")
            print("  python tests/run_tests.py failed     # Run only failed tests")
            print("  python tests/run_tests.py db         # Run database tests")
            print("  python tests/run_tests.py web        # Run web search tests")
            print("  python tests/run_tests.py utils      # Run utils tests")
            print("  python tests/run_tests.py processor  # Run processor tests")
            return 1
    else:
        # Default to running all tests
        return run_all_tests()


if __name__ == "__main__":
    sys.exit(main())
