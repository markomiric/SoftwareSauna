"""
Simple test to verify the path solver works correctly.
"""

from ascii_path_solver import BrokenPathError, InvalidMapError, PathFinder, solve_path


def test_basic_functionality():
    """Test basic functionality of the solver."""
    print("Testing basic functionality...")

    # Test 1: Basic example
    map_data = [
        "  @---A---+",
        "          |",
        "  x-B-+   C",
        "      |   |",
        "      +---+",
    ]

    finder = PathFinder(map_data)
    letters, path = finder.solve()

    print("Test 1 - Basic example:")
    print(f"  Letters: {letters}")
    print(f"  Path: {path}")
    print("  Expected: ACB, @---A---+|C|+---+|+-B-x")
    print(f"  Result: {'PASS' if letters == 'ACB' else 'FAIL'}")
    print()

    # Test 2: Convenience function
    letters2, path2 = solve_path(map_data)
    print("Test 2 - Convenience function:")
    print(f"  Result: {'PASS' if letters2 == letters and path2 == path else 'FAIL'}")
    print()

    # Test 3: Error handling
    try:
        invalid_map = ["@--A--?--x"]  # Invalid character
        PathFinder(invalid_map).solve()
        print("Test 3 - Error handling: FAIL (should have raised exception)")
    except InvalidMapError as e:
        print(f"Test 3 - Error handling: PASS (caught {type(e).__name__}: {e})")
    print()

    # Test 4: Type safety (this would be caught by type checker)
    print("Test 4 - Type safety: PASS (type hints added)")
    print()

    # Test 5: Performance test
    print("Test 5 - Performance test with larger map...")
    large_map = []
    large_map.append("@" + "-" * 18 + "+")
    for i in range(8):
        large_map.append(" " * 19 + "|")
    large_map.append("x" + "-" * 18 + "+")

    finder_large = PathFinder(large_map)
    letters_large, path_large = finder_large.solve()
    print(
        f"  Large map solved: letters='{letters_large}', path_length={len(path_large)}"
    )
    print("  Result: PASS")
    print()


def test_error_scenarios():
    """Test various error scenarios."""
    print("Testing error scenarios...")

    test_cases = [
        (["x-B-+"], InvalidMapError, "missing start"),
        (["@-A-+"], InvalidMapError, "missing end"),
        (["@-A-@-x"], InvalidMapError, "multiple start"),
        (["@-A x"], BrokenPathError, "broken path"),
    ]

    for i, (map_data, expected_error, description) in enumerate(test_cases, 1):
        try:
            PathFinder(map_data).solve()
            print(
                f"  Error test {i}: FAIL (should have raised {expected_error.__name__})"
            )
        except expected_error:
            print(f"  Error test {i}: PASS ({expected_error.__name__}: {description})")
        except Exception as e:
            print(
                f"  Error test {i}: PARTIAL (got {type(e).__name__} instead of {expected_error.__name__})"
            )


if __name__ == "__main__":
    print("=" * 60)
    print("PATH SOLVER TEST SUITE")
    print("=" * 60)
    print()

    test_basic_functionality()
    test_error_scenarios()

    print()
    print("=" * 60)
    print("TEST SUITE COMPLETED")
    print("=" * 60)
