"""
Comprehensive test suite for the ASCII Path Following Puzzle Solver.

This test suite follows pytest best practices and provides comprehensive coverage
of success cases, failure cases, and edge cases.

Author: Marko Miric
Version: 1.0
"""

from typing import List, Type

import pytest

from ascii_path_solver import (
    BrokenPathError,
    DeadEndError,
    Direction,
    ForkInPathError,
    InvalidMapError,
    PathFinder,
    PathSolverConfig,
    Position,
    solve_path,
)


class TestMapPreprocessing:
    """Test helper methods for map preprocessing."""

    @staticmethod
    def parse_map_data(map_data: str) -> List[str]:
        """
        Parse multiline map data into list of strings.

        Args:
            map_data: Multiline string representing the map

        Returns:
            List of map lines with proper preprocessing
        """
        lines = map_data.split("\n")
        # Remove empty lines from start and end only
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        # Only rstrip each line to remove trailing newlines, preserve leading spaces
        return [line.rstrip() for line in lines]


class TestPathFinderValidPaths:
    """Test cases for valid path scenarios."""

    @pytest.mark.parametrize(
        "map_data,expected_letters,expected_path",
        [
            # Basic example
            (
                """
          @---A---+
                  |
          x-B-+   C
              |   |
              +---+
        """,
                "ACB",
                "@---A---+|C|+---+|+-B-x",
            ),
            # Go straight through intersections
            (
                """
          @
          | +-C--+
          A |    |
          +---B--+
            |      x
            |      |
            +---D--+
        """,
                "ABCD",
                "@|A+---B--+|+--C-+|-||+---D--+|x",
            ),
            # Letters on turns
            (
                """
          @---A---+
                  |
          x-B-+   |
              |   |
              +---C
        """,
                "ACB",
                "@---A---+|||C---+|+-B-x",
            ),
            # Don't collect letter twice
            (
                """
             +-O-N-+
             |     |
             |   +-I-+
         @-G-O-+ | | |
             | | +-+ E
             +-+     S
                     |
                     x
        """,
                "GOONIES",
                "@-G-O-+|+-+|O||+-O-N-+|I|+-+|+-I-+|ES|x",
            ),
            # Keep direction in compact space
            (
                """
 +-L-+
 |  +A-+
@B+ ++ H
 ++    x
        """,
                "BLAH",
                "@B+++B|+-L-+A+++A-+Hx",
            ),
            # Ignore stuff after end of path
            (
                """
          @-A--+
               |
               +-B--x-C--D
        """,
                "AB",
                "@-A--+|+-B--x",
            ),
        ],
    )
    def test_valid_paths(
        self, map_data: str, expected_letters: str, expected_path: str
    ):
        """Test valid path scenarios."""
        map_lines = TestMapPreprocessing.parse_map_data(map_data)
        finder = PathFinder(map_lines)
        letters, path = finder.solve()

        assert (
            letters == expected_letters
        ), f"Expected letters '{expected_letters}', got '{letters}'"
        assert path == expected_path, f"Expected path '{expected_path}', got '{path}'"

    def test_solve_path_convenience_function(self):
        """Test the convenience function."""
        map_data = """
          @---A---+
                  |
          x-B-+   C
              |   |
              +---+
        """
        map_lines = TestMapPreprocessing.parse_map_data(map_data)
        letters, path = solve_path(map_lines)

        assert letters == "ACB"
        assert path == "@---A---+|C|+---+|+-B-x"


class TestPathFinderInvalidMaps:
    """Test cases for invalid map scenarios."""

    @pytest.mark.parametrize(
        "map_data,expected_error,error_message_contains",
        [
            # Missing start character
            (
                """
             -A---+
                  |
          x-B-+   C
              |   |
              +---+
        """,
                InvalidMapError,
                "missing start",
            ),
            # Missing end character
            (
                """
           @--A---+
                  |
            B-+   C
              |   |
              +---+
        """,
                InvalidMapError,
                "missing end",
            ),
            # Multiple start characters
            (
                """
           @--A-@-+
                  |
          x-B-+   C
              |   |
              +---+
        """,
                InvalidMapError,
                "multiple start",
            ),
            # Fork in path
            (
                """
                x-B
                  |
           @--A---+
                  |
             x+   C
              |   |
              +---+
        """,
                ForkInPathError,
                "fork in path",
            ),
            # Broken path with gap
            (
                """
           @--A-+
                |

                B-x
        """,
                BrokenPathError,
                "broken path",
            ),
            # Broken path leads to nothing
            (
                """
        @-A x
        """,
                BrokenPathError,
                "broken path",
            ),
            # Multiple starting paths
            (
                """
          x-B-@-A-x
        """,
                BrokenPathError,
                "multiple starting paths",
            ),
            # Fake turn
            (
                """
          @-A-+-B-x
        """,
                BrokenPathError,
                "fake turn",
            ),
            # Invalid character
            (
                """
        @--A--?--x
        """,
                InvalidMapError,
                "invalid character",
            ),
            # Dead end at letter
            (
                """
        @-A
          B
        x
        """,
                DeadEndError,
                "dead end at character 'a'",
            ),
        ],
    )
    def test_invalid_maps(
        self,
        map_data: str,
        expected_error: Type[Exception],
        error_message_contains: str,
    ):
        """Test invalid map scenarios."""
        map_lines = TestMapPreprocessing.parse_map_data(map_data)

        with pytest.raises(expected_error) as exc_info:
            finder = PathFinder(map_lines)
            finder.solve()

        assert error_message_contains.lower() in str(exc_info.value).lower()


class TestPathFinderEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_map(self):
        """Test empty map handling."""
        with pytest.raises(InvalidMapError):
            PathFinder([])

    def test_single_character_map(self):
        """Test single character map."""
        with pytest.raises(InvalidMapError, match="Missing end"):
            PathFinder(["@"])

    def test_minimal_valid_map(self):
        """Test minimal valid map."""
        finder = PathFinder(["@-x"])
        letters, path = finder.solve()
        assert letters == ""
        assert path == "@-x"

    def test_map_with_only_spaces(self):
        """Test map with only spaces."""
        with pytest.raises(InvalidMapError, match="Missing start"):
            PathFinder(["   ", "   ", "   "])

    def test_very_long_path(self):
        """Test handling of very long paths."""
        # Create a long horizontal path
        long_path = "@" + "-" * 50 + "x"
        finder = PathFinder([long_path])
        letters, path = finder.solve()
        assert letters == ""
        assert path == long_path


class TestPathFinderConfiguration:
    """Test configuration and customization."""

    def test_custom_config(self):
        """Test PathFinder with custom configuration."""
        custom_config = PathSolverConfig(
            COMPACT_SPACE_MAX_ROWS=2,
            COMPACT_SPACE_MAX_COLS=5,
            COMPACT_SPACE_MAX_STEPS=100,
        )

        map_lines = ["@-A-x"]
        finder = PathFinder(map_lines, config=custom_config)
        letters, path = finder.solve()

        assert letters == "A"
        assert path == "@-A-x"

    def test_compact_space_detection(self):
        """Test compact space detection logic."""
        # Small map should be detected as compact
        small_map = ["@-x"]
        finder = PathFinder(small_map)
        assert finder._is_compact_space()

        # Large map should not be compact
        large_map = ["@" + "-" * 20 + "x"] + [" " * 22] * 10
        finder = PathFinder(large_map)
        assert not finder._is_compact_space()


class TestPathFinderInternals:
    """Test internal methods and components."""

    def test_position_movement(self):
        """Test Position class movement."""
        pos = Position(5, 10)

        assert pos.move(Direction.UP) == Position(4, 10)
        assert pos.move(Direction.RIGHT) == Position(5, 11)
        assert pos.move(Direction.DOWN) == Position(6, 10)
        assert pos.move(Direction.LEFT) == Position(5, 9)

    def test_direction_opposite(self):
        """Test Direction opposite method."""
        assert Direction.UP.opposite() == Direction.DOWN
        assert Direction.RIGHT.opposite() == Direction.LEFT
        assert Direction.DOWN.opposite() == Direction.UP
        assert Direction.LEFT.opposite() == Direction.RIGHT

    def test_character_validation(self):
        """Test character validation methods."""
        finder = PathFinder(["@-x"])

        assert finder.is_valid_path_char("@")
        assert finder.is_valid_path_char("-")
        assert finder.is_valid_path_char("A")
        assert not finder.is_valid_path_char(" ")
        assert not finder.is_valid_path_char("?")

        assert finder.is_letter("A")
        assert finder.is_letter("Z")
        assert not finder.is_letter("@")
        assert not finder.is_letter("1")

    def test_get_char_at_bounds(self):
        """Test get_char_at with out of bounds positions."""
        finder = PathFinder(["@-x"])

        assert finder.get_char_at(Position(0, 0)) == "@"
        assert finder.get_char_at(Position(0, 1)) == "-"
        assert finder.get_char_at(Position(0, 2)) == "x"

        # Out of bounds should return space
        assert finder.get_char_at(Position(-1, 0)) == " "
        assert finder.get_char_at(Position(0, -1)) == " "
        assert finder.get_char_at(Position(1, 0)) == " "
        assert finder.get_char_at(Position(0, 3)) == " "


class TestPerformanceAndComplexity:
    """Test performance characteristics and complexity."""

    def test_large_map_performance(self):
        """Test performance with larger maps."""
        # Create a reasonably large map
        size = 20
        large_map = []
        large_map.append("@" + "-" * (size - 2) + "+")
        for _ in range(size - 2):
            large_map.append(" " * (size - 1) + "|")
        large_map.append("x" + "-" * (size - 2) + "+")

        finder = PathFinder(large_map)
        _, path = finder.solve()

        # Should complete without timeout
        assert path.startswith("@")
        assert path.endswith("x")

    @pytest.mark.parametrize("map_size", [5, 10, 15])
    def test_scalability(self, map_size: int):
        """Test scalability with different map sizes."""
        # Create valid rectangular maps that form a complete path
        rect_map = []
        rect_map.append("@" + "-" * (map_size - 2) + "+")
        for _ in range(map_size - 2):
            rect_map.append(" " * (map_size - 1) + "|")
        rect_map.append("x" + "-" * (map_size - 2) + "+")

        finder = PathFinder(rect_map)
        _, path = finder.solve()

        assert path.startswith("@")
        assert path.endswith("x")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
