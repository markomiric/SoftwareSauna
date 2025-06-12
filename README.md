# ASCII Path Following Puzzle Solver

A Python implementation that solves ASCII path following puzzles. Given a map with ASCII characters, it follows a path from start (`@`) to end (`x`), collecting letters along the way.

## Description

The solver follows these rules:

- Start at the `@` character
- Follow the path using `-`, `|`, and `+` characters
- Collect letters (A-Z) encountered along the path
- Stop when reaching the `x` character
- Go straight through intersections when possible
- Turn only when forced to
- Don't collect the same letter twice from the same position

## Example

```
  @---A---+
          |
  x-B-+   C
      |   |
      +---+
```

Result: Letters = "ACB", Path = "@---A---+|C|+---+|+-B-x"

## Usage

```python
from ascii_path_solver import PathFinder

map_lines = [
    "  @---A---+",
    "          |",
    "  x-B-+   C",
    "      |   |",
    "      +---+"
]

finder = PathFinder(map_lines)
letters, path = finder.solve()
print(f"Letters: {letters}")
print(f"Path: {path}")
```

## Development Setup

### 1. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

#### Minimal Setup (Testing Only)

```bash
pip install pytest
```

#### Full Development Setup (Recommended)

```bash
# Install all development tools
pip install -r requirements.txt
```

This includes:

- **pytest** - Testing framework
- **mypy** - Type checking
- **black** - Code formatting
- **isort** - Import sorting
- **flake8** - Code linting

## Running Tests

### Run All Tests

```bash
python -m pytest test_ascii_path_solver.py -v
```

### Run Integration Tests

```bash
python test_integration.py
```

## Code Quality

### Automated Quality Check

```bash
# Run all quality checks at once
python check_quality.py
```

This will check:

- ✅ Code formatting (black)
- ✅ Import sorting (isort)
- ✅ Type checking (mypy)
- ✅ Code linting (flake8)
- ✅ Unit tests (pytest)
- ✅ Integration tests

### Individual Quality Tools

```bash
# Format code
black *.py

# Sort imports
isort --profile black *.py

# Check types
mypy ascii_path_solver.py

# Lint code
flake8 *.py
```

### Before Committing

Always run the quality checker to ensure your code meets professional standards:

```bash
python check_quality.py
```

If all checks pass, you'll see:

```
🎉 All quality checks passed! Ready to commit.
```

## Development Workflow

### Quick Start

```bash
# 1. Set up environment
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 2. Make your changes
# Edit ascii_path_solver.py or tests

# 3. Check quality before committing
python check_quality.py

# 4. If all checks pass, you're ready to commit!
```

### Continuous Development

```bash
# Format code as you work
black ascii_path_solver.py

# Run tests frequently
python -m pytest test_ascii_path_solver.py -v

# Final check before commit
python check_quality.py
```

## Project Structure

### Core Files

- `ascii_path_solver.py` - Main implementation with modern Python features
- `test_ascii_path_solver.py` - Comprehensive test suite (32 tests)
- `test_integration.py` - Integration tests

### Development Tools

- `check_quality.py` - Automated quality checker
- `requirements.txt` - Development dependencies
- `pyproject.toml` - Modern Python project configuration
- `.flake8` - Code linting configuration
- `.gitignore` - Git ignore rules
