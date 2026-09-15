# Contributing to MOSAICapp

Thank you for your interest in contributing to MOSAICapp! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with:

1. A clear, descriptive title
2. Steps to reproduce the problem
3. What you expected to happen
4. What actually happened
5. Your environment (Python version, OS, browser if using the web app)
6. If possible, a minimal CSV file that reproduces the issue

### Suggesting Features

Feature requests are welcome! Please open an issue with:

1. A clear description of the feature
2. Why it would be useful for phenomenological research
3. Any ideas for implementation (optional)

### Seeking Support

If you have questions about using MOSAICapp:

1. First, check the README and existing issues for answers
2. Open a new issue with the label "question"
3. Describe what you're trying to do and where you're stuck

### Contributing Code

1. **Fork** the repository
2. **Clone** your fork locally
3. **Create a branch** for your changes: `git checkout -b feature/your-feature-name`
4. **Make your changes** and test them
5. **Commit** with a clear message describing what you changed
6. **Push** to your fork
7. **Open a Pull Request** with a description of your changes

#### Running Tests

Before submitting a pull request, please run the tests:

```bash
# Install the package with test dependencies
pip install -e ".[dev]"

# Run the fast, offline tests
CI=true pytest tests/ -v

# Run integration tests (slow, requires internet)
pytest tests/test_integration.py -v
```

New analysis methods belong in `mosaic_core` (no Streamlit imports) with tests in
`tests/`; `app.py` should only call them and display the results.

#### Code Style

- Follow PEP 8 guidelines
- Add docstrings to new functions
- Include type hints where practical

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to create a welcoming environment for researchers from all backgrounds.

## Questions?

Feel free to open an issue if anything is unclear. We appreciate your contributions!