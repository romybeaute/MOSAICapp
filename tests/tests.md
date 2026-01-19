### How to run the tests

# Install pytest if not already installed
pip install pytest

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_utils.py -v

# Run with coverage (optional)
pip install pytest-cov
pytest tests/ -v --cov=.