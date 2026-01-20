# How to run the tests

#### Install pytest if not already installed
pip install pytest

#### Run all tests
pytest tests/ -v

#### Run specific test file (eg test_core_functions.py)
pytest tests/test_core_functions.py -v


# Which tests to run:

### Test utility functions (fast)
pytest tests/test_core_functions.py -v

### Integration tests (slow, run occasionally):

- Without LLM tests
pytest tests/test_integration.py -v

- With LLM tests (needs token)
HF_TOKEN=your_token pytest tests/test_integration.py -v


