"""The analysis library must be importable without Streamlit."""

import subprocess
import sys

import pytest

MODULES = ["mosaic_core", "mosaic_core.zeroshot", "mosaic_core.comparison", "mosaic_core.metrics"]


@pytest.mark.parametrize("module", MODULES)
def test_import_does_not_load_streamlit(module):
    code = (
        "import sys\n"
        "sys.modules['streamlit'] = None  # any 'import streamlit' now raises ImportError\n"
        f"import {module}\n"
        "assert 'bertopic' not in sys.modules, 'light modules should not import BERTopic'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_core_functions_do_not_import_streamlit():
    code = (
        "import sys\n"
        "sys.modules['streamlit'] = None\n"
        "import mosaic_core.core_functions\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
