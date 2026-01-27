import os
from setuptools import setup, find_packages

# Helper to read requirements.txt
def read_requirements():
    if os.path.exists("requirements.txt"):
        with open("requirements.txt") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

def parse_requirements(filename):
    with open(filename, 'r') as f:
        # Read lines, strip whitespace, filter out comments and pip flags
        return [
            line.strip() 
            for line in f 
            if line.strip() 
            and not line.startswith('#') 
            and not line.startswith('-')
        ]

setup(
    name="MOSAICapp",
    version="0.1.0",
    description="MOSAICapp: Application for Mapping of Subjective Accounts into Interpreted Clusters",
    author="Romy Beauté",
    author_email="r.beaut@sussex.ac.uk",
    url="https://github.com/romybeaute/MOSAICapp",
    packages=find_packages(),  # find 'mosaic_core' automatically
    include_package_data=True,
    install_requires=parse_requirements("requirements.txt"),  # reads requirements.txt
    python_requires=">=3.9",
)