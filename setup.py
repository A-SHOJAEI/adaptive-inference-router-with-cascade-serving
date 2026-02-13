"""Setup script for Adaptive Inference Router with Cascade Serving."""

from pathlib import Path
from setuptools import setup, find_packages

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
def read_requirements(filename):
    """Read requirements from a file."""
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()
                if line.strip() and not line.startswith('#')]

requirements = read_requirements('requirements.txt')

# Extract version from package
def get_version():
    """Get version from package __init__.py."""
    init_file = this_directory / "src" / "adaptive_inference_router_with_cascade_serving" / "__init__.py"
    with open(init_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return "0.1.0"

setup(
    name="adaptive-inference-router-with-cascade-serving",
    version=get_version(),
    author="Research Team",
    description="A research-grade adaptive inference routing system for model cascades",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.6.0",
            "pre-commit>=3.5.0",
        ],
        "gpu": [
            "pynvml>=11.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "adaptive-router-train=adaptive_inference_router_with_cascade_serving.training.trainer:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml", "*.md", "*.txt"],
    },
    zip_safe=False,
)