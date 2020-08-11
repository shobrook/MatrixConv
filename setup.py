import sys
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from codecs import open

if sys.version_info[:3] < (3, 0, 0):
    print("Requires Python 3 to run.")
    sys.exit(1)

with open("README.md", encoding="utf-8") as file:
    readme = file.read()

setup(
    name="matrix_conv",
    description="Graph convolutional operator that uses a CNN as a filter",
    long_description=readme,
    long_description_content_type="text/markdown",
    version="v1.0.1",
    packages=["matrix_conv"],
    python_requires=">=3",
    url="https://github.com/shobrook/MatrixConv",
    author="shobrook",
    author_email="shobrookj@gmail.com",
    # classifiers=[],
    install_requires=["torch", "torch_geometric"],
    keywords=[
        "gnn",
        "graph-neural-network",
        "convolution",
        "pytorch",
        "graph",
        "cnn",
        "matrix",
        "image"
    ],
    license="MIT"
)
