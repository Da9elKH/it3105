from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='Hex Game',
    ext_modules=cythonize(["rust_mcts.pyx", "rust_hex.py"]),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)