import numpy
from Cython.Build import cythonize
from setuptools import setup

setup(
    name='Hex Game',
    ext_modules=cythonize(["modules.pyx"]),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
    annotate=True
)