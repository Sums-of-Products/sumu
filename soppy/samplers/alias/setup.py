from distutils.core import setup, Extension
from Cython.Build import cythonize

exts = Extension("alias",
                 sources=["_alias.pyx",
                          "discrete_random_variable.cpp"],
                 language='c++',
                 extra_compile_args=['-std=c++17'])

setup(ext_modules=cythonize(exts, language_level="3"))
