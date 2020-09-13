from distutils.core import setup, Extension
from Cython.Build import cythonize

exts = Extension(
    name='weight_sum',
    sources=['_weight_sum.pyx', 'weight_sum.cpp'],
    language='c++',
    extra_compile_args=['-std=c++11'])

setup(ext_modules=cythonize(exts, language_level="3"))
