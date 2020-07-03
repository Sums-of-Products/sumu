from distutils.core import setup, Extension
from Cython.Build import cythonize

exts = Extension(
    name='gadget',
    sources=['_gadget.pyx', 'gadget.cpp'],
    language='c++',
    extra_compile_args=['-std=c++11'])

setup(ext_modules=cythonize(exts, language_level="3"))
