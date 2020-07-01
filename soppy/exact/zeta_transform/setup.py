from distutils.core import setup, Extension
from Cython.Build import cythonize

exts = Extension(
    name='zeta_transform',
    sources=['_zeta_transform.pyx', 'zeta_transform.cpp'],
    language='c++',
    extra_compile_args=['-std=c++11'])

setup(ext_modules=cythonize(exts, language_level="3"))
