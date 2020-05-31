from distutils.core import setup, Extension
import os

import numpy

os.environ["CC"] = "g++"

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

setup(ext_modules=[Extension("_gadget",
                             sources=["gadget.cpp", "gadget.i"],
                             swig_opts=['-c++', '-py3'],
                             include_dirs=[numpy_include, '.'],
                             extra_compile_args=['-std=c++17'])])
