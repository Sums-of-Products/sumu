from distutils.core import setup, Extension
import os

os.environ["CC"] = "g++"

setup(ext_modules=[Extension("_zeta_transform",
                             sources=["zeta_transform.cpp", "zeta_transform.i"],
                             swig_opts=['-c++', '-py3'],
                             extra_compile_args=['-std=c++17'])])
