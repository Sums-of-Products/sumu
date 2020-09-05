from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

exts = [

    Extension(
        name='sumppy.exact.zeta_transform',
        sources=['sumppy/exact/zeta_transform/_zeta_transform.pyx',
                 'sumppy/exact/zeta_transform/zeta_transform.cpp'],
        language='c++',
        extra_compile_args=['-std=c++11']),

    # This should not be exposed really.
    # I don't know yet how to fix.
    Extension(
        name='sumppy.samplers.gadget',
        sources=['sumppy/samplers/gadget/_gadget.pyx',
                 'sumppy/samplers/gadget/gadget.cpp'],
        language='c++',
        extra_compile_args=['-std=c++11']),

    Extension("sumppy.samplers.alias",
              sources=["sumppy/samplers/alias/_alias.pyx",
                       "sumppy/samplers/alias/discrete_random_variable.cpp"],
              language='c++',
              extra_compile_args=['-std=c++17'])

]

setup(
    name="sumppy",
    author="Jussi Viinikka",
    author_email="jussi.viinikka@helsinki.fi",
    packages=["sumppy", "sumppy.exact", "sumppy.samplers"],
    # Pygobnilp scoring requires scipy, sklearn, pandas, numba.
    # With better scoring can get rid of these requirements.
    install_requires=["numpy", "scipy", "sklearn", "pandas", "numba"],
    ext_modules=cythonize(exts, language_level="3")
)
