from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

exts = [

    Extension(
        name='soppy.exact.zeta_transform',
        sources=['soppy/exact/zeta_transform/_zeta_transform.pyx',
                 'soppy/exact/zeta_transform/zeta_transform.cpp'],
        language='c++',
        extra_compile_args=['-std=c++11']),

    # This should not be exposed really.
    # I don't know yet how to fix.
    Extension(
        name='soppy.samplers.gadget',
        sources=['soppy/samplers/gadget/_gadget.pyx',
                 'soppy/samplers/gadget/gadget.cpp'],
        language='c++',
        extra_compile_args=['-std=c++11']),

    Extension("soppy.samplers.alias",
              sources=["soppy/samplers/alias/_alias.pyx",
                       "soppy/samplers/alias/discrete_random_variable.cpp"],
              language='c++',
              extra_compile_args=['-std=c++17'])

]

setup(
    name="soppy",
    author="Jussi Viinikka",
    author_email="jussi.viinikka@helsinki.fi",
    packages=["soppy", "soppy.exact", "soppy.samplers"],
    # Pygobnilp scoring requires scipy, sklearn, pandas, numba.
    # With better scoring can get rid of these requirements.
    install_requires=["numpy", "scipy", "sklearn", "pandas", "numba"],
    ext_modules=cythonize(exts, language_level="3")
)
