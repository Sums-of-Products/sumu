from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

exts = [

    Extension(
        "sumppy.scorer",
        sources=["sumppy/scorer/_scorer.pyx"],
        include_dirs=["sumppy/scorer"],
        language='c++',
        extra_compile_args=['-std=c++11', '-Wall', '-O3']),

    Extension(
        "sumppy.CandidateRestrictedScore",
        sources=["sumppy/CandidateRestrictedScore/_CandidateRestrictedScore.pyx",
                 "sumppy/zeta_transform/zeta_transform.cpp"],
        include_dirs=["sumppy/CandidateComplementScore"],
        language='c++',
        extra_compile_args=['-std=c++17']),

    Extension(
        "sumppy.DAGR",
        sources=["sumppy/DAGR/_DAGR.pyx",
                 "sumppy/zeta_transform/zeta_transform.cpp"],
        include_dirs=["sumppy/DAGR"],
        language='c++',
        extra_compile_args=['-std=c++17']),

    Extension(
        name='sumppy.zeta_transform',
        sources=['sumppy/zeta_transform/_zeta_transform.pyx',
                 'sumppy/zeta_transform/zeta_transform.cpp'],
        language='c++',
        extra_compile_args=['-std=c++11']),

    # This should not be exposed really.
    # I don't know yet how to fix.
    Extension(
        name='sumppy.weight_sum',
        sources=['sumppy/weight_sum/_weight_sum.pyx',
                 'sumppy/weight_sum/weight_sum.cpp'],
        language='c++',
        extra_compile_args=['-std=c++11']),

    Extension(
        "sumppy.alias",
        sources=["sumppy/alias/_alias.pyx",
                 "sumppy/alias/discrete_random_variable.cpp"],
        language='c++',
        extra_compile_args=['-std=c++17'])

]

setup(
    name="sumppy",
    author="Jussi Viinikka",
    author_email="jussi.viinikka@helsinki.fi",
    packages=["sumppy"],
    # Pygobnilp scoring requires scipy, sklearn, pandas, numba.
    # With better scoring can get rid of these requirements.
    install_requires=["numpy", "scipy", "sklearn", "pandas", "numba"],
    ext_modules=cythonize(exts, language_level="3")
)
