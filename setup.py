from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

exts = [

    Extension(
        "sumu.scorer",
        sources=["sumu/scores/_scorer.pyx"],
        include_dirs=["sumu/scores", numpy_include],
        language='c++',
        extra_compile_args=['-std=c++11', '-Wall', '-O3']),

    Extension(
        "sumu.CandidateRestrictedScore",
        sources=["sumu/CandidateRestrictedScore/_CandidateRestrictedScore.pyx",
                 "sumu/zeta_transform/zeta_transform.cpp"],
        include_dirs=["sumu/CandidateComplementScore"],
        language='c++',
        extra_compile_args=['-std=c++17']),

    Extension(
        "sumu.DAGR",
        sources=["sumu/DAGR/_DAGR.pyx",
                 "sumu/zeta_transform/zeta_transform.cpp"],
        include_dirs=["sumu/DAGR", numpy_include],
        language='c++',
        extra_compile_args=['-std=c++17']),

    Extension(
        name='sumu.zeta_transform',
        sources=['sumu/zeta_transform/_zeta_transform.pyx',
                 'sumu/zeta_transform/zeta_transform.cpp'],
        language='c++',
        extra_compile_args=['-std=c++11']),

    # This should not be exposed really.
    # I don't know yet how to fix.
    Extension(
        name='sumu.weight_sum',
        sources=['sumu/weight_sum/_weight_sum.pyx',
                 'sumu/weight_sum/weight_sum.cpp'],
        language='c++',
        extra_compile_args=['-std=c++11']),

    Extension(
        "sumu.alias",
        sources=["sumu/alias/_alias.pyx",
                 "sumu/alias/discrete_random_variable.cpp"],
        language='c++',
        extra_compile_args=['-std=c++17'])

]

setup(
    name="sumu",
    author="Jussi Viinikka",
    author_email="jussi.viinikka@helsinki.fi",
    packages=["sumu"],
    install_requires=["numpy", "scipy"],
    ext_modules=cythonize(exts, language_level="3")
)
