import sys
import os
import pathlib
import setuptools
from distutils.core import setup, Extension
import distutils.util
from Cython.Build import cythonize
import numpy

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text(encoding='utf-8')


COMPILE_OPTIONS = []
LINK_OPTIONS = []


if os.name == "nt":
    # This is for Windows.
    # Assumes MSVC, which doesn't seem to support c++11:
    # https://docs.microsoft.com/en-us/cpp/build/reference/std-specify-language-standard-version?view=msvc-160
    COMPILE_OPTIONS += ["/std:c++14", "/W4", "/O2", "/Zc:__cplusplus"]
if os.name == "posix":
    # This is for OS X and Linux.
    # Assumes GNU (compatible?) compiler.
    # Trying to use the oldest possible standard for maximum
    # compatibility. Maybe even older would be possible with current
    # extensions.
    COMPILE_OPTIONS += ["-std=c++11", "-Wall", "-O3"]


def is_new_osx():
    """Check whether we're on OSX >= 10.10"""
    name = distutils.util.get_platform()
    if sys.platform != "darwin":
        return False
    elif name.startswith("macosx-10"):
        minor_version = int(name.split("-")[1].split(".")[1])
        if minor_version >= 7:
            return True
        else:
            return False
    else:
        return False


if is_new_osx():
    # On Mac, use libc++ because Apple deprecated use of libstdc
    COMPILE_OPTIONS.append("-stdlib=libc++")
    LINK_OPTIONS.append("-lc++")
    # g++ (used by unix compiler on mac) links to libstdc++ as a default lib.
    # See: https://stackoverflow.com/questions/1653047/avoid-linking-to-libstdc
    LINK_OPTIONS.append("-nodefaultlibs")


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
        extra_compile_args=COMPILE_OPTIONS,
        extra_link_args=LINK_OPTIONS),

    Extension(
        "sumu.CandidateRestrictedScore",
        sources=["sumu/CandidateRestrictedScore/_CandidateRestrictedScore.pyx",
                 "sumu/zeta_transform/zeta_transform.cpp"],
        include_dirs=["sumu/CandidateComplementScore"],
        language='c++',
        extra_compile_args=COMPILE_OPTIONS,
        extra_link_args=LINK_OPTIONS),

    Extension(
        "sumu.DAGR",
        sources=["sumu/DAGR/_DAGR.pyx",
                 "sumu/zeta_transform/zeta_transform.cpp"],
        include_dirs=["sumu/DAGR"],
        language='c++',
        extra_compile_args=COMPILE_OPTIONS,
        extra_link_args=LINK_OPTIONS),

    Extension(
        name='sumu.zeta_transform',
        sources=['sumu/zeta_transform/_zeta_transform.pyx',
                 'sumu/zeta_transform/zeta_transform.cpp'],
        language='c++',
        extra_compile_args=COMPILE_OPTIONS,
        extra_link_args=LINK_OPTIONS),

    # This should not be exposed really.
    # I don't know yet how to fix.
    Extension(
        name='sumu.weight_sum',
        sources=['sumu/weight_sum/_weight_sum.pyx',
                 'sumu/weight_sum/weight_sum.cpp'],
        language='c++',
        extra_compile_args=COMPILE_OPTIONS,
        extra_link_args=LINK_OPTIONS),

    Extension(
        name='sumu.aps',
        sources=['sumu/aps/_aps.pyx',
                 'sumu/aps/aps-0.9.1/aps/simple_modular.cpp'],
        include_dirs=["sumu/aps/aps-0.9.1/aps", numpy_include],
        language='c++',
        extra_compile_args=COMPILE_OPTIONS,
        extra_link_args=LINK_OPTIONS)

]

setup(
    name="sumu",
    version="0.1.0",
    description="Library for working with probabilistic and causal graphical models",
    long_description=README,
    long_description_content_type="text/markdown",
    classifiers=['Intended Audience :: Science/Research',
                 'Intended Audience :: Developers',
                 'License :: OSI Approved',
                 'Programming Language :: C++',
                 'Programming Language :: Cython',
                 'Programming Language :: Python',
                 'Topic :: Software Development',
                 'Topic :: Scientific/Engineering',
                 'Development Status :: 2 - Pre-Alpha',
                 # How do I know if only POSIX or Unix applies?
                 'Operating System :: POSIX',
                 'Operating System :: Unix',
                 'Operating System :: MacOS',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 ('Programming Language :: Python :: '
                  'Implementation :: CPython')
                 ],
    url="https://github.com/jussiviinikka/sumu",
    author="Jussi Viinikka",
    author_email="jussi.viinikka@helsinki.fi",
    license="BSD",
    packages=["sumu", "sumu.utils", "sumu.scores"],
    install_requires=[
        "numpy",
        "scipy"
    ],
    ext_modules=cythonize(exts, language_level="3")
)
