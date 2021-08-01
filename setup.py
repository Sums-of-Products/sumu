import sys
import os
import pathlib
import setuptools
from distutils.core import setup, Extension
from distutils.command.clean import clean as Clean
import distutils.util
import shutil
from Cython.Build import cythonize
import numpy


# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text(encoding='utf-8')


COMPILE_OPTIONS = list()
LINK_OPTIONS = list()
COMPILER_DIRECTIVES = dict()
DEFINE_MACROS = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

COMPILER_DIRECTIVES['profile'] = True
COMPILER_DIRECTIVES['linetrace'] = True

# To allow coverage analysis for Cython modules
if os.environ.get("CYTHON_TRACE") == "1":
    DEFINE_MACROS.append(("CYTHON_TRACE", "1"))

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
    COMPILE_OPTIONS += ["-std=c++14", "-Wall", "-O3"]


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


# Copied from scikit-learn
class CleanCommand(Clean):
    description = "Remove build artifacts from the source tree"

    def run(self):
        Clean.run(self)
        # Remove c files if we are not within a sdist package
        cwd = os.path.abspath(os.path.dirname(__file__))
        remove_c_files = not os.path.exists(os.path.join(cwd, 'PKG-INFO'))
        if remove_c_files:
            print('Will remove generated .c files')
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk('sumu'):
            for filename in filenames:
                if any(filename.endswith(suffix) for suffix in
                       (".so", ".pyd", ".dll", ".pyc")):
                    os.unlink(os.path.join(dirpath, filename))
                    continue
                extension = os.path.splitext(filename)[1]
                if remove_c_files and extension in ['.c', '.cpp']:
                    pyx_file = str.replace(filename, extension, '.pyx')
                    if os.path.exists(os.path.join(dirpath, pyx_file)):
                        os.unlink(os.path.join(dirpath, filename))
            for dirname in dirnames:
                if dirname == '__pycache__':
                    shutil.rmtree(os.path.join(dirpath, dirname))


cmdclass = {'clean': CleanCommand}


exts = [

    Extension(
        "sumu.scorer",
        sources=["sumu/scores/_scorer.pyx"],
        include_dirs=["sumu/scores", numpy_include],
        language='c++',
        define_macros=DEFINE_MACROS,
        extra_compile_args=COMPILE_OPTIONS,
        extra_link_args=LINK_OPTIONS),

    Extension(
        name='sumu.weight_sum',
        sources=['sumu/weight_sum/_weight_sum.pyx',
                 'sumu/weight_sum/CandidateRestrictedScore.cpp',
                 'sumu/weight_sum/GroundSetIntersectSums.cpp',
                 'sumu/weight_sum/IntersectSums.cpp',
                 'sumu/weight_sum/Breal.cpp',
                 'sumu/weight_sum/common.cpp'],
        include_dirs=['sumu/weight_sum', numpy_include],
        language='c++',
        define_macros=DEFINE_MACROS,
        extra_compile_args=COMPILE_OPTIONS,
        extra_link_args=LINK_OPTIONS),

    Extension(
        "sumu.mcmc_moves",
        sources=["sumu/_mcmc_moves.pyx"],
        include_dirs=[numpy_include],
        language='c++',
        define_macros=DEFINE_MACROS,
        extra_compile_args=COMPILE_OPTIONS,
        extra_link_args=LINK_OPTIONS),

    Extension(
        name='sumu.aps',
        sources=['sumu/aps/_aps.pyx',
                 'sumu/aps/aps-0.9.1/aps/simple_modular.cpp'],
        include_dirs=["sumu/aps/aps-0.9.1/aps", numpy_include],
        language='c++',
        define_macros=DEFINE_MACROS,
        extra_compile_args=COMPILE_OPTIONS,
        extra_link_args=LINK_OPTIONS)

]

setup(
    name="sumu",
    version="0.1.1",
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
        "scipy>=1.6"
    ],
    cmdclass=cmdclass,
    ext_modules=cythonize(exts, language_level="3", compiler_directives=COMPILER_DIRECTIVES)
)
