from distutils.core import setup, Extension
from Cython.Build import cythonize


exts = [

    Extension(
        name='soppy.exact.zeta_transform',
        sources=['soppy/exact/zeta_transform/_zeta_transform.pyx',
                 'soppy/exact/zeta_transform/zeta_transform.cpp'],
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
    install_requires=["numpy"],
    ext_modules=cythonize(exts, language_level="3")
)
