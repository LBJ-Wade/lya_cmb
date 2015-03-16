#-----------------------------------------------------------------------------
#
# Author: Hongyu Zhu
# Date: 22 Feb 2015
#-----------------------------------------------------------------------------

from distutils.core import setup
from distutils.extension import Extension
# from Cython.Distutils import build_ext
from Cython.Build import cythonize

extra_args = []
# Comment/Uncomment the following line to disable/enable OpenMP for GCC-ish
# compilers.
# extra_args = ["-fopenmp"]

exts = [Extension("log_xc_mesh",
                  ["log_xc_mesh.pyx"],
                  extra_compile_args=extra_args,
                  extra_link_args=extra_args),
#        Extension("julia_cython_solution",
#                  ["julia_cython_solution.pyx"],
#                  extra_compile_args=extra_args,
#                  extra_link_args=extra_args),
        ]

setup(
    name = "chain",
#    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize(exts),
)
