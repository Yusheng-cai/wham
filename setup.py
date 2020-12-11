from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [Extension("wham.Bwham",["wham/Bwham.pyx"],\
            include_dirs=[numpy.get_include()], \
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]), \
            Extension("wham.Uwham",["wham/Uwham.pyx"],\
             include_dirs=[numpy.get_include()], \
             define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]), \
            Extension("wham.lib.numeric",["wham/lib/numeric.pyx"],\
             include_dirs=[numpy.get_include()], \
             define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]), \
            Extension("wham.lib.utils",["wham/lib/wham_utils.pyx"],\
             include_dirs=[numpy.get_include()], \
             define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])]
             
setup(name='wham',ext_modules = cythonize(extensions,compiler_directives={'language_level' : "3"}))
