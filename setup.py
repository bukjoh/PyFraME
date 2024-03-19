from setuptools import setup, Extension
import os
import numpy as np
import platform

# Define the extension module
if platform.system() == 'Windows':
    extra_compile_args = ["/openmp"]
    extra_link_args = []
else:
    extra_compile_args = ["-fopenmp"]
    extra_link_args = ["-fopenmp"]

# Determine Eigen include directory
eigen_include_dir = os.environ.get('EIGEN_INCLUDE_DIR', 'eigen3')

# Define the extension module
ext_modules = [Extension(name="pyframe.embedding.engine",
                         sources=["pyframe/embedding/engine/engine.cpp",
                                  "pyframe/embedding/engine/computation.cpp",
                                  "pyframe/embedding/engine/global.cpp"],
                         include_dirs=[np.get_include(), eigen_include_dir],
                         extra_compile_args=extra_compile_args,
                         extra_link_args=extra_link_args,
                         language="c++")]

# Setup configuration
setup(
    ext_modules=ext_modules,
)
