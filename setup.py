from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import subprocess
import numpy as np
import platform

# Define the extension module
if platform.system() == 'Windows':
    extra_compile_args = ["/openmp"]
else:
    extra_compile_args = ["-fopenmp"]

# Determine Eigen include directory
eigen_include_dir = os.environ.get('EIGEN_INCLUDE_DIR', 'eigen3')

# Define the extension module
ext_modules = [Extension(name="pyframe.embedding.cpp_interaction_tensor_element",
                         sources=["pyframe/embedding/interaction_tensor_element.cpp"],
                         include_dirs=[np.get_include(), eigen_include_dir],  # Add eigen3 directory here
                         extra_compile_args=extra_compile_args,
                         extra_link_args=extra_compile_args,
                         language="c++")]

# Eigen version to download
EIGEN_VERSION = "3.3.9"


# Define a custom build_ext command to download Eigen and include it in the build
class CustomBuildExtCommand(build_ext):
    def run(self):
        if 'EIGEN_INCLUDE_DIR' not in os.environ:
            # Download Eigen if not already present
            eigen_dir = os.path.join(self.build_lib, "eigen3")
            eigen_url = f"https://gitlab.com/libeigen/eigen/-/archive/{EIGEN_VERSION}/eigen-{EIGEN_VERSION}.tar.gz"
            eigen_tar_path = os.path.join(
                self.build_temp, f"eigen-{EIGEN_VERSION}.tar.gz")

            if not os.path.exists(eigen_dir):
                print(f"Downloading Eigen {EIGEN_VERSION}...")
                subprocess.run(["curl", "-v", "-L", eigen_url, "-o",
                                eigen_tar_path], check=True)

                # Extract the downloaded tar.gz file
                subprocess.run(["tar", "xz", "-C", self.build_temp,
                                "-f", eigen_tar_path], check=True)

                # Move the extracted Eigen directory to eigen_dir
                extracted_eigen_dir = os.path.join(self.build_temp, f"eigen-{EIGEN_VERSION}")
                os.rename(extracted_eigen_dir, eigen_dir)

            # Add the Eigen directory to include_dirs
            self.include_dirs.append(eigen_dir)

        # Continue with the build
        super().run()


# Setup configuration
setup(
    name="pyframe.embedding.cpp_interaction_tensor_element",
    #cmdclass={"build_ext": CustomBuildExtCommand},
    ext_modules=ext_modules,
    install_requires=["numpy"],
)
