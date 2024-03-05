from setuptools import Extension, setup
import numpy

setup(setup_requires=["numpy"],
      ext_modules=[
          Extension(
              name="pyframe.embedding.cpp_interaction_tensor_element",
              sources=["pyframe/embedding/interaction_tensor_element.cpp"],
              include_dirs=[numpy.get_include()],
          ),
      ]
      )
