# coding=utf-8
import os.path as path
from setuptools import setup

import pyframe

with open(path.join(path.abspath(path.dirname(__file__)), 'README.md'), encoding='utf-8') as readme_file:
    long_description = readme_file.read()

setup(name='PyFraME',
      version=pyframe.__version__,
      description='PyFraME: Python framework for Fragment-based Multiscale Embedding',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://gitlab.com/FraME-projects/PyFraME',
      download_url='https://pypi.org/project/PyFraME/',
      author='Jógvan Magnus Haugaard Olsen',
      author_email='foeroyingur@gmail.com',
      license='GPLv3+',
      classifiers=['Intended Audience :: Science/Research',
                   'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
                   'Natural Language :: English',
                   'Programming Language :: Python :: 3.6',
                   'Programming Language :: Python :: 3.7',
                   'Topic :: Scientific/Engineering :: Chemistry',
                   'Topic :: Scientific/Engineering :: Physics'
                   ],
      install_requires=['numpy', 'scipy', 'h5py'],
      python_requires='>=3.6',
      packages=['pyframe', 'pyframe.tests'],
      package_data={'pyframe': ['data/*.csv']}
      )
