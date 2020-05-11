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
      url='https://gitlab.com/FraME-projects/PyFraME/',
      project_urls={'Source': 'https://gitlab.com/FraME-projects/PyFraME/',
                    'Issue Tracker': 'https://gitlab.com/FraME-projects/PyFraME/issues',
                    'Zenodo deposits': 'https://doi.org/10.5281/zenodo.775113'},
      author=pyframe.__author__,
      author_email='foeroyingur@gmail.com',
      classifiers=['Intended Audience :: Science/Research',
                   'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
                   'Natural Language :: English',
                   'Programming Language :: Python :: 3.6',
                   'Programming Language :: Python :: 3.7',
                   'Programming Language :: Python :: 3.8',
                   'Topic :: Scientific/Engineering :: Chemistry',
                   'Topic :: Scientific/Engineering :: Physics'
                   ],
      install_requires=['numpy>=1,<2', 'scipy>=1,<2', 'h5py>=2.10,<3'],
      python_requires='>=3.6',
      packages=['pyframe', 'pyframe.tests'],
      package_data={'pyframe': ['data/*.csv']}
      )
