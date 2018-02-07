# coding=utf-8
import os.path as path
import setuptools as st

root_dir = path.abspath(path.dirname(__file__))
with open(path.join(root_dir, 'VERSION'), encoding='utf-8') as version_file:
    version = version_file.read().strip()

with open(path.join(root_dir, 'README.md'), encoding='utf-8') as readme:
    long_description = readme.read()

st.setup(name='PyFraME',
         version=version,
         description='PyFraME: Python tools for Fragment-based Multiscale Embedding',
         long_description=long_description,
         url='https://gitlab.com/FraME-projects/PyFraME',
         author='Jógvan Magnus Haugaard Olsen',
         author_email='foeroyingur@gmail.com',
         license='GPLv3+',
         classifiers=['Intended Audience :: Science/Research',
                      'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
                      'Natural Language :: English',
                      'Programming Language :: Python :: 3',
                      'Topic :: Scientific/Engineering :: Chemistry',
                      'Topic :: Scientific/Engineering :: Physics'
                      ],
         install_requires=['numpy', 'numba'],
         python_requires='>=3',
         packages=['pyframe'],
         package_data={'pyframe': ['data/*.csv']},
         data_files=['VERSION'],
         )
