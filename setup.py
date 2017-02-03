import os.path as path
import setuptools as st

root_dir = path.abspath(path.dirname(__file__))
with open(path.join(root_dir, 'VERSION'), encoding='utf-8') as version_file:
    version = version_file.read().strip()

st.setup(name='PyFraME',
         version=version,
         description='PyFraME: Python tool for Fragment-based Multiscale Embedding',
         long_description='Long description here...',
         url='https://gitlab.com/FraME-projects/PyFraME',
         author='Jogvan Magnus Haugaard Olsen',
         author_email='foeroyingur@gmail.com',
         license='GPLv3+',
         classifiers=['Development Status :: 2 - Pre-Alpha',
                      'Environment :: Console',
                      'Intended Audience :: Science/Research',
                      'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
                      'Natural Language :: English',
                      'Operating System :: POSIX :: Linux',
                      'Programming Language :: Python :: 3',
                      'Topic :: Scientific/Engineering :: Chemistry',
                      'Topic :: Scientific/Engineering :: Physics'
                      ],
         install_requires=['numpy', 'scipy'],
         packages=['pyframe'],
         package_data={'PyFraME': ['data/*.csv']},
         data_files=['VERSION'],
         entry_points={'console_scripts': ['PyFraME = pyframe.__main__:main']}
         )
