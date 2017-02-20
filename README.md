# PyFraME: Python tools for Fragment-based Multiscale Embedding calculations

## Description

PyFraME is a Python package that provides tools for setting up and running fragment-based multiscale embedding calculations.
The aim is to provide tools that can automatize the workflow of such calculations in a flexible manner. The typical workflow is as follows:
 1. a part of the total molecular system is chosen as the core region which is typically treated a high level of theory
 2. the remainder is split into a number of regions each of which can be treated at different levels of theory
 3. each region (except the core) is divided into fragments that consist of either
    - small molecules
    - or parts of larger molecules that have been fragmented into smaller computationally manageable fragments
 4. a calculation is run on each fragment to obtain fragment parameters (if necessary)
 5. all fragment parameters of all regions are assembled and constitute the embedding potential
 6. a final calculation is run on the core region using the embedding potential to model the effect from the remainder of the molecular system


## How to cite

To cite PyFraME please use a format similar to the following

"J. M. H. Olsen, *PyFraME: Python tools for Fragment-based Multiscale Embedding (version 0.1.0)*, **2017**, https://doi.org/10.5281/zenodo.293765"

where the version and DOI should of course correspond to the actual version that was used.

Bibtex entry:
```
@misc{pyframe,
	author = {Olsen, J. M. H.},
	title = {{PyFraME}: {P}ython tools for {F}ragment-based {M}ultiscale {E}mbedding (version X.X.X)},
	year = {2017},
	note = {https://doi.org/10.5281/zenodo.293765}}
```
Alternatively, BibTeX and other formats can be generated here: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.293765.svg)](https://doi.org/10.5281/zenodo.293765)


## Requirements

To use PyFraME you need:
 - [Python 3](http://www.python.org)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](http://scipy.org/)

For certain functionality you will need one or more of the following:
 - [Dalton](http://www.daltonprogram.org)
 - [LoProp for Dalton](https://github.com/vahtras/loprop)
 - [Molcas 8](http://www.molcas.org)

To run the test suite you need (note that currently there are very few tests):
 - [nose](http://nose.readthedocs.io/en/latest/)


## Installation

The source can be downloaded from [GitLab](https://gitlab.com/FraME-projects/PyFraME) or [Zenodo deposit](http://dx.doi.org/10.5281/zenodo.293765). Alternatively, you can clone the repository
```
git clone https://gitlab.com/FraME-projects/PyFraME.git
```
The package is installed by running
```
python setup.py install
```
from the PyFraME root directory. Yu may wish to add `--user` in the last line if you do not have root access / sudo rights.
Note that this will install NumPy and SciPy if they are not installed already (which can take a while).
If python3 is not your default python version, change the last command to:
```
python3 setup.py install
```

## Tests

To run the test suite type
```
nosetests
```
from the PyFraME root directory. If python3 is not your default python version, type:
```
nosetests3
```
or
```
nosetest-3
```
depending on your specific setup.

