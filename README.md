# PyFraME: Python tools for Fragment-based Multiscale Embedding

## Description

PyFraME is a Python package that provides tools for setting up and running fragment-based multiscale embedding calculations.
The workflow of such calculations is as follows:
 1. a part of the total molecular system is chosen as the core region which is typically treated a high level of theory
 2. the remainder is split into a number of regions each of which can be treated at different levels of theory
 3. each region (except the core) is divided into fragments that consist of either
    - small molecules
    - or parts of larger molecules that have been fragmented into smaller computationally manageable fragments
 4. a calculation is run on each fragment to obtain fragment parameters (if necessary)
 5. all fragment parameters of all regions are assembled and constitute the embedding potential
 6. a final calculation is run on the core region using the embedding potential to model the effect from the remainder of the molecular system


## How to cite

 J. M. H. Olsen, *PyFraME: Python tools for Fragment-based Multiscale Embedding (version X.X.X)*, **2017**, https://gitlab.com/FraME-projects/PyFraME.

Bibtex entry:
```
@misc{pyframe,
	author = {J. M. H. Olsen},
	title = {{PyFraME}: {P}ython tools for {F}ragment-based {M}ultiscale {E}mbedding (version X.X.X)},
	year = {2017},
	note = {https://gitlab.com/FraME-projects/PyFraME}
}
```


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

To install PyFrame you can either download the source and install manually:
```
git clone https://gitlab.com/FraME-projects/PyFraME.git
cd PyFrame
python setup.py install
```
where you may need to add `--user` in the last line if you do not have root access / sudo rights.
Note that this will install NumPy and Scipy if they are not installed already (which can take a while).
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
