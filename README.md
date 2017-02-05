# PyFraME: Python tool for Fragment-based Multiscale Embedding

## Description

PyFraME is a Python package that provides tools for setting up and running fragment-based multiscale embedding calculations.


## How to cite



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


## Tests

To run the test suite type
```
nosetests
```
from the PyFraME root directory.
