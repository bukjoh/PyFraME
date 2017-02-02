# PyFraME: Python tool for Fragment-based Multiscale Embedding

## Description

PyFraME is a Python package that provides tools for setting up and running fragment-based multiscale embedding calculations.


## How to cite



## Requirements

To use PyFraME you need:
 - Python 3
 - NumPy
 - SciPy

For certain functionality you will need one or more of the following:
 - Dalton (http://www.daltonprogram.org)
 - LoProp for Dalton (https://github.com/vahtras/loprop)
 - Molcas 8 (http://www.molcas.org)

To run the test suite you need (note that currently there are very few tests):
 - nose


## Installation

To install PyFrame 
```
git clone https://gitlab.com/FraME-projects/PyFraME.git
cd PyFrame
python setup.py install [--user]
```
Note that this will install NumPy and Scipy (which can take a while) if they are not installed already.


## Tests

To run the test suite type
```
nosetests
```
from the PyFraME root directory.
