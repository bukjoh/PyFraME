# Change Log

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]

### Added

- Pipfile added for use with pipenv
- Print license (`pyframe.print_license()`)
- Added [contributing guidelines](https://gitlab.com/FraME-projects/PyFraME/blob/master/CONTRIBUTING.MD)

### Fixed

- Settings, i.e. method, basis set, etc., when computing polarizabilities
  separately from multipoles using Dalton LoProp have been fixed
- Fixed bug that in some cases resulted in corrupt core regions when it consisted of two
  fragments or more

### Changed

- Minimum requirement changed to Python 3.6
- Allow extracting fragments with a single item, e.g. system.get\_fragments\_by\_chain\_id('P'),
  where it was required to always use lists before
- Updated PFP protein parameters


## 0.2.0 - 2018-20-03

### Added

- Moved the suggested BibTeX snippet into the [CITATION
  file](https://gitlab.com/FraME-projects/PyFraME/blob/master/CITATION)
- Added sodium and chloride charge and isotropic polarizability to
  solvent embedding potential
- Added calcium, magnesium, potassium, zinc and bromine charge and
  isotropic polarizability to solvent embedding potential
- New option to choose exclusion type for standard potentials,
  currently it can be either 'fragment', which excludes all
  interactions within a fragment, or 'mfcc', which is the MFCC type
  exclusions.
- Added AMBER ff94 and ff03 standard potentials
- Improve element recognition
- Surplus charge of a region treated using MFCC is now redistributed
  among all sites in the affected region, ensuring that the sum of
  partial charges is equal to the formal charge of the region
- Added averaged lipid embedding parameters from S. Witzke et al.
  J. Comput. Chem. 38 (2017) 601-611.

### Fixed

- Fixed wrong asserts which would have caused errors if octopoles or
  higher were requested
- All polarizabilities up to octopole-octopole are now present
- Errors when using CAM-B3LYP in serial Dalton calculations (which now
  always use `.DIRECT`)
- Error when LD\_LIBRARY\_PATH environment variable was not set
- Fixed erroneous exclusion lists when different parts of a molecule,
  e.g. protein, are placed in different regions
- Radius of Li, Na, Mg, K, and Ca (used in bond detection) adjusted to
  the effective ionic radius by R. D. Shannon, Acta. Cryst. A32 (1976), 751
- Naming bug when adding hydrogen-link atoms where the donor-atom name
  would erroneously have _link_ appended to it

### Changed

- Running the test suite now requires pytest (unittests can be run by typing `pytest --pyargs pyframe`)
- The default bond detection threshold factor has been set to 1.15 instead of 1.2
- Improved error message when there is a mismatch between the number of sites in
  an output file compared to the number of sites in the fragment read in from input
- Default is now to first try to find a free communication port

## 0.1.1 - 2018-02-08

### Changed

- There is now a check for empty output files from fragment
  calculations to prevent deletion of subdirectories of failed
  fragment calculations
- Now hydrogen caps between core region and other regions are always
  used
- Efficiency-related changes:
  - compute angles, distances, distance matrices using Numba jit
    decorator (introduces Numba dependency and removes SciPy dependency)
  - removed some unnecessary property decorators
- Refactored the atoms module (faster and safer)

### Fixed

- Bug in MOLCAS LoProp ('Fragment' object has no attribute 'xyz')
- Using Dalton LoProp with multipole orders lower than two no longer
  fails
- Bug in charge redistribution where negative surplus charge would not
  be redistributed

## 0.1.0 - 2017-02-20

### Added

- Initial version

