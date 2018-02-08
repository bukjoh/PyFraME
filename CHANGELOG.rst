Change Log
==========

The format is based on `Keep a Changelog <http://keepachangelog.com/>`_
and this project adheres to `Semantic
Versioning <http://semver.org/>`_.

[Unreleased]
------------

[0.1.1] - 2018-02-08
--------------------

Changed
~~~~~~~

-  There is now a check for empty output files from fragment
   calculations to prevent deletion of subdirectories of failed fragment
   calculations
-  Now hydrogen caps between core region and other regions are always
   used
-  Efficiency-related changes:

   -  compute angles, distances, distance matrices using Numba jit
      decorator (introduces Numba dependency and removes SciPy
      dependency)
   -  removed some unnecessary property decorators

-  Refactored the atoms module (faster and safer)

Fixed
~~~~~

-  Bug in MOLCAS LoProp ('Fragment' object has no attribute 'xyz')
-  Using Dalton LoProp with multipole orders lower than two no longer
   fails
-  Bug in charge redistribution where negative surplus charge would not
   be redistributed

[0.1.0] - 2017-02-20
--------------------

Added
~~~~~

-  Initial version
