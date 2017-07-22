# Change Log
The format is based on [Keep a Changelog](http://keepachangelog.com/) and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]

### Changed
- There is now a check for empty output files from fragment calculations to prevent deletion of subdirectories of failed fragment calculations
- Now hydrogen caps between core region and other regions are always used

### Fixed
- Bug in MOLCAS LoProp ('Fragment' object has no attribute 'xyz')
- Using Dalton LoProp with multipole orders lower than two no longer fails

## [0.1.0] - 2017-02-20

### Added
- Initial version
