# coding=utf-8
"""Read version from VERSION file"""

import os

root_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir)
with open(os.path.join(root_dir, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()
