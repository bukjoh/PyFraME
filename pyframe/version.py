# coding=utf-8
#
# Copyright (C) 2017-2018  Jógvan Magnus Haugaard Olsen
#
# This file is part of PyFraME.
#
# PyFraME is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyFraME is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyFraME.  If not, see <https://www.gnu.org/licenses/>.
#

"""Read version from VERSION file"""

import os

root_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(root_dir, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()
