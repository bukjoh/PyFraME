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

__all__ = ['UnknownOptionError', 'PDBError', 'PropertyNotSetError']


class UnknownOptionError(Exception):

    """Exception for class input"""

    def __init__(self, prop):
        self.prop = prop

    def __str__(self):
        return 'unknown option "{0}"'.format(self.prop)


class PDBError(Exception):

    """Exception for errors in PDB file"""

    def __init__(self, prop, line):
        self.prop = prop
        self.line = line

    def __str__(self):
        return 'unable to read {0} from:\n{1}'.format(self.prop, self.line)


class PropertyNotSetError(Exception):

    """Exception for Fragment and Atom classes"""

    def __init__(self, prop):
        self.prop = prop

    def __str__(self):
        return '{0} has not been set'.format(self.prop)
