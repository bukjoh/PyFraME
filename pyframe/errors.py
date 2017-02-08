# coding=utf-8
"""Blablabla"""

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
