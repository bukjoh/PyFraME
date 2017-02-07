# coding=utf-8
"""Test PyFraME.version.py"""

import unittest

import pyframe.version as version


class TestVersion(unittest.TestCase):

    def test_is_string(self):
        self.assertIsInstance(version.__version__, str)

if __name__ == '__main__':
    unittest.main()
