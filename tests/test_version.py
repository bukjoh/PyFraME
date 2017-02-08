# coding=utf-8
"""Test version module"""

import unittest

import os

import pyframe


class TestVersion(unittest.TestCase):

    def setUp(self):
        root_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir)
        with open(os.path.join(root_dir, 'VERSION'), encoding='utf-8') as version_file:
            self.version = version_file.read().strip()

    def test_version(self):
        self.assertIsInstance(pyframe.__version__, str)
        self.assertEqual(pyframe.__version__, self.version)

if __name__ == '__main__':
    unittest.main()
