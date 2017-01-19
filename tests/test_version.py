"""Test PyFraME.version.py"""

import unittest


class TestVersion(unittest.TestCase):

    def test_is_string(self):
        self.assertIsInstance(version.__version__, str)

if __name__ == '__main__':
    unittest.main()
