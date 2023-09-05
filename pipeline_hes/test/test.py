# -*- coding: utf-8 -*-
"""
@author: Chris Hayward
"""

import unittest
import pandas as pd

if __name__ == '__main__':
    pd.set_option('mode.chained_assignment','raise')
    test_suite = unittest.defaultTestLoader.discover('.',pattern='test_*.py')
    test_runner = unittest.runner.TextTestRunner(buffer=True, verbosity=2)
    test_runner.run(test_suite)
