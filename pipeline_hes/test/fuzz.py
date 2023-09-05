# -*- coding: utf-8 -*-
"""
@author: Chris Hayward
"""


import pandas as pd
import numpy as np
import random
import pdb

from pipeline_hes.params import params
from pipeline_hes.test import make_dummy_hes_data
from pipeline_hes.test import pipeline_test


### fuzz test

while True:
    make_dummy_hes_data.main()
    try:
        pipeline_test.main(doPlots=False)
    except Exception as e:
        print(e)
        break
    