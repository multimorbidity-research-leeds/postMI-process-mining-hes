# -*- coding: utf-8 -*-
"""
Run the pipeline on a dummy dataset.

@author: Chris Hayward
"""

# Test FULL pipeline.

from unittest.mock import patch

import unittest
import pdb
import datetime
import pandas as pd
import os
import numpy as np

from pipeline_hes import clean_hes
from pipeline_hes import filter_hes
from pipeline_hes import traces_hes
from pipeline_hes.params import params

from pipeline_hes import csv_to_parquet
from pipeline_hes import load_parquet
from pipeline_hes.test import make_dummy_hes_data



@patch('pipeline_hes.params.params.R','test_dummy')
@patch('pipeline_hes.params.params.SKIP_SAVING_COUNTS', False)
@patch('pipeline_hes.params.params.CHECK_PROP_HAZ_ASSUM', True)
@patch('pipeline_hes.params.params.FILE_HESID_PATIENT', \
      os.path.join(params.DIR_TMP,'raw_dummy_amiIDs.csv'))
@patch('pipeline_hes.params.params.FILE_HESID_CONTROLS', \
      os.path.join(params.DIR_TMP,'raw_dummy_ctlIDs.csv'))
@patch('pipeline_hes.params.params.DIR_CHECKPOINTS',params.DIR_TMP)
@patch('pipeline_hes.params.params.CONTROL_CASE_RATIO',5)
def main(doPlots=True):
    
    np.random.seed(1)
    make_dummy_hes_data.main(2000)

    # read dummy csv data...
    file = os.path.join(params.DIR_TMP,'raw_dummy.csv')
    x = csv_to_parquet.load_hes(file)
    fname = csv_to_parquet.to_parquet(x,'NIC17649_APC_0000.txt','test_dummy')

    params.DIR_RESULTS = params.DIR_RESULTS.strip('/')+'_test_dummy'

    # merge with AMI/CONTROL IDs
    load_parquet.main()

    clean_hes.main()
    filter_hes.main()
    traces_hes.main(doPlots)


if __name__ == '__main__':
    # Show full dataframes
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('mode.chained_assignment','raise')
    main()
