# -*- coding: utf-8 -*-
"""

Run the pipeline on a subset of the HES data.

@author: Chris Hayward
"""

from unittest.mock import patch

import unittest
import pdb
import pandas as pd
import datetime
import os

from pipeline_hes import clean_hes
from pipeline_hes import filter_hes
from pipeline_hes import traces_hes
from pipeline_hes.params import params

from pipeline_hes import csv_to_parquet
from pipeline_hes import load_parquet

def save_part(num_MI_subs,hes_df):
    """From the HES data, save a portion of it to run the pipeline on."""

    df = pd.read_parquet(hes_df)

    pat_ids = df.loc[~df['IS_CONTROL'],'ENCRYPTED_HESID'].\
        drop_duplicates().iloc[:num_MI_subs]
    df_pat = df.merge(pat_ids,how='right')

    # get matching ctls (using amiID)
    matchingIds = df_pat['amiID'].drop_duplicates()

    df_ctl = df.loc[df['IS_CONTROL']].merge(matchingIds)
    df = pd.concat([df_pat, df_ctl],ignore_index=True)

    df.to_parquet(
        os.path.join(params.DIR_TMP,'MATCHED_BEFORE_CLEANING_{}.gzip'.\
                  format(params.R)),compression='gzip')



@patch('pipeline_hes.params.params.R','test_subset')
@patch('pipeline_hes.params.params.SKIP_SAVING_COUNTS', False)
@patch('pipeline_hes.params.params.CHECK_PROP_HAZ_ASSUM', True)
@patch('pipeline_hes.params.params.DIR_CHECKPOINTS',params.DIR_TMP)
def main(doPlots=True):

    # Save a small bit of the data for testing
    
    # Save a subset of the HES data
    save_part(2000,
              'n:/HES_CVEPI/chrish/checkpoints/MATCHED_BEFORE_CLEANING_7304891.gzip')
    
    params.DIR_RESULTS = params.DIR_RESULTS.strip('/')+'_test_subset'
    
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
