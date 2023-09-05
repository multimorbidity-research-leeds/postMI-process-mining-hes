# -*- coding: utf-8 -*-
"""
@author: Chris Hayward
"""
# test csv_to_parquet.py

import unittest
import pandas as pd
import numpy as np
import pdb
import os
from unittest.mock import patch
from pipeline_hes.params import params
from pipeline_hes import csv_to_parquet
from pipeline_hes import load_parquet
from pipeline_hes.test import make_dummy_hes_data

import hashlib

class Test_csv_to_parquet(unittest.TestCase):

    HES_ALL = """NIC17649_APC_0809.txt
    NIC17649_APC_0910.txt
    NIC17649_APC_1011.txt
    NIC17649_APC_1112.txt
    NIC17649_APC_1213.txt
    NIC17649_APC_1314.txt
    NIC17649_APC_1415.txt
    NIC17649_APC_1516.txt
    NIC17649_APC_1617.txt"""

    TYPE_PAIRS = {
                'DIAG_01':'category',
                'DIAG_02':'category','DIAG_03':'category',
                'DIAG_04':'category','DIAG_05':'category',
                'DIAG_06':'category','DIAG_07':'category',
                'DIAG_08':'category','DIAG_09':'category',
                'DIAG_10':'category','DIAG_11':'category',
                'DIAG_12':'category','DIAG_13':'category',
                'DIAG_14':'category','DIAG_15':'category',
                'DIAG_16':'category','DIAG_17':'category',
                'DIAG_18':'category','DIAG_19':'category',
                'DIAG_20':'category',
                'ADMIMETH':'category',
                'DISMETH':'category','PROCODE':'category',
                'MYADMIDATE':'datetime64[ns]',
                'DISDATE':'datetime64[ns]','MYDOB':'datetime64[ns]', 
                'MYEPIEND':'datetime64[ns]','MYEPISTART':'datetime64[ns]',
                'EPIDUR':'float32',
                'IMD04':'float32',
                'SURVIVALTIME':'float32',
                'EPIORDER':'uint8',
                'EPISTAT':'uint8',
                'SEX':'uint8',
                'Mortality':'uint8',
                'PROVSPNOPS':'object',
                'ENCRYPTED_HESID':'object'
                }
    
    
    def setUp(self):
        np.random.seed(1)
        make_dummy_hes_data.main(500)
        self.testFile = os.path.join(params.DIR_TMP,'raw_dummy.csv')


    def make_all_categories_categorical(self,df):
        # This means that parquet will *read* dtype=category
        for col in [x[0] for x in self.TYPE_PAIRS.items() if x[1] == 'category']:
            df[col] = df[col].cat.add_categories(['XXX'])
            df.loc[0,col] = 'XXX'


    def test_write_read_dummy_data(self):
        
        x = csv_to_parquet.load_hes(self.testFile)

        # check types
        pd.testing.assert_series_equal(x.dtypes.astype('str').sort_index(),
                                        pd.Series(self.TYPE_PAIRS.values(),
                                                  index=self.TYPE_PAIRS.keys()).sort_index())
        self.make_all_categories_categorical(x)
        fname = csv_to_parquet.to_parquet(x,'test','000000')
        xBack = pd.read_parquet(fname)
        # check contents before and after writing
        pd.testing.assert_frame_equal(x,xBack)
        # check types
        pd.testing.assert_series_equal(xBack.dtypes.astype('str').sort_index(),
                                        pd.Series(self.TYPE_PAIRS.values(),
                                                  index=self.TYPE_PAIRS.keys()).sort_index())

if __name__ == '__main__':
    # Show full dataframes
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.expand_frame_repr', False)
    unittest.main(argv=['-v'],verbosity=3)

