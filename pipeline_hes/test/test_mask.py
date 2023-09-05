# -*- coding: utf-8 -*-
"""
@author: Chris Hayward
"""

# test mask_hes.py

import unittest

import numpy as np
import pandas as pd
import pdb
import importlib
from unittest.mock import patch

from pipeline_hes import mask_hes
from pipeline_hes.params import params



class Test_masks_init(unittest.TestCase):

    
    def setUp(self):
        self.df = pd.DataFrame({'INIT_ROW':np.repeat(False,3),
                                'IS_CONTROL':np.repeat(False,3),
                                'AMI':np.repeat(False,3)})        

    def test_masks_init(self):
        mask_out = mask_hes.init_mask(self.df.copy())
        mask_exp = np.array([False,False,False])
        np.testing.assert_array_equal(mask_out,mask_exp)

    def test_masks_init_change_ami(self):
        self.df['INIT_ROW'] = [False,True,True]
        self.df['AMI'] = [True,True,False]
        mask_out = mask_hes.init_mask(self.df.copy())
        mask_exp = np.array([False,True,False])
        np.testing.assert_array_equal(mask_out,mask_exp)


    def test_masks_init_change_ctl(self):
        self.df['INIT_ROW'] = [False,True,True]
        self.df['AMI'] = [True,True,False]
        self.df['IS_CONTROL'] = [True,True,True]
        mask_out = mask_hes.init_mask(self.df.copy())
        mask_exp = np.array([False,True,True])
        np.testing.assert_array_equal(mask_out,mask_exp)



class Test_ctl_ami_mask(unittest.TestCase):

    
    def setUp(self):
        self.df = pd.DataFrame({'MYEPISTART_FIRSTAMI':np.repeat(pd.to_datetime(np.nan),3),
                                'IS_CONTROL':np.repeat(True,3),})        

    def test_mask_no_change(self):
        mask_out = mask_hes.ctl_ami_mask(self.df.copy())
        mask_exp = np.array([False,False,False])
        np.testing.assert_array_equal(mask_out,mask_exp)


    def test_mask_change(self):
        self.df['MYEPISTART_FIRSTAMI'] = np.repeat(np.datetime64('2005-01-01'),3)
        mask_out = mask_hes.ctl_ami_mask(self.df.copy())
        mask_exp = np.array([True,True,True])
        np.testing.assert_array_equal(mask_out,mask_exp)
        
        



class Test_ignore_before_index_date_mask(unittest.TestCase):

    
    def setUp(self):
        self.df = pd.DataFrame({'MYEPISTART':np.repeat(pd.to_datetime(np.nan),3),
                                'MATCHED_DATE':np.repeat(pd.to_datetime(np.nan),3),})        

    def test_mask_no_change(self):
        mask_out = mask_hes.ignore_before_index_date_mask(self.df.copy())
        mask_exp = np.array([False,False,False])
        np.testing.assert_array_equal(mask_out,mask_exp)


    def test_mask_change(self):
        self.df['MYEPISTART'] = [np.datetime64('2005-01-01'),
                                   np.datetime64('2005-02-01'),
                                   np.datetime64('2005-03-01')]
        self.df['MATCHED_DATE'] = np.repeat(np.datetime64('2005-02-01'),3)
        mask_out = mask_hes.ignore_before_index_date_mask(self.df.copy())
        mask_exp = np.array([True,False,False])
        np.testing.assert_array_equal(mask_out,mask_exp)
        


class Test_controls_ignore_on_after_first_ami_mask(unittest.TestCase):

    
    def setUp(self):
        self.df = pd.DataFrame({'MYEPISTART':np.repeat(pd.to_datetime(np.nan),3),
                                'MYEPISTART_FIRSTAMI':np.repeat(pd.to_datetime(np.nan),3),
                                'IS_CONTROL':np.repeat(True,3),})        

    def test_mask_no_change(self):
        mask_out = mask_hes.controls_ignore_on_after_first_ami_mask(self.df.copy())
        mask_exp = np.array([False,False,False])
        np.testing.assert_array_equal(mask_out,mask_exp)


    def test_mask_change(self):
        self.df['MYEPISTART'] = [np.datetime64('2005-01-01'),
                                   np.datetime64('2005-02-01'),
                                   np.datetime64('2005-03-01')]
        self.df['MYEPISTART_FIRSTAMI'] = np.repeat(np.datetime64('2005-02-01'),3)
        mask_out = mask_hes.controls_ignore_on_after_first_ami_mask(self.df.copy())
        mask_exp = np.array([False,True,True])
        np.testing.assert_array_equal(mask_out,mask_exp)
        

    def test_mask_change2(self):
        self.df['MYEPISTART'] = [np.datetime64('2005-01-01'),
                                   np.datetime64('2005-02-01'),
                                   np.datetime64('2005-03-01')]
        self.df['MYEPISTART_FIRSTAMI'] = np.repeat(np.datetime64('2005-02-01'),3)
        self.df['IS_CONTROL'] = False
        mask_out = mask_hes.controls_ignore_on_after_first_ami_mask(self.df.copy())
        mask_exp = np.array([False,False,False])
        np.testing.assert_array_equal(mask_out,mask_exp)





class Test_ignore_diag_not_A_to_N_mask(unittest.TestCase):

    
    def setUp(self):
        self.df = pd.DataFrame({'DIAG_01':['A00','N99','O00','Z99','N99'],})        

    def test_mask_no_change(self):
        mask_out = mask_hes.ignore_diag_not_A_to_N_mask(self.df.copy())
        mask_exp = np.array([False,False,True,True,False])
        np.testing.assert_array_equal(mask_out,mask_exp)




class Test_invalid_epiorder_mask(unittest.TestCase):

    
    def setUp(self):
        self.df = pd.DataFrame({'EPIORDER':['nan',np.nan,1,'99'],})        

    def test_mask_no_change(self):
        mask_out = mask_hes.invalid_epiorder_mask(self.df.copy())
        mask_exp = np.array([True,True,False,True])
        np.testing.assert_array_equal(mask_out,mask_exp)




class Test_invalid_diag_mask_R69(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame()
        
        diags = ['A00','R69X','O00','R69','N99']
        self.df['DIAG_01'] = diags
        self.df = self.df.astype({'DIAG_01':'category'})
        for d in params.SEC_DIAG_COLS:
            self.df[d] = diags
            self.df = self.df.astype({d:'category'})
            
    def test_mask_no_change(self):
        mask_out = mask_hes.invalid_primary_diag_mask_R69(self.df.copy())
        mask_exp = np.array([False,True,False,True,False])
        np.testing.assert_array_equal(mask_out,mask_exp)

    def test_mask_change(self):
        self.df['DIAG_20'] = 'R69a'
        mask_out = mask_hes.invalid_primary_diag_mask_R69(self.df.copy())
        mask_exp = np.array([False,True,False,True,False])
        np.testing.assert_array_equal(mask_out,mask_exp)





class Test_invalid_diag_mask_diag01_nan(unittest.TestCase):

    
    def setUp(self):
        self.df = pd.DataFrame({'DIAG_01':['A00',np.nan,'nan','Z99','N99'],})        

    def test_mask_no_change(self):
        mask_out = mask_hes.invalid_primary_diag_mask_nan(self.df.copy())
        mask_exp = np.array([False,True,True,False,False])
        np.testing.assert_array_equal(mask_out,mask_exp)


class Test_invalid_date_mask_nan(unittest.TestCase):

    
    def setUp(self):
        self.df = pd.DataFrame({'DUMMY':[pd.to_datetime(np.nan),
                                   np.datetime64('1801-01-02'),
                                   np.datetime64('1801-01-01')],})

    def test_mask_no_change(self):
        mask_out = mask_hes._invalid_date_mask_nan(self.df.copy(),'DUMMY')
        mask_exp = np.array([True,False,True])
        np.testing.assert_array_equal(mask_out,mask_exp)


class Test_invalid_disdate_mask(unittest.TestCase):

    
    def setUp(self):
        self.df = pd.DataFrame({'DISDATE':[pd.to_datetime(np.nan),
                                   np.datetime64('1801-01-02'),
                                   np.datetime64('1801-01-01')],})

    def test_mask_no_change(self):
        mask_out = mask_hes.invalid_disdate_mask(self.df.copy())
        mask_exp = np.array([False,False,True])
        np.testing.assert_array_equal(mask_out,mask_exp)



if __name__ == '__main__':
    # Show full dataframes
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.expand_frame_repr', False)
    unittest.main(argv=['-v'],verbosity=3)


