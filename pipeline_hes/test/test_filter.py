# -*- coding: utf-8 -*-
"""
@author: Chris Hayward
"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pdb
import os

from pipeline_hes import filter_hes
from pipeline_hes.params import params




class Test_ignore_invalid_epiorder(unittest.TestCase):
    
    def setUp(self):
        times = np.arange('2005-01', '2005-10', dtype='datetime64[M]')
        
        self.df = pd.DataFrame({'ENCRYPTED_HESID': [0,0,0,1,1,1,2,2,2],
                                'EPIORDER':[1,1,1,
                                            1,1,1,
                                            1,1,1],})
        self.df['IGNORE'] = False
    
    def test_ignore_invalid_epiorder_noChange(self):
        df_tmp = self.df.copy()
        filter_hes.ignore_invalid_epiorder(df_tmp)
        pd.testing.assert_frame_equal(self.df, df_tmp)
        

    def test_remove_subjects_invalid_epiorder_bad1(self):
        df_tmp = self.df.copy()
        df_tmp.loc[0,'EPIORDER'] = '99'
        df_tmp.loc[3,'EPIORDER'] = 'nan'
        df_tmp.loc[6,'EPIORDER'] = np.nan
        
        df_exp = df_tmp.copy()
        
        filter_hes.ignore_invalid_epiorder(df_tmp)
        
        df_exp['IGNORE'] = np.concatenate([
                                    [True,False,False],
                                    [True,False,False],
                                    [True,False,False]],axis=None)        
        pd.testing.assert_frame_equal(df_exp, df_tmp)





class Test_ignore_ignore_epistart_epiend(unittest.TestCase):
    
    def setUp(self):
        times = np.arange('2005-01', '2005-10', dtype='datetime64[M]')
        
        self.df = pd.DataFrame({'ENCRYPTED_HESID': [0,0,0,1,1,1,2,2,2],
                                'MYEPISTART':times,
                                'MYEPIEND':times})
        self.df['IGNORE'] = False
    
    def test_ignore_ignore_epistart_epiend_noChange(self):
        df_tmp = self.df.copy()
        filter_hes.ignore_invalid_epistart_epiend(df_tmp)
        pd.testing.assert_frame_equal(self.df, df_tmp)
        
    def test_ignore_ignore_epistart_epiend_bad_epistart(self):
        self.df.loc[0,'MYEPISTART'] = np.nan
        self.df.loc[3,'MYEPISTART'] = np.datetime64('1801-01-01')
        self.df.loc[6,'MYEPISTART'] = np.datetime64('1800-01-01')
        df_tmp = self.df.copy()
        filter_hes.ignore_invalid_epistart_epiend(df_tmp)
        
        df_expected = self.df.copy()
        df_expected['IGNORE'] = [True,False,False,
                                 True,False,False,
                                 True,False,False]
        pd.testing.assert_frame_equal(df_expected, df_tmp, check_dtype=False)


    def test_ignore_ignore_epistart_epiend_bad_epiend(self):
        self.df.loc[0,'MYEPIEND'] = np.nan
        self.df.loc[3,'MYEPIEND'] = np.datetime64('1801-01-01')
        self.df.loc[6,'MYEPIEND'] = np.datetime64('1800-01-01')
        df_tmp = self.df.copy()
        filter_hes.ignore_invalid_epistart_epiend(df_tmp)
        
        df_expected = self.df.copy()
        df_expected['IGNORE'] = [True,False,False,
                                 True,False,False,
                                 True,False,False]
        pd.testing.assert_frame_equal(df_expected, df_tmp, check_dtype=False)


class Test_ignore_invalid_primary_diag(unittest.TestCase):
    
    def setUp(self):
        times = np.arange('2005-01', '2005-10', dtype='datetime64[M]')
        
        self.df = pd.DataFrame({'ENCRYPTED_HESID': [0,0,0,1,1,1,2,2,2],})
        self.df['IGNORE'] = False

        diags = ['A00','B50','C99',
                  'A00','B50','C99',
                  'A00','B50','C99']
        self.df['DIAG_01'] = diags
        self.df  = self.df.astype({'DIAG_01':'category'})
        for d in params.SEC_DIAG_COLS:
            self.df [d] = diags
            self.df = self.df.astype({d:'category'})
            

    def test_ignore_invalid_primary_diag_nochange(self):
        df_tmp = self.df.copy()
        df_exp = df_tmp.copy()
        filter_hes.ignore_invalid_primary_diag(df_tmp)
        pd.testing.assert_frame_equal(df_tmp, df_exp, check_like=True,
                                      check_categorical=False)


    def test_ignore_invalid_primary_diag_pri(self):
        df_tmp = self.df.copy()

        df_tmp['DIAG_01'] = df_tmp['DIAG_01'].cat.add_categories('R69')
        df_tmp['DIAG_01'] = df_tmp['DIAG_01'].cat.add_categories('R69X')
        df_tmp['DIAG_01'] = df_tmp['DIAG_01'].cat.add_categories('R69X3')
        df_tmp.loc[0,'DIAG_01'] = 'R69'
        df_tmp.loc[3,'DIAG_01'] = 'R69X'
        df_tmp.loc[6,'DIAG_01'] = 'R69X3'
        df_tmp.loc[8,'DIAG_01'] = np.nan        
        df_exp = df_tmp.copy()
        
        filter_hes.ignore_invalid_primary_diag(df_tmp)
        df_exp['IGNORE'] = np.concatenate([
                                    [True,False,False],
                                    [True,False,False],
                                    [True,False,True]],axis=None)
        pd.testing.assert_frame_equal(df_tmp, df_exp, check_like=True,
                                      check_categorical=False)

    def test_ignore_invalid_primary_diag_sec(self):
        df_tmp = self.df.copy()

        df_tmp['DIAG_02'] = df_tmp['DIAG_02'].cat.add_categories('R69')
        df_tmp['DIAG_03'] = df_tmp['DIAG_03'].cat.add_categories('R69X')
        df_tmp['DIAG_19'] = df_tmp['DIAG_19'].cat.add_categories('R69X3')
        df_tmp.loc[0,'DIAG_02'] = 'R69'
        df_tmp.loc[3,'DIAG_03'] = 'R69X'
        df_tmp.loc[6,'DIAG_19'] = 'R69X3'
        df_tmp.loc[8,'DIAG_20'] = np.nan
        
        df_exp = df_tmp.copy()
        
        filter_hes.ignore_invalid_primary_diag(df_tmp)
        # pd.testing.assert_frame_equal(
        #     self.df.loc[self.df['ENCRYPTED_HESID']!=0].reset_index(drop=True),
        #     df_tmp, check_dtype=False, check_categorical=False)
        
        pd.testing.assert_frame_equal(df_tmp, df_exp, check_like=True,
                                      check_categorical=False)


class Test_ignore_irrelevant_events(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0,
                                                    1,1,1,
                                                    2,2,2],
                                'IS_CONTROL':np.concatenate([
                                    np.repeat(False,3),
                                    np.repeat(True,3),
                                    np.repeat(True,3)],axis=None),
                                'MYEPISTART':np.concatenate([
                                    np.arange('2005-01', '2005-04', dtype='datetime64[M]'),
                                    np.arange('2006-01', '2006-04', dtype='datetime64[M]'),
                                    np.arange('2007-01', '2007-04', dtype='datetime64[M]'),],axis=None),
                                'MYEPISTART_FIRSTAMI':np.concatenate([
                                    np.repeat(np.datetime64('2005-03-01'),3),
                                    np.repeat(np.datetime64(None),3),
                                    np.repeat(np.datetime64('2007-03-01'),3)],axis=None),
                                'MATCHED_DATE':np.concatenate([
                                    np.repeat(np.datetime64('2005-03-01'),3),
                                    np.repeat(np.datetime64('2006-02-01'),3),
                                    np.repeat(np.datetime64('2007-02-01'),3)],axis=None),
                                'IGNORE':np.repeat(False,9),
                                })

    def test_controls_ignore_on_after_first_ami(self):
        df_tmp = self.df.copy()
        filter_hes.controls_ignore_on_after_first_ami(df_tmp)
        df_exp = self.df.copy()
        df_exp['IGNORE'] = np.concatenate([
                                    np.repeat(False,3),
                                    np.repeat(False,3),
                                    [False,False,True]],axis=None)
        pd.testing.assert_frame_equal(df_tmp, df_exp, check_like=True)


    def test_ignore_before_index_date(self):
        df_tmp = self.df.copy()
        filter_hes.ignore_before_index_date(df_tmp)
        df_exp = self.df.copy()
        df_exp['IGNORE'] = np.concatenate([
                                    [True,True,False],
                                    [True,False,False],
                                    [True,False,False]],axis=None)
        pd.testing.assert_frame_equal(df_tmp, df_exp, check_like=True)
    
    
    # def test_ignore_admimeth(self):
    #     df_tmp = self.df.copy()
    #     filter_hes.ignore_admimeth(df_tmp)

    #     df_exp = self.df.copy()
    #     df_exp['IGNORE'] = np.concatenate([
    #                                 [False,True,False],
    #                                 [False,False,False],
    #                                 [True,False,False]],axis=None)
        
    #     pd.testing.assert_frame_equal(df_tmp, df_exp, check_like=True)


class Test_ignore_rare_diags(unittest.TestCase):
    
    def setUp(self):
        # #####params.USE_ACUTE_CHRONIC_CODE_FOR_DETERMINING_RARE_DIAGS = False

        nsubs = 10000
        # make a massive dataframe
        self.df = pd.DataFrame({'ENCRYPTED_HESID':np.arange(nsubs),
                                'IS_CONTROL':np.repeat(False,nsubs),
                                'DIAG_01':['A00'] * nsubs,
                                'DIAG_02':['B00'] * nsubs,
                                'IGNORE':np.repeat(False,nsubs)})
        self.df  = self.df.astype({'DIAG_01':'category'})
        self.df  = self.df.astype({'DIAG_02':'category'})
        
        
    @patch('pipeline_hes.params.params.RARE_EVENT_PRC_THRESH',.1)
    def test_ignore_rare_diags_no_rare(self):
        
        df_tmp = self.df.copy()
        df_out = filter_hes.ignore_rare_diags_combined_simple(df_tmp)
        df_out = df_out.sort_values('ENCRYPTED_HESID').reset_index(drop=True)

        df_exp = self.df.copy()      
        pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)


    @patch('pipeline_hes.params.params.RARE_EVENT_PRC_THRESH',.1)
    def test_ignore_rare_diags_with_rare_pri_and_sec(self):
        
        # 9 subjects
        self.df['DIAG_01'] = self.df['DIAG_01'].cat.add_categories('XXX')
        self.df['DIAG_02'] = self.df['DIAG_02'].cat.add_categories('XXX')
        self.df.loc[slice(0,8),'DIAG_02'] = 'XXX'
        self.df.loc[[1],'DIAG_01'] = 'XXX'
        df_tmp = self.df.copy()
                
        df_out = filter_hes.ignore_rare_diags_combined_simple(df_tmp)
        df_out = df_out.sort_values('ENCRYPTED_HESID').reset_index(drop=True)
        df_exp = self.df.copy()
        df_exp.loc[1,'IGNORE'] = True
        #pdb.set_trace()
        pd.testing.assert_frame_equal(df_out.sort_values('ENCRYPTED_HESID'), df_exp, check_like=True)

    @patch('pipeline_hes.params.params.RARE_EVENT_PRC_THRESH',.1)
    def test_ignore_rare_diags_no_rare_pri_and_sec(self):
        
        self.df['DIAG_01'] = self.df['DIAG_01'].cat.add_categories('XXX')
        self.df['DIAG_02'] = self.df['DIAG_02'].cat.add_categories('XXX')
        self.df.loc[slice(0,9),'DIAG_02'] = 'XXX'
        self.df.loc[[1],'DIAG_01'] = 'XXX'
        df_tmp = self.df.copy()
        
        df_out = filter_hes.ignore_rare_diags_combined_simple(df_tmp)        
        df_out = df_out.sort_values('ENCRYPTED_HESID').reset_index(drop=True)
        
        df_exp = self.df.copy()
        #pdb.set_trace()
        pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)


    @patch('pipeline_hes.params.params.RARE_EVENT_PRC_THRESH',1)
    def test_ignore_rare_diags_thresh1_rare(self):
                
        self.df['DIAG_01'] = self.df['DIAG_01'].cat.add_categories('XXX')
        self.df.loc[slice(0,98),'DIAG_01'] = 'XXX'
        df_tmp = self.df.copy()
        df_out = filter_hes.ignore_rare_diags_combined_simple(df_tmp)        
        df_out = df_out.sort_values('ENCRYPTED_HESID').reset_index(drop=True)
        df_exp = self.df.copy()
        df_exp.loc[slice(0,98),'IGNORE'] = True
        pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)

    @patch('pipeline_hes.params.params.RARE_EVENT_PRC_THRESH',1)
    def test_ignore_rare_diags_thresh1_notRare(self):
                
        self.df['DIAG_01'] = self.df['DIAG_01'].cat.add_categories('XXX')
        self.df.loc[slice(0,99),'DIAG_01'] = 'XXX'
        df_tmp = self.df.copy()
        df_out = filter_hes.ignore_rare_diags_combined_simple(df_tmp)        
        df_out = df_out.sort_values('ENCRYPTED_HESID').reset_index(drop=True)
        df_exp = self.df.copy()
        pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)





class Test_ignore_rare_diags_mix_ami_controls(unittest.TestCase):
    
    def setUp(self):

        nsubs = 20000
        # make a massive dataframe
        self.df = pd.DataFrame({'ENCRYPTED_HESID':np.arange(nsubs),
                                'IS_CONTROL':np.concatenate([
                                    np.repeat(False,nsubs/2),
                                    np.repeat(True,nsubs/2)]),
                                'DIAG_01':['A00'] * nsubs,
                                'DIAG_02':['B00'] * nsubs,
                                'IGNORE':np.repeat(False,nsubs)})
        self.df  = self.df.astype({'DIAG_01':'category'})
        self.df  = self.df.astype({'DIAG_02':'category'})
        
        
    @patch('pipeline_hes.params.params.RARE_EVENT_PRC_THRESH',.1)
    def test_ignore_rare_diags_no_rare(self):
        df_tmp = self.df.copy()
        df_out = filter_hes.ignore_rare_diags_combined_simple(df_tmp)
        df_out = df_out.sort_values('ENCRYPTED_HESID').reset_index(drop=True)

        df_exp = self.df.copy()      
        pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)


    @patch('pipeline_hes.params.params.RARE_EVENT_PRC_THRESH',.1)
    def test_ignore_rare_diags_with_rare_pri_and_sec(self):
        
        # 9 subjects
        self.df['DIAG_01'] = self.df['DIAG_01'].cat.add_categories('XXX')
        self.df['DIAG_02'] = self.df['DIAG_02'].cat.add_categories('XXX')
        self.df.loc[slice(0,8),'DIAG_02'] = 'XXX'
        self.df.loc[[1],'DIAG_01'] = 'XXX'
        df_tmp = self.df.copy()
                
        df_out = filter_hes.ignore_rare_diags_combined_simple(df_tmp)
        df_out = df_out.sort_values('ENCRYPTED_HESID').reset_index(drop=True)
        df_exp = self.df.copy()
        df_exp.loc[1,'IGNORE'] = True
        #pdb.set_trace()
        pd.testing.assert_frame_equal(df_out.sort_values('ENCRYPTED_HESID'), df_exp, check_like=True)


    @patch('pipeline_hes.params.params.RARE_EVENT_PRC_THRESH',.1)
    def test_ignore_rare_diags_no_rare_pri_and_sec(self):
        
        self.df['DIAG_01'] = self.df['DIAG_01'].cat.add_categories('XXX')
        self.df['DIAG_02'] = self.df['DIAG_02'].cat.add_categories('XXX')
        self.df.loc[slice(0,9),'DIAG_02'] = 'XXX'
        self.df.loc[[1],'DIAG_01'] = 'XXX'
        df_tmp = self.df.copy()
        
        df_out = filter_hes.ignore_rare_diags_combined_simple(df_tmp)        
        df_out = df_out.sort_values('ENCRYPTED_HESID').reset_index(drop=True)
        
        df_exp = self.df.copy()
        #pdb.set_trace()
        pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)


    @patch('pipeline_hes.params.params.RARE_EVENT_PRC_THRESH',1)
    def test_ignore_rare_diags_thresh1_rare(self):
                
        self.df['DIAG_01'] = self.df['DIAG_01'].cat.add_categories('XXX')
        self.df.loc[slice(0,98),'DIAG_01'] = 'XXX'
        df_tmp = self.df.copy()
        df_out = filter_hes.ignore_rare_diags_combined_simple(df_tmp)        
        df_out = df_out.sort_values('ENCRYPTED_HESID').reset_index(drop=True)
        df_exp = self.df.copy()
        df_exp.loc[slice(0,98),'IGNORE'] = True
        pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)


    @patch('pipeline_hes.params.params.RARE_EVENT_PRC_THRESH',1)
    def test_ignore_rare_diags_thresh1_notRare(self):
                
        self.df['DIAG_01'] = self.df['DIAG_01'].cat.add_categories('XXX')
        self.df.loc[slice(0,99),'DIAG_01'] = 'XXX'
        df_tmp = self.df.copy()
        df_out = filter_hes.ignore_rare_diags_combined_simple(df_tmp)        
        df_out = df_out.sort_values('ENCRYPTED_HESID').reset_index(drop=True)
        df_exp = self.df.copy()
        pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)




class Test_flatten_sec_diags_into_pri_large(unittest.TestCase):
    
    def setUp(self):
        self.nsubs = 100
        # make a massive dataframe
        self.df = pd.DataFrame({'ENCRYPTED_HESID':np.arange(self.nsubs),
                                'DIAG_01':['A00'] * self.nsubs,
                                'DIAG_02':['B00'] * self.nsubs,
                                'ACUTE_01':['A'] * self.nsubs,
                                'ACUTE_02':['C'] * self.nsubs,
                                'EPIORDER':[1] * self.nsubs})


    def test_flatten_sec_diags_to_pri_large(self):
        
        df_out = filter_hes.flatten_sec_diags_into_pri(self.df)

        self.assertEqual(df_out.shape[0],self.nsubs*2)
        self.assertFalse('DIAG_02' in df_out.columns)
        self.assertFalse('ACUTE_02' in df_out.columns)
        
        #pdb.set_trace()
        np.testing.assert_array_equal(
            df_out.loc[0:self.nsubs-1,'DIAG_01'].drop_duplicates(), ['A00'])
        np.testing.assert_array_equal(
            df_out.loc[self.nsubs:,'DIAG_01'].drop_duplicates(), ['B00'])
        
        np.testing.assert_array_equal(
            df_out.loc[0:self.nsubs-1,'ACUTE_01'].drop_duplicates(), ['A'])
        np.testing.assert_array_equal(
            df_out.loc[self.nsubs:,'ACUTE_01'].drop_duplicates(), ['C'])

        np.testing.assert_array_equal(
            df_out.loc[0:self.nsubs-1,'COL_IDX'].drop_duplicates(), 0)
        np.testing.assert_array_equal(
            df_out.loc[self.nsubs:,'COL_IDX'].drop_duplicates(), 1)

        np.testing.assert_array_equal(
            df_out['EPIORDER'].drop_duplicates(), [1])
        np.testing.assert_array_equal(
            df_out['EPIORDER'].isna().sum(), [0])
        


class Test_flatten_sec_diags_into_pri_small(unittest.TestCase):
    
    def setUp(self):
        self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0,0],
                                'EPIORDER':[1,2,3,4],
                                'IGNORE':[False,True,True,False],
                                'DIAG_01':['A00','I21','C00','B00'],
                                'ACUTE_01':['A','A','C','X']})
        for i,col in enumerate(params.SEC_DIAG_COLS):
            self.df[col] = ['A00','I21','C00',np.nan]
            self.df['ACUTE_{:02d}'.format(i+2)] = ['A','A','C','X']


    def test_flatten_sec_diags_into_pri_small(self):
        
        df_out = filter_hes.flatten_sec_diags_into_pri(self.df)

        np.testing.assert_array_equal(sorted(df_out.columns),
                                      sorted(['ACUTE_01', 'AMI', 'COL_IDX', 'DIAG_01', 'ENCRYPTED_HESID',
                                              'EPIORDER', 'IGNORE']))

        np.testing.assert_array_equal(df_out.loc[0:3,'DIAG_01'],['A00','I21','C00','B00'])
        col_idx = 1
        for i in np.arange(4,61,3):
            np.testing.assert_array_equal(df_out.iloc[i:i+3]['DIAG_01'],['A00','I21','C00'])
            np.testing.assert_array_equal(df_out.iloc[i:i+3]['ACUTE_01'],['A','A','C',])
            np.testing.assert_array_equal(df_out.iloc[i:i+3]['EPIORDER'],[1,2,3,])
            np.testing.assert_array_equal(df_out.iloc[i:i+3]['IGNORE'],[False,True,True])
            np.testing.assert_array_equal(df_out.iloc[i:i+3]['COL_IDX'],[col_idx,col_idx,col_idx])
            np.testing.assert_array_equal(df_out.iloc[i:i+3]['AMI'],[False,True,False,])
            col_idx = col_idx + 1
    

class Test_full_runthrough(unittest.TestCase):
    
    def setUp(self):
        # # this is necessary to refresh all param options
        times = np.array([np.datetime64('2005-03-01'),
                          np.datetime64('2005-01-01'),
                          np.datetime64('2005-04-01'),
                          np.datetime64('2005-02-01'),
                          np.datetime64('2010-06-01'),
                          np.datetime64('2010-07-01'),
                          np.datetime64('2010-08-01'),
                          np.datetime64('2005-06-01'),
                          np.datetime64('2005-07-01'),
                          np.datetime64('2005-08-01'),
                          np.datetime64('2005-09-01'),
                          np.datetime64('2005-10-01'),
                          np.datetime64('2005-11-01')])
        matched = np.array([np.datetime64('2005-01-01'),
                            np.datetime64('2005-01-01'),
                            np.datetime64('2005-01-01'),
                            np.datetime64('2005-01-01'),
                            np.datetime64('2010-07-01'),
                            np.datetime64('2010-07-01'),
                            np.datetime64('2010-07-01'),
                            np.datetime64('2005-07-01'),
                            np.datetime64('2005-07-01'),
                            np.datetime64('2005-07-01'),
                            np.datetime64('2005-10-01'),
                            np.datetime64('2005-10-01'),
                            np.datetime64('2005-10-01')])
        firstAMI = np.array([np.datetime64('2005-01-01'),
                            np.datetime64('2005-01-01'),
                            np.datetime64('2005-01-01'),
                            np.datetime64('2005-01-01'),
                            np.datetime64('2010-07-01'),
                            np.datetime64('2010-07-01'),
                            np.datetime64('2010-07-01'),
                            pd.to_datetime(np.nan),
                            pd.to_datetime(np.nan),
                            pd.to_datetime(np.nan),
                            np.datetime64('2005-10-01'),
                            np.datetime64('2005-10-01'),
                            np.datetime64('2005-10-01')])
        deathdate = np.array([
            pd.to_datetime(np.nan),
            pd.to_datetime(np.nan),
            pd.to_datetime(np.nan),
            pd.to_datetime(np.nan),
            np.datetime64('2011-07-01'),
            np.datetime64('2011-07-01'),
            np.datetime64('2011-07-01'),
            np.datetime64('2011-07-01'),
            np.datetime64('2011-07-01'),
            np.datetime64('2011-07-01'),
            pd.to_datetime(np.nan),
            pd.to_datetime(np.nan),
            pd.to_datetime(np.nan),])

        # CONTROL WITH AMI (Alive)
        # Control WITH AMI (Dead - but will be treated as an alive subject)
        # Control without AMI
        # AMI patient
        self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0,0,
                                                    1,1,1,
                                                    2,2,2,
                                                    3,3,3],
                                'MYADMIDATE':times,
                                'DISDATE':times,
                                'MYEPISTART':times,
                                'MYEPIEND':times,
                                'MATCHED_DATE':matched,
                                'MYEPISTART_FIRSTAMI':firstAMI,
                                'DEATHDATE':deathdate,
                                'AMI':[True,False,True,False,
                                        False,False,True,
                                        False,False,False,
                                        False,True,False],
                                'IS_CONTROL':[True,True,True,True,
                                              True,True,True,
                                              True,True,True,
                                              False,False,False],
                                'amiID':[10,10,10,10,
                                          13,13,13,
                                          11,11,11,
                                          12,12,12],
                                'Mortality':[0,0,0,0,
                                              1,1,1,
                                              1,1,1,
                                              0,0,0],
                                'SEX':[0,0,0,0,
                                        0,0,0,
                                        1,1,1,
                                        1,1,1],
                                'MYDOB':np.append(
                                    np.repeat(np.datetime64('2000-01-01'),4),
                                    [np.repeat(np.datetime64('1940-01-01'),3),
                                      np.repeat(np.datetime64('1990-01-01'),3),
                                      np.repeat(np.datetime64('1970-01-01'),3),]),
                                'IMD04':[0,0,0,0,
                                          4,4,4,
                                          3,3,3,
                                          1,1,1],
                                'PROCODE':np.append(
                                    np.repeat('TXX',4),
                                    [np.repeat('MXX',3),
                                      np.repeat('AXX',3),
                                      np.repeat('TXX',3),]),
                                'SURVIVALTIME':[np.nan,np.nan,np.nan,np.nan,
                                                50,30,20,
                                                90,60,30,
                                                np.nan,np.nan,np.nan,],
                                'PROVSPNOPS':['AYY','BYY','CYY','DYY',
                                              'BYY','CYY','DYY',
                                              'CYY','DYY','EYY',
                                              'DYY','EYY','FYY',],
                                'EPIORDER':[1,1,1,1,
                                            1,1,1,
                                            1,1,1,
                                            1,1,1,],
                                'EPISTAT':[3,3,3,3,
                                            3,3,3,
                                            3,3,3,
                                            3,3,3,],
                                'ADMIMETH':['0','0','0','0',
                                            '0','0','0',
                                            '0','0','0',
                                            '0','0','0',]
                                })
        
        # DIAG cols
        diags_pri = ['I21','I22','I23','I211',
                      'A00','B00','i21A',
                      'I24','O11','a000',
                      'A00','B00','N99']
        diag_sec = ['B00','C00','X00','I211',
                    'A00','B00','K24',
                    'J42','O11','a000',
                    'A00','D49','N99']
        self.df ['DIAG_01'] = diags_pri
        self.df  = self.df.astype({'DIAG_01':'category'})
        for d in params.SEC_DIAG_COLS:
            self.df[d] = diag_sec
            self.df = self.df.astype({d:'category'})
    
        # ACUTE cols
        acute_pri = ['A','C','X','A',
                      'A','A','A',
                      'C','C','C',
                      'A','C','A']
        acute_sec = ['A','C','X','A',
                      'A','A','A',
                      'C','C','C',
                      'A','C','A']
        self.df ['ACUTE_01'] = acute_pri
        self.df  = self.df.astype({'ACUTE_01':'category'})
        for d in params.SEC_ACUTE_COLS:
            self.df[d] = acute_sec
            self.df = self.df.astype({d:'category'})
            
    
    
    @patch('pipeline_hes.params.params.LIMIT_TO_TIME_IGNORE_THESE', False)
    def test_full_runthrough(self):
        self._sub_test_full_runthrough()

    @patch('pipeline_hes.params.params.LIMIT_TO_TIME_IGNORE_THESE', '<6m')
    def test_full_runthrough2(self):
        self._sub_test_full_runthrough()
            
    @patch('pipeline_hes.params.params.LIMIT_TO_TIME_IGNORE_THESE', '>6m')
    def test_full_runthrough3(self):
        self._sub_test_full_runthrough()

    @patch('pipeline_hes.params.params.CHAPTER_HEADINGS_USE_GRANULAR', False)
    def test_full_runthrough4(self):
        self._sub_test_full_runthrough()
        
    @patch('pipeline_hes.params.params.CHAPTER_HEADINGS_USE_GRANULAR', True)
    def test_full_runthrough5(self):
        self._sub_test_full_runthrough()
        
    @patch('pipeline_hes.params.params.RARE_EVENT_PRC_THRESH', 0.1)
    def test_full_runthrough6(self):
        self._sub_test_full_runthrough()
        
    @patch('pipeline_hes.params.params.RARE_EVENT_PRC_THRESH', 0.01)
    def test_full_runthrough7(self):
        self._sub_test_full_runthrough()

    @patch('pipeline_hes.params.params.SKIP_SAVING_COUNTS', True)
    def _sub_test_full_runthrough(self):
        (df_out,_) = filter_hes.run_filtering(self.df)
        return df_out




if __name__ == '__main__':
    # Show full dataframes
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.expand_frame_repr', False)
    unittest.main(argv=['-v'],verbosity=3)

