# -*- coding: utf-8 -*-
"""
@author: Chris Hayward
"""


# test clean deathdate

import unittest
import numpy as np
import pandas as pd
import pdb

from pipeline_hes import clean_hes
from pipeline_hes import clean_hes_deathdate
from pipeline_hes import clean_hes_prepare




# #######
# This is for using JUST THE LAST disdate
# #######
class Test_set_death_date_LAST_DISDATE(unittest.TestCase):
    
    def setUp(self):
        self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0],
                                'IS_CONTROL':[False,False,False],
                                'Mortality': [0,0,0],
                                'SURVIVALTIME':[np.nan,np.nan,np.nan],
                                'MYEPISTART_FIRSTAMI':
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                'MYEPIEND':np.repeat(np.datetime64('2005-01-01'),3),
                                'DISDATE':np.repeat(np.datetime64('2005-01-01'),3),})

    
    def test_set_death_date(self):
        self.df['SURVIVALTIME'] = np.array([300,200,10,])
        self.df['Mortality'] = 1

        # mean of latest disdates (20 days)
        df_out = clean_hes_deathdate.set_death_date( self.df.copy())
        death_expected = np.datetime64('2005-01-01')
        
        np.testing.assert_array_equal(df_out['DEATHDATE'],
                                      np.array(np.repeat(death_expected,3),dtype='datetime64[ns]'))


    def test_set_death_date_different_disdate(self):
        times = np.arange('2005-01', '2005-04', dtype='datetime64[M]')
        self.df['DISDATE'] = times
        self.df['SURVIVALTIME'] = np.array([900,600,30,])
        self.df['Mortality'] = 1

        df_out = clean_hes_deathdate.set_death_date(self.df.copy())
        expected_death = np.datetime64('2005-03-01')
        
        np.testing.assert_array_equal(df_out['DEATHDATE'].values,
             np.array(np.repeat(expected_death,3),dtype='datetime64[ns]'))



    def test_set_death_date_full_nan(self):
        self.df['DISDATE'] = np.datetime64(None)
        self.df['MYEPIEND'] = np.datetime64(None)
        self.df['SURVIVALTIME'] = np.array([90,600,300,])
        self.df['Mortality'] = 1

        df_out = clean_hes_deathdate.set_death_date(self.df.copy())
        expected_death = np.datetime64(None)
        
        np.testing.assert_array_equal(df_out['DEATHDATE'].values,
            np.array(np.repeat(expected_death,3),dtype='datetime64[ns]'))


    def test_set_death_date_nan_disdate_only(self):
        self.df['DISDATE'] = np.datetime64(None)
        self.df['SURVIVALTIME'] = np.array([90,600,300,])
        self.df['Mortality'] = 1

        df_out = clean_hes_deathdate.set_death_date(self.df.copy())
        expected_death = np.datetime64('2005-10-01')
        
        #pdb.set_trace()
        np.testing.assert_array_equal(df_out['DEATHDATE'].values,
            np.array(np.repeat(expected_death,3),dtype='datetime64[ns]'))


    def test_set_death_date_part_nan_disdate_only(self):
        times = np.arange('2005-01', '2005-04', dtype='datetime64[M]')
        times[1:3] = np.datetime64(None)
        self.df['DISDATE'] = times
        self.df['SURVIVALTIME'] = np.array([90,600,300,])
        self.df['Mortality'] = 1

        df_out = clean_hes_deathdate.set_death_date(self.df.copy())
        expected_death = np.datetime64('2005-10-01')
        
        np.testing.assert_array_equal(df_out['DEATHDATE'].values,
            np.array(np.repeat(expected_death,3),dtype='datetime64[ns]'))


    def test_set_death_date_nan_survivaltime(self):
        self.df['SURVIVALTIME'] = np.array([900,60,np.nan,])
        self.df['Mortality'] = 1

        df_out = clean_hes_deathdate.set_death_date(self.df.copy())
        expected_death = np.datetime64('2005-03-01')
        
        np.testing.assert_array_equal(df_out['DEATHDATE'].values,
            np.array(np.repeat(expected_death,3),dtype='datetime64[ns]'))

        
    def test_set_death_date_no_death(self):
        self.df['SURVIVALTIME'] = np.array([np.nan,np.nan,np.nan])
        df_out = clean_hes_deathdate.set_death_date(self.df.copy())
        self.assertTrue(df_out['DEATHDATE'].isna().all())


    def test_set_death_date_multiple_subjects(self):
        
        self.df['Mortality'] = 1
        
        df_sub1 = self.df.copy()
        df_sub1['SURVIVALTIME'] = np.array([900,600,32,])

        df_sub2 = self.df.copy()
        df_sub2['ENCRYPTED_HESID'] = 1
        df_sub2['DISDATE'] = np.repeat(np.datetime64('2010-01-01'),3)
        df_sub2['SURVIVALTIME'] = np.array([300,300,0,])        
        
        df_tmp = pd.concat([df_sub1, df_sub2])
        df_out = clean_hes_deathdate.set_death_date(df_tmp)

        expected_death1 = np.datetime64('2005-02-01')
            
        expected_death2 = np.datetime64('2010-01-01')
          
        #pdb.set_trace()
        np.testing.assert_array_equal(
            df_out.loc[df_out['ENCRYPTED_HESID']==0,'DEATHDATE'].values,
            np.array(np.repeat(expected_death1,3),dtype='datetime64[ns]'))
        np.testing.assert_array_equal(
            df_out.loc[df_out['ENCRYPTED_HESID']==1,'DEATHDATE'].values,
            np.array(np.repeat(expected_death2,3),dtype='datetime64[ns]'))
        

class Test_backfill_disdate(unittest.TestCase):
    
    def setUp(self):
        self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0],})

    # ############
    # BACKFILL
    # ############

    def test_backfill_disdate_multiple_subjects_and_spells(self):

        times = np.arange('2005-01', '2005-04', dtype='datetime64[M]')
        
        df1 = self.df.copy()
        df1['ENCRYPTED_HESID'] = [0,0,0]
        df1['DISDATE'] = np.array([np.datetime64(None),
                          np.datetime64(None),
                          np.datetime64('2005-03-01')])
        df1['PROVSPNOPS'] = 'XXX'
        df1['EPIORDER'] = [1,2,3]

        # Set final row as DISDATE=nan
        df2 = self.df.copy()
        df2['ENCRYPTED_HESID'] = [1,1,1]        
        df2['DISDATE'] = np.datetime64(None)
        df2['PROVSPNOPS'] = 'XXX'
        df2['EPIORDER'] = [1,2,3]
        
        
        df3 = self.df.copy()
        df3['ENCRYPTED_HESID'] = [1,1,1]        
        df3['DISDATE'] = np.array([np.datetime64(None),
                          np.datetime64(None),
                          np.datetime64('2010-03-01')])
        df3['PROVSPNOPS'] = 'YYY'
        df3['EPIORDER'] = [1,2,3]


        df4 = self.df.copy()
        df4['ENCRYPTED_HESID'] = [1,1,1]        
        df4['DISDATE'] = np.array([np.datetime64(None),
                          np.datetime64(None),
                          np.datetime64('2011-03-01')])
        df4['PROVSPNOPS'] = 'ZZZ'
        df4['EPIORDER'] = [1,2,3]

        df = pd.concat([df1,df2,df3,df4], ignore_index=True)

        df_out = clean_hes_deathdate.backfill_disdate(df)

        np.testing.assert_array_equal(df_out['DISDATE'].values,
                                      np.array(
                                      np.concatenate(
                                          [np.datetime64('2005-03-01'),
                                          np.datetime64('2005-03-01'),
                                          np.datetime64('2005-03-01'),
                                          np.datetime64(None),
                                          np.datetime64(None),
                                          np.datetime64(None),
                                          np.datetime64('2010-03-01'),
                                          np.datetime64('2010-03-01'),
                                          np.datetime64('2010-03-01'),
                                          np.datetime64('2011-03-01'),
                                          np.datetime64('2011-03-01'),
                                          np.datetime64('2011-03-01'),],axis=None),dtype='datetime64[ns]'))


    def test_backfill_disdate_nan_final(self):
        df1 = self.df.copy()
        df1['DISDATE'] = np.array([np.datetime64(None),
                          np.datetime64(None),
                          np.datetime64('2005-03-01')])
        df1['PROVSPNOPS'] = 'XXX'
        df1['EPIORDER'] = [1,2,3]

        # Set all rows as DISDATE=nan
        df2 = self.df.copy()
        df2['DISDATE'] = np.datetime64(None)
        df2['PROVSPNOPS'] = 'YYY'
        df2['EPIORDER'] = [1,2,3]

        df3 = self.df.copy()
        df3['DISDATE'] = np.array([np.datetime64(None),
                          np.datetime64(None),
                          np.datetime64('2007-03-01')])
        df3['PROVSPNOPS'] = 'ZZZ'
        df3['EPIORDER'] = [1,2,3]
        
        df = pd.concat([df1,df2,df3], ignore_index=True)
        
        df_out = clean_hes_deathdate.backfill_disdate(df.copy())
        
        
        np.testing.assert_array_equal(df_out['DISDATE'].values,
                                      np.array(
                                      np.concatenate(
                                          [np.datetime64('2005-03-01'),
                                          np.datetime64('2005-03-01'),
                                          np.datetime64('2005-03-01'),
                                          np.repeat(np.datetime64(None),3),
                                          np.datetime64('2007-03-01'),
                                          np.datetime64('2007-03-01'),
                                          np.datetime64('2007-03-01'),],axis=None),dtype='datetime64[ns]'))


    def test_backfill_disdate_invalid_disdate(self):
        df1 = self.df.copy()
        df1['DISDATE'] = np.array([np.datetime64(None),
                          np.datetime64(None),
                          np.datetime64('2005-03-01')])
        df1['PROVSPNOPS'] = 'XXX'
        df1['EPIORDER'] = [1,2,3]

        # Bad date
        df2 = self.df.copy()
        df2['DISDATE'] = np.array([np.datetime64(None),
                          np.datetime64(None),
                          np.datetime64('1801-01-01')])
        df2['PROVSPNOPS'] = 'YYY'
        df2['EPIORDER'] = [1,2,3]

        df3 = self.df.copy()
        df3['DISDATE'] = np.array([np.datetime64(None),
                          np.datetime64(None),
                          np.datetime64('2007-03-01')])
        df3['PROVSPNOPS'] = 'ZZZ'
        df3['EPIORDER'] = [1,2,3]
        
        df = pd.concat([df1,df2,df3], ignore_index=True)
        
        df_out = clean_hes_deathdate.backfill_disdate(df.copy())
        #pdb.set_trace()
        np.testing.assert_array_equal(df_out['DISDATE'].values,
                                      np.array(
                                      np.concatenate(
                                          [np.repeat(np.datetime64('2005-03-01'),3),
                                          np.repeat(np.datetime64(None),3),
                                          np.datetime64('2007-03-01'),
                                          np.datetime64('2007-03-01'),
                                          np.datetime64('2007-03-01'),],axis=None),dtype='datetime64[ns]'))




class Test_remove_nan_deathdate(unittest.TestCase):
    
    def setUp(self):
        self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0],})

    def test_remove_nan_deathdate(self):
        
        df1 = self.df.copy()
        
        # deathdate is erroneously NAN for this subject
        df1['DEATHDATE'] = np.datetime64(None)
        df1['Mortality'] = 1
        
        df2 = self.df.copy()
        df2['ENCRYPTED_HESID'] = 1
        df2['DEATHDATE'] = np.datetime64('2005-01-01')
        df2['Mortality'] = 1

        df = pd.concat([df1,df2], ignore_index=True)

        df_out = clean_hes_deathdate.remove_dead_with_nan_deathdate(df.copy())
        pd.testing.assert_frame_equal(df2, df_out)




# class Test_remove_nan_disdate(unittest.TestCase):
    
#     def setUp(self):
#         self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0],})

#     def test_remove_nan_disdate(self):
        
#         df1 = self.df.copy()
        
#         # deathdate is erroneously NAN for this subject
#         df1['DISDATE'] = np.datetime64(None)
#         df1['Mortality'] = 1
        
#         df2 = self.df.copy()
#         df2['ENCRYPTED_HESID'] = 1
#         df2['DISDATE'] = np.datetime64('2005-01-01')
#         df2['Mortality'] = 1

#         df = pd.concat([df1,df2]).reset_index(drop=True)

#         df_out = clean_hes_deathdate.remove_subjects_nan_disdate(df.copy())
#         pd.testing.assert_frame_equal(df2, df_out)




class Test_remove_subjects_with_early_deathdate(unittest.TestCase):
    
    def setUp(self):
        self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0],})

    def test_remove_subjects_with_early_deathdate(self):
        
        df1 = self.df.copy()
        df1['ENCRYPTED_HESID'] = 1
        df1['DISDATE'] = np.datetime64('2005-01-01')
        df1['DEATHDATE'] = np.datetime64('2005-01-01')
        df1['Mortality'] = 1
        
        df2 = self.df.copy()
        df2['ENCRYPTED_HESID'] = 2
        df2['DISDATE'] = np.datetime64('2005-12-01')
        df2['DEATHDATE'] = np.datetime64('2005-11-01')
        df2['Mortality'] = 1
        

        df = pd.concat([df1,df2],ignore_index=True)

        df_out = clean_hes_deathdate.remove_subjects_with_early_deathdate(df.copy())
        #pdb.set_trace()
        pd.testing.assert_frame_equal(df1, df_out)


class Test_set_death_date_main(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0],
                                'Mortality': [0,0,0],
                                'SURVIVALTIME':[np.nan,np.nan,np.nan],
                                'MYEPIEND':
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                'MYEPISTART_FIRSTAMI':
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                'DISDATE': np.array([np.datetime64(None),
                                                     np.datetime64('2005-01-01'),
                                                     np.datetime64('2005-02-01')]),})
    
    def test_set_death_date_main_ami(self):        
        self.df['SURVIVALTIME'] = np.array([60,40,20,])
        self.df['Mortality'] = np.array([1,1,1,])
        self.df['MYEPISTART_FIRSTAMI'] = np.nan
        self.df['IS_CONTROL'] = False

        # call
        df_out = clean_hes_deathdate.set_death_date(self.df.copy())

        df_exp = self.df.copy()
        death_expected = np.datetime64('2005-02-01')
        #np.timedelta64(20,'D') + pd.to_timedelta(28.25/2,'D')
        df_exp['DEATHDATE'] = death_expected
        
        pd.testing.assert_frame_equal(df_exp, df_out)



    def test_set_death_date_main_ctl_no_ami(self):        
        self.df['SURVIVALTIME'] = np.array([60,40,20,])
        self.df['Mortality'] = np.array([1,1,1,])
        self.df['MYEPISTART_FIRSTAMI'] = np.nan
        self.df['IS_CONTROL'] = True

        # call
        df_out = clean_hes_deathdate.set_death_date(self.df.copy())

        df_exp = self.df.copy()
        death_expected = np.datetime64('2005-02-01')
            #np.timedelta64(20,'D') + pd.to_timedelta(28.25/2,'D')
        df_exp['DEATHDATE'] = death_expected
        
        pd.testing.assert_frame_equal(df_exp, df_out)

        
        
    def test_set_death_date_main_ctl_with_ami(self):
        self.df['SURVIVALTIME'] = np.array([60,40,20,])
        # controls with AMI are always set to not-dead
        self.df['Mortality'] = np.array([0,0,0])
        self.df['IS_CONTROL'] = True
        # call
        df_out = clean_hes_deathdate.set_death_date(self.df.copy())

        df_exp = self.df.copy()
        df_exp['DEATHDATE'] = np.datetime64(None)
        
        pd.testing.assert_frame_equal(df_exp, df_out)


    def test_mark_controls_with_ami_as_alive(self):
        # controls with AMI are always set to not-dead
        self.df['IS_CONTROL'] = True
        self.df['Mortality'] = 1
        # call
        df_out = clean_hes_deathdate.mark_controls_with_ami_as_alive(self.df.copy())

        df_exp = self.df.copy()
        df_exp['Mortality'] = 0
        
        pd.testing.assert_frame_equal(df_exp, df_out)

    def test_set_negative_survivaltime_to_zero(self):
        # controls with AMI are always set to not-dead
        self.df['SURVIVALTIME'] = np.array([60,40,-20,])
        # call
        df_tmp = self.df.copy()
        clean_hes_deathdate.set_negative_survivaltime_to_zero(df_tmp)

        df_exp = self.df.copy()
        df_exp['SURVIVALTIME'] = np.array([60,40,0,])
        
        pd.testing.assert_frame_equal(df_exp, df_tmp)


if __name__ == '__main__':
    # Show full dataframes
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.expand_frame_repr', False)
    unittest.main(argv=['-v'],verbosity=3)

