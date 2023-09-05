# -*- coding: utf-8 -*-
"""
@author: Chris Hayward
"""

# test_censoring


import unittest
import numpy as np
import pandas as pd
import pdb

from pipeline_hes import censoring_hes
from pipeline_hes.params import params

from unittest.mock import patch



class Test_add_censor_col(unittest.TestCase):
    
    def setUp(self):
        
        times = np.array([np.datetime64('2005-03-01'),
                          np.datetime64('2005-01-01'),
                          np.datetime64('2005-04-01'),
                          np.datetime64('2005-02-01'),
                          
                          np.datetime64('2005-06-01'),
                          np.datetime64('2005-07-01'),
                          np.datetime64('2005-08-01'),
                          
                          np.datetime64('2005-09-01'),
                          np.datetime64('2005-10-01'),
                          np.datetime64('2005-11-01')])
        
        firstami = np.array([np.datetime64('2005-03-01'),
                          np.datetime64('2005-03-01'),
                          np.datetime64('2005-03-01'),
                          np.datetime64('2005-03-01'),
                          
                          np.datetime64(None),
                          np.datetime64(None),
                          np.datetime64(None),
                          
                          np.datetime64('2005-10-01'),
                          np.datetime64('2005-10-01'),
                          np.datetime64('2005-10-01')])
        
        init = np.array([np.datetime64('2005-01-01'),
                          np.datetime64('2005-01-01'),
                          np.datetime64('2005-01-01'),
                          np.datetime64('2005-01-01'),
                          
                          np.datetime64('2005-07-01'),
                          np.datetime64('2005-07-01'),
                          np.datetime64('2005-07-01'),
                          
                          np.datetime64('2005-10-01'),
                          np.datetime64('2005-10-01'),
                          np.datetime64('2005-10-01')])

        
        # ctl/ctl/ctl/ctl, ctl(no-ami)/ctl(no-ami)/ctl(no-ami), pat/pat/pat
        self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0,0,
                                                   1,1,1,
                                                   2,2,2],
                                'MYEPISTART':times,
                                'AMI':[True,False,True,False,
                                       False,False,False,
                                       False,True,False],
                                'IS_CONTROL':[True,True,True,True,
                                              True,True,True,
                                              False,False,False],
                                'MYEPISTART_FIRSTAMI':firstami,
                                'MATCHED_DATE':init})


        
    def test_add_censor_col(self):
        # set as ALIVE
        self.df['DEATHDATE'] = np.array(np.repeat(np.nan,10),dtype='datetime64[ns]')
        self.df['Mortality'] = 0
        df_tmp = self.df.copy()
        censoring_hes.add_censor_col(df_tmp)
        
        censor_ctl_first_ami = (self.df.loc[0:3,'MYEPISTART_FIRSTAMI']).values
        # assert that the censor col is correct
        np.testing.assert_array_equal(df_tmp['CENSOR'].values,
                                      np.array(
                                          np.concatenate([
                                              censor_ctl_first_ami-np.timedelta64(1,'s'),
                                              np.repeat(np.datetime64('2017-03-27'),3),
                                              np.repeat(np.datetime64('2017-03-27'),3)],axis=None),
                                              dtype='datetime64[ns]'))
        
        
    def test_add_censor_col_dead(self):
        censor_ctl_first_ami = (self.df.loc[0:3,'MYEPISTART_FIRSTAMI']).values
        self.df['DEATHDATE'] = np.concatenate([censor_ctl_first_ami,
                                               np.repeat(np.datetime64('2006-01-01'),3),
                                               np.repeat(np.datetime64('2008-01-01'),3)],axis=None)
        self.df['Mortality'] = 1
        df_tmp = self.df.copy()
        censoring_hes.add_censor_col(df_tmp)   
        
        np.testing.assert_array_equal(df_tmp['CENSOR'].values,
                                      np.array(
                                          np.concatenate([
                                              censor_ctl_first_ami-np.timedelta64(1,'s'),
                                              np.repeat(np.datetime64('2006-01-01'),3)+\
                                                      np.timedelta64(1,'s'),
                                              np.repeat(np.datetime64('2008-01-01'),3)+\
                                                      np.timedelta64(1,'s')],axis=None),
                                              dtype='datetime64[ns]'))



class Test_landmark_alteration(unittest.TestCase):
    

    def setUp(self):
        
        times = np.array([np.datetime64('2005-03-01'),
                          np.datetime64('2005-01-01'),
                          np.datetime64('2005-04-01'),
                          np.datetime64('2005-02-01'),
                          
                          np.datetime64('2005-06-01'),
                          np.datetime64('2005-07-01'),
                          np.datetime64('2005-08-01'),
                          
                          np.datetime64('2005-09-01'),
                          np.datetime64('2005-10-01'),
                          np.datetime64('2005-11-01')])
        #
        init = np.array([np.datetime64('2005-01-01'),
                          np.datetime64('2005-01-01'),
                          np.datetime64('2005-01-01'),
                          np.datetime64('2005-01-01'),
                          
                          np.datetime64('2005-07-01'),
                          np.datetime64('2005-07-01'),
                          np.datetime64('2005-07-01'),
                          
                          np.datetime64('2005-10-01'),
                          np.datetime64('2005-10-01'),
                          np.datetime64('2005-10-01')])
        
        censor_just_before_ami = np.repeat(np.datetime64('2005-03-01'),4)-np.timedelta64(1,'s')
        censor = np.array(
            np.concatenate([censor_just_before_ami,
                            np.repeat(np.datetime64('2017-03-27'),3),
                            np.repeat(np.datetime64('2017-03-27'),3)],axis=None),
            dtype='datetime64[ns]')

        
        # ctl/ctl/ctl/ctl, ctl(no-ami)/ctl(no-ami)/ctl(no-ami), pat/pat/pat
        self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0,0,
                                                   1,1,1,
                                                   2,2,2],
                                'MYADMIDATE':times,
                                'MATCHED_DATE':init,
                                'CENSOR':censor,
                                'Mortality':np.repeat(1,10),
                                'SURVIVALTIME':np.repeat(1,10),
                                'IGNORE':np.repeat(False,10)})

    @patch('pipeline_hes.params.params.LIMIT_TO_TIME_IGNORE_THESE', '<6m')
    def test_landmark_alteration_ignore_less_than_6m(self):
        # Ignore between 0-6 months from INIT
        df_tmp = self.df.copy()
                        
        censoring_hes.landmark_alteration(df_tmp)
        np.testing.assert_array_equal(df_tmp['IGNORE'].values,
                                      np.repeat(True,10))
        


    @patch('pipeline_hes.params.params.LIMIT_TO_TIME_IGNORE_THESE', '<6m')
    def test_landmark_alteration_ignore_less_than_6m_later_dates(self):
        # Ignore between 0-6 months from INIT
        df_tmp = self.df.copy()

        # more than 6months
        df_tmp.loc[[6], 'MYADMIDATE'] = df_tmp.loc[[5], 'MATCHED_DATE'].array + \
            np.timedelta64(28,'W')
        # less than 6months
        df_tmp.loc[[9], 'MYADMIDATE'] = df_tmp.loc[[8], 'MATCHED_DATE'].array + \
            np.timedelta64(24,'W')
        
        censoring_hes.landmark_alteration(df_tmp)

        np.testing.assert_array_equal(df_tmp['IGNORE'].values,
                                      [True,True,True,True,
                                        True,True,False,
                                        True,True,True])


    @patch('pipeline_hes.params.params.LIMIT_TO_TIME_IGNORE_THESE', '>6m')
    def test_landmark_alteration_ignore_after_6m(self):
        # Ignore after 6m from INIT
        df_tmp = self.df.copy()
                
        censoring_hes.landmark_alteration(df_tmp)
        np.testing.assert_array_equal(df_tmp['IGNORE'].values,
                                      np.repeat(False,10))
        
        # changed CENSOR
        exp_censor = np.concatenate([
            df_tmp.loc[0:3, 'CENSOR'].array,
                          np.array([np.datetime64('2006-01-01'),
                          np.datetime64('2006-01-01'),
                          np.datetime64('2006-01-01'),
                          np.datetime64('2006-04-01'),
                          np.datetime64('2006-04-01'),
                          np.datetime64('2006-04-01')])])
        
        np.testing.assert_array_equal(df_tmp['CENSOR'].values,
                                      exp_censor)
        
        # changed Mortality
        exp_mortality = np.concatenate([
            np.repeat(1,4),
                          np.repeat(0,6)])
        np.testing.assert_array_equal(df_tmp['Mortality'].values,
                                      exp_mortality)

        
    @patch('pipeline_hes.params.params.LIMIT_TO_TIME_IGNORE_THESE', '>6m')
    def test_landmark_alteration_ignore_after_6m_later_dates(self):
        # Ignore after 6m from INIT
        df_tmp = self.df.copy()
        
        # more than 6months
        df_tmp.loc[[6], 'MYADMIDATE'] = df_tmp.loc[[5], 'MATCHED_DATE'].array + \
            np.timedelta64(28,'W')
        # less than 6months
        df_tmp.loc[[9], 'MYADMIDATE'] = df_tmp.loc[[8], 'MATCHED_DATE'].array + \
            np.timedelta64(24,'W')
        
        censoring_hes.landmark_alteration(df_tmp)
        np.testing.assert_array_equal(df_tmp['IGNORE'].values,
                                      [False,False,False,False,
                                        False,False,True,
                                        False,False,False])
        # changed CENSOR
        exp_censor = np.concatenate([
            df_tmp.loc[0:3, 'CENSOR'].array,
                          np.array([np.datetime64('2006-01-01'),
                          np.datetime64('2006-01-01'),
                          np.datetime64('2006-01-01'),
                          np.datetime64('2006-04-01'),
                          np.datetime64('2006-04-01'),
                          np.datetime64('2006-04-01')])])
        np.testing.assert_array_equal(df_tmp['CENSOR'].values,
                                      exp_censor)

        # changed Mortality
        exp_mortality = np.concatenate([
            np.repeat(1,4),
                          np.repeat(0,6)])
        np.testing.assert_array_equal(df_tmp['Mortality'].values,
                                      exp_mortality)

        
            
class Test_add_censor_event(unittest.TestCase):
    
    def setUp(self):
        self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0,
                                                   1,1,1],
                                
                                'DUR':[1,1,1,1,1,1],
                                'Mortality':[1,1,1,0,0,0],
                                'INIT_AGE':[40,40,40,40,40,40],
                                'IMD04':[0,0,0,0,0,0],
                                'SEX':[1,1,1,0,0,0],
                                'CENSOR':np.append(
                                    np.repeat(np.datetime64('2006-01-01'),3),
                                    np.repeat(np.datetime64('2011-01-01'),3),),
                                'IS_CONTROL':np.repeat(False,6),
                                'MATCHED_DATE':np.append(
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2010-01-01'),3),),

                                'PROVSPNOPS':[1,1,1,
                                             2,2,2,],
                               'PROCODE':[1,1,1,
                                          2,2,2,],
                                'COL_IDX':np.repeat(np.array(0,dtype=np.uint8),6),
                                'MYEPISTART':np.append(
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2010-01-01'),3),),
                                'MYEPIEND':np.append(
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2010-01-01'),3),),
                                'DISDATE':np.append(
                                    np.repeat(np.datetime64('2005-02-01'),3),
                                    np.repeat(np.datetime64('2010-02-01'),3),),
                                'DIAG_01':['A00','B00','C00',
                                           'A00','B00','X00'],
                                'SURVIVALTIME':[30,20,10,np.nan,np.nan,np.nan],
                                'AMI':np.repeat(False,6),
                                'EPIORDER':np.repeat(np.array(1,dtype=np.uint8),6),
                                'INIT_ROW':np.repeat(False,6),
                                'IGNORE':np.repeat(False,6)})
        self.df = self.df.astype({'DIAG_01':'category'})
    
    
    def test_add_censor_event(self):
        df_out = censoring_hes.add_censor_event(self.df)

        df_exp = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0,0,
                                                  1,1,1,1],
                                'DUR':[1,1,1,1,1,1,1,1],
                                'Mortality':[1,1,1,1,
                                             0,0,0,0],
                                'INIT_AGE':[40,40,40,40,40,40,40,40],
                                'IMD04':[0,0,0,0,0,0,0,0],
                                'SEX':[1,1,1,1,0,0,0,0],
                                'CENSOR':np.concatenate([
                                    np.repeat(np.datetime64('2006-01-01'),4),
                                    np.repeat(np.datetime64('2011-01-01'),4),],axis=None),
                                'IS_CONTROL':np.repeat(False,8),
                                'MATCHED_DATE':np.append(
                                    np.repeat(np.datetime64('2005-01-01'),4),
                                    np.repeat(np.datetime64('2010-01-01'),4),),
                                
                               'PROVSPNOPS':[1,1,1,3,
                                             2,2,2,3,],
                               'PROCODE':[1,1,1,3,
                                          2,2,2,3,],
                               'COL_IDX':[0,0,0,255,0,0,0,255],
                                'MYEPISTART':np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.datetime64('2006-01-01'),
                                    np.repeat(np.datetime64('2010-01-01'),3),
                                    np.datetime64('2011-01-01')],axis=None),
                                'MYEPIEND':np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.datetime64('2006-01-01'),
                                    np.repeat(np.datetime64('2010-01-01'),3),
                                    np.datetime64('2011-01-01')],axis=None),
                                'DISDATE':np.concatenate([
                                    np.repeat(np.datetime64('2005-02-01'),3),
                                    np.datetime64('2006-01-01'),
                                    np.repeat(np.datetime64('2010-02-01'),3),
                                    np.datetime64('2011-01-01')],axis=None),
                                'DIAG_01':['A00','B00','C00',params.CENSOR_CODE,
                                           'A00','B00','X00',params.CENSOR_CODE,],
                                'SURVIVALTIME':[30,20,10,0,
                                                np.nan,np.nan,np.nan,np.nan],

                                'AMI':np.repeat(False,8),
                                'EPIORDER':[1,1,1,255,
                                            1,1,1,255],
                                'INIT_ROW':np.repeat(False,8),
                                'IGNORE':np.repeat(False,8)})
        df_exp = df_exp.astype({'EPIORDER':np.uint8})
        df_exp = df_exp.astype({'COL_IDX':np.uint8})
        df_exp = df_exp.astype({'DIAG_01':'category'})
        #pdb.set_trace()
        pd.testing.assert_frame_equal(
            df_out.sort_values('ENCRYPTED_HESID').reset_index(drop=True),
            df_exp, check_like=True, check_categorical=False)





class Test_add_followup_duration_col(unittest.TestCase):


    def setUp(self):

        times = np.array([np.datetime64('2005-01-01'),
                          np.datetime64('2005-01-01'),
                          np.datetime64('2005-01-01'),
                          np.datetime64('2005-01-01'),
                          
                          np.datetime64('2005-07-01'),
                          np.datetime64('2005-07-01'),
                          np.datetime64('2005-07-01'),
                          
                          np.datetime64('2005-10-01'),
                          np.datetime64('2005-10-01'),
                          np.datetime64('2005-10-01')])
        
        censor_just_before_ami = np.repeat(np.datetime64('2005-03-01'),4)-\
            np.timedelta64(1,'s')
        censor = np.array(
            np.concatenate([censor_just_before_ami,
                            np.repeat(np.datetime64('2017-03-27'),3),
                            np.repeat(np.datetime64('2017-03-27'),3)],axis=None),
            dtype='datetime64[ns]')
        
        # ctl/ctl/ctl/ctl, ctl(no-ami)/ctl(no-ami)/ctl(no-ami), pat/pat/pat
        self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0,0,
                                                   1,1,1,
                                                   2,2,2],
                                'EVENT_INIT':[True,False,False,False,
                                              True,False,False,
                                              True,False,False,],
                                'MATCHED_DATE':times,
                                'CENSOR':censor,})

    def test_add_followup_duration_col(self):
        df_tmp = self.df.copy()
        censoring_hes.add_followup_duration_col(df_tmp)
        
        # 1: '2005-01-01' -> '2005-03-01' = ~2
        # 2: '2005-07-01' -> '2017-03-27' = 11years + 9 months (141)
        # 3: '2005-10-01' -> '2017-03-27' = 11years + 6 months (138)
        #pdb.set_trace()
        
        y1 = ((pd.to_datetime('2005-03-01') - pd.to_timedelta('1s') \
               - pd.to_datetime('2005-01-01')) / \
            np.timedelta64(1,'Y'))
        y2 = ((pd.to_datetime('2017-03-27') - pd.to_datetime('2005-07-01')) / \
            np.timedelta64(1,'Y'))
        y3 = ((pd.to_datetime('2017-03-27') - pd.to_datetime('2005-10-01')) / \
            np.timedelta64(1,'Y'))
        
        np.testing.assert_array_almost_equal(df_tmp['DUR'].values,
                                      np.array([y1,y1,y1,y1,
                                                y2,y2,y2,
                                                y3,y3,y3]))




if __name__ == '__main__':
    # Show full dataframes
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.expand_frame_repr', False)
    unittest.main(argv=['-v'],verbosity=3)
