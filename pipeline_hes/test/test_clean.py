# -*- coding: utf-8 -*-
"""
@author: Chris Hayward
"""

import unittest
# from unittest.mock import patch
import numpy as np
import pandas as pd
import pdb

from pipeline_hes import clean_hes
from pipeline_hes import clean_hes_prepare
from pipeline_hes.params import params

from unittest.mock import patch
        




# class Test_remove_invalid_epistart(unittest.TestCase):
    
#     def setUp(self):
#         times = np.arange('2005-01', '2005-10', dtype='datetime64[M]')
#         self.df = pd.DataFrame({'ENCRYPTED_HESID': [0,0,0,1,1,1,2,2,2],
#                                 'MYEPISTART':times,
#                                 'MYEPIEND':times,
#                                 'PROVSPNOPS':np.arange(9,dtype=float),
#                                 'EPIORDER':np.arange(9,dtype=float)})
#         self.df['IGNORE'] = False
        

#     def test_ignore_invalid_epistart_nochange(self):
#         df_tmp = self.df.copy()
#         df_exp = df_tmp.copy()
#         clean_hes.remove_rows_nan_provspnops_epistart_epiend_epiorder(df_tmp)
#         pd.testing.assert_frame_equal(df_exp, df_tmp)
        

#     def test_ignore_invalid_epistart(self):
#         df_tmp = self.df.copy()
#         df_tmp.loc[0,'MYEPISTART'] = np.nan
#         df_exp = df_tmp.copy()
#         clean_hes.remove_rows_nan_provspnops_epistart_epiend_epiorder(df_tmp)
#         pd.testing.assert_frame_equal(self.df.loc[1:], df_tmp)

#     def test_ignore_invalid_epiend(self):
#         df_tmp = self.df.copy()
#         df_tmp.loc[0,'MYEPIEND'] = np.nan
#         df_exp = df_tmp.copy()
#         clean_hes.remove_rows_nan_provspnops_epistart_epiend_epiorder(df_tmp)
#         pd.testing.assert_frame_equal(self.df.loc[1:], df_tmp)

#     def test_ignore_invalid_provspnops(self):
#         df_tmp = self.df.copy()
#         df_tmp.loc[0,'PROVSPNOPS'] = np.nan
#         df_exp = df_tmp.copy()
#         clean_hes.remove_rows_nan_provspnops_epistart_epiend_epiorder(df_tmp)
#         pd.testing.assert_frame_equal(self.df.loc[1:], df_tmp)
        
#     def test_ignore_invalid_epiorder(self):
#         df_tmp = self.df.copy()
#         df_tmp.loc[0,'EPIORDER'] = np.nan
#         df_exp = df_tmp.copy()
#         clean_hes.remove_rows_nan_provspnops_epistart_epiend_epiorder(df_tmp)
#         pd.testing.assert_frame_equal(self.df.loc[1:], df_tmp)



class Test_remove_unfinished_episodes(unittest.TestCase):
    
    def test_remove_unfinished_episodes(self):
        df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0,1,1,1,2,2,2],
                           'EPISTAT':[1,3,3,3,3,3,1,1,1]
                               })

        df_expected = pd.DataFrame({'ENCRYPTED_HESID':[0,0,1,1,1],
                                    },index=[1,2,3,4,5])
        
        clean_hes.remove_unfinished_episodes(df)
        pd.testing.assert_frame_equal(df_expected, df)



class Test_set_init_procode_imd04(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0,1,1,1,2,2,2],
                               'IS_CONTROL':[1,1,1,1,1,1,0,0,0],
                               'AMI':[False,False,False,
                                      False,False,False,
                                      False,True,False],
                               'MYEPISTART':
                                   np.concatenate(
                                              [np.datetime64('2005-01-01'),
                                              np.datetime64('2005-02-01'),
                                              np.datetime64('2005-03-01'),
                                              np.datetime64('2005-04-01'),
                                              np.datetime64('2005-05-01'),
                                              np.datetime64('2005-06-01'),
                                              np.datetime64('2005-07-01'),
                                              np.datetime64('2005-07-01'),
                                              np.datetime64('2005-07-01'),],axis=None),
                               'INIT_ROW':[True,False,False,
                                           True,False,False,
                                           True,True,False],
                                  'PROCODE':[0,4,2,
                                             3,4,7,
                                             2,1,9],
                                  'IMD04':[0,4,2,
                                           3,4,7,
                                           2,1,9],
                               })
        self.df = self.df.astype({'PROCODE':float})
        self.df = self.df.astype({'IMD04':float})


    def test_set_init_procode_nochange(self):
        df_in = self.df.copy()
        df_out = clean_hes.set_init_procode(df_in)
        df_expected = self.df.copy()
        df_expected['PROCODE'] = [0,0,0,3,3,3,1,1,1]
        df_expected = df_expected.astype({'PROCODE':float})
                
        pd.testing.assert_frame_equal(df_expected, df_out)
        
        
    def test_set_init_procode(self):
        df_in = self.df.copy()
        df_in.loc[0,'PROCODE'] = np.nan
        df_in.loc[8,'PROCODE'] = np.nan # non-AMI event, so kept

        df_out = clean_hes.set_init_procode(df_in)
        
        df_expected = self.df.copy()
        df_expected['PROCODE'] = [np.nan,np.nan,np.nan,3,3,3,1,1,1]
        df_expected = df_expected.astype({'PROCODE':float})
        
        pd.testing.assert_frame_equal(df_expected, df_out)
        
    def test_set_init_procode2(self):
        df_in = self.df.copy()

        df_in.loc[0,'PROCODE'] = np.nan
        df_in.loc[7,'PROCODE'] = np.nan # AMI event, so removed

        df_out = clean_hes.set_init_procode(df_in)
        
        df_expected = self.df.copy()
        df_expected['PROCODE'] = [np.nan,np.nan,np.nan,3,3,3,np.nan,np.nan,np.nan]
        df_expected = df_expected.astype({'PROCODE':float})

        pd.testing.assert_frame_equal(df_expected, df_out)
        


    def test_set_init_deprivation_nochange(self):
        df_in = self.df.copy()

        df_out = clean_hes.set_init_deprivation(df_in)
        
        df_expected = self.df.copy()
        df_expected['IMD04'] = [0,0,0,3,3,3,1,1,1]
        df_expected = df_expected.astype({'IMD04':float})

        pd.testing.assert_frame_equal(df_expected, df_out)

    def test_set_init_deprivation(self):
        df_in = self.df.copy()

        df_in.loc[0,'IMD04'] = np.nan
        df_in.loc[8,'IMD04'] = np.nan # non-AMI event, so kept

        df_out = clean_hes.set_init_deprivation(df_in)
        
        df_expected = self.df.copy()
        df_expected['IMD04'] = [np.nan,np.nan,np.nan,3,3,3,1,1,1]
        df_expected = df_expected.astype({'IMD04':float})

        pd.testing.assert_frame_equal(df_expected, df_out)
        
    def test_set_init_deprivation2(self):
        df_in = self.df.copy()

        df_in.loc[0,'IMD04'] = np.nan
        df_in.loc[7,'IMD04'] = np.nan # AMI event, so removed

        df_out = clean_hes.set_init_deprivation(df_in)
        
        df_expected = self.df.copy()
        df_expected['IMD04'] = [np.nan,np.nan,np.nan,3,3,3,np.nan,np.nan,np.nan,]
        df_expected = df_expected.astype({'IMD04':float})

        pd.testing.assert_frame_equal(df_expected, df_out)
        
        

class Test_remove_subjects_nan_spell_id(unittest.TestCase):
    
    
    def setUp(self):
        self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0,1,1,1,2,2,2],
                                'PROVSPNOPS': [0,0,0,1,1,1,2,2,2],
                               })
    
    def test_remove_subjects_nan_spell_id(self):
        df_out = clean_hes.remove_subjects_nan_spell_id(self.df.copy())
        df_exp = self.df.copy()
        df_exp = df_exp.astype({'PROVSPNOPS':np.uint8})
        pd.testing.assert_frame_equal(df_exp, df_out)


    def test_remove_subjects_nan_spell_id_nan(self):
        self.df['PROVSPNOPS'] = [np.nan,0,0,1,np.nan,1,2,2,2]
        df_out = clean_hes.remove_subjects_nan_spell_id(self.df.copy())
        df_exp = self.df.loc[self.df['ENCRYPTED_HESID']==2].reset_index(drop=True)
        df_exp = df_exp.astype({'PROVSPNOPS':np.uint8})
        pd.testing.assert_frame_equal(df_exp, df_out)



class Test_check_single_option_within_subjects(unittest.TestCase):
    
    def setUp(self):
        self.df = pd.DataFrame({
            'ENCRYPTED_HESID':[0,0,0,0,0,
                               1,1,1,
                               2,2,2,
                               3,3,3],
            'Mortality':[0,0,0,0,0,
                         1,1,1,
                         1,1,1,
                         0,0,0],
            'SEX':[0,0,0,0,0,
                   0,0,0,
                   1,1,1,
                   1,1,1],
            'MYDOB':np.append(np.repeat(np.datetime64('2005-01-01'),5),
                              [np.repeat(np.datetime64('1990-01-01'),3),
                               np.repeat(np.datetime64('1980-12-25'),3),
                               np.repeat(np.datetime64('1970-01-01'),3),])})
        self.df = self.df.astype({'Mortality':np.uint8, 'SEX':np.uint8})


    @patch('pipeline_hes.params.params.WITHIN_SUBJECT_SINGLE_VALUE_THRESHOLD', 75)
    def test_check_single_option_within_subjects(self):
        df_out = clean_hes.check_single_option_within_subjects(self.df.copy())
        pd.testing.assert_frame_equal(self.df, df_out, check_like=True)

    @patch('pipeline_hes.params.params.WITHIN_SUBJECT_SINGLE_VALUE_THRESHOLD', 75)
    def test_check_single_option_within_subjects_change_mortality(self):
        self.df.loc[0:2,'Mortality'] = 1
        self.df.loc[5:7,'Mortality'] = np.nan
        df_out = clean_hes.check_single_option_within_subjects(self.df)
        np.testing.assert_array_equal(df_out['ENCRYPTED_HESID'],
                                      [2,2,2,3,3,3])
        
    @patch('pipeline_hes.params.params.WITHIN_SUBJECT_SINGLE_VALUE_THRESHOLD', 75)
    def test_check_single_option_within_subjects_change_sex(self):
        self.df.loc[0:2,'SEX'] = 1
        self.df.loc[5:7,'SEX'] = np.nan
        df_out = clean_hes.check_single_option_within_subjects(self.df.copy())
        np.testing.assert_array_equal(df_out['ENCRYPTED_HESID'],
                                      [2,2,2,3,3,3])
        
    @patch('pipeline_hes.params.params.WITHIN_SUBJECT_SINGLE_VALUE_THRESHOLD', 75)
    def test_check_single_option_within_subjects_change_dob(self):
        self.df.loc[0:2,'MYDOB'] = np.datetime64('2005-01-02')
        self.df.loc[5:7,'MYDOB'] = np.nan
        df_out = clean_hes.check_single_option_within_subjects(self.df.copy())
        np.testing.assert_array_equal(df_out['ENCRYPTED_HESID'],
                                      [2,2,2,3,3,3])



    @patch('pipeline_hes.params.params.WITHIN_SUBJECT_SINGLE_VALUE_THRESHOLD', 75)
    def test_check_single_option_within_subjects_change_mortality_majority(self):
        df_tmp = self.df.copy()
        df_tmp.loc[0,'Mortality'] = 1
        df_out = clean_hes.check_single_option_within_subjects(df_tmp)
        np.testing.assert_array_equal(df_out['ENCRYPTED_HESID'],
                                      self.df['ENCRYPTED_HESID'])
        np.testing.assert_array_equal(df_out['Mortality'],
                                      self.df['Mortality'])
        
    @patch('pipeline_hes.params.params.WITHIN_SUBJECT_SINGLE_VALUE_THRESHOLD', 75)
    def test_check_single_option_within_subjects_change_sex_majority(self):
        df_tmp = self.df.copy()
        df_tmp.loc[0,'SEX'] = 1
        df_out = clean_hes.check_single_option_within_subjects(df_tmp)
        np.testing.assert_array_equal(df_out['ENCRYPTED_HESID'],
                                      self.df['ENCRYPTED_HESID'])
        np.testing.assert_array_equal(df_out['SEX'],
                                      self.df['SEX'])
        
    @patch('pipeline_hes.params.params.WITHIN_SUBJECT_SINGLE_VALUE_THRESHOLD', 75)
    def test_check_single_option_within_subjects_change_dob_majority(self):
        df_tmp = self.df.copy()
        df_tmp.loc[0,'MYDOB'] = np.datetime64('2005-01-02')
        df_out = clean_hes.check_single_option_within_subjects(df_tmp)
        np.testing.assert_array_equal(df_out['ENCRYPTED_HESID'],
                                      self.df['ENCRYPTED_HESID'])
        np.testing.assert_array_equal(df_out['MYDOB'],
                                      self.df['MYDOB'])
        
    @patch('pipeline_hes.params.params.WITHIN_SUBJECT_SINGLE_VALUE_THRESHOLD', 75)
    def test_check_single_option_within_subjects_change_mortality_majority_nan(self):
        df_tmp = self.df.copy()
        df_tmp.loc[0,'Mortality'] = np.nan
        df_out = clean_hes.check_single_option_within_subjects(df_tmp)
        np.testing.assert_array_equal(df_out['ENCRYPTED_HESID'],
                                      self.df['ENCRYPTED_HESID'])
        np.testing.assert_array_equal(df_out['Mortality'],
                                      self.df['Mortality'])
        
    @patch('pipeline_hes.params.params.WITHIN_SUBJECT_SINGLE_VALUE_THRESHOLD', 75)
    def test_check_single_option_within_subjects_change_sex_majority_nan(self):
        df_tmp = self.df.copy()
        df_tmp.loc[0,'SEX'] = np.nan
        df_out = clean_hes.check_single_option_within_subjects(df_tmp)
        np.testing.assert_array_equal(df_out['ENCRYPTED_HESID'],
                                      self.df['ENCRYPTED_HESID'])
        np.testing.assert_array_equal(df_out['SEX'],
                                      self.df['SEX'])
        
    @patch('pipeline_hes.params.params.WITHIN_SUBJECT_SINGLE_VALUE_THRESHOLD', 75)
    def test_check_single_option_within_subjects_change_dob_majority_nan(self):
        df_tmp = self.df.copy()
        df_tmp.loc[0,'MYDOB'] = np.nan
        df_out = clean_hes.check_single_option_within_subjects(df_tmp)
        np.testing.assert_array_equal(df_out['ENCRYPTED_HESID'],
                                      self.df['ENCRYPTED_HESID'])
        np.testing.assert_array_equal(df_out['MYDOB'],
                                      self.df['MYDOB'])
        



# class Test_remove_subjects_with_dup_within_spell_events(unittest.TestCase):
    
#     def setUp(self):
#         self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0,
#                                                     1,1,1],
#                                 'PROVSPNOPS':['X','X','X',
#                                               'Y','Y','Y'],
#                                 'PROCODE':['A','A','A',
#                                             'B','B','B'],
#                                 'EPIORDER':[1,2,3,
#                                             1,2,3],
#                                 'EPISTAT':[3,3,3,
#                                             3,3,3],
#                                 })
        
#     def test_remove_subjects_with_dup_within_spell_events(self):
#         df_tmp = self.df.copy()
#         df_out = clean_hes.remove_subjects_with_dup_within_spell_events(df_tmp.copy())
#         df_exp = df_tmp.copy()
#         pd.testing.assert_frame_equal(df_out,df_exp)
        
        
#     def test_remove_subjects_with_dup_within_spell_events_epiorder(self):
#         df_tmp = self.df.copy()
#         df_tmp['EPIORDER'] = [1,2,3,1,1,2]
#         df_out = clean_hes.remove_subjects_with_dup_within_spell_events(df_tmp.copy())
#         df_exp = df_tmp.copy()
#         df_exp = df_exp.loc[df_exp['ENCRYPTED_HESID']==0]
#         pd.testing.assert_frame_equal(df_out,df_exp)
        
#     def test_remove_subjects_with_dup_within_spell_events_epistat(self):
#         df_tmp = self.df.copy()
#         df_tmp['EPISTAT'] = [3,3,3,3,1,3]
#         df_out = clean_hes.remove_subjects_with_dup_within_spell_events(df_tmp.copy())
#         df_exp = df_tmp.copy()
#         pd.testing.assert_frame_equal(df_out,df_exp)

#     def test_remove_subjects_with_dup_within_spell_events_procode(self):
#         df_tmp = self.df.copy()
#         df_tmp['PROCODE'] = ['A','A','X',
#                               'B','B','B']
#         df_out = clean_hes.remove_subjects_with_dup_within_spell_events(df_tmp.copy())
#         df_exp = df_tmp.copy()
#         pd.testing.assert_frame_equal(df_out,df_exp)
        
#     def test_remove_subjects_with_dup_within_spell_events_provspnops(self):
#         df_tmp = self.df.copy()
#         df_tmp['PROVSPNOPS'] = ['X','X','Z',
#                                 'Y','Y','Y']
#         df_out = clean_hes.remove_subjects_with_dup_within_spell_events(df_tmp.copy())
#         df_exp = df_tmp.copy()
#         pd.testing.assert_frame_equal(df_out,df_exp)
        



class Test_remove_using_amiID(unittest.TestCase):

    
    def setUp(self):
        self.df_ami = pd.DataFrame({'ENCRYPTED_HESID': [0,0,0, 1,1,1,],
                                    'amiID':[10,10,10, 11,11,11],
                                    'IS_CONTROL':np.repeat(False,6)})
        self.df_ctl = pd.DataFrame({'ENCRYPTED_HESID': np.repeat(np.arange(102,112),3),
                                    'amiID':np.append(np.repeat(10,15),
                                                      np.repeat(11,15)),
                                    'IS_CONTROL':np.repeat(True,30)})
     
    @patch('pipeline_hes.params.params.CONTROL_CASE_RATIO', 5)
    def test_remove_using_amiID_ami_control_NO_CHANGE(self):
        df = pd.concat([self.df_ami,self.df_ctl]).reset_index(drop=True)
        df_out = clean_hes.remove_using_amiID(df.copy())
        df_exp = df.copy().reset_index(drop=True)
        pd.testing.assert_frame_equal(
            df_out.sort_values('ENCRYPTED_HESID').reset_index(drop=True),
            df_exp.sort_values('ENCRYPTED_HESID').reset_index(drop=True))

        
    @patch('pipeline_hes.params.params.CONTROL_CASE_RATIO', 5)
    def test_remove_using_amiID_ami_control_REMOVE_CONTROL(self):

        # remove ONE control (disrupt the 1:5 ratio)
        df = pd.concat([self.df_ami,self.df_ctl.iloc[3:]]).reset_index(drop=True)

        df_out = clean_hes.remove_using_amiID(df.copy())

        df_exp = df.loc[df['amiID']==11].reset_index(drop=True)

        pd.testing.assert_frame_equal(
            df_out.sort_values('ENCRYPTED_HESID').reset_index(drop=True),
            df_exp.sort_values('ENCRYPTED_HESID').reset_index(drop=True))

    @patch('pipeline_hes.params.params.CONTROL_CASE_RATIO', 5)
    def test_remove_using_amiID_ami_control_INSERT_CONTROL(self):

        # add a control
        add_ctl = self.df_ctl.iloc[:3].copy()
        add_ctl['ENCRYPTED_HESID'] = 999   
        df = pd.concat([self.df_ami,self.df_ctl,add_ctl]).reset_index(drop=True)
        
        df_out = clean_hes.remove_using_amiID(df.copy())

        df_exp = df.iloc[:df.shape[0]-3]

        pd.testing.assert_frame_equal(
            df_out.sort_values('ENCRYPTED_HESID').reset_index(drop=True),
            df_exp.sort_values('ENCRYPTED_HESID').reset_index(drop=True))


    # 2:10
    @patch('pipeline_hes.params.params.CONTROL_CASE_RATIO', 5)
    def test_remove_using_amiID_ami_control_REMOVE_CONTROL_double(self):

        df_ami2 = self.df_ami.copy()
        df_ami2['ENCRYPTED_HESID'] = df_ami2['ENCRYPTED_HESID'] + \
            df_ami2['ENCRYPTED_HESID'].max() + 1
        df_ctl2 = self.df_ctl.copy()
        df_ctl2['ENCRYPTED_HESID'] = df_ctl2['ENCRYPTED_HESID'] + \
            df_ctl2['ENCRYPTED_HESID'].max() + 1

        # remove ONE control (disrupt the 1:5 ratio)
        df = pd.concat([self.df_ami,
                        df_ami2,
                        self.df_ctl.iloc[3:],
                        df_ctl2]).reset_index(drop=True)

        df_out = clean_hes.remove_using_amiID(df.copy())

        # one missing AMI, 4 missing controls
        self.assertEqual(df_out['ENCRYPTED_HESID'].drop_duplicates().shape[0],
                         df['ENCRYPTED_HESID'].drop_duplicates().shape[0]-1-4)
        self.assertEqual(df_out.loc[df_out['amiID']==11].\
                         drop_duplicates('ENCRYPTED_HESID').shape[0],
                         df.loc[df['amiID']==11].\
                         drop_duplicates('ENCRYPTED_HESID').shape[0])
        self.assertEqual(df_out.loc[df_out['amiID']==10].\
                         drop_duplicates('ENCRYPTED_HESID').shape[0],
                         df.loc[df['amiID']==10].\
                         drop_duplicates('ENCRYPTED_HESID').shape[0]-1-4)
        self.assertEqual(df_out.loc[df_out['IS_CONTROL']].\
                         drop_duplicates('ENCRYPTED_HESID').shape[0],
                         df.loc[df['IS_CONTROL']].\
                         drop_duplicates('ENCRYPTED_HESID').shape[0]-4)
        self.assertEqual(df_out.loc[~df_out['IS_CONTROL']].\
                         drop_duplicates('ENCRYPTED_HESID').shape[0],
                         df.loc[~df['IS_CONTROL']].\
                         drop_duplicates('ENCRYPTED_HESID').shape[0]-1)
            
    # 2:10
    @patch('pipeline_hes.params.params.CONTROL_CASE_RATIO', 5)
    def test_remove_using_amiID_ami_control_INSERT_CONTROL_double(self):

        df_ami2 = self.df_ami.copy()
        df_ami2['ENCRYPTED_HESID'] = df_ami2['ENCRYPTED_HESID'] + \
            df_ami2['ENCRYPTED_HESID'].max() + 1
        df_ctl2 = self.df_ctl.copy()
        df_ctl2['ENCRYPTED_HESID'] = df_ctl2['ENCRYPTED_HESID'] + \
            df_ctl2['ENCRYPTED_HESID'].max() + 1
            
        # add a control
        add_ctl = self.df_ctl.iloc[:3].copy()
        add_ctl['ENCRYPTED_HESID'] = 999   

        # remove ONE control (disrupt the 1:5 ratio)
        df = pd.concat([self.df_ami,
                        df_ami2,
                        self.df_ctl,
                        add_ctl,
                        df_ctl2]).reset_index(drop=True)

        df_out = clean_hes.remove_using_amiID(df.copy())

        # one missing control
        self.assertEqual(df_out['ENCRYPTED_HESID'].drop_duplicates().shape[0],
                         df['ENCRYPTED_HESID'].drop_duplicates().shape[0]-1)
        self.assertEqual(df_out.loc[df_out['amiID']==11].\
                         drop_duplicates('ENCRYPTED_HESID').shape[0],
                         df.loc[df['amiID']==11].\
                         drop_duplicates('ENCRYPTED_HESID').shape[0])
        self.assertEqual(df_out.loc[df_out['amiID']==10].\
                         drop_duplicates('ENCRYPTED_HESID').shape[0],
                         df.loc[df['amiID']==10].\
                         drop_duplicates('ENCRYPTED_HESID').shape[0]-1)
        self.assertEqual(df_out.loc[df_out['IS_CONTROL']].\
                         drop_duplicates('ENCRYPTED_HESID').shape[0],
                         df.loc[df['IS_CONTROL']].\
                         drop_duplicates('ENCRYPTED_HESID').shape[0]-1)
        self.assertEqual(df_out.loc[~df_out['IS_CONTROL']].\
                         drop_duplicates('ENCRYPTED_HESID').shape[0],
                         df.loc[~df['IS_CONTROL']].\
                         drop_duplicates('ENCRYPTED_HESID').shape[0])
            
        #df_exp = df_exp.loc[df_exp['ENCRYPTED_HESID']!=4]

        # pdb.set_trace()
        # pd.testing.assert_frame_equal(
        #     df_out.sort_values('ENCRYPTED_HESID').reset_index(drop=True),
        #     df_exp.sort_values('ENCRYPTED_HESID').reset_index(drop=True))


    

        


class Test_remove_subjects_using_dates(unittest.TestCase):
    
    def setUp(self):
        event_times = np.arange('2005-01', '2005-10', dtype='datetime64[M]')
        earliest_times = np.append(np.repeat(np.datetime64('2005-01-01'), 3),
                                   [np.repeat(np.datetime64('2005-04-01'), 3),
                                   np.repeat(np.datetime64('2005-07-01'), 3)])
        matched_times = np.append(np.repeat(np.datetime64('2005-01-01'), 3),
                                  [np.repeat(np.datetime64('2005-05-01'), 3),
                                   np.repeat(np.datetime64('2005-09-01'), 3)])

        # final control does not have AMI
        ami_times = np.append(np.repeat(np.datetime64('2005-01-01'), 3),
                              [np.repeat(np.datetime64('2005-06-01'), 3),
                               np.repeat(np.datetime64(None), 3)])
        
        # AMI (*01,02,03)
        # Control with AMI (04,*05,06)
        # Control without AMI (07,08,*09)
        self.df = pd.DataFrame({'ENCRYPTED_HESID': [0,0,0,1,1,1,2,2,2],
                                'MYEPISTART':event_times,
                                'MYEPISTART_EARLIEST':earliest_times,
                                'MATCHED_DATE': matched_times,
                                'MYEPISTART_FIRSTAMI':ami_times,
                                'IS_CONTROL':np.append(np.repeat(False,3),
                                                       [np.repeat(True,3),
                                                       np.repeat(True,3)])})


    # #############
    # remove_subjects_with_no_event_on_matched_date
    # #############
    def test_remove_subjects_with_no_event_on_matched_date_no_change(self):
        df_out = \
            clean_hes.remove_subjects_with_no_event_on_matched_date(self.df.copy())
        df_expected = self.df.copy()
        df_expected['INIT_ROW'] = df_expected['MATCHED_DATE'] == df_expected['MYEPISTART']
        pd.testing.assert_frame_equal(df_expected, df_out)
        
        
    def test_remove_subjects_with_no_event_on_matched_date_bad1(self):
        df_tmp = self.df.copy()
        
        # make the matched date WAY OFF
        df_tmp['MATCHED_DATE'] = \
            np.append(np.repeat(np.datetime64('2005-01-01'), 3),
                      [np.repeat(np.datetime64('2004-05-01'), 3),
                       np.repeat(np.datetime64('2005-09-01'), 3)])
        
        df_out = clean_hes.\
            remove_subjects_with_no_event_on_matched_date(df_tmp.copy())
        df_expected = self.df.loc[self.df['ENCRYPTED_HESID']!=1].reset_index(drop=True)
        df_expected['INIT_ROW'] = df_expected['MATCHED_DATE'] == df_expected['MYEPISTART']
        pd.testing.assert_frame_equal(df_expected, df_out)

        
    # @patch('pipeline_hes.params.params.MATCHING_INITIAL_EVENT_THRESHOLD_DAYS', 32)
    # def test_remove_subjects_with_no_event_on_matched_date_fuzzy1(self):
    #     df_tmp = self.df.copy()
    #     # remove event with matched date
    #     df_tmp = df_tmp.loc[
    #         df_tmp['MYEPISTART']!=np.datetime64('2005-05-01')]
        
    #     df_out = clean_hes.\
    #         remove_subjects_with_no_event_on_matched_date(df_tmp.copy())
        
    #     df_expected = df_tmp.copy()
    #     df_expected['INIT_ROW'] = \
    #         [True,False,False,
    #          True,True,
    #          False,False,True]
    #     pd.testing.assert_frame_equal(
    #         df_expected.sort_values('ENCRYPTED_HESID').reset_index(drop=True),
    #         df_out.sort_values('ENCRYPTED_HESID').reset_index(drop=True))


    #@patch('pipeline_hes.params.params.MATCHING_INITIAL_EVENT_THRESHOLD_DAYS', 32)
    # def test_remove_subjects_with_no_event_on_matched_date_fuzzy2(self):
    #     df_tmp = self.df.copy()
    #     df_tmp = df_tmp.loc[
    #         df_tmp['MYEPISTART']!=np.datetime64('2005-05-01')]
    #     # one close month, one far month
    #     df_tmp.loc[
    #         df_tmp['MATCHED_DATE']==np.datetime64('2005-05-01'),'MYEPISTART'] =\
    #             [np.datetime64('2005-04-01'),np.datetime64('2005-07-01')]
    #     df_out = clean_hes.\
    #         remove_subjects_with_no_event_on_matched_date(df_tmp.copy())
        
    #     df_expected = df_tmp.copy()
    #     df_expected['INIT_ROW'] = \
    #         [True,False,False,
    #          True,False,
    #          False,False,True]
    #     #pdb.set_trace()
    #     pd.testing.assert_frame_equal(
    #         df_expected.sort_values('ENCRYPTED_HESID').reset_index(drop=True),
    #         df_out.sort_values('ENCRYPTED_HESID').reset_index(drop=True))
        

    def test_remove_subjects_with_no_event_on_matched_date_ok1(self):
        df_tmp = self.df.copy()
        df_tmp = df_tmp.loc[
            df_tmp['MYEPISTART']!=np.datetime64('2005-04-01')].reset_index(drop=True)
        
        df_out = clean_hes.\
            remove_subjects_with_no_event_on_matched_date(df_tmp.copy())
        df_expected = df_tmp.copy()
        df_expected['INIT_ROW'] = df_expected['MATCHED_DATE'] == df_expected['MYEPISTART']
        pd.testing.assert_frame_equal(df_expected, df_out)
    



    # #################
    # remove_ami_subjects_where_ami_date_does_not_match_checked_ami_date
    # #################

    def test_remove_ami_subjects_where_first_ami_date_is_not_matched_date_no_change(self):
        df_tmp = self.df.copy()
        clean_hes.\
            remove_ami_subjects_where_first_ami_date_is_not_matched_date(self.df.copy())
        pd.testing.assert_frame_equal(self.df, df_tmp)
        
    def test_remove_ami_subjects_where_first_ami_date_is_not_matched_date_bad1(self):
        df_tmp = self.df.copy()
        df_tmp.loc[df_tmp['ENCRYPTED_HESID']==0, 'MYEPISTART_FIRSTAMI'] = \
            np.datetime64('2005-02-01')

        df_expected = self.df.loc[self.df['ENCRYPTED_HESID']!=0]
        clean_hes.\
            remove_ami_subjects_where_first_ami_date_is_not_matched_date(df_tmp)
        pd.testing.assert_frame_equal(df_expected, df_tmp)

    def test_remove_ami_subjects_where_first_ami_date_is_not_matched_date_ok1(self):
        df_tmp = self.df.copy()
        df_tmp.loc[df_tmp['ENCRYPTED_HESID']==0, 'MYEPISTART_FIRSTAMI'] = \
            np.datetime64('2005-02-01')
        df_tmp.loc[df_tmp['ENCRYPTED_HESID']==0, 'MATCHED_DATE'] = \
            np.datetime64('2005-02-01')

        df_expected = df_tmp.copy()
        clean_hes.\
            remove_ami_subjects_where_first_ami_date_is_not_matched_date(df_tmp)
        pd.testing.assert_frame_equal(df_expected, df_tmp)


    # ################
    # remove_controls_with_ami_matched_date_not_before_first_ami
    # ################

    def test_remove_controls_with_ami_matched_date_not_before_first_ami_no_change(self):
        df_tmp = self.df.copy()
        clean_hes.remove_controls_with_ami_matched_date_not_before_first_ami(df_tmp)
        pd.testing.assert_frame_equal(self.df, df_tmp)

    def test_remove_controls_with_ami_matched_date_not_before_first_ami_bad1(self):
        df_tmp = self.df.copy()
        # put first AMI date before matched date
        df_tmp.loc[df_tmp['ENCRYPTED_HESID']==1,'MYEPISTART_FIRSTAMI'] =\
            np.datetime64('2005-04-01')
        
        df_out = clean_hes.\
            remove_controls_with_ami_matched_date_not_before_first_ami(df_tmp)
        df_expected = self.df.loc[self.df['ENCRYPTED_HESID']!=1]
        pd.testing.assert_frame_equal(df_expected, df_tmp)
        
    def test_remove_controls_with_ami_matched_date_not_before_first_ami_bad2(self):
        df_tmp = self.df.copy()
        # put first AMI date ON matched date
        df_tmp.loc[df_tmp['ENCRYPTED_HESID']==1,'MYEPISTART_FIRSTAMI'] =\
            np.datetime64('2005-05-01')
        
        df_expected = self.df.loc[self.df['ENCRYPTED_HESID']!=1]
        clean_hes.\
            remove_controls_with_ami_matched_date_not_before_first_ami(df_tmp)
        pd.testing.assert_frame_equal(df_expected, df_tmp)

    def test_remove_controls_with_ami_matched_date_not_before_first_ami_ok1(self):
        df_tmp = self.df.copy()
        df_tmp.loc[df_tmp['ENCRYPTED_HESID']==1,'MYEPISTART_FIRSTAMI'] =\
            np.datetime64('2005-06-01')
        
        df_expected = df_tmp.copy()
        clean_hes.\
            remove_controls_with_ami_matched_date_not_before_first_ami(df_tmp)
        pd.testing.assert_frame_equal(df_expected, df_tmp)




if __name__ == '__main__':
    # Show full dataframes
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.expand_frame_repr', False)
    unittest.main(argv=['-v'],verbosity=3)

