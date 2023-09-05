# -*- coding: utf-8 -*-
"""
@author: Chris Hayward
"""


import unittest
import numpy as np
import pandas as pd
import pdb

from pipeline_hes import clean_hes
from pipeline_hes import clean_hes_prepare


class Test_add_min_max_spell_times(unittest.TestCase):

    def setUp(self):
        # just one subject is fine...
        # The important thing is how we sort WITHIN subjects
        self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0,
                                                   0,0,0,
                                                   0,0,0],
                                'PROVSPNOPS':['XXX','XXX','XXX',
                                              'YYY','YYY','YYY',
                                              'ZZZ','ZZZ','ZZZ',],
                                'MYEPISTART':np.concatenate([
                                    [np.datetime64('2007-01-01'),
                                     np.datetime64(None),
                                     np.datetime64('2007-03-01')],
                                    np.repeat(np.datetime64('2006-01-01'),3),
                                    np.repeat(np.datetime64('2007-01-01'),3),
                                    ],axis=None),
                                'MYEPIEND':np.concatenate([
                                    [np.datetime64('2007-01-01'),
                                     np.datetime64(None),
                                     np.datetime64('2007-03-01')],
                                    np.repeat(np.datetime64('2006-01-01'),3),
                                    np.repeat(np.datetime64('2007-01-01'),3),
                                    ],axis=None),
                                'SURVIVALTIME':[60,60,60,
                                                50,20,20,
                                                30,np.nan,10,]})
        
    def test_add_min_max_spell_times(self):
        df_out = clean_hes_prepare.add_min_max_spell_times(self.df.copy())      
        df_exp = self.df.copy()
        df_exp['MYEPISTART_SPELLMIN'] = np.concatenate([
                                    np.repeat(np.datetime64('2007-01-01'),3),
                                    np.repeat(np.datetime64('2006-01-01'),3),
                                    np.repeat(np.datetime64('2007-01-01'),3),
                                    ],axis=None)
        df_exp['MYEPISTART_SPELLMAX'] = np.concatenate([
                                    np.repeat(np.datetime64('2007-03-01'),3),
                                    np.repeat(np.datetime64('2006-01-01'),3),
                                    np.repeat(np.datetime64('2007-01-01'),3),
                                    ],axis=None)
        
        df_exp['MYEPIEND_SPELLMIN'] = np.concatenate([
                                    np.repeat(np.datetime64('2007-01-01'),3),
                                    np.repeat(np.datetime64('2006-01-01'),3),
                                    np.repeat(np.datetime64('2007-01-01'),3),
                                    ],axis=None)
        df_exp['MYEPIEND_SPELLMAX'] = np.concatenate([
                                    np.repeat(np.datetime64('2007-03-01'),3),
                                    np.repeat(np.datetime64('2006-01-01'),3),
                                    np.repeat(np.datetime64('2007-01-01'),3),
                                    ],axis=None)
        df_exp['SURVIVALTIME_SPELLMIN'] = np.array([60,60,60,
                                           20,20,20,
                                           10,10,10,],dtype=float)
        df_exp['SURVIVALTIME_SPELLMAX'] = np.array([60,60,60,
                                           50,50,50,
                                           30,30,30,],dtype=float)
        pd.testing.assert_frame_equal(df_out,df_exp,check_like=True)




class Test_prepare_df(unittest.TestCase):
    
    def test_set_age(self):
        df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0],
                           'MATCHED_DATE':np.repeat(np.datetime64('2017-01-01'),3),
                           'MYDOB':np.datetime64('1900-01-01')})
        # expected 117 years old
        clean_hes_prepare.set_age(df)
        np.testing.assert_array_equal(df['INIT_AGE'], np.array([117,117,117]))
        
        
    def test_trim_diags_set_to_upper_replace_ami(self):
        df = pd.DataFrame({'DIAG_01':['I21', 'I22', 'I23', 'I211',
                                      'I24', 'I20', 'a000', np.nan],})
        for i in range(2,21):
            df['DIAG_{:02d}'.format(i)] = df['DIAG_01']
        
        clean_hes_prepare.trim_all_diags_set_to_upper_replace_ami(df)
        
        for i in range(2,21):
            pd.testing.assert_series_equal(df['DIAG_{:02d}'.format(i)],
                                          pd.Series(['I21', 'I22', 'I23', 'I21',
                                                    'I24','I20','A00',np.nan],
                                                    name='DIAG_{:02d}'.format(i),
                                                    dtype=object))

    def test_convert_all_diag_cols_to_category(self):
        df = pd.DataFrame({'DIAG_01':['I21', 'I21', 'I23', 'I23',
                                      'I24', 'I20', 'a000', np.nan],})
        for i in range(2,21):
            df['DIAG_{:02d}'.format(i)] = df['DIAG_01']
        
        df_out = clean_hes_prepare.convert_all_diag_cols_to_category(df.copy())
        
        for i in range(2,21):
            np.testing.assert_equal(df_out['DIAG_{:02d}'.format(i)].dtype,
                pd.CategoricalDtype(categories=['I20', 'I21', 'I23', 'I24', 'a000'],
                                    ordered=False))


        
    # def test_extend_spell_id(self):
    #     df = pd.DataFrame({'PROCODE':['ABB', 'ABB', 'XYY',],
    #                        'PROVSPNOPS':['XX', 'YY', 'ZZ'],})
    #     clean_hes_prepare.extend_spell_id(df)
    #     np.testing.assert_array_equal(df['PROVSPNOPS'].values,
    #                                   np.array(['XXABB', 'YYABB', 'ZZXYY']))

    # def test_replace_deprivation_score(self):
    #     df = pd.DataFrame({'IMD04_DECILE':['Least deprived 10%',
    #                                        'Less deprived 10-20%', 
    #                                        'Less deprived 20-30%',
    #                                        'Less deprived 30-40%',
    #                                        'Less deprived 40-50%',
    #                                        'More deprived 40-50%',
    #                                        'More deprived 30-40%',
    #                                        'More deprived 20-30%',
    #                                        'More deprived 10-20%',
    #                                        'Most deprived 10%'],})
    #     clean_hes.replace_deprivation_score(df)
    #     np.testing.assert_array_equal(df['IMD04_DECILE'].values,
    #                                   np.array([0,0,1,1,2,2,3,3,4,4]))

    


class Test_add_new_cols(unittest.TestCase):
    
    def setUp(self):
        
        times = np.array([np.datetime64('2005-04-01'),
                          np.datetime64('2005-01-01'),
                          np.datetime64('2005-03-01'),
                          np.datetime64('2005-02-01'),
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
                          np.datetime64('2005-07-01'),
                          np.datetime64('2005-07-01'),
                          np.datetime64('2005-07-01'),
                          np.datetime64('2005-10-01'),
                          np.datetime64('2005-10-01'),
                          np.datetime64('2005-10-01')])
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
        earliest = np.array([np.datetime64('2005-01-01'),
                          np.datetime64('2005-01-01'),
                          np.datetime64('2005-01-01'),
                          np.datetime64('2005-01-01'),
                          np.datetime64('2005-06-01'),
                          np.datetime64('2005-06-01'),
                          np.datetime64('2005-06-01'),
                          np.datetime64('2005-09-01'),
                          np.datetime64('2005-09-01'),
                          np.datetime64('2005-09-01'),])

        
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
                                'MATCHED_DATE':matched,
                                'MYEPISTART_FIRSTAMI':firstami,
                                'MYEPISTART_EARLIEST':earliest,})
        #'MYDOB':np.repeat(np.datetime64('1990-01-01'),10)


    def test_add_first_ami_date(self):
        self.df = self.df.drop(columns=['MYEPISTART_FIRSTAMI'])
        self.df = self.df.drop(columns=['MYEPISTART_EARLIEST'])
        self.df = self.df.drop(columns=['MATCHED_DATE'])

        # subjects with AMI: _FIRSTAMI = first ami date (using column 'AMI')
        # controls without AMI: _FIRSTAMI = NaT
        df_out = clean_hes_prepare.add_first_ami_date(self.df.copy())
        df_expected = self.df.copy()
        df_expected['MYEPISTART_FIRSTAMI'] = \
           np.concatenate([np.repeat(np.datetime64('2005-03-01'),4),
                           np.repeat(np.datetime64(None),3),
                           np.repeat(np.datetime64('2005-10-01'),3)],axis=None)
        pd.testing.assert_frame_equal(df_expected, df_out)


    # Make sure an exception is raised
    def test_add_first_ami_date_missing_ami(self):
        self.df = self.df.drop(columns=['MYEPISTART_FIRSTAMI'])
        self.df = self.df.drop(columns=['MYEPISTART_EARLIEST'])
        self.df = self.df.drop(columns=['MATCHED_DATE'])

        # set all events to AMI=false for AMI subject
        self.df.loc[self.df['ENCRYPTED_HESID']==2,'AMI'] = False
        # For ami subjects (IS_CONTROL==False), the first ami event is based on AMI==True
        # Any ami subjects without _FIRSTAMI set will raise an exception
        self.assertRaises(Exception, clean_hes_prepare.add_first_ami_date, self.df.copy())



        
        
if __name__ == '__main__':
    # Show full dataframes
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.expand_frame_repr', False)
    unittest.main(argv=['-v'],verbosity=3)

