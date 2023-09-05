# -*- coding: utf-8 -*-
"""
@author: Chris Hayward
"""

# test load_parquet



import unittest
import pandas as pd
import numpy as np
import pdb

from pipeline_hes import load_parquet
from pipeline_hes.params import params
from unittest.mock import patch


class Test_set_nans_as_mynan_str(unittest.TestCase):
    """Ensure that values which represent nan are replaced with my nan placeholder."""
    
    def setUp(self):
        self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0,
                                                   1,1,1],
                                'PROVSPNOPS':[np.nan, 'na', 'aaa',
                                              'xxx', 'nan', 'non'],
                                'ADMIMETH':[np.nan,'C','C',
                                           'D','na','non'],
                                'DISMETH':['J','J','J',
                                           'K','null','nan'],
                                'PROCODE':[np.nan, 'na', 'aaa',
                                           'xxx', 'nan', 'non'],
                                })
        
        diags = ['A00','NA','NaN',
                 np.nan,'E00','noN']
        self.df['DIAG_01'] = diags

        self.df = self.df.astype({'DIAG_01':'category'})
        for d in params.SEC_DIAG_COLS:
            self.df[d] = diags
            self.df = self.df.astype({d:'category'})

        
    def test_set_nans_as_mynan_str(self):
        df_tmp = self.df.copy()
        df_out = load_parquet.set_nans_as_str(df_tmp.copy())

        # Expected
        df_exp = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0,
                                                  1,1,1],
                               'PROVSPNOPS':['MYNAN', 'MYNAN', 'aaa',
                                             'xxx', 'MYNAN', 'MYNAN'],
                                'ADMIMETH':['MYNAN','C','C',
                                            'D','MYNAN','MYNAN'],
                                'DISMETH':['J','J','J',
                                           'K','MYNAN','MYNAN'],
                                'PROCODE':['MYNAN', 'MYNAN', 'aaa',
                                           'xxx', 'MYNAN', 'MYNAN'],
                                })
        
        diags = ['A00','MYNAN','MYNAN','MYNAN','E00','MYNAN']
        df_exp['DIAG_01'] = diags
        df_exp = df_exp.astype({'DIAG_01':'category'})
        for d in params.SEC_DIAG_COLS:
            df_exp[d] = diags
            df_exp = df_exp.astype({d:'category'})
            
        pd.testing.assert_frame_equal(df_out,df_exp,check_categorical=False)


class Test_set_mynan_as_npnan(unittest.TestCase):
    """Ensure that nan placeholders are replaced with numpy nans."""
    
    def setUp(self):
        # Expected
        self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0,
                                                  1,1,1],
                                'PROVSPNOPS':['MYNAN', 'MYNAN', 'aaa',
                                             'xxx', 'MYNAN', 'MYNAN'],
                                'ADMIMETH':['C','C','C',
                                           'D','MYNAN','D'],
                                'DISMETH':['J','J','J',
                                           'K','K','MYNAN'],
                                'PROCODE':['MYNAN', 'MYNAN', 'aaa',
                                           'xxx', 'MYNAN', 'MYNAN'],
                                })
        
        diags = ['A00','B00','MYNAN','MYNAN','E00','D00']
        self.df ['DIAG_01'] = diags
        self.df  = self.df.astype({'DIAG_01':'category'})
        for d in params.SEC_DIAG_COLS:
            self.df [d] = diags
            self.df = self.df.astype({d:'category'})

    def test_set_mynan_as_npnan(self):
        df_tmp = self.df.copy()
        load_parquet.set_nans_as_npnan(df_tmp)
        # Expected
        df_exp = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0,
                                                  1,1,1],
                               'PROVSPNOPS':[np.nan, np.nan, 'aaa',
                                             'xxx', np.nan, np.nan],
                                'ADMIMETH':['C','C','C',
                                           'D',np.nan,'D'],
                                'DISMETH':['J','J','J',
                                           'K','K',np.nan],
                                'PROCODE':[np.nan, np.nan, 'aaa',
                                           'xxx', np.nan, np.nan],
                                })
        diags = ['A00','B00',np.nan,np.nan,'E00','D00']
        df_exp['DIAG_01'] = diags
        df_exp = df_exp.astype({'DIAG_01':'category'})
        for d in params.SEC_DIAG_COLS:
            df_exp[d] = diags
            df_exp = df_exp.astype({d:'category'})
            
        pd.testing.assert_frame_equal(df_tmp,df_exp,check_categorical=False)

class Test_set_types_numeric(unittest.TestCase):
    """Check that variable types are set correctly."""
    
    def setUp(self):
        # Expected
        self.df = pd.DataFrame({'amiID':[1,2,3],
                                'ENCRYPTED_HESID':[1,2,3],
                                'SURVIVALTIME':[1,2,np.nan],
                                'IMD04':[1.0,2.1,3],
                                'EPIDUR':[1,2,3]})
            
    def test_set_types_numeric(self):
        df_tmp = self.df.copy()
        load_parquet.set_types_numeric(df_tmp)
        df_exp = self.df.copy()
        
        np.testing.assert_array_equal(df_tmp['amiID'].values,
                                      np.array([1,2,3],dtype=np.uint8))
        np.testing.assert_array_equal(df_tmp['ENCRYPTED_HESID'].values,
                                      np.array([1,2,3],dtype=np.uint8))
        np.testing.assert_array_equal(df_tmp['SURVIVALTIME'].values,
                                      np.array([1,2,np.nan],dtype=np.float32))
        np.testing.assert_array_equal(df_tmp['IMD04'].values,
                                      np.array([1.0,2.1,3],dtype=np.float32))
        np.testing.assert_array_equal(df_tmp['EPIDUR'].values,
                                      np.array([1.1,2,3],dtype=np.uint8))
        

class Test_clean_diag(unittest.TestCase):
    """Replace characters in diagnosis codes."""

    def setUp(self):

        diags = ['I21-','I211+','I222~',
                 'I22 ','I23','I233*',
                 'B900','B90.','B93!','D6941','D694','d69',np.nan]
        self.df = pd.DataFrame({'ENCRYPTED_HESID':np.zeros(len(diags)),
                                'DIAG_01': diags})
        self.df = self.df.astype({'DIAG_01':'category'})
        for d in params.SEC_DIAG_COLS:
            self.df[d] = diags
            self.df = self.df.astype({d:'category'})

    def test_clean_diag(self):
        df_tmp = self.df.copy()
        df_out = load_parquet.clean_diag(df_tmp)
        
        # EXPECTED:
        diags = ['I21','I211','I222',
                 'I22','I23','I233',
                 'B900','B90','B93','D6941','D694','D69',np.nan]
        df_exp = pd.DataFrame({'ENCRYPTED_HESID':np.zeros(len(diags)),
                                'DIAG_01': diags})
        
        df_exp = df_exp.astype({'DIAG_01':'category'})
        
        for d in params.SEC_DIAG_COLS:
            df_exp[d] = diags
            df_exp = df_exp.astype({d:'category'})
        pd.testing.assert_frame_equal(df_out,df_exp,check_categorical=False)



class Test_mark_ami(unittest.TestCase):

    def setUp(self):
        diags = ['I21','I211','I22',
                 'I222','I23','I233',
                 'B900','B90','I24','I20','D694','I2']
        self.df = pd.DataFrame({'ENCRYPTED_HESID':np.zeros(len(diags)),
                                'DIAG_01': diags})
        self.df = self.df.astype({'DIAG_01':'category'})
        for d in params.SEC_DIAG_COLS:
            self.df[d] = diags
            self.df = self.df.astype({d:'category'})

    def test_mark_ami(self):
        df_tmp = self.df.copy()
        load_parquet.mark_ami(df_tmp)
        df_exp = self.df.copy()
        df_exp['AMI'] = False
        df_exp.loc[0:5,'AMI'] = True
        pd.testing.assert_frame_equal(df_tmp,df_exp)

    def test_mark_ami_change_sec(self):
        df_tmp = self.df.copy()
        df_tmp['DIAG_01'] = 'AAA'
        df_tmp['DIAG_20'] = 'I21'
        df_tmp = df_tmp.astype({'DIAG_01':'category'})
        df_tmp = df_tmp.astype({'DIAG_20':'category'})
        
        load_parquet.mark_ami(df_tmp)
        
        df_exp = self.df.copy()
        df_exp['DIAG_01'] = 'AAA'
        df_exp['DIAG_20'] = 'I21'
        df_exp = df_exp.astype({'DIAG_01':'category'})
        df_exp = df_exp.astype({'DIAG_20':'category'})
        df_exp['AMI'] = True
        pd.testing.assert_frame_equal(df_tmp,df_exp,check_categorical=False)



class Test_replace_codes(unittest.TestCase):
    
    def test_replace_procode(self):
        df = pd.DataFrame({'PROCODE':['ABB', 'ABBB', 'XYYY', 'XYY', 'CBB', 'XYY', np.nan],})
        df_out = load_parquet.replace_procode(df)
        np.testing.assert_array_equal(df_out['PROCODE'].values,
                                      np.array([0,0,1,1,2,1,np.nan],dtype=np.float32))

    def test_replace_spellid(self):
        df = pd.DataFrame({'PROVSPNOPS':['ABB', 'ABBB', 'XYY', 'ABB', 'XYY',np.nan, 'ABB'],
                           'PROCODE':['ABB', 'ABB', np.nan, 'ABBB', 'XXX', 'XXX', 'ABB'],})
        df_out = load_parquet.replace_spellid(df)
        np.testing.assert_array_equal(df_out['PROVSPNOPS'].values,
                                      np.array([0,1,np.nan,2,3,np.nan,0]))


class Test_acute_chronic(unittest.TestCase):

    #@patch('pipeline_hes.params.params.R', '000000')
    def test_aggregate_codes(self):
        
        # 'B380'	Acute pulmonary coccidioidomycosis	'A'
        # 'B381'	Chronic pulmonary coccidioidomycosis	'C'
        # 'B382'	Pulmonary coccidioidomycosis, unspecified	'A'
        # 'B383'	Cutaneous coccidioidomycosis	'A'
        # 'B384'	Coccidioidomycosis meningitis	'A'
        # 'B387'	Disseminated coccidioidomycosis	'A'
        # 'B3881'	Prostatic coccidioidomycosis	'A'
        # 'B3889'	Other forms of coccidioidomycosis	'A'
        # 'B389'	Coccidioidomycosis, unspecified	'A'

        # 'B390'	Acute pulmonary histoplasmosis capsulati	'A'
        # 'B391'	Chronic pulmonary histoplasmosis capsulati	'A'
        # 'B392'	Pulmonary histoplasmosis capsulati, unspecified	'A'
        # 'B393'	Disseminated histoplasmosis capsulati	'A'
        # 'B394'	Histoplasmosis capsulati, unspecified	'A'
        # 'B395'	Histoplasmosis duboisii	'A'
        # 'B399'	Histoplasmosis, unspecified	'A'
        
        # 'B900'	Sequelae of central nervous system tuberculosis	'C'
        # 'B901'	Sequelae of genitourinary tuberculosis	'C'
        # 'B902'	Sequelae of tuberculosis of bones and joints	'C'
        # 'B908'	Sequelae of tuberculosis of other organs	'C'
        # 'B909'	Sequelae of respiratory and unspecified tuberculosis	'C'

        # B93 = doesnt exist
        
        # 'D690'	Allergic purpura	'A'
        # 'D691'	Qualitative platelet defects	'C'
        # 'D692'	Other nonthrombocytopenic purpura	'A'
        # 'D693'	Immune thrombocytopenic purpura	'C'
        # 'D6941'	Evans syndrome	'C'
        # 'D6942'	Congenital and hereditary thrombocytopenia purpura	'C'
        # 'D6949'	Other primary thrombocytopenia	'C'
        # 'D6951'	Posttransfusion purpura	'A'
        # 'D6959'	Other secondary thrombocytopenia	'A'
        # 'D696'	Thrombocytopenia, unspecified	'C'
        # 'D698'	Other specified hemorrhagic conditions	'A'
        # 'D699'	Hemorrhagic condition, unspecified	'A'

        diags = ['B38','B381','B382','B388','B388888',
                 'B390A','B390','B39',
                 'B900A','B900','B90',
                 'B93','B930',
                 'D6941','D694','D69']
        df = pd.DataFrame({'DIAG_01': diags})
        for i in range(2,21):
            df['DIAG_{:02}'.format(i)] = diags
        df = df.astype('category')
        df_chronic = load_parquet.merge_with_acute_chronic_codes(df)
        
        for i in range(1,21):
            diagField = 'DIAG_{:02}'.format(i)
            acuteField = 'ACUTE_{:02}'.format(i)
            df_exp = pd.DataFrame({diagField: diags,
                                   acuteField: ['X','C','A','A','A',
                                                'A','A','A',
                                                'C','C','C',
                                                np.nan,np.nan,
                                                'C','C','X']})
            df_exp = df_exp.astype({diagField:'category'})
            pd.testing.assert_frame_equal(df_chronic[[diagField,acuteField]],df_exp)
            



if __name__ == '__main__':
    # Show full dataframes
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.expand_frame_repr', False)
    unittest.main(argv=['-v'],verbosity=3)

