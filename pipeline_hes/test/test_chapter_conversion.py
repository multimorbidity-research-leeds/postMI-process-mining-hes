# -*- coding: utf-8 -*-
"""
@author: Chris Hayward
"""



# MISC TESTS


import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pdb

from pipeline_hes import parse_chapters
from pipeline_hes.params import params
from pipeline_hes import load_parquet



class Test_GetDiagConversionDict(unittest.TestCase):
    
    def setUp(self):
        self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0,],
                                'DIAG_01':['A00','B49','C99',],
                                })
        
    def _addRow(self,row):
        self.df = self.df.astype({'DIAG_01':str})
        self.df.loc[self.df.shape[0], ['MYADMIDATE', 'DIAG_01', 'ENCRYPTED_HESID']] = row
        self.df = self.df.astype({'DIAG_01':'category'})
        
    def test_diag_converstion(self):
        chaptersRange = [('A','A',0,99), ('B','B',0,99), ('C','C',0,99)]
        chaptersAbbrv_extended = ['X', 'Y', 'Z']
        conv = parse_chapters.build_diag_conversion_dict(self.df['DIAG_01'], chaptersRange, chaptersAbbrv_extended)
        self.assertDictEqual(conv, {params.AMI_CODE:params.AMI_CODE,
                                    'A00':'X', 'B49':'Y', 'C99':'Z',
                                    params.CENSOR_CODE:params.CENSOR_CODE})
        
    def test_diag_converstion_missing(self):
        chaptersRange = [('A','A',0,99), ('B','B',0,99)]
        chaptersAbbrv_extended = ['X', 'Y']
        conv = parse_chapters.build_diag_conversion_dict(self.df['DIAG_01'], chaptersRange, chaptersAbbrv_extended)
        self.assertDictEqual(conv, {params.AMI_CODE:params.AMI_CODE,
                                    'A00':'X', 'B49':'Y', 'C99':'nomatch',
                                    params.CENSOR_CODE:params.CENSOR_CODE})

    def test_diag_converstion_missing2(self):
        chaptersRange = [('A','A',0,99), ('B','B',0,99), ('F','F',0,99)]
        chaptersAbbrv_extended = ['X', 'Y', 'Z']
        self._addRow([np.datetime64('2005-01'), 'D00', 10])
        conv = parse_chapters.build_diag_conversion_dict(self.df['DIAG_01'], chaptersRange, chaptersAbbrv_extended)
        self.assertDictEqual(conv, {params.AMI_CODE:params.AMI_CODE,
                                    'A00':'X', 'B49':'Y', 'C99':'nomatch', 'D00':'nomatch',
                                    params.CENSOR_CODE:params.CENSOR_CODE})

    def test_diag_converstion_grouped(self):
        chaptersRange = [('A','C',0,99)]
        chaptersAbbrv_extended = ['X']
        conv = parse_chapters.build_diag_conversion_dict(self.df['DIAG_01'], chaptersRange, chaptersAbbrv_extended)
        self.assertDictEqual(conv, {params.AMI_CODE:params.AMI_CODE,
                                    'A00':'X', 'B49':'X', 'C99':'X',
                                    params.CENSOR_CODE:params.CENSOR_CODE})

    def test_diag_converstion_some_grouped(self):
        chaptersRange = [('A','B',0,30), ('B','C',31,99)]
        chaptersAbbrv_extended = ['X', 'Y']
        conv = parse_chapters.build_diag_conversion_dict(self.df['DIAG_01'], 
                                                       chaptersRange,
                                                       chaptersAbbrv_extended)
        self.assertDictEqual(conv, {params.AMI_CODE:params.AMI_CODE,
                                    'A00':'X', 'B49':'Y', 'C99':'Y',
                                    params.CENSOR_CODE:params.CENSOR_CODE})

    def test_diag_converstion_some_grouped2(self):
        chaptersRange = [('A','B',0,59), ('B','C',60,99)]
        chaptersAbbrv_extended = ['X', 'Y']
        conv = parse_chapters.build_diag_conversion_dict(self.df['DIAG_01'], 
                                                       chaptersRange, 
                                                       chaptersAbbrv_extended)
        self.assertDictEqual(conv, {params.AMI_CODE:params.AMI_CODE,
                                    'A00':'X', 'B49':'X', 'C99':'Y',
                                    params.CENSOR_CODE:params.CENSOR_CODE})


class Test_apply_diag_conversion_dict(unittest.TestCase):
    
    def setUp(self):
        self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0,0,],
                                'DIAG_01':['A00','B49','C85','C99'],
                                })
        self.df = self.df.astype({'DIAG_01':'category'})
        
    @patch('pipeline_hes.params.params.CHAPTER_HEADINGS_USE_GRANULAR',False)
    def test_apply_diag_conversion_dict(self):
        df_out = parse_chapters.apply_diag_conversion_dict(self.df)
        np.testing.assert_array_equal(df_out['DIAG_01_CONV_LOW'],
                                      ['A00-A09','B35-B49','C81-C96','nomatch'])
        np.testing.assert_array_equal(df_out['DIAG_01_CONV_HIGH'],
                                      ['A00-B99','A00-B99','C00-D48','C00-D48'])
        np.testing.assert_array_equal(df_out['DIAG_01_ALT2'],
                                      ['A00','B49','C85','C98'])
        
    @patch('pipeline_hes.params.params.CHAPTER_HEADINGS_USE_GRANULAR',True)
    def test_apply_diag_conversion_dict(self):
        df_tmp = self.df.copy()
        parse_chapters.apply_diag_conversion_dict(df_tmp)
        np.testing.assert_array_equal(df_tmp['DIAG_01_CONV_LOW'],
                                      ['A00-A09','B35-B49','C81-C96','nomatch'])
        np.testing.assert_array_equal(df_tmp['DIAG_01_CONV_HIGH'],
                                      ['A00-B99','A00-B99','C00-D48','C00-D48'])
        np.testing.assert_array_equal(df_tmp['DIAG_01'],
                                      ['A00','B49','C85','C99'])


class Test_parse_chapter_txt_file(unittest.TestCase):
    
    def setUp(self):
        self.df = pd.DataFrame({'DIAG_01':['A00',
                                           'B49',
                                           'C99',
                                           'D50',
                                           'E11',
                                           'F01',
                                           'G20',
                                           'H00',
                                           'I21',
                                           'X00'],})
        self.df['ENCRYPTED_HESID']=0


    def _check_tbl(self, _chaptersRaw):
        chaptersRange, chaptersAbbrv_extended = \
            parse_chapters.parse_chapter_txt_file(_chaptersRaw)
        tbl_conv = parse_chapters.build_diag_conversion_dict(self.df['DIAG_01'],
                                                       chaptersRange,
                                                       chaptersAbbrv_extended)
        #pdb.set_trace()
        x=pd.Series(tbl_conv.values(), index=tbl_conv.keys())
        x=x.loc[x!=params.CHAPTER_NO_MATCH]
        x=x.loc[x!=params.AMI_CODE]
        x=x.loc[x!=params.CENSOR_CODE]
        
        # #####
        # Check first letter in range
        # ####
        #pdb.set_trace()
        self.assertTrue(
            pd.Series(np.array([y[0] for y in x.index]) >= \
                      np.array([y[0] for y in x.values])).all())
        self.assertTrue(
            pd.Series(np.array([y[0] for y in x.index]) <= \
                      np.array([y[4] for y in x.values])).all())
        
        # #####
        # Check the number
        # #####
        ### Equal to first letter
        x1 = x.loc[np.array([y[0] for y in x.index]) == np.array([y[0] for y in x.values])]
        # Check number greater than initial
        #pdb.set_trace()
        self.assertTrue(
            pd.Series(np.array([int(y[1:3]) for y in x1.index]) >= \
                      np.array([int(y[1:3]) for y in x1.values])).all())
        
        ### Equal to last letter
        x2 = x.loc[np.array([y[0] for y in x.index]) == \
                   np.array([y[4] for y in x.values])]
        # Check number less than final cutoff
        self.assertTrue(
            pd.Series(np.array([int(y[1:3]) for y in x2.index]) <=\
                      np.array([int(y[5:7]) for y in x2.values])).all())
        ### Equal to no letter (in-between)
        x3 = x.loc[np.logical_and(
            np.array([y[0] for y in x.index]) != np.array([y[0] for y in x.values]),
            np.array([y[0] for y in x.index]) != np.array([y[4] for y in x.values]))]
        self.assertTrue(
            pd.Series(np.array([int(y[1:3]) for y in x3.index]) >= 0).all())
        self.assertTrue(
            pd.Series(np.array([int(y[1:3]) for y in x3.index]) <= 99).all())
        
        #pdb.set_trace()

        
    def test_chapter_conversion_coarse(self):
        _chaptersRaw_coarse = parse_chapters._read_chapter_file_highlevel()
        self._check_tbl(_chaptersRaw_coarse)

    def test_chapter_conversion_granular(self):
        _chaptersRaw_granular = parse_chapters._read_chapter_file_lowlevel()
        self._check_tbl(_chaptersRaw_granular)



if __name__ == '__main__':
    # Show full dataframes
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.expand_frame_repr', False)
    unittest.main(argv=['-v'],verbosity=3)

