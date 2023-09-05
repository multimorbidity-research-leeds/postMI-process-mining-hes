# -*- coding: utf-8 -*-
"""
@author: Chris Hayward
"""

# Test filter sort

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pdb

from pipeline_hes import filter_hes_sort
from pipeline_hes.params import params


REPEATS=10

class Test_count_number_of_individuals_with_undetermined_spell_order(unittest.TestCase):
    
    
    def setUp(self):
        # just one subject is fine...
        # The important thing is how we sort WITHIN subjects
        self.df = pd.DataFrame({'ENCRYPTED_HESID':np.repeat(0,18),
                                'SURVIVALTIME_SPELLMIN':np.repeat(60,18),
                                'SURVIVALTIME_SPELLMAX':np.repeat(60,18),
                                'MYEPISTART_SPELLMIN':
                                    np.repeat(np.datetime64('2005-01-01'),18),
                                'MYEPISTART_SPELLMAX':
                                    np.repeat(np.datetime64('2005-01-01'),18),
                                'MYEPIEND_SPELLMIN':
                                    np.repeat(np.datetime64('2005-01-01'),18),
                                'MYEPIEND_SPELLMAX':
                                    np.repeat(np.datetime64('2005-01-01'),18),
                                'DISDATE':
                                    np.repeat(np.datetime64('2005-01-01'),18),
                                'PROVSPNOPS':np.repeat('XXX',18),
                                })
    
    # one subject, one spell
    def test_count_number_of_individuals_with_undetermined_spell_order1(self):
        df_tmp = self.df.copy()
        frac_ambiguous = filter_hes_sort.\
            count_number_of_individuals_with_undetermined_spell_order(df_tmp)

        self.assertEqual(frac_ambiguous[0],0)
        self.assertEqual(frac_ambiguous[1],1)

    # one subject, two spells
    def test_count_number_of_individuals_with_undetermined_spell_order2(self):
        df_tmp = self.df.copy()
        df_tmp['PROVSPNOPS'] = np.repeat(['XXX','YYY'],9)

        frac_ambiguous = filter_hes_sort.\
            count_number_of_individuals_with_undetermined_spell_order(df_tmp)

        self.assertEqual(frac_ambiguous[0],1)
        self.assertEqual(frac_ambiguous[1],1)


    # 18 subjects, one spell each
    def test_count_number_of_individuals_with_undetermined_spell_order3(self):
        df_tmp = self.df.copy()
        df_tmp['ENCRYPTED_HESID'] = np.array(np.linspace(0,17,18),dtype=int)

        frac_ambiguous = filter_hes_sort.\
            count_number_of_individuals_with_undetermined_spell_order(df_tmp)

        self.assertEqual(frac_ambiguous[0],0)
        self.assertEqual(frac_ambiguous[1],18)
        
    # 2 subjects, one spell each
    def test_count_number_of_individuals_with_undetermined_spell_order4(self):
        df_tmp = self.df.copy()
        df_tmp['ENCRYPTED_HESID'] = np.repeat([1,2],9)
        df_tmp['PROVSPNOPS'] = np.repeat(['XXX','YYY'],9)

        frac_ambiguous = filter_hes_sort.\
            count_number_of_individuals_with_undetermined_spell_order(df_tmp)
        self.assertEqual(frac_ambiguous[0],0)
        self.assertEqual(frac_ambiguous[1],2)
        
    # 2 subjects, one with two spells, one with one
    def test_count_number_of_individuals_with_undetermined_spell_order5(self):
        df_tmp = self.df.copy()
        df_tmp['ENCRYPTED_HESID'] = np.repeat([1,2],9)
        df_tmp['PROVSPNOPS'] = np.concatenate([
            np.repeat('XXX',4),
            np.repeat('YYY',5),
            np.repeat('YYY',4),
            np.repeat('YYY',5)])

        frac_ambiguous = filter_hes_sort.\
            count_number_of_individuals_with_undetermined_spell_order(df_tmp)
        self.assertEqual(frac_ambiguous[0],1)
        self.assertEqual(frac_ambiguous[1],2)



class Test_strictly_order_events(unittest.TestCase):
    
    def setUp(self):
        # just one subject is fine...
        # The important thing is how we sort WITHIN subjects
        self.df = pd.DataFrame({'ENCRYPTED_HESID':np.repeat(0,18),
                                'SURVIVALTIME_SPELLMIN':np.repeat(60,18),
                                'SURVIVALTIME_SPELLMAX':np.repeat(60,18),
                                'MYEPISTART_SPELLMIN':np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),6),
                                    np.repeat(np.datetime64('2005-01-01'),6),
                                    np.repeat(np.datetime64('2005-01-01'),6),
                                    ],axis=None),
                                'MYEPISTART_SPELLMAX':np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),6),
                                    np.repeat(np.datetime64('2005-01-01'),6),
                                    np.repeat(np.datetime64('2005-01-01'),6),
                                    ],axis=None),
                                'MYEPIEND_SPELLMIN':np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),6),
                                    np.repeat(np.datetime64('2005-01-01'),6),
                                    np.repeat(np.datetime64('2005-01-01'),6),
                                    ],axis=None),
                                'MYEPIEND_SPELLMAX':np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),6),
                                    np.repeat(np.datetime64('2005-01-01'),6),
                                    np.repeat(np.datetime64('2005-01-01'),6),
                                    ],axis=None),
                                'DISDATE':np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),6),
                                    np.repeat(np.datetime64('2005-01-01'),6),
                                    np.repeat(np.datetime64('2005-01-01'),6),
                                    ],axis=None),
                                'AMB_ORDER':[1,1,1,1,1,1,
                                             2,2,2,2,2,2,
                                             3,3,3,3,3,3],
                                'EPIORDER':[1,1,2,2,3,3,
                                            1,1,2,2,3,3,
                                            1,1,2,2,3,3,],
                                'COL_IDX':[1,2,1,2,1,2,
                                           1,2,1,2,1,2,
                                           1,2,1,2,1,2],
                                'PROVSPNOPS':['XXX','XXX','XXX','XXX','XXX','XXX',
                                              'YYY','YYY','YYY','YYY','YYY','YYY',
                                              'ZZZ','ZZZ','ZZZ','ZZZ','ZZZ','ZZZ',],
                                })


    def test_strictly_order_events_nochange(self):
        df_tmp = self.df.copy()
        filter_hes_sort.strictly_order_events(df_tmp)

        df_exp = self.df.copy()
        # expected order is INT
        df_exp['ORDER'] = np.arange(df_exp.shape[0])

        #pdb.set_trace()
        pd.testing.assert_frame_equal(df_tmp, df_exp, check_like=True)


    # AMB_ORDER causes the spells to become ordered as originally ordered
    def test_strictly_order_events_shuffle_spells(self):
        df_orig = self.df.copy()
        df = pd.concat([df_orig.loc[6:11],
                        df_orig.loc[0:5],
                        df_orig.loc[12:17],], ignore_index=True).reset_index(drop=True)
        df_tmp = df.copy()
        filter_hes_sort.strictly_order_events(df_tmp)
        
        df_exp = self.df.copy()        
        # expected order is INT
        df_exp['ORDER'] = np.arange(df_exp.shape[0])
        pd.testing.assert_frame_equal(df_tmp, df_exp, check_like=True)


    # col_idx becomes ordered (1,2,1,2,1,2)
    def test_strictly_order_events_shuffle_col_idx(self):
        df_orig = self.df.copy()
        df_orig.loc[0:5,'COL_IDX'] = [2,1,2,1,2,1]
        
        df_tmp = df_orig.copy()
        filter_hes_sort.strictly_order_events(df_tmp)
        
        df_exp = self.df.copy()        
        # expected order is INT
        df_exp['ORDER'] = np.arange(df_exp.shape[0])
        pd.testing.assert_frame_equal(df_tmp, df_exp, check_like=True)


    # epiorder becomes ordered [1,1,2,2,3,3]
    def test_strictly_order_events_shuffle_epiorder(self):
        df_orig = self.df.copy()

        df_orig.loc[0:5,'EPIORDER'] = [3,1,2,3,1,2]
        df_orig.loc[0:5,'COL_IDX'] = [1,1,1,2,2,2]
        
        df_tmp = self.df.copy()
        filter_hes_sort.strictly_order_events(df_tmp)
        
        #pdb.set_trace()
        df_exp = self.df.copy()        
        # expected order is INT
        df_exp['ORDER'] = np.arange(df_exp.shape[0])
        pd.testing.assert_frame_equal(df_tmp, df_exp, check_like=True)


    # AMB_ORDER causes a new ordering of spells
    def test_strictly_order_events_shuffle_amb_order(self):
        df_orig = self.df.copy()
        df_orig['AMB_ORDER'] = [2,2,2,2,2,2,
                                1,1,1,1,1,1,
                                3,3,3,3,3,3]
        
        df_tmp = df_orig.copy()
        filter_hes_sort.strictly_order_events(df_tmp)
        df_exp = self.df.copy()
        # expected order is INT
        df_exp = pd.concat([df_exp.loc[6:11],
                            df_exp.loc[0:5],
                            df_exp.loc[12:17],], ignore_index=True).reset_index(drop=True)
        df_exp['AMB_ORDER'] = [1,1,1,1,1,1,
                               2,2,2,2,2,2,
                               3,3,3,3,3,3]
        df_exp['ORDER'] = np.arange(df_exp.shape[0])
        pd.testing.assert_frame_equal(df_tmp, df_exp, check_like=True)


    # DISDATE causes a new ordering of spells
    def test_strictly_order_events_shuffle_disdate(self):
        df_orig = self.df.copy()
        df_orig['DISDATE'] = np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),6),
                                    np.repeat(np.datetime64('2007-01-01'),6),
                                    np.repeat(np.datetime64('2006-01-01'),6),
                                    ],axis=None)
        
        df_tmp = df_orig.copy()
        filter_hes_sort.strictly_order_events(df_tmp)
        
        df_exp = self.df.copy()        
        # expected order is INT
        df_exp = pd.concat([df_exp.loc[0:5],
                            df_exp.loc[12:17],
                            df_exp.loc[6:11],], ignore_index=True).reset_index(drop=True)
        df_exp['DISDATE'] = np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),6),
                                    np.repeat(np.datetime64('2006-01-01'),6),
                                    np.repeat(np.datetime64('2007-01-01'),6),
                                    ],axis=None)

        df_exp['ORDER'] = np.arange(df_exp.shape[0])
        pd.testing.assert_frame_equal(df_tmp, df_exp, check_like=True)
        

    # # MYADMIDATE causes a new ordering of spells
    # def test_strictly_order_events_shuffle_myadmidates(self):
    #     df_orig = self.df.copy()
    #     df_orig['MYADMIDATE'] = np.concatenate([
    #                                 np.repeat(np.datetime64('2005-01-01'),6),
    #                                 np.repeat(np.datetime64('2007-01-01'),6),
    #                                 np.repeat(np.datetime64('2006-01-01'),6),
    #                                 ],axis=None)
        
    #     df_out = filter_hes_sort.strictly_order_events(df_orig)
        
    #     df_exp = self.df.copy()        
    #     # expected order is INT
    #     df_exp = pd.concat([df_exp.loc[0:5],
    #                         df_exp.loc[12:17],
    #                         df_exp.loc[6:11],]).reset_index(drop=True)
    #     df_exp['MYADMIDATE'] = np.concatenate([
    #                                 np.repeat(np.datetime64('2005-01-01'),6),
    #                                 np.repeat(np.datetime64('2006-01-01'),6),
    #                                 np.repeat(np.datetime64('2007-01-01'),6),
    #                                 ],axis=None)

    #     df_exp['ORDER'] = np.arange(df_exp.shape[0], dtype=np.int64)
    #     pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)


    # SURVIVALTIME causes a new ordering of spells
    def test_strictly_order_events_shuffle_survivaltime_spellmin(self):
        df_orig = self.df.copy()
        df_orig['SURVIVALTIME_SPELLMIN'] = [30,30,30,30,30,30,
                                            10,10,10,10,10,10,
                                            40,40,40,40,40,40]
        df_tmp = df_orig.copy()
        filter_hes_sort.strictly_order_events(df_tmp)
        df_exp = self.df.copy()        
        # expected order is INT
        df_exp = pd.concat([df_exp.loc[12:17],
                            df_exp.loc[0:5],
                            df_exp.loc[6:11],], ignore_index=True)
        df_exp['SURVIVALTIME_SPELLMIN'] = [40,40,40,40,40,40,
                                           30,30,30,30,30,30,
                                           10,10,10,10,10,10,]
        df_exp['ORDER'] = np.arange(df_exp.shape[0])
        pd.testing.assert_frame_equal(df_tmp, df_exp, check_like=True)


    # SURVIVALTIME causes a new ordering of spells
    def test_strictly_order_events_shuffle_survivaltime_spellmax(self):
        df_orig = self.df.copy()
        df_orig['SURVIVALTIME_SPELLMAX'] = [30,30,30,30,30,30,
                                            10,10,10,10,10,10,
                                            40,40,40,40,40,40]

        df_tmp = df_orig.copy()
        filter_hes_sort.strictly_order_events(df_tmp)
        df_exp = self.df.copy()        
        # expected order is INT
        df_exp = pd.concat([df_exp.loc[12:17],
                            df_exp.loc[0:5],
                            df_exp.loc[6:11],], ignore_index=True)
        df_exp['SURVIVALTIME_SPELLMAX'] = [40,40,40,40,40,40,
                                           30,30,30,30,30,30,
                                           10,10,10,10,10,10,]
        df_exp['ORDER'] = np.arange(df_exp.shape[0])
        pd.testing.assert_frame_equal(df_tmp, df_exp, check_like=True)


    # MYEPISTART causes a new ordering of spells
    def test_strictly_order_events_shuffle_epistart_spellmin(self):
        df_orig = self.df.copy()
        df_orig['MYEPISTART_SPELLMIN'] = np.concatenate([
                                    np.repeat(np.datetime64('2006-01-01'),6),
                                    np.repeat(np.datetime64('2007-01-01'),6),
                                    np.repeat(np.datetime64('2005-01-01'),6),
                                    ],axis=None)
        df_tmp = df_orig.copy()
        filter_hes_sort.strictly_order_events(df_tmp)
        df_exp = self.df.copy()        
        # expected order is INT
        df_exp = pd.concat([df_exp.loc[12:17],
                            df_exp.loc[0:5],
                            df_exp.loc[6:11],], ignore_index=True)
        df_exp['MYEPISTART_SPELLMIN'] = np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),6),
                                    np.repeat(np.datetime64('2006-01-01'),6),
                                    np.repeat(np.datetime64('2007-01-01'),6),
                                    ],axis=None)
        df_exp['ORDER'] = np.arange(df_exp.shape[0])
        pd.testing.assert_frame_equal(df_tmp, df_exp, check_like=True)


    # MYEPISTART causes a new ordering of spells
    def test_strictly_order_events_shuffle_epistart_spellmax(self):
        df_orig = self.df.copy()
        df_orig['MYEPISTART_SPELLMAX'] = np.concatenate([
                                    np.repeat(np.datetime64('2006-01-01'),6),
                                    np.repeat(np.datetime64('2007-01-01'),6),
                                    np.repeat(np.datetime64('2005-01-01'),6),
                                    ],axis=None)
        df_tmp = df_orig.copy()
        filter_hes_sort.strictly_order_events(df_tmp)
        df_exp = self.df.copy()        
        # expected order is INT
        df_exp = pd.concat([df_exp.loc[12:17],
                            df_exp.loc[0:5],
                            df_exp.loc[6:11],], ignore_index=True)
        df_exp['MYEPISTART_SPELLMAX'] = np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),6),
                                    np.repeat(np.datetime64('2006-01-01'),6),
                                    np.repeat(np.datetime64('2007-01-01'),6),
                                    ],axis=None)
        df_exp['ORDER'] = np.arange(df_exp.shape[0])
        pd.testing.assert_frame_equal(df_tmp, df_exp, check_like=True)


    def test_strictly_order_events_shuffle_epiend_spellmin(self):
        df_orig = self.df.copy()
        df_orig['MYEPIEND_SPELLMIN'] = np.concatenate([
                                    np.repeat(np.datetime64('2006-01-01'),6),
                                    np.repeat(np.datetime64('2007-01-01'),6),
                                    np.repeat(np.datetime64('2005-01-01'),6),
                                    ],axis=None)
        df_tmp = df_orig.copy()
        filter_hes_sort.strictly_order_events(df_tmp)
        df_exp = self.df.copy()        
        # expected order is INT
        df_exp = pd.concat([df_exp.loc[12:17],
                            df_exp.loc[0:5],
                            df_exp.loc[6:11],], ignore_index=True)
        df_exp['MYEPIEND_SPELLMIN'] = np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),6),
                                    np.repeat(np.datetime64('2006-01-01'),6),
                                    np.repeat(np.datetime64('2007-01-01'),6),
                                    ],axis=None)
        df_exp['ORDER'] = np.arange(df_exp.shape[0])
        pd.testing.assert_frame_equal(df_tmp, df_exp, check_like=True)


    def test_strictly_order_events_shuffle_epiend_spellmax(self):
        df_orig = self.df.copy()
        df_orig['MYEPIEND_SPELLMAX'] = np.concatenate([
                                    np.repeat(np.datetime64('2006-01-01'),6),
                                    np.repeat(np.datetime64('2007-01-01'),6),
                                    np.repeat(np.datetime64('2005-01-01'),6),
                                    ],axis=None)
        df_tmp = df_orig.copy()
        filter_hes_sort.strictly_order_events(df_tmp)
        df_exp = self.df.copy()        
        # expected order is INT
        df_exp = pd.concat([df_exp.loc[12:17],
                            df_exp.loc[0:5],
                            df_exp.loc[6:11],], ignore_index=True)
        df_exp['MYEPIEND_SPELLMAX'] = np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),6),
                                    np.repeat(np.datetime64('2006-01-01'),6),
                                    np.repeat(np.datetime64('2007-01-01'),6),
                                    ],axis=None)
        df_exp['ORDER'] = np.arange(df_exp.shape[0])
        pd.testing.assert_frame_equal(df_tmp, df_exp, check_like=True)


    def test_strictly_order_events_shuffle_disdate(self):
        df_orig = self.df.copy()
        df_orig['DISDATE'] = np.concatenate([
                                    np.repeat(np.datetime64('2006-01-01'),6),
                                    np.repeat(np.datetime64('2007-01-01'),6),
                                    np.repeat(np.datetime64('2005-01-01'),6),
                                    ],axis=None)
        df_tmp = df_orig.copy()
        filter_hes_sort.strictly_order_events(df_tmp)
        df_exp = self.df.copy()        
        # expected order is INT
        df_exp = pd.concat([df_exp.loc[12:17],
                            df_exp.loc[0:5],
                            df_exp.loc[6:11],], ignore_index=True)
        df_exp['DISDATE'] = np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),6),
                                    np.repeat(np.datetime64('2006-01-01'),6),
                                    np.repeat(np.datetime64('2007-01-01'),6),
                                    ],axis=None)
        df_exp['ORDER'] = np.arange(df_exp.shape[0])
        pd.testing.assert_frame_equal(df_tmp, df_exp, check_like=True)


    # Multiple subjects
    def test_strictly_order_events_mulitple_subjects(self):
        df_sub1 = self.df.copy()
        df_sub2 = self.df.copy()
        df_sub2['ENCRYPTED_HESID'] = 1
       
        df_in = pd.concat([df_sub1,df_sub2], ignore_index=True)
        df_tmp = df_in.copy()
        filter_hes_sort.strictly_order_events(df_tmp)
        # no change
        df_exp = df_in.copy()
        df_exp['ORDER'] = np.arange(df_exp.shape[0])
        pd.testing.assert_frame_equal(df_tmp, df_exp, check_like=True)



    # Multiple subjects
    def test_strictly_order_events_mulitple_subjects_shuffle(self):
        df_sub1 = self.df.copy()
        df_sub2 = self.df.copy()
        df_sub2['ENCRYPTED_HESID'] = 1
       
        df_in = pd.concat([df_sub1,df_sub2], ignore_index=True)
                
        # shuffle
        for i in range(REPEATS):
            df_in_shuffled = df_in.copy()
            tmp_order = np.arange(df_in_shuffled.shape[0])
            np.random.default_rng(i).shuffle(tmp_order)
            df_in_shuffled['TMP_ORDER'] = tmp_order
            df_in_shuffled = df_in_shuffled.sort_values('TMP_ORDER').reset_index(drop=True)
            df_in_shuffled.drop(columns=['TMP_ORDER'], inplace=True)
           
            df_tmp = df_in_shuffled.copy()
            filter_hes_sort.strictly_order_events(df_tmp)
            df_exp = df_in.copy()
            
            #pdb.set_trace()
            
            df_exp['ORDER'] = np.arange(df_exp.shape[0])
            pd.testing.assert_frame_equal(df_tmp, df_exp, check_like=True)



    ##################### HIERARCHY CHECK
    

    # # MYADMIDATE causes a new ordering of spells
    # # DISDATE and SURVIVALTIME and AMB_ORDER have no effect
    # def test_strictly_order_events_shuffle_myadmidate_no_effect_other(self):
    #     df_orig = self.df.copy()
    #     df_orig['MYADMIDATE'] = np.concatenate([
    #                                 np.repeat(np.datetime64('2005-01-01'),6),
    #                                 np.repeat(np.datetime64('2007-01-01'),6),
    #                                 np.repeat(np.datetime64('2006-01-01'),6),
    #                                 ],axis=None)
    #     # no effect
    #     df_orig['DISDATE'] = np.concatenate([
    #                                 np.repeat(np.datetime64('2007-01-01'),6),
    #                                 np.repeat(np.datetime64('2006-01-01'),6),
    #                                 np.repeat(np.datetime64('2005-01-01'),6),
    #                                 ],axis=None)
    #     # no effect
    #     df_orig['SURVIVALTIME_SPELLAVG'] = [40,40,40,40,40,40,
    #                                        10,10,10,10,10,10,
    #                                        30,30,30,30,30,30,]
        
    #     df_orig['MYEPISTART_SPELLAVG'] = np.concatenate([
    #                                 np.repeat(np.datetime64('2006-01-01'),6),
    #                                 np.repeat(np.datetime64('2007-01-01'),6),
    #                                 np.repeat(np.datetime64('2005-01-01'),6),
    #                                 ],axis=None)

    #     # no effect
    #     df_orig['AMB_ORDER'] = [2,2,2,2,2,2,
    #                             1,1,1,1,1,1,
    #                             3,3,3,3,3,3,]
    #     df_out = filter_hes_sort.strictly_order_events(df_orig)
        
    #     df_exp = self.df.copy()        
    #     # expected order is INT
    #     df_exp = pd.concat([df_exp.loc[0:5],
    #                         df_exp.loc[12:17],
    #                         df_exp.loc[6:11],]).reset_index(drop=True)
    #     df_exp['MYADMIDATE'] = np.concatenate([
    #                                 np.repeat(np.datetime64('2005-01-01'),6),
    #                                 np.repeat(np.datetime64('2006-01-01'),6),
    #                                 np.repeat(np.datetime64('2007-01-01'),6),
    #                                 ],axis=None)
    #     df_exp['DISDATE'] = np.concatenate([
    #                                 np.repeat(np.datetime64('2007-01-01'),6),
    #                                 np.repeat(np.datetime64('2005-01-01'),6),
    #                                 np.repeat(np.datetime64('2006-01-01'),6),
    #                                 ],axis=None)
    #     df_exp['SURVIVALTIME_SPELLAVG'] = [40,40,40,40,40,40,
    #                                        30,30,30,30,30,30,
    #                                        10,10,10,10,10,10,]
    #     df_exp['MYEPISTART_SPELLAVG'] = np.concatenate([
    #                                 np.repeat(np.datetime64('2006-01-01'),6),
    #                                 np.repeat(np.datetime64('2005-01-01'),6),
    #                                 np.repeat(np.datetime64('2007-01-01'),6),
    #                                 ],axis=None)

    #     df_exp['AMB_ORDER'] = [2,2,2,2,2,2,
    #                            3,3,3,3,3,3,
    #                            1,1,1,1,1,1,]

    #     df_exp['ORDER'] = np.arange(df_exp.shape[0], dtype=np.int64)
    #     pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)



class Test_strictly_order_events_sort_order(unittest.TestCase):
    
    def setUp(self):
        # just one subject is fine...
        # The important thing is how we sort WITHIN subjects
        self.df = pd.DataFrame({'ENCRYPTED_HESID':np.repeat(0,18),
                                'SURVIVALTIME_SPELLMIN':
                                    [40,40,40,40,40,40,
                                     30,30,30,30,30,30,
                                     10,10,10,10,10,10,],
                                'SURVIVALTIME_SPELLMAX':
                                    [40,40,40,40,40,40,
                                     30,30,30,30,30,30,
                                     10,10,10,10,10,10,],
                                'MYEPISTART_SPELLMIN':np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),6),
                                    np.repeat(np.datetime64('2006-01-01'),6),
                                    np.repeat(np.datetime64('2007-01-01'),6),
                                    ],axis=None),
                                'MYEPISTART_SPELLMAX':np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),6),
                                    np.repeat(np.datetime64('2006-01-01'),6),
                                    np.repeat(np.datetime64('2007-01-01'),6),
                                    ],axis=None),
                                'MYEPIEND_SPELLMIN':np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),6),
                                    np.repeat(np.datetime64('2006-01-01'),6),
                                    np.repeat(np.datetime64('2007-01-01'),6),
                                    ],axis=None),
                                'MYEPIEND_SPELLMAX':np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),6),
                                    np.repeat(np.datetime64('2006-01-01'),6),
                                    np.repeat(np.datetime64('2007-01-01'),6),
                                    ],axis=None),
                                'DISDATE':np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),6),
                                    np.repeat(np.datetime64('2006-01-01'),6),
                                    np.repeat(np.datetime64('2007-01-01'),6),
                                    ],axis=None),
                                'AMB_ORDER':[1,1,1,1,1,1,
                                             2,2,2,2,2,2,
                                             3,3,3,3,3,3],
                                'EPIORDER':[1,1,2,2,3,3,
                                            1,1,2,2,3,3,
                                            1,1,2,2,3,3,],
                                'COL_IDX':[1,2,1,2,1,2,
                                           1,2,1,2,1,2,
                                           1,2,1,2,1,2],
                                'PROVSPNOPS':['XXX','XXX','XXX','XXX','XXX','XXX',
                                              'YYY','YYY','YYY','YYY','YYY','YYY',
                                              'ZZZ','ZZZ','ZZZ','ZZZ','ZZZ','ZZZ',],
                                })


    def test_strictly_order_events_sort_order(self):
        ordered_cols = np.array(['SURVIVALTIME_SPELLMIN',
                                'SURVIVALTIME_SPELLMAX',
                                'MYEPISTART_SPELLMIN',
                                'MYEPISTART_SPELLMAX',
                                'MYEPIEND_SPELLMIN',
                                'MYEPIEND_SPELLMAX',
                                'DISDATE',
                                'AMB_ORDER'])
        # swap the spells,
        df_shuff = pd.concat([self.df.loc[6:11],
                               self.df.loc[0:5],
                               self.df.loc[12:17]],ignore_index=True)

        for i,col in enumerate(ordered_cols):
            print(col)
            
            # set column of interest as ordered (stays ordered from here on)
            df_shuff[col] = self.df[col].copy()          
            df_tmp = df_shuff.copy()
            filter_hes_sort.strictly_order_events(df_tmp)

            # check subordinate cols havent changed
            for j in range(i+1,len(ordered_cols)):
                print(' {}'.format(ordered_cols[j]))

                # remains unordered
                np.testing.assert_array_equal(df_shuff[ordered_cols[j]],
                                              df_tmp[ordered_cols[j]])
                



    ###################

    # def test_strictly_order_events_rely_myadmidate(self):

    #     # should have no effect
    #     self.df['AMB_ORDER'] = [2,3,1,
    #                             1,2,3,
    #                             1,3,2,]
    #     df_out = filter_hes_sort.strictly_order_events(self.df.copy())
        
    #     df_exp = self.df.copy()
    #     df_exp['_MYADMIDATE'] = df_exp['MYADMIDATE']
        
    #     # expected order is INT
    #     df_exp['MYADMIDATE'] = np.arange(df_exp.shape[0], dtype=np.int64)
        
        
    #     pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)


    # def test_strictly_order_events_rely_amb_order(self):
    #     self.df['MYADMIDATE'] = np.concatenate([
    #                                 np.repeat(np.datetime64('2005-01-01'),3),
    #                                 np.repeat(np.datetime64('2006-01-01'),3),
    #                                 np.repeat(np.datetime64('2007-01-01'),3),
    #                                 ],axis=None, dtype='datetime64[M]')
    #     self.df['SURVIVALTIME'] = [np.nan,np.nan,np.nan,
    #                                np.nan,np.nan,np.nan,
    #                                np.nan,np.nan,np.nan]
    #     self.df['DISDATE'] = self.df['MYADMIDATE']
    #     self.df['AMB_ORDER'] = [2,3,1,
    #                             1,2,3,
    #                             1,3,2,]

        
    #     df_out = filter_hes_sort.strictly_order_events(self.df.copy())
        
        
    #     df_exp = self.df.copy()
    #     df_exp['_MYADMIDATE'] = df_exp['MYADMIDATE']
        
    #     # expected order is INT
    #     df_exp['MYADMIDATE'] = np.arange(df_exp.shape[0], dtype=np.int64)
    #     df_exp['AMB_ORDER'] = [1,2,3,
    #                            1,2,3,
    #                            1,2,3,]
        
    #     pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)


    # def test_strictly_order_events_rely_mixed(self):
    #     self.df['MYADMIDATE'] = np.concatenate([
    #                                 [np.datetime64('2005-01-01'),
    #                                  np.datetime64('2005-01-01'),
    #                                  np.datetime64('2005-01-01'), ],
    #                                 [np.datetime64('2006-01-01'),
    #                                  np.datetime64('2006-01-01'),
    #                                  np.datetime64('2006-01-01'), ],
    #                                 np.arange('2007-01', '2007-04', dtype='datetime64[M]'),
    #                                 ],axis=None)
        
    #     self.df['SURVIVALTIME'] = [np.nan,np.nan,np.nan,
    #                                10,30,40,
    #                                np.nan,np.nan,np.nan]
        
    #     self.df['DISDATE'] = np.concatenate([
    #                                 [np.datetime64('2005-01-01'),
    #                                  np.datetime64('2005-03-01'),
    #                                  np.datetime64('2005-02-01'), ],
    #                                 [np.datetime64('2006-01-01'),
    #                                  np.datetime64('2006-01-01'),
    #                                  np.datetime64('2006-01-01'), ],
    #                                 np.arange('2007-01', '2007-04', dtype='datetime64[M]'),
    #                                 ],axis=None)

    #     self.df['AMB_ORDER'] = [2,3,1,
    #                             1,2,3,
    #                             1,3,2,]

        
    #     df_out = filter_hes_sort.strictly_order_events(self.df.copy())
        
        
        
    #     df_exp = self.df.copy()
    #     df_exp['_MYADMIDATE'] = df_exp['MYADMIDATE']
        
    #     # expected order is INT
    #     df_exp['MYADMIDATE'] = np.arange(df_exp.shape[0], dtype=np.int64)
    #     df_exp['SURVIVALTIME'] = [np.nan,np.nan,np.nan,
    #                                40,30,10,
    #                                np.nan,np.nan,np.nan]
    #     df_exp['DISDATE'] = np.concatenate([
    #                                 [np.datetime64('2005-01-01'),
    #                                  np.datetime64('2005-02-01'),
    #                                  np.datetime64('2005-03-01'), ],
    #                                 [np.datetime64('2006-01-01'),
    #                                  np.datetime64('2006-01-01'),
    #                                  np.datetime64('2006-01-01'), ],
    #                                 np.arange('2007-01', '2007-04', dtype='datetime64[M]'),
    #                                 ],axis=None)
    #     df_exp['AMB_ORDER'] = [2,1,3,
    #                            3,2,1,
    #                            1,3,2,]        
        
    #     pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)




# This has a RANDOM component
class Test_shuffle_and_assign_final_order(unittest.TestCase):
    
    def setUp(self):
        # just one subject is fine...
        # The important thing is how we sort WITHIN subjects
        self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0,
                                                   0,0,0,
                                                   0,0,0],
                                'PROVSPNOPS':['XXX','XXX','XXX',
                                              'YYY','YYY','YYY',
                                              'ZZZ','ZZZ','ZZZ',],
                                'DIAG_01':['A00','B00','C00',
                                            'A00','B00','C00',
                                            'A00','B00','C00',],
                                'MYEPISTART':np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2006-01-01'),3),
                                    np.repeat(np.datetime64('2007-01-01'),3),
                                    ],axis=None),
                                'DISDATE':np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2006-01-01'),3),
                                    np.repeat(np.datetime64('2007-01-01'),3),
                                    ],axis=None),
                                'SURVIVALTIME_SPELLMIN':[60.0,60.0,60.0,
                                                         60.0,60.0,60.0,
                                                         60.0,60.0,60.0,],
                                'SURVIVALTIME_SPELLMAX':[60.0,60.0,60.0,
                                                         60.0,60.0,60.0,
                                                         60.0,60.0,60.0,],
                                'MYEPISTART_SPELLMIN':np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    ],axis=None),
                                'MYEPISTART_SPELLMAX':np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    ],axis=None),
                                'MYEPIEND_SPELLMIN':np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    ],axis=None),
                                'MYEPIEND_SPELLMAX':np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    ],axis=None),
                                'EPIORDER':[1,2,3,
                                            1,2,3,
                                            1,2,3],
                                'COL_IDX':[1,1,1,
                                           1,1,1,
                                           1,1,1],
                                'IGNORE':np.repeat(False,9),
                                })
        
    def test_shuffle_and_assign_final_order_no_change(self):
        
        # try X times
        for i in range(REPEATS):
            df_out = filter_hes_sort.shuffle_and_assign_final_order(self.df.copy())        
            df_out = df_out.drop(columns=['AMB_ORDER'])
            
            #expected:
            df_exp = self.df.copy()
            df_exp['ORDER'] = np.array(np.arange(df_exp.shape[0]))
            pd.testing.assert_frame_equal(df_out,df_exp,check_like=True)
        
        
    def test_shuffle_and_assign_final_order_ambiguous_reverse(self):
        
        self.df['DISDATE'] = np.concatenate([
                                    np.repeat(np.datetime64('2007-01-01'),3),
                                    np.repeat(np.datetime64('2006-01-01'),3),
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    ],axis=None)
        
        # try X times
        for i in range(REPEATS):
                
            df_out = filter_hes_sort.shuffle_and_assign_final_order(self.df.copy())
            df_out = df_out.drop(columns=['AMB_ORDER']) # not used
    
            # check order sorted.
            np.testing.assert_array_equal(df_out['ORDER'],np.arange(9))
            
            df_exp = pd.concat([self.df.loc[6:8].copy(),
                                self.df.loc[3:5].copy(),
                                self.df.loc[0:2].copy()], ignore_index=True)
            df_exp['ORDER'] = np.array(np.arange(df_exp.shape[0]))
    
            pd.testing.assert_frame_equal(df_out,df_exp,check_like=True)    
            
            # ensure spell IDs are kept together
            self.assertEqual(df_out.loc[0:2,'PROVSPNOPS'].drop_duplicates().shape[0],1)
            self.assertEqual(df_out.loc[3:5,'PROVSPNOPS'].drop_duplicates().shape[0],1)
            self.assertEqual(df_out.loc[6:8,'PROVSPNOPS'].drop_duplicates().shape[0],1)


    def test_shuffle_and_assign_final_order_ambiguous_same_dates(self):
        self.df['DISDATE'] = np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    ],axis=None)
        
        # try X times
        for i in range(REPEATS):
            df_out = filter_hes_sort.shuffle_and_assign_final_order(self.df.copy())
    
            # check AMB_ORDER are sorted.
            np.testing.assert_array_equal(df_out['AMB_ORDER'],np.sort(df_out['AMB_ORDER']))
            # Check order sorted
            np.testing.assert_array_equal(df_out['ORDER'],np.arange(9))
            df_out = df_out.drop(columns=['AMB_ORDER'])
            df_out = df_out.drop(columns=['ORDER'])
            
            # expected
            np.testing.assert_array_equal(df_out['EPIORDER'],self.df['EPIORDER'])
            np.testing.assert_array_equal(df_out['DIAG_01'],self.df['DIAG_01'])

            # ensure spell IDs are kept together
            self.assertEqual(df_out.loc[0:2,'PROVSPNOPS'].drop_duplicates().shape[0],1)
            self.assertEqual(df_out.loc[3:5,'PROVSPNOPS'].drop_duplicates().shape[0],1)
            self.assertEqual(df_out.loc[6:8,'PROVSPNOPS'].drop_duplicates().shape[0],1)



    def test_shuffle_and_assign_final_order_ambiguous_same_dates_shuffle_epiorder(self):

        self.df['DISDATE'] = np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    ],axis=None)

        self.df['EPIORDER'] = [1,3,2,
                              3,2,1,
                              1,2,3]
        # try X times
        for i in range(REPEATS):
            df_out = filter_hes_sort.shuffle_and_assign_final_order(self.df.copy())
            # check AMB_ORDER are sorted.
            np.testing.assert_array_equal(df_out['AMB_ORDER'],np.sort(df_out['AMB_ORDER']))
            # Check order sorted
            np.testing.assert_array_equal(df_out['ORDER'],np.arange(9))
            df_out = df_out.drop(columns=['AMB_ORDER'])
            df_out = df_out.drop(columns=['ORDER'])
            
            # expected
            np.testing.assert_array_equal(df_out['EPIORDER'],
                                          [1,2,3,
                                           1,2,3,
                                           1,2,3])
            # ensure spell IDs are kept together
            self.assertEqual(df_out.loc[0:2,'PROVSPNOPS'].drop_duplicates().shape[0],1)
            self.assertEqual(df_out.loc[3:5,'PROVSPNOPS'].drop_duplicates().shape[0],1)
            self.assertEqual(df_out.loc[6:8,'PROVSPNOPS'].drop_duplicates().shape[0],1)




    def test_shuffle_and_assign_final_order_ambiguous_same_dates_change_col_idx(self):
        self.df['DISDATE'] = np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    ],axis=None)
        self.df['EPIORDER'] = [1,1,1,
                              1,1,1,
                              1,2,3]
        self.df['COL_IDX'] = [1,2,3,
                              3,2,1,
                              1,3,2]
        # try X times
        for i in range(REPEATS):
            df_out = filter_hes_sort.shuffle_and_assign_final_order(self.df.copy())
            # check AMB_ORDER are sorted.
            np.testing.assert_array_equal(df_out['AMB_ORDER'],
                                          np.sort(df_out['AMB_ORDER']))
            # Check order sorted
            np.testing.assert_array_equal(df_out['ORDER'],np.arange(9))
            df_out = df_out.drop(columns=['AMB_ORDER'])
            df_out = df_out.drop(columns=['ORDER'])
            
            # EPIORDER should always be sorted
            np.testing.assert_array_equal(
                df_out.loc[df_out['PROVSPNOPS']=='XXX','EPIORDER'],[1,1,1])
            np.testing.assert_array_equal(
                df_out.loc[df_out['PROVSPNOPS']=='YYY','EPIORDER'],[1,1,1])
            np.testing.assert_array_equal(
                df_out.loc[df_out['PROVSPNOPS']=='ZZZ','EPIORDER'],[1,2,3])

            # COL_IDX may be sorted, depending on EPIORDER
            np.testing.assert_array_equal(
                df_out.loc[df_out['PROVSPNOPS']=='XXX','COL_IDX'],[1,2,3])
            np.testing.assert_array_equal(
                df_out.loc[df_out['PROVSPNOPS']=='YYY','COL_IDX'],[1,2,3])
            np.testing.assert_array_equal(
                df_out.loc[df_out['PROVSPNOPS']=='ZZZ','COL_IDX'],[1,3,2])
            
            # ensure spell IDs are kept together
            self.assertEqual(df_out.loc[0:2,'PROVSPNOPS'].\
                             drop_duplicates().shape[0],1)
            self.assertEqual(df_out.loc[3:5,'PROVSPNOPS'].\
                             drop_duplicates().shape[0],1)
            self.assertEqual(df_out.loc[6:8,'PROVSPNOPS'].\
                             drop_duplicates().shape[0],1)



        # # survivaltime should be sorted for second subject
        # np.testing.assert_array_equal(df_out.loc[3:5,'SURVIVALTIME'],
        #                               np.flip(np.sort(df_out.loc[3:5,'SURVIVALTIME'])))
        # np.testing.assert_array_equal(df_out.loc[3:5,'DIAG_01'],
        #                               ['C00','B00','A00'])
        
        # # DISDATE should be sorted for third subject
        # np.testing.assert_array_equal(df_out.loc[6:9,'DISDATE'],
        #                               np.sort(df_out.loc[6:9,'DISDATE']))
        # np.testing.assert_array_equal(df_out.loc[6:9,'DIAG_01'],
        #                               ['B00','C00','A00'])

        # np.testing.assert_array_equal(df_out['MYADMIDATE'],
        #                               np.array(np.arange(df_out.shape[0]),dtype=np.int64))



        
#     def test_handle_ambiguous_sequences_ambiguous_using_sec_diags(self):
        
#         params.AMB_STRICT=False
#         self.df['MYADMIDATE'] = np.concatenate([
#                                     [np.datetime64('2005-01-01'),
#                                      np.datetime64('2005-01-01'),
#                                      np.datetime64('2005-01-01'), ],
#                                     [np.datetime64('2006-01-01'),
#                                      np.datetime64('2006-01-01'),
#                                      np.datetime64('2006-01-01'), ],
#                                     np.arange('2007-01', '2007-04', dtype='datetime64[M]'),
#                                     ],axis=None)
        
#         self.df['SURVIVALTIME'] = np.nan

#         # PRI, followed by 2 SEC diags
#         self.df.loc[0:2, 'ROW_IDX'] = [1,1,1,]
#         self.df.loc[0:2, 'COL_IDX'] = [1,3,2,] # will be returned sorted, as AMB_ORDER will be same     
        
#         df_out = filter_hes_sort.handle_ambiguous_sequences(self.df.copy())
        
#         # check AMB_ORDER is same for first subject.
#         self.assertEqual(df_out.loc[0:2,'AMB_ORDER'].drop_duplicates().shape[0], 1)
#         # check COL_IDX is sorted
#         np.testing.assert_array_equal(df_out.loc[0:2,'COL_IDX'],
#                                       np.sort(df_out.loc[0:2,'COL_IDX']))

#         np.testing.assert_array_equal(df_out.loc[0:2,'DIAG_01'],
#                                       ['A00','C00','B00'])

#         # check rest
#         np.testing.assert_array_equal(df_out.loc[3:5,'AMB_ORDER'],
#                                       np.sort(df_out.loc[3:5,'AMB_ORDER']))
        
#         np.testing.assert_array_equal(df_out['MYADMIDATE'],
#                                       np.array(np.arange(df_out.shape[0]),dtype=np.int64))



class Test_set_initial_event(unittest.TestCase):
    
    def setUp(self):
        times = np.arange('2005-01', '2005-04', dtype='datetime64[M]')
        times = np.tile(times,3)
        self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0,1,1,1,2,2,2],
                                'INIT_ROW':[True,False,False,
                                            True,False,False,
                                            True,False,False],
                                'DIAG_01':np.tile(['A00','B00','C00'],3),
                                'DIAG_01_ALT1':np.tile(['A00','B00','C00'],3),
                                'DIAG_01_ALT2':np.tile(['A00','B00','C00'],3),
                                'ORDER':np.arange(9),
                                'AMI':np.repeat(False,9),
                                'IGNORE':np.repeat(False,9),
                                })
        self.df = self.df.astype({'DIAG_01':'category',
                                  'DIAG_01_ALT1':'category',
                                  'DIAG_01_ALT2':'category'})
        self.repeats = 10
        
        #pdb.set_trace()
        
    def _assertEarlierAreIgnored(self,df):
        # ensure that all earlier events (those before AMI/initial) have IGNORED==True
        df_init = df.loc[df['CHOSEN_INIT'], ['ENCRYPTED_HESID', 'ORDER']]
        df = df.merge(df_init, on='ENCRYPTED_HESID', how='left', suffixes=('','_TEST'))
        tmp = df.loc[df['ORDER']<df['ORDER_TEST'],'IGNORE']
        if tmp.shape[0]>0:
            # check all previous ignored
            self.assertTrue(tmp.all())
        else:
            print('WARNING: no earlier rows')
            
    def _assertDiagChanged(self,df):
        self.assertTrue((df.loc[df['CHOSEN_INIT'],'DIAG_01']==params.AMI_INIT).all())
        self.assertTrue((df.loc[df['CHOSEN_INIT'],'DIAG_01_ALT1']==params.AMI_INIT).all())
        self.assertTrue((df.loc[df['CHOSEN_INIT'],'DIAG_01_ALT2']==params.AMI_INIT).all())

    # def _assertLaterAreNotIgnored(self,df):
    #     # ensure that all earlier events (those before AMI/initial) have IGNORED==True
    #     df_init = df.loc[df['DIAG_01']==params.AMI_INIT, ['ENCRYPTED_HESID', 'ORDER']]
    #     df = df.merge(df_init, on='ENCRYPTED_HESID', how='left', suffixes=('','_TEST'))
    #     tmp = df.loc[df['ORDER']>df['ORDER_TEST'], 'IGNORE'].drop_duplicates()
    #     if tmp.shape[0]>0:
    #         self.assertFalse(tmp.iloc[0])
    #         self.assertEqual(tmp.shape[0], 1)
    #     else:
    #         print('WARNING: no later rows')
        
    # #######
    # TESTS
    # #######
    
    # specific dates (first, default date)
    @patch('pipeline_hes.params.params.INITIAL_EPISODE_RANDOM_SEED', 0)
    def test_set_initial_event_control(self):
        self.df['IS_CONTROL'] = True
        #pdb.set_trace()
        for i in np.arange(self.repeats):
            df = filter_hes_sort.set_initial_random(self.df.copy())
            self.assertTrue((df.loc[[0,3,6], 'CHOSEN_INIT']).all())
            self.assertFalse((df.loc[[1,2,4,5,7,8], 'CHOSEN_INIT']).any())
            self.assertFalse(df.loc[df['CHOSEN_INIT'], 'IGNORE'].any())
            self._assertEarlierAreIgnored(df)
            self._assertDiagChanged(df)


    # specific dates (controls) - move date to 2005-02-01
    @patch('pipeline_hes.params.params.INITIAL_EPISODE_RANDOM_SEED', 0)
    def test_set_initial_event_control_init_date_change(self):
        self.df['IS_CONTROL'] = True
        self.df['INIT_ROW'] = [False,True,False,
                               False,True,False,
                               False,True,False]
        for i in np.arange(self.repeats):
            df = filter_hes_sort.set_initial_random(self.df.copy())
            self.assertTrue((df.loc[[1,4,7], 'CHOSEN_INIT']).all())
            self.assertFalse((df.loc[[0,2,3,5,6,8], 'CHOSEN_INIT']).any())
            self.assertFalse(df.loc[df['CHOSEN_INIT'], 'IGNORE'].any())
            self._assertEarlierAreIgnored(df)
            self._assertDiagChanged(df)
            
            
    # random choice (controls)
    @patch('pipeline_hes.params.params.INITIAL_EPISODE_RANDOM_SEED', 0)
    def test_set_initial_event_control_random(self):
        self.df['IS_CONTROL'] = True
        # all candidates
        self.df['INIT_ROW'] = True

        # for different random seeds, check that previous events are marked as ignored
        for i in np.arange(self.repeats):
            df = filter_hes_sort.set_initial_random(self.df.copy())
            self.assertFalse(df.loc[df['CHOSEN_INIT'],'IGNORE'].any())
            self.assertEqual(df.loc[df['CHOSEN_INIT']].shape[0],3)
            self.assertEqual(df.loc[df['CHOSEN_INIT']].\
                             drop_duplicates(subset=['ENCRYPTED_HESID']).shape[0],3)
            self._assertEarlierAreIgnored(df)
            self._assertDiagChanged(df)
            

    # random choice (controls) - FULL IGNORE
    @patch('pipeline_hes.params.params.INITIAL_EPISODE_RANDOM_SEED', 0)
    def test_set_initial_event_control_random_full_ignore(self):
        self.df['IS_CONTROL'] = True
        self.df['IGNORE'] = True
        # all candidates
        self.df['INIT_ROW'] = True

        # for different random seeds, check that previous events are marked as ignored
        for i in np.arange(self.repeats):
            df = filter_hes_sort.set_initial_random(self.df.copy())
            self.assertFalse(df.loc[df['CHOSEN_INIT'],'IGNORE'].any())
            self.assertEqual(df.loc[df['CHOSEN_INIT']].shape[0],3)
            self.assertEqual(df.loc[df['CHOSEN_INIT']].\
                             drop_duplicates(subset=['ENCRYPTED_HESID']).shape[0],3)
            self._assertEarlierAreIgnored(df)
            self._assertDiagChanged(df)
            

    # # ensures that only IGNORE=False events are selected
    # def test_set_initial_event_control_ignore_false(self):
    #     self.df['IS_CONTROL'] = True
    #     self.df['IGNORE'] = True
    #     self.df.loc[[0,4,8],'IGNORE'] = False
    #     # all candidates
    #     self.df['INIT_ROW'] = True

    #     for i in np.arange(self.repeats):
    #         df = filter_hes_sort.set_initial_random(self.df.copy())
    #         self.assertTrue((df.loc[[0,4,8], 'CHOSEN_INIT']).all())
    #         self.assertFalse((df.loc[[1,2,3,5,6,7], 'CHOSEN_INIT']).any())
    #         self.assertFalse(df.loc[df['CHOSEN_INIT'], 'IGNORE'].any())
    #         self._assertEarlierAreIgnored(df)
    #         self._assertDiagChanged(df)


    # mix of controls and AMI
    @patch('pipeline_hes.params.params.INITIAL_EPISODE_RANDOM_SEED', 0)
    def test_set_initial_event_control_and_ami(self):
        self.df.loc[[0,1,2],'IS_CONTROL'] = True
        self.df.loc[[3,4,5,6,7,8],'IS_CONTROL'] = False
        self.df.loc[[3,8],'AMI'] = True
        self.df['INIT_ROW'] = True

        for i in np.arange(self.repeats):
            df = filter_hes_sort.set_initial_random(self.df.copy())
            # one initial event for control
            self.assertEqual((df.loc[0:2,'CHOSEN_INIT']).sum(),1)
            
            # ami check
            self.assertTrue((df.loc[[3,8], 'CHOSEN_INIT']).all())
            self.assertFalse((df.loc[[4,5,6,7], 'CHOSEN_INIT']).any())
            
            self.assertFalse(df.loc[df['CHOSEN_INIT'], 'IGNORE'].any())
            self._assertEarlierAreIgnored(df)
            self._assertDiagChanged(df)


    # (AMI) first events are selected
    @patch('pipeline_hes.params.params.INITIAL_EPISODE_RANDOM_SEED', 0)
    def test_set_initial_event_ami_head(self):
        self.df['IS_CONTROL'] = False
        self.df['AMI'] = True

        for i in np.arange(self.repeats):
            df = filter_hes_sort.set_initial_random(self.df.copy())
            # ensure that only AMI events are marked as AMI/initial
            self.assertTrue((df.loc[[0,3,6], 'CHOSEN_INIT']).all())
            self.assertFalse((df.loc[[1,2,4,5,7,8], 'CHOSEN_INIT']).any())
            self.assertTrue(df.loc[df['CHOSEN_INIT'], 'AMI'].all())
            self.assertFalse(df.loc[df['CHOSEN_INIT'], 'IGNORE'].any())
            self._assertEarlierAreIgnored(df)
            self._assertDiagChanged(df)


    # (AMI) last events are selected
    @patch('pipeline_hes.params.params.INITIAL_EPISODE_RANDOM_SEED', 0)
    def test_set_initial_event_ami_last(self):
        self.df['IS_CONTROL'] = False
        self.df['AMI'] = False
        self.df.loc[[2,5,8],'AMI'] = True
        self.df.loc[2,'IGNORE'] = True
        # all candidates
        self.df['INIT_ROW'] = True

        for i in np.arange(self.repeats):
            df = filter_hes_sort.set_initial_random(self.df.copy())
            # ensure that only AMI events are marked as AMI/initial
            self.assertTrue((df.loc[[2,5,8], 'CHOSEN_INIT']).all())
            self.assertFalse((df.loc[[0,1,3,4,6,7], 'CHOSEN_INIT']).any())
            self.assertTrue(df.loc[df['CHOSEN_INIT'], 'AMI'].all())
            self.assertFalse(df.loc[df['CHOSEN_INIT'], 'IGNORE'].any())
            self._assertEarlierAreIgnored(df)
            self._assertDiagChanged(df)


    # (AMI) all events have AMI, but only the first events must be chosen (first AMI)
    @patch('pipeline_hes.params.params.INITIAL_EPISODE_RANDOM_SEED', 0)
    def test_set_initial_event_ami_all(self):
        self.df['IS_CONTROL'] = False
        self.df['AMI'] = True
        # all candidates
        self.df['INIT_ROW'] = True

        for i in np.arange(self.repeats):
            df = filter_hes_sort.set_initial_random(self.df.copy())
            # ensure that only AMI events are marked as AMI/initial
            self.assertTrue((df.loc[[0,3,6], 'CHOSEN_INIT']).all())
            self.assertFalse((df.loc[[1,2,4,5,7,8], 'CHOSEN_INIT']).any())
            self.assertTrue(df.loc[df['CHOSEN_INIT'], 'AMI'].all())
            self.assertFalse(df.loc[df['CHOSEN_INIT'], 'IGNORE'].any())
            self._assertEarlierAreIgnored(df)
            self._assertDiagChanged(df)
            
            
    # (AMI) specific events only
    @patch('pipeline_hes.params.params.INITIAL_EPISODE_RANDOM_SEED', 0)
    def test_set_initial_event_only_ami_specific_rows(self):
        self.df['IS_CONTROL'] = False
        self.df.loc[[0,1,4,8],'AMI'] = True
        # We set the first row to IGNORE, but this should still be selected as initial.
        # It's also important to combine this with two options for initial (0,1 are AMI=True)
        # We dont want to select the second row - ALWAYS the first AMI is chosen.
        self.df.loc[[0],'IGNORE'] = True
        # all candidates
        self.df['INIT_ROW'] = True

        # for different random seeds, check that previous events are marked as ignored
        for i in np.arange(self.repeats):
            df = filter_hes_sort.set_initial_random(self.df.copy())
            #pdb.set_trace()
            # # ensure that only AMI events are marked as AMI/initial
            self.assertTrue((df.loc[[0,4,8], 'CHOSEN_INIT']).all())
            self.assertFalse((df.loc[[1,2,3,5,6,7], 'CHOSEN_INIT']).any())
            self.assertTrue(df.loc[df['CHOSEN_INIT'], 'AMI'].all())
            self.assertFalse(df.loc[df['CHOSEN_INIT'],'IGNORE'].any())
            self._assertEarlierAreIgnored(df)
            self._assertDiagChanged(df)

    @patch('pipeline_hes.params.params.INITIAL_EPISODE_RANDOM_SEED', 0)
    # one AMI subject without any AMI (gives error)
    def test_set_initial_event_set_some_to_no_ami(self):
        self.df.loc[[0,1,2],'IS_CONTROL'] = False
        self.df.loc[[0,1,2],'AMI'] = False
        self.assertRaises(Exception, filter_hes_sort.set_initial_random, self.df.copy(), 42)


    # epiorder DOESNT matter
    # Many AMI subjects do not have their first AMI as EPIORDER==1 event.
    # Thus, setting the initial event for controls can also be a non EPIORDER==1 event
    @patch('pipeline_hes.params.params.INITIAL_EPISODE_RANDOM_SEED', 0)
    def test_set_initial_event_epiorder(self):
        self.df['EPIORDER'] = 2
        #mix
        self.df.loc[[0,1,2],'IS_CONTROL'] = True
        self.df.loc[[3,4,5,6,7,8],'IS_CONTROL'] = False
        self.df.loc[[3,8],'AMI'] = True
        self.df['MYADMIDATE'] = np.datetime64('2005-03-01')
        self.df['MYADMIDATE_INIT'] = np.datetime64('2005-03-01')
        # all candidates
        self.df['INIT_ROW'] = True
   
        for i in np.arange(self.repeats):
            df = filter_hes_sort.set_initial_random(self.df.copy())
            self.assertEqual(df.loc[df['CHOSEN_INIT']].shape[0],3)
            self.assertEqual(df.loc[df['CHOSEN_INIT']].\
                             drop_duplicates(subset=['ENCRYPTED_HESID']).shape[0],3)
            self.assertFalse(df.loc[df['CHOSEN_INIT'],'IGNORE'].any())
            self._assertEarlierAreIgnored(df)
            self._assertDiagChanged(df)
        
    # colidx DOESNT matter
    # After flattening, the AMI events may be in a secondary position (COL_IDX>0)
    @patch('pipeline_hes.params.params.INITIAL_EPISODE_RANDOM_SEED', 0)
    def test_set_initial_event_epiorder(self):
        self.df['COL_IDX'] = 2
        #mix
        self.df.loc[[0,1,2],'IS_CONTROL'] = True
        self.df.loc[[3,4,5,6,7,8],'IS_CONTROL'] = False
        self.df.loc[[3,8],'AMI'] = True
        self.df['MYADMIDATE'] = np.datetime64('2005-03-01')
        self.df['MYADMIDATE_INIT'] = np.datetime64('2005-03-01')
        # all candidates
        self.df['INIT_ROW'] = True

        for i in np.arange(self.repeats):
            df = filter_hes_sort.set_initial_random(self.df.copy())
            self.assertEqual(df.loc[df['CHOSEN_INIT']].shape[0],3)
            self.assertEqual(df.loc[df['CHOSEN_INIT']].\
                             drop_duplicates(subset=['ENCRYPTED_HESID']).shape[0],3)
            self.assertFalse(df.loc[df['CHOSEN_INIT'],'IGNORE'].any())
            self._assertEarlierAreIgnored(df)
            self._assertDiagChanged(df)
        
        
    # # colidx does matter
    # def test_set_initial_event_colidx_all1(self):
    #     self.df['COL_IDX'] = 1
    #     self.df['IS_CONTROL'] = True        
    #     self.assertRaises(Exception, filter_hes_sort.set_initial_random, self.df.copy(), 42)

        
    # # epiorder/colidx doesnt matter
    # def test_set_initial_event_epiorder_colidx_onesub(self):
    #     self.df.loc[[0,1,2],'EPIORDER'] = 2
    #     self.df.loc[[0,1,2],'COLIDX'] = 1
    #     self.df['IS_CONTROL'] = True
        
    #     for i in np.arange(self.repeats):
    #         df = filter_hes_sort.set_initial_random(self.df.copy())
    #         self.assertTrue((df.loc[[0,3,6], 'CHOSEN_INIT']).all())
    #         self.assertFalse((df.loc[[1,2,4,5,7,8], 'CHOSEN_INIT']).any())
    #         self.assertFalse(df.loc[df['CHOSEN_INIT'],'IGNORE'].any())
    #         self._assertEarlierAreIgnored(df)



    # TODO def test_set_first_ami_index(self):


    # # set the first subject so that no initial event candidate exists (no EPIORDER==1, no COL_IDX==0)
    # def test_set_initial_event_set_some_to_no_init_event_epiorder(self):
    #     self.df.loc[[0,1,2],'EPIORDER'] = 2
    #     self.assertRaises(Exception, filter_hes_sort.set_initial_random, self.df.copy(), 42)

        
    # set the first subject so that no initial event candidate exists (no COL_IDX==0)
    # def test_set_initial_event_set_some_to_no_init_event_colidx_onesub(self):
    #     self.df.loc[[0,1,2],'COL_IDX'] = 1
    #     self.assertRaises(Exception, filter_hes_sort.set_initial_random, self.df.copy(), 42)

    # def test_set_initial_event_set_some_to_no_init_event_colidx(self):
    #     self.df.loc[[0,3,6],'COL_IDX'] = 1
    #     self.assertRaises(Exception, filter_hes_sort.set_initial_random, self.df.copy(), 42)


    # def test_set_initial_event_set_some_to_no_init_event_colidx(self):
    #     self.df.loc[[0,3,6],'COL_IDX'] = 1
    #     self.assertRaises(Exception, filter_hes_sort.set_initial_random, self.df.copy(), 42)


    # def test_set_initial_event_colidx_ok_head(self):
    #     self.df.loc[[1,2,4,5,7,8],'COL_IDX'] = 1
    #     # all candidates
    #     self.df['MYADMIDATE_INIT'] = np.datetime64('2005-03-01')
    #     self.df['MYADMIDATE'] = np.datetime64('2005-03-01')             

    #     for i in np.arange(self.repeats):
    #         df = filter_hes_sort.set_initial_random(self.df.copy())
    #         self.assertTrue((df.loc[[0,3,6], 'DIAG_01'].drop_duplicates()==params.AMI_INIT).all())
    #         self.assertTrue((df.loc[[1,2,4,5,7,8], 'DIAG_01']!=params.AMI_INIT).all())
    #         self.assertEqual(df.loc[df['DIAG_01']==params.AMI_INIT, 'IGNORE'].drop_duplicates().shape[0],1)
    #         self._assertEarlierAreIgnored(df)
    #         self._assertLaterAreNotIgnored(df)

    # def test_set_initial_event_colidx_ok_tail(self):
    #     self.df.loc[[0,1,3,4,6,7],'COL_IDX'] = 1
    #     # all candidates
    #     self.df['MYADMIDATE_INIT'] = np.datetime64('2005-03-01')
    #     self.df['MYADMIDATE'] = np.datetime64('2005-03-01')             

    #     for i in np.arange(self.repeats):
    #         df = filter_hes_sort.set_initial_random(self.df.copy())
    #         self.assertTrue((df.loc[[2,5,8], 'DIAG_01'].drop_duplicates()==params.AMI_INIT).all())
    #         self.assertTrue((df.loc[[0,1,3,4,6,7], 'DIAG_01']!=params.AMI_INIT).all())
    #         self.assertEqual(df.loc[df['DIAG_01']==params.AMI_INIT, 'IGNORE'].drop_duplicates().shape[0],1)
    #         self._assertEarlierAreIgnored(df)
    #         self._assertLaterAreNotIgnored(df)


        
class Test_ignore_close_events_single_subject(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0,
                                                   0,0,0,
                                                   0,0,0],
                                'DIAG_01':['A00','B00','C00',
                                           'A00','B00','C00',
                                           'A00','B00','C00'],
                                'MYEPISTART':np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2005-02-01'),3),
                                    np.repeat(np.datetime64('2005-03-01'),3),
                                    ],axis=None),
                                'MYEPIEND':np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2005-02-01'),3),
                                    np.repeat(np.datetime64('2005-03-01'),3),
                                    ],axis=None),
                                'IGNORE':np.repeat(False,9),
                                'ORDER':np.arange(9),
                                })
        self.df['DIAG_01'] = self.df['DIAG_01'].astype('category')


    def test_ignore_close_events_default(self):
        df_out = filter_hes_sort.ignore_close_events(self.df.copy())
        df_exp = self.df.copy()
        df_exp['IGNORE'] = [False,False,False,
                            True,True,True,
                            True,True,True]
        pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)


    def test_ignore_close_events_lengthen_gaps(self):
        
        self.df['MYEPISTART'] = np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2005-04-01'),3),
                                    np.repeat(np.datetime64('2005-05-01'),3),
                                    ],axis=None)
        self.df['MYEPIEND'] = np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2005-04-01'),3),
                                    np.repeat(np.datetime64('2005-05-01'),3),
                                    ],axis=None)

        df_out = filter_hes_sort.ignore_close_events(self.df.copy())
        
        df_exp = self.df.copy()
        df_exp['IGNORE'] = [False,False,False,
                            False,False,False,
                            True,True,True]
        pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)


    def test_ignore_close_events_same_diags(self):
        self.df['DIAG_01'] = ['B00','B00','B00',
                              'B00','B00','B00',
                              'B00','B00','B00']
        self.df['DIAG_01'] = self.df['DIAG_01'].astype('category')

        df_out = filter_hes_sort.ignore_close_events(self.df.copy())
        df_exp = self.df.copy()
        # initial ami events are replaced with AMI (so will change repetitions)
        df_exp['IGNORE'] = [False,True,True,
                            True,True,True,
                            True,True,True]
        #df_exp['IGNORE'] = df_exp['DUP']
        #pdb.set_trace()
        pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)        


    def test_ignore_close_events_same_diags_gaps(self):
        self.df['DIAG_01'] = ['B00','B00','B00',
                              'B00','B00','B00',
                              'B00','B00','B00']
        self.df['DIAG_01'] = self.df['DIAG_01'].astype('category')

        # adjacent months
        self.df['MYEPISTART'] = np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2005-04-01'),3),
                                    np.repeat(np.datetime64('2005-07-01'),3),
                                    ],axis=None)
        self.df['MYEPIEND'] = np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2005-04-01'),3),
                                    np.repeat(np.datetime64('2005-07-01'),3),
                                    ],axis=None)

        df_out = filter_hes_sort.ignore_close_events(self.df.copy())
        df_exp = self.df.copy()
        # initial ami events are replaced with AMI (so will change repetitions)
        df_exp['IGNORE'] = [False,True,True,
                            False,True,True,
                            False,True,True]
        #df_exp['IGNORE'] = df_exp['DUP']
        #pdb.set_trace()
        pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)        



    def test_ignore_close_events_different_diags(self):
        self.df['DIAG_01'] = ['B00','B00','B00',
                              'B00','Z00','B00',
                              'B00','Z00','B00']
        self.df['DIAG_01'] = self.df['DIAG_01'].astype('category')
        
        #pdb.set_trace()
        df_out =  filter_hes_sort.ignore_close_events(self.df.copy())
        df_exp = self.df.copy()
        
        # initial ami events are replaced with AMI (so will change repetitions)
        df_exp['IGNORE'] = [False,True,True,
                            True,False,True,
                            True,True,True]
        #df_exp['IGNORE'] = df_exp['DUP']
        #pdb.set_trace()
        pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)


    def test_ignore_close_events_big_spell_gaps(self):
        self.df['DIAG_01'] = ['B00','C00','C00',
                              'C00','C00','C00',
                              'C00','C00','B00']
        self.df['DIAG_01'] = self.df['DIAG_01'].astype('category')

        
        self.df['MYEPISTART'] = np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2010-01-01'),3),
                                    np.repeat(np.datetime64('2015-01-01'),3),
                                    ],axis=None)
        self.df['MYEPIEND'] = np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2010-01-01'),3),
                                    np.repeat(np.datetime64('2015-01-01'),3),
                                    ],axis=None)
        
        df_out =  filter_hes_sort.ignore_close_events(self.df.copy())
        
        df_exp = self.df.copy()
        df_exp['IGNORE'] = [False,False,True,
                            False,True,True,
                            False,True,False]
        #pdb.set_trace()
        pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)        


    def test_ignore_close_events_same_date(self):
        self.df['DIAG_01'] = ['B00','C00','C00',
                              'C00','C00','C00',
                              'C00','C00','B00']
        self.df['DIAG_01'] = self.df['DIAG_01'].astype('category')
        
        self.df['MYEPISTART'] = np.datetime64('2005-01-01')
        self.df['MYEPIEND'] = np.datetime64('2005-01-01')
        
        df_out =  filter_hes_sort.ignore_close_events(self.df.copy())
                
        df_exp = self.df.copy()
        df_exp['IGNORE'] = [False,False,True,
                          True,True,True,
                          True,True,True]
        pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)        


    def test_ignore_close_events_last_changed(self):
        self.df['DIAG_01'] = ['B00','C00','C00',
                              'C00','C00','C00',
                              'C00','C00','B00']
        self.df['DIAG_01'] = self.df['DIAG_01'].astype('category')
        
        self.df['MYEPISTART'] = np.datetime64('2005-01-01')
        self.df['MYEPIEND'] = np.datetime64('2005-01-01')
        
        self.df.loc[8, 'MYEPISTART'] = np.datetime64('2005-04-01')
        self.df.loc[8, 'MYEPIEND'] = np.datetime64('2005-04-01')
        
        df_out =  filter_hes_sort.ignore_close_events(self.df.copy())
        
        df_exp = self.df.copy()
        df_exp['IGNORE'] = [False,False,True,
                          True,True,True,
                          True,True,False]
        #pdb.set_trace()
        pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)


    def test_ignore_close_events_with_sec_diag(self):
        self.df['DIAG_01'] = ['C00','C00','D00',
                              'C00','X00','C00',
                              'C00','C00','Q00']
        self.df['DIAG_02'] = ['B00','C00','D00',
                              'X00','C00','C00',
                              'C00','Q00','D00']
        self.df['DIAG_01'] = self.df['DIAG_01'].astype('category')
        self.df['DIAG_02'] = self.df['DIAG_02'].astype('category')

        
        df_out =  filter_hes_sort.ignore_close_events(self.df.copy())
        
        df_exp = self.df.copy()
        df_exp['IGNORE'] = [False,True,False,
                            True,True,True,
                            True,True,True]
        #pdb.set_trace()
        pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)


    def test_ignore_close_events_with_sec_diag_last_changed_small_gap(self):
        self.df['DIAG_01'] = ['C00','C00','D00',
                              'C00','X00','C00',
                              'C00','C00','Q00']
        self.df['DIAG_02'] = ['B00','C00','D00',
                              'X00','C00','C00',
                              'C00','Q00','D00']
        self.df['DIAG_01'] = self.df['DIAG_01'].astype('category')
        self.df['DIAG_02'] = self.df['DIAG_02'].astype('category')
        
        self.df['MYEPISTART'] = np.datetime64('2005-01-01')
        self.df['MYEPIEND'] = np.datetime64('2005-01-01')
        self.df.loc[8,'MYEPISTART'] = np.datetime64('2005-03-01')
        self.df.loc[8,'MYEPIEND'] = np.datetime64('2005-03-01')
        
        df_out =  filter_hes_sort.ignore_close_events(self.df.copy())
        
        df_exp = self.df.copy()
        df_exp['IGNORE'] = [False,True,False,
                            True,True,True,
                            True,True,True]
        #pdb.set_trace()
        pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)


    def test_ignore_close_events_with_sec_diag_last_changed_big_gap(self):
        self.df['DIAG_01'] = ['C00','C00','D00',
                              'C00','X00','C00',
                              'C00','C00','Q00']
        self.df['DIAG_02'] = ['B00','C00','D00',
                              'X00','C00','C00',
                              'C00','Q00','D00']
        self.df['DIAG_01'] = self.df['DIAG_01'].astype('category')
        self.df['DIAG_02'] = self.df['DIAG_02'].astype('category')
        
        self.df['MYEPISTART'] = np.datetime64('2005-01-01')
        self.df['MYEPIEND'] = np.datetime64('2005-01-01')
        self.df.loc[8,'MYEPISTART'] = np.datetime64('2005-04-01')
        self.df.loc[8,'MYEPIEND'] = np.datetime64('2005-04-01')
        
        df_out =  filter_hes_sort.ignore_close_events(self.df.copy())
        
        df_exp = self.df.copy()
        df_exp['IGNORE'] = [False,True,False,
                            True,True,True,
                            True,True,False]
        #pdb.set_trace()
        pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)

    def test_ignore_close_events_with_sec_diag_big_gap(self):
        self.df['DIAG_01'] = ['C00','C00','D00',
                              'C00','X00','C00',
                              'C00','C00','Q00']
        self.df['DIAG_02'] = ['B00','C00','D00',
                              'X00','C00','C00',
                              'C00','Q00','D00']
        self.df['DIAG_01'] = self.df['DIAG_01'].astype('category')
        self.df['DIAG_02'] = self.df['DIAG_02'].astype('category')
        
        self.df['MYEPISTART'] = np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2010-01-01'),3),
                                    np.repeat(np.datetime64('2015-01-01'),3),
                                    ],axis=None)
        self.df['MYEPIEND'] = np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2010-01-01'),3),
                                    np.repeat(np.datetime64('2015-01-01'),3),
                                    ],axis=None)
        
        df_out =  filter_hes_sort.ignore_close_events(self.df.copy())
        
        df_exp = self.df.copy()
        df_exp['IGNORE'] = [False,True,False,
                            False,True,True,
                            False,True,True]
        #pdb.set_trace()
        pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)


    def test_ignore_close_events_with_sec_diag20(self):
        self.df['DIAG_01'] = ['C00','C00','D00',
                              'C00','X00','C00',
                              'C00','C00','Q00']
        self.df['DIAG_20'] = ['B00','C00','D00',
                              'X00','C00','C00',
                              'C00','Q00','D00']
        self.df['DIAG_01'] = self.df['DIAG_01'].astype('category')
        self.df['DIAG_20'] = self.df['DIAG_20'].astype('category')
        
        df_out =  filter_hes_sort.ignore_close_events(self.df.copy())
        
        df_exp = self.df.copy()
        df_exp['IGNORE'] = [False,True,False,
                            True,True,True,
                            True,True,True]
        #pdb.set_trace()
        pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)


class Test_ignore_close_events_mulitple_subjects(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0,
                                                   1,1,1,
                                                   2,2,2],
                                'DIAG_01':['A00','B00','C00',
                                           'A00','B00','C00',
                                           'A00','B00','C00'],
                                'MYEPISTART':np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2005-02-01'),3),
                                    np.repeat(np.datetime64('2005-03-01'),3),
                                    ],axis=None),
                                'MYEPIEND':np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2005-02-01'),3),
                                    np.repeat(np.datetime64('2005-03-01'),3),
                                    ],axis=None),
                                'IGNORE':np.repeat(False,9),
                                'ORDER':np.arange(9),
                                })
        self.df['DIAG_01'] = self.df['DIAG_01'].astype('category')


    def test_ignore_close_events_no_repeats(self):
        df_exp = self.df.copy()
        df_exp['IGNORE'] = False
        df_out = filter_hes_sort.ignore_close_events(self.df.copy())

        pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)



    def test_ignore_close_events_same(self):
        self.df['DIAG_01'] = ['A00','A00','A00',
                              'A00','A00','A00',
                              'A00','A00','A00']
        self.df['DIAG_01'] = self.df['DIAG_01'].astype('category')
        df_out = filter_hes_sort.ignore_close_events(self.df.copy())
        df_exp = self.df.copy()
        df_exp['IGNORE'] =  [False,True,True,
                              False,True,True,
                              False,True,True]
        #pdb.set_trace()
        pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)


    def test_ignore_close_events_with_repeats_some_ignored(self):
        self.df['DIAG_01'] = ['A00','B00','B00',
                              'Y00','Y00','Y00',
                              'B00','B00','Y00']
        self.df['DIAG_01'] = self.df['DIAG_01'].astype('category')
        self.df['IGNORE'] = [True,False,False,
                              True,False,True,
                              True,False,False]
        df_out = filter_hes_sort.ignore_close_events(self.df.copy())
        df_exp = self.df.copy()
        df_exp['IGNORE'] =  [True,False,True,
                              True,True,True,
                              True,True,False]
        #pdb.set_trace()
        pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)


    def test_ignore_close_events_gaps(self):
        self.df['DIAG_01'] = ['A00','B00','B00',
                              'Y00','Y00','Y00',
                              'B00','B00','Y00']
        self.df['DIAG_01'] = self.df['DIAG_01'].astype('category')

        # adjacent months
        self.df['MYEPISTART'] = np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2006-02-01'),3),
                                    np.repeat(np.datetime64('2007-03-01'),3),
                                    ],axis=None)
        self.df['MYEPIEND'] = np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2006-02-01'),3),
                                    np.repeat(np.datetime64('2007-03-01'),3),
                                    ],axis=None)   

        df_out = filter_hes_sort.ignore_close_events(self.df.copy())

        df_exp = self.df.copy()
        df_exp['IGNORE'] =  [False,False,True,
                              False,True,True,
                              False,True,False]

        #pdb.set_trace()
        pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)


    def test_ignore_close_events_with_sec(self):
        self.df['DIAG_01'] = ['A00','B00','B00',
                              'B00','Y00','Y00',
                              'A00','B00','Y00']
        self.df['DIAG_02'] = ['B00','B00','B00',
                              'Y00','Y00','Y00',
                              'A00','B00','Y00']
        self.df['DIAG_01'] = self.df['DIAG_01'].astype('category')
        self.df['DIAG_02'] = self.df['DIAG_02'].astype('category')


        df_out = filter_hes_sort.ignore_close_events(self.df.copy())

        df_exp = self.df.copy()
        df_exp['IGNORE'] =  [False,True,True,
                              False,True,True,
                              False,False,False]
        pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)


    def test_ignore_close_events_with_sec_same_dates(self):
        self.df['DIAG_01'] = ['A00','B00','B00',
                              'B00','Y00','Y00',
                              'B00','B00','Y00']
        self.df['DIAG_02'] = ['C00','B00','B00',
                              'Y00','Y00','Y00',
                              'B00','B00','Y00']
        self.df['DIAG_01'] = self.df['DIAG_01'].astype('category')
        self.df['DIAG_02'] = self.df['DIAG_02'].astype('category')

        self.df['MYEPISTART'] = np.repeat(np.datetime64('2005-01-01'),9)
        self.df['MYEPIEND'] = np.repeat(np.datetime64('2005-01-01'),9) 
        df_out = filter_hes_sort.ignore_close_events(self.df.copy())

        df_exp = self.df.copy()
        df_exp['IGNORE'] =  [False,False,True,
                              False,True,True,
                              False,True,False]
        pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)


    def test_ignore_close_events_with_sec_change_gaps(self):
        self.df['DIAG_01'] = ['A00','B00','B00',
                              'B00','Y00','X00',
                              'B00','B00','Y00']
        self.df['DIAG_02'] = ['B00','B00','B00',
                              'Y00','X00','Y00',
                              'B00','B00','Y00']
        self.df['DIAG_01'] = self.df['DIAG_01'].astype('category')
        self.df['DIAG_02'] = self.df['DIAG_02'].astype('category')

        # adjacent months
        self.df['MYEPISTART'] = np.concatenate([
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    [np.datetime64('2005-01-01'),
                                     np.datetime64('2005-04-01'),
                                     np.datetime64('2005-06-01'),],
                                    np.repeat(np.datetime64('2007-03-01'),3),
                                    ],axis=None)
        self.df['MYEPIEND'] = np.concatenate([
                                    np.repeat(np.datetime64('2005-02-01'),3),
                                    [np.datetime64('2005-01-01'),
                                     np.datetime64('2005-04-01'),
                                     np.datetime64('2005-06-01'),],
                                    np.repeat(np.datetime64('2007-03-01'),3),
                                    ],axis=None)   

        df_out = filter_hes_sort.ignore_close_events(self.df.copy())

        df_exp = self.df.copy()
        df_exp['IGNORE'] =  [False,True,True,
                              False,False,True,
                              False,True,False]
        pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)


    def test_ignore_close_events_with_sec20(self):
        self.df['DIAG_01'] = ['A00','B00','B00',
                              'B00','Y00','Y00',
                              'B00','B00','Y00']
        self.df['DIAG_20'] = ['C00','B00','B00',
                              'Y00','Y00','Y00',
                              'B00','B00','Y00']
        self.df['DIAG_01'] = self.df['DIAG_01'].astype('category')
        self.df['DIAG_20'] = self.df['DIAG_20'].astype('category')

        self.df['MYEPISTART'] = np.repeat(np.datetime64('2005-01-01'),9)
        self.df['MYEPIEND'] = np.repeat(np.datetime64('2005-01-01'),9) 
        df_out = filter_hes_sort.ignore_close_events(self.df.copy())

        df_exp = self.df.copy()
        df_exp['IGNORE'] =  [False,False,True,
                              False,True,True,
                              False,True,False]
        pd.testing.assert_frame_equal(df_out, df_exp, check_like=True)


class Test_acute_chronic(unittest.TestCase):

    def test_ignore_repeated_chronic_full_ignore(self):
        diags_01 = ['A00', 'B00', 'C00', 'A00', 'Z00']
        diags_02 = ['B00', 'C00', 'Z00', 'Z00', 'B00']
        chronic_01 = ['C', 'C', 'C', 'C', 'C']
        chronic_02 = ['C', 'C', 'A', 'C', 'C']
        ids = ['ID1', 'ID1', 'ID1', 'ID1', 'ID1']
        order = range(5)
        df = pd.DataFrame({'_DIAG_01': diags_01,
                            'DIAG_01': diags_01,
                            'ACUTE_01': chronic_01,
                            'DIAG_02': diags_02,
                            'ACUTE_02': chronic_02,
                            'ENCRYPTED_HESID':ids,
                            'ORDER':order},)
        #print(df)
        x=filter_hes_sort.only_keep_first_chronic_occurrence(df)
        #print(x)
        pd.testing.assert_frame_equal(\
              x,
              pd.DataFrame({'_DIAG_01': {0: 'A00', 1: 'B00', 2: 'C00', 3: 'A00', 4: 'Z00'},
                            'DIAG_01': {0: 'A00', 1: 'B00', 2: 'C00', 3: 'A00', 4: 'Z00'},
                            'ACUTE_01': {0: 'C', 1: 'C', 2: 'C', 3: 'C', 4: 'C'},
                            'DIAG_02': {0: 'B00', 1: 'C00', 2: 'Z00', 3: 'Z00', 4: 'B00'},
                            'ACUTE_02': {0: 'C', 1: 'C', 2: 'A', 3: 'C', 4: 'C'},
                            'ENCRYPTED_HESID': {0: 'ID1', 1: 'ID1', 2: 'ID1', 3: 'ID1', 4: 'ID1'},
                            'ORDER': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
                            'IGNORE': {0: np.nan, 1: True, 2: True, 3: True, 4: True}},))
        #print(x.to_dict())
        
    def test_ignore_repeated_chronic_repeated_acute_in_sec(self):
        diags_01 = ['A00', 'B00', 'C00', 'A00', 'Z00']
        diags_02 = ['B00', 'C00', 'Z00', 'Z00', 'B00']
        chronic_01 = ['C', 'C', 'C', 'C', 'C']
        chronic_02 = ['C', 'C', 'A', 'A', 'C']
        ids = ['ID1', 'ID1', 'ID1', 'ID1', 'ID1']
        order = range(5)
        df = pd.DataFrame({'_DIAG_01': diags_01,
                            'DIAG_01': diags_01,
                            'ACUTE_01': chronic_01,
                            'DIAG_02': diags_02,
                            'ACUTE_02': chronic_02,
                            'ENCRYPTED_HESID':ids,
                            'ORDER':order},)
        #print(df)
        x=filter_hes_sort.only_keep_first_chronic_occurrence(df)
        #print(x)
        pd.testing.assert_frame_equal(\
              x,
              pd.DataFrame({'_DIAG_01': {0: 'A00', 1: 'B00', 2: 'C00', 3: 'A00', 4: 'Z00'},
                            'DIAG_01': {0: 'A00', 1: 'B00', 2: 'C00', 3: 'A00', 4: 'Z00'},
                            'ACUTE_01': {0: 'C', 1: 'C', 2: 'C', 3: 'C', 4: 'C'},
                            'DIAG_02': {0: 'B00', 1: 'C00', 2: 'Z00', 3: 'Z00', 4: 'B00'},
                            'ACUTE_02': {0: 'C', 1: 'C', 2: 'A', 3: 'A', 4: 'C'},
                            'ENCRYPTED_HESID': {0: 'ID1', 1: 'ID1', 2: 'ID1', 3: 'ID1', 4: 'ID1'},
                            'ORDER': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
                            'IGNORE': {0: np.nan, 1: True, 2: True, 3: True, 4: np.nan}},))
        #print(x.to_dict())
            
    def test_ignore_repeated_chronic_one_secondary(self):
        diags_01 = ['A00', 'B00', 'C00', 'A00', 'Z00']
        diags_02 = ['C00', 'B00', 'Z00', 'A00', 'B00']
        chronic_01 = ['C', 'C', 'C', 'C', 'C']
        chronic_02 = ['C', 'C', 'A', 'C', 'C']
        ids = ['ID1', 'ID1', 'ID1', 'ID1', 'ID1']
        order = range(5)
        df = pd.DataFrame({'_DIAG_01': diags_01,
                            'DIAG_01': diags_01,
                            'ACUTE_01': chronic_01,
                            'DIAG_02': diags_02,
                            'ACUTE_02': chronic_02,
                            'ENCRYPTED_HESID':ids,
                            'ORDER':order},)
        #print(df)
        x=filter_hes_sort.only_keep_first_chronic_occurrence(df)
        #print(x)
        pd.testing.assert_frame_equal(\
              x,
              pd.DataFrame({'_DIAG_01': {0: 'A00', 1: 'B00', 2: 'C00', 3: 'A00', 4: 'Z00'},
                            'DIAG_01': {0: 'A00', 1: 'B00', 2: 'C00', 3: 'A00', 4: 'Z00'},
                            'ACUTE_01': {0: 'C', 1: 'C', 2: 'C', 3: 'C', 4: 'C'},
                            'DIAG_02': {0: 'C00', 1: 'B00', 2: 'Z00', 3: 'A00', 4: 'B00'},
                            'ACUTE_02': {0: 'C', 1: 'C', 2: 'A', 3: 'C', 4: 'C'},
                            'ENCRYPTED_HESID': {0: 'ID1', 1: 'ID1', 2: 'ID1', 3: 'ID1', 4: 'ID1'},
                            'ORDER': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
                            'IGNORE': {0: np.nan, 1: np.nan, 2: True, 3: True, 4: np.nan}},))
        #print(x.to_dict())
        
        
    def test_ignore_repeated_chronic_no_secondary_influence(self):
        diags_01 = ['A00', 'B00', 'C00', 'A00', 'Z00']
        diags_02 = ['C00', 'B00', 'Z00', 'A00', 'B00']
        chronic_01 = ['C', 'C', 'C', 'C', 'C']
        chronic_02 = ['A', 'C', 'A', 'C', 'C']
        ids = ['ID1', 'ID1', 'ID1', 'ID1', 'ID1']
        order = range(5)
        df = pd.DataFrame({'_DIAG_01': diags_01,
                            'DIAG_01': diags_01,
                            'ACUTE_01': chronic_01,
                            'DIAG_02': diags_02,
                            'ACUTE_02': chronic_02,
                            'ENCRYPTED_HESID':ids,
                            'ORDER':order},)
        #print(df)
        x=filter_hes_sort.only_keep_first_chronic_occurrence(df)
        #print(x)
        pd.testing.assert_frame_equal(\
              x,
              pd.DataFrame({'_DIAG_01': {0: 'A00', 1: 'B00', 2: 'C00', 3: 'A00', 4: 'Z00'},
                            'DIAG_01': {0: 'A00', 1: 'B00', 2: 'C00', 3: 'A00', 4: 'Z00'},
                            'ACUTE_01': {0: 'C', 1: 'C', 2: 'C', 3: 'C', 4: 'C'},
                            'DIAG_02': {0: 'C00', 1: 'B00', 2: 'Z00', 3: 'A00', 4: 'B00'},
                            'ACUTE_02': {0: 'A', 1: 'C', 2: 'A', 3: 'C', 4: 'C'},
                            'ENCRYPTED_HESID': {0: 'ID1', 1: 'ID1', 2: 'ID1', 3: 'ID1', 4: 'ID1'},
                            'ORDER': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
                            'IGNORE': {0: np.nan, 1: np.nan, 2: np.nan, 3: True, 4: np.nan}},))
        #print(x.to_dict())
        
    def test_ignore_repeated_chronic_no_secondary_influence_none_ignored(self):
        diags_01 = ['A00', 'B00', 'C00', 'A00', 'Z00']
        diags_02 = ['C00', 'A00', 'Z00', 'A00', 'B00']
        chronic_01 = ['A', 'C', 'C', 'C', 'C']
        chronic_02 = ['A', 'A', 'A', 'C', 'C']
        ids = ['ID1', 'ID1', 'ID1', 'ID1', 'ID1']
        order = range(5)
        df = pd.DataFrame({'_DIAG_01': diags_01,
                            'DIAG_01': diags_01,
                            'ACUTE_01': chronic_01,
                            'DIAG_02': diags_02,
                            'ACUTE_02': chronic_02,
                            'ENCRYPTED_HESID':ids,
                            'ORDER':order},)
        #print(df)
        x=filter_hes_sort.only_keep_first_chronic_occurrence(df)
        #print(x)
        df_exp = pd.DataFrame({'_DIAG_01': {0: 'A00', 1: 'B00', 2: 'C00', 3: 'A00', 4: 'Z00'},
                            'DIAG_01': {0: 'A00', 1: 'B00', 2: 'C00', 3: 'A00', 4: 'Z00'},
                            'ACUTE_01': {0: 'A', 1: 'C', 2: 'C', 3: 'C', 4: 'C'},
                            'DIAG_02': {0: 'C00', 1: 'A00', 2: 'Z00', 3: 'A00', 4: 'B00'},
                            'ACUTE_02': {0: 'A', 1: 'A', 2: 'A', 3: 'C', 4: 'C'},
                            'ENCRYPTED_HESID': {0: 'ID1', 1: 'ID1', 2: 'ID1', 3: 'ID1', 4: 'ID1'},
                            'ORDER': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
                            'IGNORE': {0: np.nan, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan}},)
        df_exp = df_exp.astype({'IGNORE':object})
        pd.testing.assert_frame_equal(x,df_exp)
        #print(x.to_dict())
        
    def test_ignore_repeated_chronic_none_ignored(self):
        diags_01 = ['A00', 'B00', 'C00', 'Y00', 'Z00']
        diags_02 = ['C00', 'B00', 'Z00', 'A00', 'B00']
        chronic_01 = ['C', 'C', 'C', 'C', 'C']
        chronic_02 = ['A', 'C', 'A', 'C', 'C']
        ids = ['ID1', 'ID1', 'ID1', 'ID1', 'ID1']
        order = range(5)
        df = pd.DataFrame({'_DIAG_01': diags_01,
                            'DIAG_01': diags_01,
                            'ACUTE_01': chronic_01,
                            'DIAG_02': diags_02,
                            'ACUTE_02': chronic_02,
                            'ENCRYPTED_HESID':ids,
                            'ORDER':order},)
        #print(df)
        x=filter_hes_sort.only_keep_first_chronic_occurrence(df)
        #print(x)
        
        df_exp = pd.DataFrame({'_DIAG_01': {0: 'A00', 1: 'B00', 2: 'C00', 3: 'Y00', 4: 'Z00'},
                            'DIAG_01': {0: 'A00', 1: 'B00', 2: 'C00', 3: 'Y00', 4: 'Z00'},
                            'ACUTE_01': {0: 'C', 1: 'C', 2: 'C', 3: 'C', 4: 'C'},
                            'DIAG_02': {0: 'C00', 1: 'B00', 2: 'Z00', 3: 'A00', 4: 'B00'},
                            'ACUTE_02': {0: 'A', 1: 'C', 2: 'A', 3: 'C', 4: 'C'},
                            'ENCRYPTED_HESID': {0: 'ID1', 1: 'ID1', 2: 'ID1', 3: 'ID1', 4: 'ID1'},
                            'ORDER': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
                            'IGNORE': {0: np.nan, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan}},)
        df_exp = df_exp.astype({'IGNORE':object})
        pd.testing.assert_frame_equal(x,df_exp)
        #print(x.to_dict())
        
    def test_ignore_repeated_chronic_same(self):
        diags_01 = ['A00', 'A00', 'A00', 'A00', 'A00']
        diags_02 = ['C00', 'A00', 'Z00', 'A00', 'B00']
        chronic_01 = ['A', 'A', 'A', 'A', 'A']
        chronic_02 = ['A', 'C', 'A', 'C', 'C']
        ids = ['ID1', 'ID1', 'ID1', 'ID1', 'ID1']
        order = range(5)
        df = pd.DataFrame({'_DIAG_01': diags_01,
                            'DIAG_01': diags_01,
                            'ACUTE_01': chronic_01,
                            'DIAG_02': diags_02,
                            'ACUTE_02': chronic_02,
                            'ENCRYPTED_HESID':ids,
                            'ORDER':order},)
        #print(df)
        x=filter_hes_sort.only_keep_first_chronic_occurrence(df)
        #print(x)
        
        df_exp = pd.DataFrame({'_DIAG_01': {0: 'A00', 1: 'A00', 2: 'A00', 3: 'A00', 4: 'A00'},
                            'DIAG_01': {0: 'A00', 1: 'A00', 2: 'A00', 3: 'A00', 4: 'A00'},
                            'ACUTE_01': {0: 'A', 1: 'A', 2: 'A', 3: 'A', 4: 'A'},
                            'DIAG_02': {0: 'C00', 1: 'A00', 2: 'Z00', 3: 'A00', 4: 'B00'},
                            'ACUTE_02': {0: 'A', 1: 'C', 2: 'A', 3: 'C', 4: 'C'},
                            'ENCRYPTED_HESID': {0: 'ID1', 1: 'ID1', 2: 'ID1', 3: 'ID1', 4: 'ID1'},
                            'ORDER': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
                            'IGNORE': {0: np.nan, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan}},)
        df_exp = df_exp.astype({'IGNORE':object})
        pd.testing.assert_frame_equal(x,df_exp)
        #print(x.to_dict())
        
        
    # This is the important test:
    # - repeated chronic secondary (Q00) should not cause an IGNORE
    def test_ignore_repeated_chronic_two_secondary(self):
        diags_01 = ['A00', 'A00', 'C00', 'A00', 'Z00', 'Y00']
        diags_02 = ['Q00', 'B00', 'Z00', 'Z00', 'Q00', 'C00']
        diags_03 = ['C00', 'B00', 'Z00', 'Z00', 'Y00', 'A00']
        chronic_01 = ['C', 'A', 'C', 'C', 'C', 'C']
        chronic_02 = ['C', 'C', 'A', 'A', 'C', 'C']
        chronic_03 = ['C', 'C', 'A', 'A', 'A', 'C']
        ids = ['ID1', 'ID1', 'ID1', 'ID1', 'ID1', 'ID1']
        order = range(6)
        df = pd.DataFrame({'_DIAG_01': diags_01,
                           'DIAG_01': diags_01,
                           'ACUTE_01': chronic_01,
                           'DIAG_02': diags_02,
                           'ACUTE_02': chronic_02,
                           'DIAG_03': diags_03,
                           'ACUTE_03': chronic_03,
                           'ENCRYPTED_HESID':ids,
                           'ORDER':order},)
        #print(df)
        x=filter_hes_sort.only_keep_first_chronic_occurrence(df)
        #print(x)
        #print(x.to_dict())
        #pdb.set_trace()
        pd.testing.assert_frame_equal(\
              x,
              pd.DataFrame({'_DIAG_01': {0: 'A00', 1: 'A00', 2: 'C00', 3: 'A00', 4: 'Z00', 5: 'Y00'},
                            'DIAG_01': {0: 'A00', 1: 'A00', 2: 'C00', 3: 'A00', 4: 'Z00', 5: 'Y00'},
                            'ACUTE_01': {0: 'C', 1: 'A', 2: 'C', 3: 'C', 4: 'C', 5: 'C'},
                            'DIAG_02': {0: 'Q00', 1: 'B00', 2: 'Z00', 3: 'Z00', 4: 'Q00', 5: 'C00'},
                            'ACUTE_02': {0: 'C', 1: 'C', 2: 'A', 3: 'A', 4: 'C', 5: 'C'},
                            'DIAG_03': {0: 'C00', 1: 'B00', 2: 'Z00', 3: 'Z00', 4: 'Y00', 5: 'A00'},
                            'ACUTE_03': {0: 'C', 1: 'C', 2: 'A', 3: 'A', 4: 'A', 5: 'C'},
                            'ENCRYPTED_HESID': {0: 'ID1', 1: 'ID1', 2: 'ID1', 3: 'ID1', 4: 'ID1', 5: 'ID1'},
                            'ORDER': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
                            'IGNORE': {0: np.nan, 1: np.nan, 2: True, 3: True, 4: np.nan, 5: np.nan}}))

    def test_ignore_repeated_chronic_no_secondaries(self):
        diags_01 = ['A00', 'B00', 'C00', 'A00', 'B00']
        chronic_01 = ['C', 'A', 'C', 'C', 'C']
        ids = ['ID1', 'ID1', 'ID1', 'ID1', 'ID1']
        order = range(5)
        df = pd.DataFrame({'_DIAG_01': diags_01,
                            'DIAG_01': diags_01,
                            'ACUTE_01': chronic_01,
                            'ENCRYPTED_HESID':ids,
                            'ORDER':order},)
        #print(df)
        x=filter_hes_sort.only_keep_first_chronic_occurrence(df)
        #print(x)
        pd.testing.assert_frame_equal(\
              x,
              pd.DataFrame({'_DIAG_01': {0: 'A00', 1: 'B00', 2: 'C00', 3: 'A00', 4: 'B00'},
                            'DIAG_01': {0: 'A00', 1: 'B00', 2: 'C00', 3: 'A00', 4: 'B00'},
                            'ACUTE_01': {0: 'C', 1: 'A', 2: 'C', 3: 'C', 4: 'C'},
                            'ENCRYPTED_HESID': {0: 'ID1', 1: 'ID1', 2: 'ID1', 3: 'ID1', 4: 'ID1'},
                            'ORDER': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
                            'IGNORE': {0: np.nan, 1: np.nan, 2: np.nan, 3: True, 4: np.nan}},))

    def test_ignore_repeated_chronic_no_secondaries2(self):
        diags_01 = ['A00', 'A00', 'A00', 'A00', 'A00']
        chronic_01 = ['A', 'A', 'C', 'A', 'C']
        ids = ['ID1', 'ID1', 'ID1', 'ID1', 'ID1']
        order = range(5)
        df = pd.DataFrame({'_DIAG_01': diags_01,
                            'DIAG_01': diags_01,
                            'ACUTE_01': chronic_01,
                            'ENCRYPTED_HESID':ids,
                            'ORDER':order},)
        #print(df)
        x=filter_hes_sort.only_keep_first_chronic_occurrence(df)
        
        df_exp = \
              pd.DataFrame({'_DIAG_01': {0: 'A00', 1: 'A00', 2: 'A00', 3: 'A00', 4: 'A00'},
                            'DIAG_01': {0: 'A00', 1: 'A00', 2: 'A00', 3: 'A00', 4: 'A00'},
                            'ACUTE_01': {0: 'A', 1: 'A', 2: 'C', 3: 'A', 4: 'C'},
                            'ENCRYPTED_HESID': {0: 'ID1', 1: 'ID1', 2: 'ID1', 3: 'ID1', 4: 'ID1'},
                            'ORDER': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
                            'IGNORE': {0: np.nan, 1: np.nan, 2: np.nan, 3: np.nan, 4: True}},)
        df_exp = df_exp.astype({'IGNORE':object})
        pd.testing.assert_frame_equal(x,df_exp)


    def test_ignore_repeated_chronic_no_secondaries3(self):
        diags_01 = ['A00', 'A00', 'A00', 'A00', 'A00']
        chronic_01 = ['A', 'A', 'C', 'A', 'A']
        ids = ['ID1', 'ID1', 'ID1', 'ID1', 'ID1']
        order = range(5)
        df = pd.DataFrame({'_DIAG_01': diags_01,
                            'DIAG_01': diags_01,
                            'ACUTE_01': chronic_01,
                            'ENCRYPTED_HESID':ids,
                            'ORDER':order},)
        #print(df)
        x=filter_hes_sort.only_keep_first_chronic_occurrence(df)
        
        df_exp = \
              pd.DataFrame({'_DIAG_01': {0: 'A00', 1: 'A00', 2: 'A00', 3: 'A00', 4: 'A00'},
                            'DIAG_01': {0: 'A00', 1: 'A00', 2: 'A00', 3: 'A00', 4: 'A00'},
                            'ACUTE_01': {0: 'A', 1: 'A', 2: 'C', 3: 'A', 4: 'A'},
                            'ENCRYPTED_HESID': {0: 'ID1', 1: 'ID1', 2: 'ID1', 3: 'ID1', 4: 'ID1'},
                            'ORDER': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
                            'IGNORE': {0: np.nan, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan}},)
        df_exp = df_exp.astype({'IGNORE':object})
        pd.testing.assert_frame_equal(x,df_exp)



class Test_convert_chapter_headings(unittest.TestCase):
    
    def setUp(self):
        self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0,
                                                   1,1,1],
                                'MYADMIDATE':np.append(
                                    np.repeat(np.datetime64('2005-01-01'),3),
                                    np.repeat(np.datetime64('2010-01-01'),3),),
                                'DIAG_01':['A00','B49','C99',
                                           'N99','X99','O00'],
                                'IGNORE':np.repeat(False,6)})
        self.df = self.df.astype({'DIAG_01':'category'})


    @patch('pipeline_hes.params.params.CHAPTER_HEADINGS_USE_GRANULAR', True)
    def test_useGranularToConvert(self):

        df_out = filter_hes_sort.convert_chapters_ignore_no_match(self.df.copy())
        
        df_exp = self.df.copy()
        df_exp['DIAG_01'] = ['A00-A09','B35-B49',params.CHAPTER_NO_MATCH,
                             'N99-N99',params.CHAPTER_NO_MATCH,params.CHAPTER_NO_MATCH]
        df_exp['DIAG_01_ALT1'] = ['A00-B99','A00-B99','C00-D48',
                                  'N00-N99',params.CHAPTER_NO_MATCH,params.CHAPTER_NO_MATCH]
        df_exp['DIAG_01_ALT2'] = ['A00','B49','C99',
                                  'N99','X99','O00']
        df_exp['_DIAG_01'] = ['A00','B49','C99',
                              'N99','X99','O00']
        df_exp = df_exp.astype({'DIAG_01':'category'})
        df_exp = df_exp.astype({'_DIAG_01':'category'})
        df_exp = df_exp.astype({'DIAG_01_ALT1':'category'})
        df_exp = df_exp.astype({'DIAG_01_ALT2':'category'})
        df_exp['IGNORE'] = [False,False,True,
                            False,True,True]

        pd.testing.assert_frame_equal(df_out,df_exp,check_like=True)

    @patch('pipeline_hes.params.params.CHAPTER_HEADINGS_USE_GRANULAR', False)
    def test_useCoarseToConvert(self):

        df_out = filter_hes_sort.convert_chapters_ignore_no_match(self.df.copy())
        
        df_exp = self.df.copy()
        df_exp['DIAG_01'] = ['A00-B99','A00-B99','C00-D48',
                             'N00-N99',params.CHAPTER_NO_MATCH,params.CHAPTER_NO_MATCH]
        df_exp['DIAG_01_ALT1'] = ['A00-A09','B35-B49',params.CHAPTER_NO_MATCH,
                                 'N99-N99',params.CHAPTER_NO_MATCH,params.CHAPTER_NO_MATCH]
        df_exp['DIAG_01_ALT2'] = ['A00','B49','C99',
                                  'N99','X99','O00']
        df_exp['_DIAG_01'] = ['A00','B49','C99',
                              'N99','X99','O00']
        
        df_exp = df_exp.astype({'DIAG_01':'category'})
        df_exp = df_exp.astype({'_DIAG_01':'category'})
        df_exp = df_exp.astype({'DIAG_01_ALT1':'category'})
        df_exp = df_exp.astype({'DIAG_01_ALT2':'category'})
        df_exp['IGNORE'] = [False,False,False,
                            False,True,True]
        #pdb.set_trace()
        pd.testing.assert_frame_equal(df_out,df_exp,check_like=True)




# class Test_add_followup_duration_col(unittest.TestCase):


#     def setUp(self):
        
#         # times = np.array([np.datetime64('2005-03-01'),
#         #                   np.datetime64('2005-01-01'),
#         #                   np.datetime64('2005-04-01'),
#         #                   np.datetime64('2005-02-01'),
                          
#         #                   np.datetime64('2005-06-01'),
#         #                   np.datetime64('2005-07-01'),
#         #                   np.datetime64('2005-08-01'),
                          
#         #                   np.datetime64('2005-09-01'),
#         #                   np.datetime64('2005-10-01'),
#         #                   np.datetime64('2005-11-01')])
#         # #
#         epistart = np.array([np.datetime64('2005-01-01'),
#                           np.datetime64('2005-01-01'),
#                           np.datetime64('2005-01-01'),
#                           np.datetime64('2005-01-01'),
                          
#                           np.datetime64('2005-07-01'),
#                           np.datetime64('2005-07-01'),
#                           np.datetime64('2005-07-01'),
                          
#                           np.datetime64('2005-10-01'),
#                           np.datetime64('2005-10-01'),
#                           np.datetime64('2005-10-01')])
        
#         censor_just_before_ami = np.repeat(np.datetime64('2005-03-01'),4)-\
#             np.timedelta64(1,'s')
#         censor = np.array(
#             np.concatenate([censor_just_before_ami,
#                             np.repeat(np.datetime64('2017-03-27'),3),
#                             np.repeat(np.datetime64('2017-03-27'),3)],axis=None),
#             dtype='datetime64[ns]')
        
#         # ctl/ctl/ctl/ctl, ctl(no-ami)/ctl(no-ami)/ctl(no-ami), pat/pat/pat
#         self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0,0,
#                                                    1,1,1,
#                                                    2,2,2],
#                                 'CHOSEN_INIT':[True,False,False,False,
#                                               True,False,False,
#                                               True,False,False,],
#                                 'MYEPISTART':epistart,
#                                 'CENSOR':censor,})

#     def test_add_followup_duration_col(self):

#         df_out = filter_hes_sort.add_followup_duration_col(self.df.copy())
        
#         # 1: '2005-01-01' -> '2005-03-01' = ~2
#         # 2: '2005-07-01' -> '2017-03-27' = 11years + 9 months (141)
#         # 3: '2005-10-01' -> '2017-03-27' = 11years + 6 months (138)
#         #pdb.set_trace()
#         np.testing.assert_array_almost_equal(df_out['DUR'].values,
#                                       np.array([59,59,59,59,
#                                                 4287,4287,4287,
#                                                 4195,4195,4195]))




if __name__ == '__main__':
    # Show full dataframes
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.expand_frame_repr', False)
    unittest.main(argv=['-v'],verbosity=3)

