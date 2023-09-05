# -*- coding: utf-8 -*-
"""
@author: Chris Hayward
"""

import unittest
import numpy as np
import pandas as pd
import pdb

from pipeline_hes import traces_hes
from pipeline_hes import filter_hes_sort
from pipeline_hes.params import params
from pipeline_hes import adj_matrix

from unittest.mock import patch

# ###############



# The DEPENDENCY THRESH for the B to C conn is (9-1) / (10+1) = 0.7272...
def get_small_example(diag1='A00',diag2='B49',diag3='C99'):
    # hes id (10 cases, 3 events each)
    hesids = np.repeat(np.arange(10),3)
    diagNormal = np.array([diag1, diag2, diag3])
    diagNoise = np.array([diag1, diag3, diag2])
    diag = np.tile(np.array(diagNormal),9)
    diag = np.append(diag,diagNoise)
    # 3 months
    times = np.arange('2005-01', '2005-04', dtype='datetime64[M]')
    times = np.tile(times,10)
    df = pd.DataFrame(data={'MYEPISTART':times, 'DIAG_01':diag,'ENCRYPTED_HESID':hesids})
    df = df.astype({'DIAG_01':'category'})
    df['DUR'] = 4 # four months
    return df



class Test_adj_matrix_generation(unittest.TestCase):
    
    def setUp(self):
        self.df = get_small_example()

    def test_dfg(self):
        df_from_to = adj_matrix.get_directly_follows_counts(self.df)
        pd.testing.assert_series_equal(\
              df_from_to,
              pd.Series([9,9,1,1],
                        index=pd.MultiIndex.from_tuples([('A00','B49'),
                                                         ('B49','C99'),
                                                         ('A00','C99'),
                                                         ('C99','B49')],
                                                        names=['From', 'To'])))

    @patch('pipeline_hes.params.params.CHAPTER_HEADINGS_USE_GRANULAR', False)
    def test_adj_coarse(self):
        #pdb.set_trace()
        df = filter_hes_sort.convert_chapters_ignore_no_match(self.df)
        df_adj = adj_matrix.generate_dfg_adj_from_df(df)
        df_adj = df_adj.loc[['A00-B99','C00-D48'],['A00-B99','C00-D48']]
        pd.testing.assert_frame_equal(\
              df_adj,
              pd.DataFrame({'A00-B99':(9.0,1.0), 'C00-D48':(10.0,0.0)},
                           index=['A00-B99','C00-D48']))

    @patch('pipeline_hes.params.params.CHAPTER_HEADINGS_USE_GRANULAR', True)
    def test_adj_granular(self):
        
        #C99 is not in the granular list
        # Replace with C90
        self.df = get_small_example(diag3='C90')
        
        df = filter_hes_sort.convert_chapters_ignore_no_match(self.df)
        df_adj = adj_matrix.generate_dfg_adj_from_df(df)
        df_adj = df_adj.loc[['A00-A09','B35-B49','C81-C96'],['A00-A09','B35-B49','C81-C96']]
        pd.testing.assert_frame_equal(\
              df_adj,
              pd.DataFrame({'A00-A09':(0.0,0.0,0.0),
                            'B35-B49':(9.0,0.0,1.0),
                            'C81-C96':(1.0,9.0,0.0)},
                           index=['A00-A09','B35-B49','C81-C96']))









class Test_remove_rare_traces_using_IS_CONTROL(unittest.TestCase):
    
    def setUp(self):
        self.n = 1000
        self.ctl_vars_per_sub = pd.DataFrame({'variant':\
                                  np.concatenate([np.repeat('A,B,C',self.n/2),
                                                 np.repeat('D,E,F',self.n/2),
                                                 np.repeat('G,H,I',20)],axis=None),
                                'IS_CONTROL':np.repeat(True,self.n+20),
                                })
        # Fewer AMI (enables testing)
        self.pat_vars_per_sub = pd.DataFrame({'variant':\
                                  np.concatenate([np.repeat('A,B,C',self.n/4),
                                                 np.repeat('D,E,F',self.n/4),
                                                 np.repeat('G,H,I',10)],axis=None),
                                'IS_CONTROL':np.repeat(False,self.n/2+10),
                                })
    
    
    @patch('pipeline_hes.params.params.AMI_TRACE_COUNTS_PRC_THRESHOLD',
           (100*10/510))
   # @patch('pipeline_hes.params.params.CTL_TRACE_COUNTS_LOWER_LIMIT', 20)
    # remove based on ~IS_CONTROL (which may not be AMI, depending on data split choice)
    # Doing this means the non-rare traces are consistent across choice of data split
    def test_remove_rare_traces_using_IS_CONTROL_1(self):
        non_rare_traces_out = traces_hes.get_subset_of_traces_for_figure(
            self.ctl_vars_per_sub,self.pat_vars_per_sub)
        non_rare_exp = pd.DataFrame({'count':[self.n/4,self.n/4,10]},
                                    index=['A,B,C','D,E,F','G,H,I'],
                                    dtype=np.int64)
        pd.testing.assert_frame_equal(non_rare_traces_out.sort_index(),
                                      non_rare_exp.sort_index())

    # just over the 10 subject limit
    @patch('pipeline_hes.params.params.AMI_TRACE_COUNTS_PRC_THRESHOLD',
           (100*11/510))
    #@patch('pipeline_hes.params.params.CTL_TRACE_COUNTS_LOWER_LIMIT', 20)
    def test_remove_rare_traces_using_IS_CONTROL_change_prc(self):
        non_rare_traces_out = traces_hes.get_subset_of_traces_for_figure(
            self.ctl_vars_per_sub,self.pat_vars_per_sub)
        non_rare_exp = pd.DataFrame({'count':[self.n/4,self.n/4]},
                                    index=['A,B,C','D,E,F'],
                                    dtype=np.int64)
        pd.testing.assert_frame_equal(non_rare_traces_out.sort_index(),
                                      non_rare_exp.sort_index())


# NO LONGER DOING THIS (HIDING SOME TRACES)
    # @patch('pipeline_hes.params.params.AMI_TRACE_COUNTS_PRC_THRESHOLD',
    #        (100*1/510))
    # @patch('pipeline_hes.params.params.CTL_TRACE_COUNTS_LOWER_LIMIT', 21)
    # def test_remove_rare_traces_using_IS_CONTROL_change_ctl_limit(self):
    #     non_rare_traces_out = traces_hes.get_subset_of_traces_for_figure(
    #         self.ctl_vars_per_sub,self.pat_vars_per_sub)
    #     non_rare_exp = pd.DataFrame({'count':[self.n/4,self.n/4]},
    #                                 index=['A,B,C','D,E,F'],
    #                                 dtype=np.int64)
    #     pd.testing.assert_frame_equal(non_rare_traces_out.sort_index(),
    #                                   non_rare_exp.sort_index())


class Test_GetVariants(unittest.TestCase):

    def setUp(self):
        
        n = 10
        self.df = pd.DataFrame({'ENCRYPTED_HESID':np.repeat(np.arange(n),3),
                                'ORDER':np.arange(3*n),
                                'DIAG_01':np.tile([params.AMI_INIT,
                                                   'B00', params.CENSOR_CODE],n),
                                'DIAG_01_ALT1':np.tile([params.AMI_INIT,
                                                        'B00_ALT1', params.CENSOR_CODE],n),
                                'DIAG_01_ALT2':np.tile([params.AMI_INIT,
                                                        'B00_ALT2', params.CENSOR_CODE],n),
                                'IS_CONTROL':np.repeat(False,3*n),
                                'IS_PATIENT':np.repeat(1,3*n),
                                'IGNORE':np.repeat(False,3*n),
                                'DUR':np.ones(3*n),
                                'SEX':np.repeat(1,3*n),
                                'INIT_AGE':np.repeat(20,3*n),
                                'IMD04':np.repeat(1.1,3*n),
                                'MATCHED_DATE':np.repeat(np.datetime64('2005-01-01'),3*n),
                                'Mortality':np.repeat(0,3*n),
                                'PROCODE':np.repeat(1,3*n),
                                })
        self.df = self.df.astype({'DIAG_01':'category'})
        self.df = self.df.astype({'DIAG_01_ALT1':'category'})
        self.df = self.df.astype({'DIAG_01_ALT2':'category'})
        


    def addRow(self,row):
        self.df = self.df.astype({'DIAG_01':str})
        self.df.loc[self.df.shape[0], ['ORDER', 'DIAG_01', 'ENCRYPTED_HESID']] = row
        self.df = self.df.astype({'DIAG_01':'category'})


    def test_noChange(self):
        variants_count, variants_per_subject = \
            traces_hes.get_variants(self.df.copy())
        
        traceStr = '{},{},{}'.format(params.AMI_INIT,'B00',params.CENSOR_CODE)
        traceStrALT1 = '{},{},{}'.format(params.AMI_INIT,'B00_ALT1',params.CENSOR_CODE)
        traceStrALT2 = '{},{},{}'.format(params.AMI_INIT,'B00_ALT2',params.CENSOR_CODE)
        #pdb.set_trace()
        # Check counts
        df_exp = pd.DataFrame.from_dict({traceStr:[10,1,10*1],},
                                        orient='index',
                                        columns=['count', 'num_events', 'num_events_subs'])
        pd.testing.assert_frame_equal(variants_count, df_exp, check_like=True)
        
        # Check per subject
        v_per_sub_exp = pd.DataFrame(
                  {'case': np.array(np.arange(10),dtype=np.int64),
                   'variant': np.repeat(traceStr,10),
                   'variant_ALT1': np.repeat(traceStrALT1,10),
                   'variant_ALT2': np.repeat(traceStrALT2,10),
                   'num_events': np.array(np.repeat(1,10),dtype=np.int64),
                   'events_per_month': np.array(np.repeat(1,10),dtype=np.float64),
                   'IS_CONTROL':np.repeat(False,10),
                   'IS_PATIENT':np.repeat(1,10),
                   'DUR':np.ones(10),
                   'SEX':np.repeat(1,10),
                   'INIT_AGE':np.repeat(20,10),
                   'IMD04':np.repeat(1.1,10),
                   'MATCHED_DATE':np.repeat(np.datetime64('2005-01-01'),10),
                   'Mortality':np.repeat(0,10),
                   'PROCODE':np.repeat(1,10),})
        
        
        pd.testing.assert_frame_equal(variants_per_subject,v_per_sub_exp,check_like=True)

    def test_replace_first_trace(self):
        
        df_tmp = self.df.copy()
        df_tmp['DIAG_01'] = df_tmp['DIAG_01'].cat.add_categories('C00')
        df_tmp['DIAG_01_ALT1'] = df_tmp['DIAG_01_ALT1'].cat.add_categories('C00_ALT1')
        df_tmp['DIAG_01_ALT2'] = df_tmp['DIAG_01_ALT2'].cat.add_categories('C00_ALT2')
        
        df_tmp.loc[0, ['ENCRYPTED_HESID','ORDER','DIAG_01','DIAG_01_ALT1','DIAG_01_ALT2','IS_CONTROL','DUR']] = \
            [0,2,params.CENSOR_CODE,params.CENSOR_CODE,params.CENSOR_CODE, False, 1.0]
        df_tmp.loc[1, ['ENCRYPTED_HESID','ORDER','DIAG_01','DIAG_01_ALT1','DIAG_01_ALT2','IS_CONTROL','DUR']] = \
            [0,1,'C00', 'C00_ALT1','C00_ALT2',False, 1.0]
        df_tmp.loc[2, ['ENCRYPTED_HESID','ORDER','DIAG_01','DIAG_01_ALT1','DIAG_01_ALT2','IS_CONTROL','DUR']] = \
            [0,0,params.AMI_INIT, params.AMI_INIT, params.AMI_INIT, False, 1.0]

        variants_count, variants_per_subject = traces_hes.get_variants(df_tmp)
        
        # check trace counts:
        traceStr = '{},{},{}'.format(params.AMI_INIT,'B00',params.CENSOR_CODE)
        traceStr_ALT1 = '{},{},{}'.format(params.AMI_INIT,'B00_ALT1',params.CENSOR_CODE)
        traceStr_ALT2 = '{},{},{}'.format(params.AMI_INIT,'B00_ALT2',params.CENSOR_CODE)
        newTrace = '{},{},{}'.format(params.AMI_INIT,'C00',params.CENSOR_CODE)
        newTrace_ALT1 = '{},{},{}'.format(params.AMI_INIT,'C00_ALT1',params.CENSOR_CODE)
        newTrace_ALT2 = '{},{},{}'.format(params.AMI_INIT,'C00_ALT2',params.CENSOR_CODE)

        df_exp = pd.DataFrame.from_dict({traceStr:[9,1,9*1],
                                          newTrace:[1,1,1*1],},
                                          orient='index',
                                          columns=['count', 'num_events', 'num_events_subs'])
        
        pd.testing.assert_frame_equal(variants_count, df_exp, check_like=True)

        # check traces per sub:
        v_per_sub_exp = pd.DataFrame(
                  {'case': np.array(np.arange(10),dtype=np.int64),
                    'variant': np.append(newTrace,np.repeat(traceStr,9)),
                    'variant_ALT1': np.append(newTrace_ALT1,np.repeat(traceStr_ALT1,9)),
                    'variant_ALT2': np.append(newTrace_ALT2,np.repeat(traceStr_ALT2,9)),
                    'num_events': np.array(np.repeat(1,10),dtype=np.int64),
                    'events_per_month': np.array(np.repeat(1,10),dtype=np.float64),
                    'IS_CONTROL':np.repeat(False,10),
                    'IS_PATIENT':np.repeat(1,10),
                    'DUR':np.ones(10),
                    'SEX':np.repeat(1,10),
                    'INIT_AGE':np.repeat(20,10),
                    'IMD04':np.repeat(1.1,10),
                    'MATCHED_DATE':np.repeat(np.datetime64('2005-01-01'),10),
                    'Mortality':np.repeat(0,10),
                    'PROCODE':np.repeat(1,10),})
        

        pd.testing.assert_frame_equal(variants_per_subject,v_per_sub_exp,check_like=True)
            
    def test_newTrace_long(self):
        
        df_tmp = self.df.copy()
        df_tmp['DIAG_01'] = df_tmp['DIAG_01'].cat.add_categories('C00')
        df_tmp['DIAG_01_ALT1'] = df_tmp['DIAG_01_ALT1'].cat.add_categories('C00')
        df_tmp['DIAG_01_ALT2'] = df_tmp['DIAG_01_ALT2'].cat.add_categories('C00')
        
        new_trace = '{},{},{},{}'.format(params.AMI_INIT,'B00','C00',params.CENSOR_CODE)
        new_trace_ALT1 = '{},{},{},{}'.format(params.AMI_INIT,'B00_ALT1','C00_ALT1',params.CENSOR_CODE)
        new_trace_ALT2 = '{},{},{},{}'.format(params.AMI_INIT,'B00_ALT2','C00_ALT2',params.CENSOR_CODE)
        
        df_new_trace = pd.DataFrame({'ENCRYPTED_HESID':np.repeat(10,4),
                                     'ORDER':[1,2,3,4],
                                     'DIAG_01':new_trace.split(','),
                                     'DIAG_01_ALT1':new_trace_ALT1.split(','),
                                     'DIAG_01_ALT2':new_trace_ALT2.split(','),
                                     'IS_CONTROL':np.repeat(False,4),
                                     'IS_PATIENT':np.repeat(1,4),
                                     'IGNORE':np.repeat(False,4),
                                     'IGNORE_ALT':np.repeat(False,4),
                                     'DUR':np.repeat(1,4),
                                     'SEX':np.repeat(1,4),
                                     'INIT_AGE':np.repeat(20,4),
                                     'IMD04':np.repeat(1.1,4),
                                     'MATCHED_DATE':np.repeat(np.datetime64('2005-01-01'),4),
                                     'Mortality':np.repeat(0,4),
                                     'PROCODE':np.repeat(1,4),})
        # pdb.set_trace()
        df_tmp = pd.concat([df_tmp,df_new_trace]).reset_index(drop=True)
        
        # needed for column checks later
        self.df = df_tmp.copy()
                
        variants_count, variants_per_subject = traces_hes.get_variants(self.df.copy())
        
        # check trace counts:
        traceStr = '{},{},{}'.format(params.AMI_INIT,'B00',params.CENSOR_CODE)
        traceStr_ALT1 = '{},{},{}'.format(params.AMI_INIT,'B00_ALT1',params.CENSOR_CODE)
        traceStr_ALT2 = '{},{},{}'.format(params.AMI_INIT,'B00_ALT2',params.CENSOR_CODE)
        df_exp = pd.DataFrame.from_dict({traceStr:[10,1,10*1],
                                          new_trace:[1,2,2*1],},
                                          orient='index',
                                          columns=['count', 'num_events', 'num_events_subs'])
        pd.testing.assert_frame_equal(variants_count, df_exp, check_like=True)
        
        # check traces per sub:
        v_per_sub_exp = pd.DataFrame(
                  {'case': np.array(np.arange(11),dtype=np.int64),
                    'variant': np.append(np.repeat(traceStr,10),new_trace),
                    'variant_ALT1': np.append(np.repeat(traceStr_ALT1,10),new_trace_ALT1),
                    'variant_ALT2': np.append(np.repeat(traceStr_ALT2,10),new_trace_ALT2),
                    'num_events': np.append(np.array(np.repeat(1,10),dtype=np.int64),2),
                    'events_per_month': np.append(np.array(np.repeat(1,10),dtype=np.float64),2),
                    'IS_CONTROL':np.repeat(False,11),
                    'IS_PATIENT':np.repeat(1,11),
                    'DUR':np.ones(11),
                    'SEX':np.repeat(1,11),
                    'INIT_AGE':np.repeat(20,11),
                    'IMD04':np.repeat(1.1,11),
                    'MATCHED_DATE':np.repeat(np.datetime64('2005-01-01'),11),
                    'Mortality':np.repeat(0,11),
                    'PROCODE':np.repeat(1,11),})
        
        pd.testing.assert_frame_equal(variants_per_subject,v_per_sub_exp,check_like=True)
            

    def test_bad_trace_no_CENSOR(self):
        df_tmp = self.df.copy()
        df_tmp['DIAG_01'] = df_tmp['DIAG_01'].cat.add_categories('C00')
        new_trace = '{},{},{}'.format(params.AMI_INIT,'B00','C00')
        df_new_trace = pd.DataFrame({'ENCRYPTED_HESID':np.repeat(10,3),
                                      'ORDER':np.repeat(0,3),
                                      'DIAG_01':new_trace.split(','),
                                      '_DIAG_01_ALT':new_trace.split(','),
                                      'IS_CONTROL':np.repeat(False,3),
                                      'IS_PATIENT':np.repeat(1,3),
                                      'IGNORE':np.repeat(False,3),
                                      'IGNORE_ALT':np.repeat(False,3),
                                      'DUR':np.repeat(1,3)})
        df_tmp = pd.concat([df_tmp,df_new_trace]).reset_index(drop=True)

        self.assertRaises(Exception, traces_hes.get_variants, df_tmp)


    def test_bad_trace_no_AMIINIT(self):
        df_tmp = self.df.copy()
        new_trace = '{},{},{}'.format('B00','C00',params.CENSOR_CODE)
        df_new_trace = pd.DataFrame({'ENCRYPTED_HESID':np.repeat(10,3),
                                      'ORDER':np.repeat(0,3),
                                      'DIAG_01':new_trace.split(','),
                                      'DIAG_01_ALT1':new_trace.split(','),
                                      'DIAG_01_ALT2':new_trace.split(','),
                                      'IS_CONTROL':np.repeat(False,3),
                                      'IS_PATIENT':np.repeat(1,3),
                                      'IGNORE':np.repeat(False,3),
                                      'IGNORE_ALT':np.repeat(False,3),
                                      'DUR':np.repeat(1,3)})
        df_tmp = pd.concat([df_tmp,df_new_trace]).reset_index(drop=True)
        self.assertRaises(Exception, traces_hes.get_variants, df_tmp)


    def test_check_against_manual_trace_creation_using_groupby(self):
        
        df_tmp = self.df.copy()
        df_tmp['DIAG_01'] = df_tmp['DIAG_01'].cat.add_categories('C00')
        df_tmp['DIAG_01_ALT1'] = df_tmp['DIAG_01_ALT1'].cat.add_categories('C00')
        df_tmp['DIAG_01_ALT2'] = df_tmp['DIAG_01_ALT2'].cat.add_categories('C00')
        
        df_tmp.loc[2, ['ENCRYPTED_HESID','ORDER','DIAG_01','DIAG_01_ALT1','DIAG_01_ALT2','IS_CONTROL','DUR']] = \
            [0,0,params.AMI_INIT, params.AMI_INIT,params.AMI_INIT, False, 1.0]
        df_tmp.loc[1, ['ENCRYPTED_HESID','ORDER','DIAG_01','DIAG_01_ALT1','DIAG_01_ALT2','IS_CONTROL','DUR']] = \
            [0,1,'C00','C00','C00', False, 1.0]
        df_tmp.loc[0, ['ENCRYPTED_HESID','ORDER','DIAG_01','DIAG_01_ALT1','DIAG_01_ALT2','IS_CONTROL','DUR']] = \
            [0,2,params.CENSOR_CODE,params.CENSOR_CODE,params.CENSOR_CODE, False, 1.0]
        
        #pdb.set_trace()
        df_in = df_tmp.copy()
        variants_count, variants_per_subject = traces_hes.get_variants(df_in)
        
        # ########
        # Double check the trajectories...
        # ########
        my_variants_per_subject = df_tmp.\
            sort_values(['ENCRYPTED_HESID','ORDER'])[['ENCRYPTED_HESID','DIAG_01']].\
                groupby('ENCRYPTED_HESID').apply(
            lambda x: ','.join(x['DIAG_01']))
        my_variants_per_subject.name = 'variant'
        #pdb.set_trace()
        pd.testing.assert_series_equal(variants_per_subject['variant'],
                                        my_variants_per_subject.reset_index(drop=True))

        np.testing.assert_array_equal(my_variants_per_subject.value_counts(),
                                      variants_count['count'])
    
    
    
    # ########################################
    

    # def test_repeatDateChange(self):
    #     self.addRow([np.datetime64('2010-01'), 'A00', 10])
    #     self.addRow([np.datetime64('2006-01'), 'D00', 10])
    #     self.addRow([np.datetime64('2007-01'), 'D00', 10])
    #     self.addRow([np.datetime64('2008-01'), 'X00', 10])
    #     self.addRow([np.datetime64('2009-01'), 'D00', 10])
    #     variants_count, variants_per_subject = traces_hes.get_variants(self.df)
    #     pd.testing.assert_frame_equal(\
    #           variants_count, pd.DataFrame.from_dict({'A00,B49,C99':[9,2,9*2],
    #                                                   'A00,C99,B49':[1,2,1*2],
    #                                                   'D00,D00,X00,D00,A00':[1,4,1*4]},
    #                                                   orient='index',
    #                                                   columns=['count', 'num_events', 'num_events_subs']),
    #           check_like=True)
            
    #     v_per_sub_part = pd.DataFrame(
    #               {'case': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10},
    #                 'variant': {0: 'A00,B49,C99', 1: 'A00,B49,C99', 2: 'A00,B49,C99',
    #                             3: 'A00,B49,C99', 4: 'A00,B49,C99', 5: 'A00,B49,C99',
    #                             6: 'A00,B49,C99', 7: 'A00,B49,C99', 8: 'A00,B49,C99',
    #                             9: 'A00,C99,B49', 10: 'D00,D00,X00,D00,A00'},
    #                 'num_events': {0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 4}})
    #     pd.testing.assert_series_equal(\
    #           variants_per_subject['variant'], v_per_sub_part['variant'])
    #     pd.testing.assert_series_equal(\
    #           variants_per_subject['num_events'], v_per_sub_part['num_events'])
            
    # def test_same(self):
    #     self.addRow([np.datetime64('2008-03'), 'C99', 10])
    #     self.addRow([np.datetime64('2008-02'), 'C99', 10])
    #     self.addRow([np.datetime64('2008-01'), 'C99', 10])
    #     self.addRow([np.datetime64('2009-01'), 'B49', 10])
    #     self.addRow([np.datetime64('2007-01'), 'A00', 10])
    #     variants_count, variants_per_subject = traces_hes.get_variants(self.df)
    #     #pdb.set_trace()
    #     pd.testing.assert_frame_equal(\
    #           variants_count, pd.DataFrame.from_dict({'A00,B49,C99':[9,2,9*2],
    #                                                   'A00,C99,B49':[1,2,1*2],
    #                                                   'A00,C99,C99,C99,B49':[1,4,1*4]},
    #                                                  orient='index',
    #                                                  columns=['count', 'num_events', 'num_events_subs']),
    #           check_like=True)
            
    # def test_new(self):
    #     self.addRow([np.datetime64('2009-01'), 'D49', 10])
    #     variants_count, variants_per_subject = traces_hes.get_variants(self.df)
    #     pd.testing.assert_frame_equal(\
    #           variants_count, pd.DataFrame.from_dict({'A00,B49,C99':[9,2,9*2],
    #                                                   'A00,C99,B49':[1,2,1*2],
    #                                                   'D49':[1,0,1*0]},
    #                                                  orient='index',
    #                                                  columns=['count', 'num_events', 'num_events_subs']),
    #           check_like=True)
            
    # def test_new2(self):
    #     self.addRow([np.datetime64('2009-01'), 'D49', 10])
    #     self.addRow([np.datetime64('2009-01'), 'D49', 10])
    #     self.addRow([np.datetime64('2009-01'), 'D49', 10])
    #     variants_count, variants_per_subject = traces_hes.get_variants(self.df)
    #     pd.testing.assert_frame_equal(\
    #           variants_count, pd.DataFrame.from_dict({'A00,B49,C99':[9,2,9*2],
    #                                                   'A00,C99,B49':[1,2,1*2],
    #                                                   'D49,D49,D49':[1,2,1*2]},
    #                                                  orient='index',
    #                                                  columns=['count', 'num_events', 'num_events_subs']),
    #           check_like=True)
            
    # def test_month(self):
    #     self.addRow([np.datetime64('2009-03'), 'Z49', 10])
    #     self.addRow([np.datetime64('2009-01'), 'X49', 10])
    #     self.addRow([np.datetime64('2009-02'), 'Y49', 10])
    #     self.addRow([np.datetime64('2008-02'), 'Y49', 11])
    #     self.addRow([np.datetime64('2008-03'), 'Z49', 11])
    #     self.addRow([np.datetime64('2008-01'), 'Y49', 11])
    #     variants_count, variants_per_subject = traces_hes.get_variants(self.df)
    #     pd.testing.assert_frame_equal(\
    #           variants_count, pd.DataFrame.from_dict({'A00,B49,C99':[9,2,9*2],
    #                                                   'A00,C99,B49':[1,2,1*2],
    #                                                   'X49,Y49,Z49':[1,2,1*2],
    #                                                   'Y49,Y49,Z49':[1,2,1*2]},
    #                                                  orient='index',
    #                                                  columns=['count', 'num_events', 'num_events_subs']),
    #           check_like=True)




if __name__ == '__main__':
    # Show full dataframes
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.expand_frame_repr', False)
    unittest.main(argv=['-v'],verbosity=3)

