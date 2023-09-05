# -*- coding: utf-8 -*-
"""
@author: Chris Hayward
"""

# test trace stats

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pdb

from pipeline_hes.params import params
from pipeline_hes import trace_stats

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

import os
import subprocess


class Test_HR(unittest.TestCase):
    def setUp(self):
        self.covariates = ['as.factor(IS_PATIENT)','as.factor(SEX)','INIT_AGE','IMD04','MATCHED_DATE']
        
        
    def conv_to_r_obj(self,df):
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_from_pd_df = robjects.conversion.py2rpy(df)
        return r_from_pd_df
    
    
    def test_first(self):
        n = 1000
        df = pd.DataFrame.from_dict({'IS_PATIENT':np.tile([0,1], n),
                                     'MATCHED_DATE':np.tile(np.repeat(np.datetime64('2005-01-01'),2),n)})
        df['IS_PATIENT'] = df['IS_PATIENT'].astype(np.uint8)

        # ensures a starting point is found for the Cox model
        #offset = (pd.Series(df.index).map(lambda x: int(x / 4)) / df.shape[0])
        offset = np.arange(df.shape[0]) / df.shape[0]
        
        np.random.default_rng(0).shuffle(offset)        
        df['PROCODE'] = offset > .5
        df['PROCODE'] = df['PROCODE'].astype(np.uint8)
        np.random.default_rng(0).shuffle(offset)        
        df['Mortality'] = offset > .5        
        df['Mortality'] = df['Mortality'].astype(np.uint8)
        np.random.default_rng(0).shuffle(offset)        
        df['SEX'] = offset > .5        
        df['SEX'] = df['SEX'].astype(np.uint8) + 1
        np.random.default_rng(0).shuffle(offset)        
        df['DUR'] = offset*100
        np.random.default_rng(0).shuffle(offset)
        df['IMD04'] = offset*5
        np.random.default_rng(0).shuffle(offset)
        df['INIT_AGE'] = offset*20
        np.random.default_rng(0).shuffle(offset)
        df['MATCHED_DATE'] = (df['MATCHED_DATE'] + (pd.to_timedelta(offset*1400,'D')))
        df['MATCHED_DATE'] = df['MATCHED_DATE'].dt.year - 2000


        r_from_pd_df = self.conv_to_r_obj(df.copy())
        
        # bonferroni
        n_traces = 10        
        df_out = trace_stats.callR_cox(r_from_pd_df,
                                       self.covariates)
        
        tmp_csv_in = os.path.join(params.DIR_TMP,'tmp_HR_test.csv')
        tmp_csv_out = os.path.join(params.DIR_TMP,'tmp_HR_test_out.csv')
        tmp_r_script = os.path.join(params.DIR_TMP,'test_trace_stats_HR.R')
        
        # cross-check with pure R script
        df.to_csv(tmp_csv_in)
        with open(tmp_r_script,'w') as f:
            f.write(r"""
suppressPackageStartupMessages(library(coxme))
library(coxme)
x<-read.csv('{}')
res<-coxme(Surv(DUR, Mortality) ~ as.factor(IS_PATIENT) + as.factor(SEX) + INIT_AGE + IMD04 + MATCHED_DATE + (1|PROCODE), data=x)
write.csv(res$coefficients, '{}')""".format(tmp_csv_in,tmp_csv_out).replace('\\','\\\\'))
        
        # clear output
        with open(tmp_csv_out, 'w') as f:
            f.write('')
        retcode = subprocess.call(['Rscript', tmp_r_script])
        r_out = pd.read_csv(tmp_csv_out)
        r_IS_PATIENT_HR = np.exp(r_out['x']).iloc[0]
        
        # check raw R with rpy2 call
        self.assertEqual(np.round(df_out['HR'],7), np.round(r_IS_PATIENT_HR,7))
    
    
class Test_RMST(unittest.TestCase):
    
    def setUp(self):
        self.hazard_column = 'IS_PATIENT'
        
        
    def conv_to_r_obj(self,df):
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_from_pd_df = robjects.conversion.py2rpy(df)
        return r_from_pd_df
    
    
    def test_rmst_1(self):
        # half alive right at the end
        df = pd.DataFrame.from_dict({'Mortality':[0,0,1,1,0,0,1,1],
                                     'DUR':[1,1,1,1,1,1,1,1],
                                     'IS_PATIENT':[0,0,0,0,1,1,1,1]})    
        r_from_pd_df = self.conv_to_r_obj(df)
        df_out = trace_stats.callR_restricted_mean_surv_time(r_from_pd_df)
        df_exp = pd.Series({'RMST0':1.0,
                            'RMST1':1.0,
                            'tau':1.0,})        
        pd.testing.assert_series_equal(df_out[['RMST0','RMST1','tau']], df_exp)
        

    def test_rmst_reduce_dur_part(self):
        # half alive for the final 10% of DUR
        df = pd.DataFrame.from_dict({'Mortality':[0,0,1,1,0,0,1,1],
                                     'DUR':[1,1,.9,.9,1,1,1,1],
                                     'IS_PATIENT':[0,0,0,0,1,1,1,1]})    
        r_from_pd_df = self.conv_to_r_obj(df)
        
        df_out = trace_stats.callR_restricted_mean_surv_time(r_from_pd_df)
        df_exp = pd.Series({'RMST0':.95,
                            'RMST1':1,
                            'tau':1,})        
        pd.testing.assert_series_equal(df_out[['RMST0','RMST1','tau']], df_exp)


    def test_rmst_reduce_dur_almost_all_die(self):
        # one alive for the final 10%
        df = pd.DataFrame.from_dict({'Mortality':[1,1,1,0,0,0,1,1],
                                     'DUR':[.9,.9,.9,1,1,1,1,1],
                                     'IS_PATIENT':[0,0,0,0,1,1,1,1]})    
        r_from_pd_df = self.conv_to_r_obj(df)
        
        df_out = trace_stats.callR_restricted_mean_surv_time(r_from_pd_df)
        df_exp = pd.Series({'RMST0':.925,
                            'RMST1':1,
                            'tau':1,})        
        pd.testing.assert_series_equal(df_out[['RMST0','RMST1','tau']], df_exp)


    def test_rmst_reduce_dur_all_die(self):
        df = pd.DataFrame.from_dict({'Mortality':[1,1,1,1,0,0,1,1],
                                     'DUR':[.9,.9,.9,.9,1,1,1,1],
                                     'IS_PATIENT':[0,0,0,0,1,1,1,1]})    
        r_from_pd_df = self.conv_to_r_obj(df)
        
        df_out = trace_stats.callR_restricted_mean_surv_time(r_from_pd_df)
        df_exp = pd.Series({'RMST0':.9,
                            'RMST1':1,
                            'tau':1,})        
        pd.testing.assert_series_equal(df_out[['RMST0','RMST1','tau']], df_exp)


    def test_rmst_reduce_tau(self):
        # half die halfway through
        # reduce tau
        df = pd.DataFrame.from_dict({'Mortality':[0,0,1,1,0,0,1,1],
                                     'DUR':[.9,.9,.45,.45,.9,.9,.9,.9],
                                     'IS_PATIENT':[0,0,0,0,1,1,1,1]})    
        r_from_pd_df = self.conv_to_r_obj(df)
        
        df_out = trace_stats.callR_restricted_mean_surv_time(r_from_pd_df)
        df_exp = pd.Series({'RMST0':.9*.75,
                            'RMST1':.9,
                            'tau':.9,})        
        pd.testing.assert_series_equal(df_out[['RMST0','RMST1','tau']], df_exp)


    def test_rmst_reduce_tau2(self):
        # half die halfway through
        # reduce tau
        df = pd.DataFrame.from_dict({'Mortality':[0,0,1,1,0,0,1,1],
                                     'DUR':[.9,.9,.25,.25,.5,.5,.5,.25],
                                     'IS_PATIENT':[0,0,0,0,1,1,1,1]})    
        r_from_pd_df = self.conv_to_r_obj(df)
        
        df_out = trace_stats.callR_restricted_mean_surv_time(r_from_pd_df)
        df_exp = pd.Series({'RMST0':.5*.75,
                            'RMST1':.5*.875,
                            'tau':.5,})        
        pd.testing.assert_series_equal(df_out[['RMST0','RMST1','tau']], df_exp)



class Test_enough_variants_have_resulted_in_death(unittest.TestCase):
    

    def test_enough_variants_have_resulted_in_death_FALSE(self):
        
        variants_part = pd.DataFrame()
        variants_part['variant'] = np.repeat('AAA,BBB',1000)
        variants_part['Mortality'] = np.repeat(0,1000)
        # need at least 20*5 deaths
        variants_part.loc[:98,'Mortality'] = 1
        self.assertFalse(trace_stats.enough_variants_have_resulted_in_death(variants_part))

    def test_enough_variants_have_resulted_in_death_TRUE(self):
        
        variants_part = pd.DataFrame()
        variants_part['variant'] = np.repeat('AAA,BBB',1000)
        variants_part['Mortality'] = np.repeat(0,1000)
        # need at least 20*5 deaths
        variants_part.loc[:99,'Mortality'] = 1
        self.assertTrue(trace_stats.enough_variants_have_resulted_in_death(variants_part))



class Test_relative_risk(unittest.TestCase):
    

    def test_relative_risk_same(self):
        # AMI (events)
        eP = 100
        # AMI (all events)
        nP = 100
        # CONTROL (events)
        eC = 100
        # CONTROL (all events)
        nC = 100
        
        alpha = 0.05
        # For Bonferroni correction
        numTraces = 10
        
        (rr_out, rr_out_ci, rr_out_ci_corrected) = \
            trace_stats.relative_risk(eP,nP,eC,nC,alpha,numTraces)
        self.assertEqual(rr_out,1/1)
        self.assertEqual(rr_out_ci[0],1)
        self.assertEqual(rr_out_ci[1],1)
        self.assertEqual(rr_out_ci_corrected[0],1)
        self.assertEqual(rr_out_ci_corrected[1],1)
        
        
    def test_relative_risk_same_boost_ctl(self):
        # AMI (events)
        eP = 100
        # AMI (all events)
        nP = 100
        # CONTROL (events)
        eC = 200
        # CONTROL (all events)
        nC = 200
        
        alpha = 0.05
        # For Bonferroni correction
        numTraces = 10
        
        (rr_out, rr_out_ci, rr_out_ci_corrected) = \
            trace_stats.relative_risk(eP,nP,eC,nC,alpha,numTraces)
        self.assertEqual(rr_out,1/1)
        self.assertEqual(rr_out_ci[0],1)
        self.assertEqual(rr_out_ci[1],1)
        self.assertEqual(rr_out_ci_corrected[0],1)
        self.assertEqual(rr_out_ci_corrected[1],1)
        
        
    def test_relative_risk_all_controls(self):
        # AMI (events)
        eP = 50
        # AMI (all events)
        nP = 100
        # CONTROL (events)
        eC = 200
        # CONTROL (all events)
        nC = 200
        
        alpha = 0.05
        # For Bonferroni correction
        numTraces = 10
        
        (rr_out, rr_out_ci, rr_out_ci_corrected) = \
            trace_stats.relative_risk(eP,nP,eC,nC,alpha,numTraces)
        self.assertEqual(rr_out,.5/1)
        self.assertTrue(rr_out_ci[0]<rr_out)
        self.assertTrue(rr_out_ci[1]>rr_out)
        self.assertTrue(rr_out_ci_corrected[0]<rr_out_ci[0])
        self.assertTrue(rr_out_ci_corrected[1]>rr_out_ci[1])
        
    
    
    def test_relative_risk_some(self):
        # AMI (events)
        eP = 50
        # AMI (all events)
        nP = 100
        # CONTROL (events)
        eC = 75
        # CONTROL (all events)
        nC = 100
        
        alpha = 0.05
        # For Bonferroni correction
        numTraces = 10
        
        (rr_out, rr_out_ci, rr_out_ci_corrected) = \
            trace_stats.relative_risk(eP,nP,eC,nC,alpha,numTraces)
        self.assertEqual(rr_out,.5/.75)
        self.assertTrue(rr_out_ci[0]<rr_out)
        self.assertTrue(rr_out_ci[1]>rr_out)
        self.assertTrue(rr_out_ci_corrected[0]<rr_out_ci[0])
        self.assertTrue(rr_out_ci_corrected[1]>rr_out_ci[1])
        
    
    
    # externally verified RR and CI
    def test_relative_risk_exact_CI(self):
        # AMI (events)
        eP = 100
        # AMI (all events)
        nP = 140
        # CONTROL (events)
        eC = 20
        # CONTROL (all events)
        nC = 100
        
        alpha = 0.05
        # For Bonferroni correction
        numTraces = 10
        
        (rr_out, rr_out_ci, rr_out_ci_corrected) = \
            trace_stats.relative_risk(eP,nP,eC,nC,alpha,numTraces)
        self.assertEqual(np.round(rr_out,2),3.57)
        self.assertTrue(np.round(rr_out_ci[0],2)==2.38)
        self.assertTrue(np.round(rr_out_ci[1],2)==5.36)
        
    
    
    

if __name__ == '__main__':
    # Show full dataframes
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.expand_frame_repr', False)
    unittest.main(argv=['-v'],verbosity=3)

