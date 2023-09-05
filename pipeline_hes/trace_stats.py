# -*- coding: utf-8 -*-
"""
Calculates statistics on disease trajectories. In particular, runs Cox
regression by calling R functions. Also calculates relative risks, errors, and
restricted mean survival time.

@author: Chris Hayward
"""

import numpy as np
import pandas as pd
import pdb
from scipy import stats
    
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

from pipeline_hes.params import params
from pipeline_hes import plot_utils

# IS_PATIENT=True for MI cohort
# IS_PATIENT=False for Control cohort

HAZARD_COLUMN = 'IS_PATIENT'

def relative_risk(eP,nP,eC,nC,alpha,numTraces):
    """Calculate relative risk and the standard error / CI."""
    ratio_p = eP / nP
    ratio_c = eC / nC
    rr = ratio_p / ratio_c

    # Sampling distribution of the log(rr)
    se_logrr = np.sqrt((nC-eC)/(nC*eC) + (nP-eP)/(nP*eP))
    
    # Confidence interval
    nrm_z_score = stats.norm.ppf(1-(alpha/2))
    rr_ci = np.exp([np.log(rr)-se_logrr*nrm_z_score,
                    np.log(rr)+se_logrr*nrm_z_score])
    
    # Confidence interval (bonferroni corrected)
    nrm_z_score_strict = stats.norm.ppf(1-((alpha/numTraces)/2))
    rr_ci_corrected = np.exp([np.log(rr)-se_logrr*nrm_z_score_strict,
                              np.log(rr)+se_logrr*nrm_z_score_strict])
    
    return rr, rr_ci, rr_ci_corrected


def calc_rr(variants_controls, variants_patients, common_traces,
            nC, nP, ctl_name, pat_name):
    """Append relative risks to the trajectory dataframe."""

    # Adding 'pat' and 'ctl' suffixes allows me to RELIABLY figure out which
    # Columns to display when plotting later on
    cols = ['TOTALpat_{}'.format(pat_name), '#{}'.format(pat_name),'%{}'.format(pat_name), 
            'TOTALctl_{}'.format(ctl_name), '#{}'.format(ctl_name), '%{}'.format(ctl_name),
            'RR','RR_CIl','RR_CIu','RR_CIl_corrected','RR_CIu_corrected']
    # Relative risk for traces in patients
    df_rr = pd.DataFrame(columns=cols).astype(float)
    for i in range(common_traces.shape[0]):
        
        v = common_traces.iloc[i].name

        if v in variants_patients.index:
            eP = variants_patients.loc[v]['count']
            df_rr.loc[v,['TOTALpat_{}'.format(pat_name),
                         '#{}'.format(pat_name),
                         '%{}'.format(pat_name)]] = \
                [nP, int(eP), 100*eP/nP]
        
        if v in variants_controls.index:
            eC = variants_controls.loc[v]['count']
            df_rr.loc[v,['TOTALctl_{}'.format(ctl_name),
                         '#{}'.format(ctl_name),
                         '%{}'.format(ctl_name)]] = \
                [nC, int(eC), 100*eC/nC]
            
        # if this sequence post AMI is also a control sequence, compute RR
        if v in variants_patients.index and v in variants_controls.index:
            rr,rr_ci,rr_ci_corrected = relative_risk(eP,nP,eC,nC,0.05,common_traces.shape[0])
            df_rr.loc[v,['RR','RR_CIl','RR_CIu',
                         'RR_CIl_corrected','RR_CIu_corrected']] = [rr, rr_ci[0], rr_ci[1],
                                                                    rr_ci_corrected[0],
                                                                    rr_ci_corrected[1]]
    print(df_rr)
    return df_rr

    
def callR_restricted_mean_surv_time(r_from_pd_df):
    """For all subjects following a single trajectory, calculate the RMST."""

    robjects.r('suppressPackageStartupMessages(library(survRM2))')
    robjects.globalenv['x'] = r_from_pd_df

    # tau (truncation time point) defaults to min(max(DUR_pat,DUR_ctl))
    formula = """res<-rmst2(
        time=x$DUR,
        status=x$Mortality,
        arm=x${},)""".format(HAZARD_COLUMN)
    try:
        robjects.r(formula)
    except Exception as ex:
        print(ex)
        return None

    robjects.r('print(res)')
    r_tmp = robjects.globalenv['res']
    r_dict = dict(zip(r_tmp.names, r_tmp))
    
    # Sanity checks
    if not (np.array(list(r_dict.keys())) == \
            np.array(['tau', 'note', 'RMST.arm1', 'RMST.arm0', 'unadjusted.result'])).all():
        raise Exception('Unexpected RMST names')

    exp_names = np.array(['result', 'rmst', 'rmtl', 'tau', 'rmst.var', 'fit'])
    if not (np.array(r_dict['RMST.arm0'].names) == exp_names).all():
        raise Exception('Unexpected RMST names')
    if not (np.array(r_dict['RMST.arm1'].names) == exp_names).all():
        raise Exception('Unexpected RMST names')        

    exp_names = np.array(['Est.', 'se', 'lower .95', 'upper .95'])
    if not (np.array(r_dict['RMST.arm0'][1].names) == exp_names).all():
        raise Exception('Unexpected RMST names')
    if not (np.array(r_dict['RMST.arm1'][1].names) == exp_names).all():
        raise Exception('Unexpected RMST names')

    # matrix of values:
    if not (np.array(r_dict['unadjusted.result'].names[0]) == \
            np.array(['RMST (arm=1)-(arm=0)', 'RMST (arm=1)/(arm=0)','RMTL (arm=1)/(arm=0)'], dtype='<U20')).all():
        raise Exception('Unexpected RMST names')
    if not (np.array(r_dict['unadjusted.result'].names[1]) == \
            np.array(['Est.', 'lower .95', 'upper .95', 'p'], dtype='<U9')).all():
        raise Exception('Unexpected RMST names')

    # Restricted Mean Survival Time (RMST) by arm
    #           Est.    se lower .95 upper .95
    rmst0 = r_dict['RMST.arm0'][1][0] # hazard col = 0 (arm=0)
    rmst1 = r_dict['RMST.arm1'][1][0] # hazard col = 1 (arm=1)
    
    p_diff_unadj = np.array(r_dict['unadjusted.result'])[0][-1]# r_dict['unadjusted.result'][-3]

    return pd.Series(
        (rmst0,
         rmst1,
         r_dict['RMST.arm0'][1][2],
         r_dict['RMST.arm0'][1][3],
         r_dict['RMST.arm1'][1][2],
         r_dict['RMST.arm1'][1][3],
         r_dict['tau'][0],
         p_diff_unadj),
        ('RMST0','RMST1',
         'RMST0_CIl','RMST0_CIu',
         'RMST1_CIl','RMST1_CIu',
         'tau','p_diff_unadj'))


def callR_flexsurvspline_grid(hes_part_DEBUG,r_from_pd_df,covariates):
    """Call the flexible parametric survival function for a single
    trajectory. Do this over several knot and scale choices, and select the
    HR which minimizes BIC over these choices."""
    n_knots_choices = [0,1,2,3,4]
    scale_choices = ['odds', 'hazard', 'normal']
    
    fit_dist_choices = np.empty((len(n_knots_choices),len(scale_choices),5))
    fit_dist_choices[:] = np.nan

    robjects.globalenv['x_flexsurv_tmp'] = r_from_pd_df
    
    for i,knot_choice in enumerate(n_knots_choices):
        for j,scale_choice in enumerate(scale_choices):

            print("""----------------------------
                  flexsurvspline: knots={}, scale={}""".\
                  format(knot_choice,scale_choice))

            # Run Spline model
            try:
                robjects.r("""fss<-flexsurvspline(
                    formula = Surv(DUR, Mortality) ~ {},
                    data=x_flexsurv_tmp,
                    k={},
                    cl={},
                    scale="{}")""".\
                    format(' + '.join(covariates), knot_choice, 1-0.05, scale_choice))
            except Exception as ex:
                print(ex)
                continue

            robjects.r('print(fss)')
            r_tmp_fss = robjects.globalenv['fss']
            fss_dict = dict(zip(r_tmp_fss.names, r_tmp_fss))
            
            
            # sanity check
            if len(fss_dict.keys()) != 32:
                raise Exception('Unexpected flexsurvspline length')
            ngamma = knot_choice+2
            ncovars = 5 if not (params.ONLY_ONE_SEX in ['M','F']) else 4
            if len(np.array(fss_dict['res'])) != ngamma+ncovars:
                raise Exception('Unexpected flexsurvspline length')


            # Save the key values
            fit_dist_choices[i,j,0] = fss_dict['AIC'][0]
            fit_dist_choices[i,j,1] = robjects.r('BIC(fss)')[0]

            # get the IS_PATIENT coefficient (apply exp too)
            # knot_choice+2: skips the first N gamma values, and takes the first coeff (for IS_PATIENT)
            fit_dist_choices[i,j,2] = np.exp(np.array(fss_dict['res']))[knot_choice+2,0]
            # Lower and upper CI
            fit_dist_choices[i,j,3] = np.exp(np.array(fss_dict['res']))[knot_choice+2,1]
            fit_dist_choices[i,j,4] = np.exp(np.array(fss_dict['res']))[knot_choice+2,2]

    print(fit_dist_choices)
            

    # get exp val (and lower/upper CI) for min BIC
    
    BIC_vals = fit_dist_choices[:,:,1]
    hr_vals = fit_dist_choices[:,:,2]
    ciL_vals = fit_dist_choices[:,:,3]
    ciU_vals = fit_dist_choices[:,:,4]
    
    # do not select BIC values which are associated with nan hr, or CI's
    BIC_vals[np.isnan(hr_vals)] = np.nan
    BIC_vals[np.isnan(ciL_vals)] = np.nan
    BIC_vals[np.isnan(ciU_vals)] = np.nan
    
    minBIC_ij = np.unravel_index(np.nanargmin(BIC_vals), BIC_vals.shape)
    
    hr_ci = fit_dist_choices[minBIC_ij[0],minBIC_ij[1],[2,3,4]]
    return pd.Series(
        ([hr_ci[0],
          hr_ci[1],
          hr_ci[2],
          hr_ci[1],
          hr_ci[2],
          np.nan]),
        ('HR', 'HR_CIl', 'HR_CIu',
         'HR_CIl_corrected', 'HR_CIu_corrected',
         'HR_old',))


def callR_check_pha(hes_part,covariates):
    """Check the 'proportional hazards' assumption for a trajectory. Used
    to determine whether to run a flexible parametric model instead of the Cox
    model."""
    # Check the proportional hazards assumption
    # zph will fail if not enough data is present
    
    try:
        robjects.r('pha<-cox.zph(res)')
    except Exception:
        print('Failed to execute pha<-cox.zph(res)')
        return None
    
    robjects.r('print(pha)')

    r_tmp_zph = robjects.globalenv['pha']
    r_dict_zph = dict(zip(r_tmp_zph.names, r_tmp_zph))
    
    # Return if obeys PHA
    p_val_pha = np.array(r_dict_zph['table'])[0,-1]
    if p_val_pha>(0.05):
        return None

    print('\n!!!\nProportional Hazards Assumption does NOT hold.\n!!!\n')

    # Fix short durations
    # Causes problems with the flexsurv...    
    with localconverter(robjects.default_converter + pandas2ri.converter):
        tmp_dat = hes_part.copy()

        # move them all slightly 
        tmp_dat['DUR'] = tmp_dat['DUR'] + np.linspace(1e-5,1e-4,tmp_dat.shape[0])
        
        r_from_pd_df = robjects.conversion.py2rpy(tmp_dat.copy())
    
    # FLEXSURV Loop
    flexsurv_hr_ci_min_BIC = callR_flexsurvspline_grid(hes_part,r_from_pd_df,
                                                       covariates)

    return flexsurv_hr_ci_min_BIC


def callR_cox(r_from_pd_df,covariates):
    """Run a Cox regression model for a single trajectory (trace)."""
    
    robjects.r('suppressPackageStartupMessages(library(coxme))')
    robjects.r('suppressPackageStartupMessages(library(flexsurv))')
    robjects.r('suppressPackageStartupMessages(library(muhaz))')

    robjects.globalenv['x'] = r_from_pd_df


    # build and run formula
    formula = 'res<-coxme(Surv(DUR, Mortality) ~ {} + (1 | PROCODE), data=x)'.\
        format(' + '.join(covariates))
    print('Hazard, running formula: {}'.format(formula))

    try:
        robjects.r(formula)
    except Exception:
        print('Failed to execute R-Formula: {}'.format(formula))
        return None
        
    robjects.r('summary(res)')
    
    r_tmp = robjects.globalenv['res']
    r_dict = dict(zip(r_tmp.names, r_tmp))
    
    coeffs = r_dict['coefficients']
    # Get the standard error
    robjects.r('se<-sqrt(diag(vcov(res)))')
    coeffs_se = np.array(robjects.globalenv['se'])
    
    # convert SE to CI (95%)
    alpha=0.05
    coeffs_ci = coeffs_se * stats.norm.ppf(1-(alpha/2))

    # bonferroni correction
    coeffs_ci_corrected = [np.nan]# coeffs_se * stats.norm.ppf(1-(alpha_strict/2))

    return pd.Series(
        ([np.exp(coeffs[0]),
          np.exp(coeffs[0] - coeffs_ci[0]),
          np.exp(coeffs[0] + coeffs_ci[0]),
          np.exp(coeffs[0] - coeffs_ci_corrected[0]),
          np.exp(coeffs[0] + coeffs_ci_corrected[0]),
          np.nan]),
        ('HR', 'HR_CIl', 'HR_CIu',
         'HR_CIl_corrected', 'HR_CIu_corrected',
         'HR_old',))

    
def callR_hazard_ratio(hes_part_DEBUG,
                       r_from_pd_df,
                       covariates):
    """Wrapper function, running a Cox model, and if the PHA fails, instead
    runs a flexible parametric model instead to obtain the hazard ratio for
    MI versus matched controls."""

    # Sanity:
    # ! hazard column MUST be first item
    # .. this means that the output values related to the hazard column will be in position ZERO
    covariates = np.append([HAZARD_COLUMN], covariates[covariates!=HAZARD_COLUMN])

    # change to factors
    covariates[covariates==HAZARD_COLUMN] = 'as.factor({})'.format(HAZARD_COLUMN)
    covariates[covariates=='SEX'] = 'as.factor(SEX)'

    # COX model
    hr_dat = callR_cox(r_from_pd_df,covariates)

    # Return empty if fail
    if hr_dat is None:
        return pd.Series(np.repeat(np.nan,6),
                         ('HR', 'HR_CIl', 'HR_CIu',
                          'HR_CIl_corrected', 'HR_CIu_corrected',
                          'HR_old',))
    
    # Check PHA - replace with flexsurv HR value if PHA does not hold
    if params.CHECK_PROP_HAZ_ASSUM:
        flexsurv_hr_cl = callR_check_pha(hes_part_DEBUG,covariates)
        
        # replace Cox HR with FLEXSURV HR if PHA did not hold
        if not (flexsurv_hr_cl is None):
            hr_old = hr_dat['HR'].copy() # for debugging really
            hr_dat = flexsurv_hr_cl
            hr_dat['HR_old'] = hr_old
        
    return hr_dat



def select_trace_rows(variants_controls_per_subject,
                      variants_patients_per_subject,
                      thisVariant):
    """From the list of all trajectories (aka, variants) and their stats,
    get the subset of rows for a particular trajectory from both cohorts."""
    
    # Get all controls for this variant
    ctl_part = variants_controls_per_subject.loc[
        variants_controls_per_subject['variant'] == thisVariant]

    # Get all patients for this variant
    pat_part = variants_patients_per_subject.loc[
        variants_patients_per_subject['variant'] == thisVariant]
    
    # if a sequence doesnt exist, skip
    if ctl_part.shape[0]==0 or pat_part.shape[0]==0:
        return None

    # Concat 
    hes_part = pd.concat([ctl_part,pat_part]).reset_index(drop=True)
    
    return hes_part


def init_covariates(variants_controls_per_subject,
                     variants_patients_per_subject):
    """The covariates to be used in the models used to calculate the hazard
    ratio."""
    
    covariates_ORIG = np.array([HAZARD_COLUMN,
                                'SEX',
                                'INIT_AGE',
                                'IMD04',
                                'MATCHED_DATE'], dtype=object)
    
    # Remove SEX if only looking at males OR females
    if pd.concat([variants_controls_per_subject['SEX'],
                  variants_patients_per_subject['SEX']]).drop_duplicates().shape[0]==1:
        covariates_ORIG = covariates_ORIG[covariates_ORIG!='SEX']
    
    return covariates_ORIG


def enough_variants_have_resulted_in_death(v):
    """Check that a sufficient number of individuals have died, enabling a
    calculation of the hazard ratio (events per variable)."""
    
    num_explanatory_vars = 5 if not (params.ONLY_ONE_SEX in ['M','F']) else 4
    n_limit = 20
    # check that enough subjects have died
    if np.sum(v['Mortality']==1) < (num_explanatory_vars * n_limit):
        return False
    return True


# For each non-rare sequence for patients, calc HR
def hazard_ratio_per_trace(variants_controls_per_subject,
                           variants_patients_per_subject,
                           common_traces):
    """Calculations the hazard ratio and RMST for each trajectory."""
    
    covariates_ORIG = init_covariates(variants_controls_per_subject,
                                        variants_patients_per_subject)

    # Trace crossed with [hr,se_lower,se_upper]
    df_cox = pd.DataFrame(
        index=pd.MultiIndex.from_product([common_traces.index,
                                          ('HR',
                                           'HR_CIl', 'HR_CIu',
                                           'HR_CIl_corrected', 'HR_CIu_corrected',
                                           'HR_old',
                                           'RMST0',
                                           'RMST1',
                                           'RMST0_CIl',
                                           'RMST0_CIu',
                                           'RMST1_CIl',
                                           'RMST1_CIu',
                                           'tau',
                                           'p_diff_unadj',)],
                                         names=('variant', 'stat')),
        dtype=float,
        columns=[HAZARD_COLUMN])
    
    # to save R plots
    plot_utils.create_fig_folder()
    
    n_traces = common_traces.shape[0]
    
    for i in range(n_traces):        
        thisVariant = common_traces.iloc[i].name
        print('{}\nHazard for: {}'.format('-'*50,thisVariant))
    
        # Get the rows for this trace
        variants_part = select_trace_rows(variants_controls_per_subject,
                                     variants_patients_per_subject,
                                     thisVariant)
        
        if variants_part is None:
            continue
        
        n_subs_for_trace = variants_part.shape[0]
        
        print('number of subjects: {}'.format(n_subs_for_trace))

        # Handle the coefficients
        covariates = covariates_ORIG.copy()
    
        # If only females OR males are present, then drop SEX from fomula
        # (some N codes are sex specific - e.g. male genitals related N40-N53)
        if variants_part['SEX'].drop_duplicates().shape[0] == 1:
            covariates = covariates[covariates!='SEX']
        
        # Select columns
        variants_part = variants_part[np.append(['DUR','Mortality','PROCODE'],covariates)]
        
        # Convert init date to year only
        variants_part['MATCHED_DATE'] = variants_part['MATCHED_DATE'].dt.year - 2000
        
        # Ensure that there are enough FACTOR choices
        # - Prop-hazard assumption check will fail if not enough events
        if not enough_variants_have_resulted_in_death(variants_part):
            print('!!! Insufficient data size. Skipping (fill with NaN).')
            continue

        # convert to R dataframe
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_from_pd_df = robjects.conversion.py2rpy(variants_part)

        # Run COX-Mixed-effects model (with PROCODE as the random effect)
        # also pass in the variants_part df (just for DEUBUGGING only)
        series_hr = callR_hazard_ratio(variants_part,
                                       r_from_pd_df, covariates)
        
        df_cox.loc[pd.MultiIndex.from_product(
            [(thisVariant,),series_hr.index]),HAZARD_COLUMN] = series_hr.values

        # Get RMST
        series_rmst = callR_restricted_mean_surv_time(r_from_pd_df)
        
        if series_rmst is not None:
            df_cox.loc[pd.MultiIndex.from_product(
                [(thisVariant,),series_rmst.index]),HAZARD_COLUMN] = series_rmst.values
        
        #%%
        robjects.r('warnings()')

    # Just get the PATIENT Hazard ratio (hazard for AMI)
    # - unstack does a pivot operation (moves the CI...etc multi-index level to the columns)
    # - droplevel(axis=1) removes 'IS_PATIENT' level from MI in columns
    hr = (df_cox.loc[:,[HAZARD_COLUMN,]]).unstack(level=1).droplevel(0,axis=1)
    
    return hr

