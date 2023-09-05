# -*- coding: utf-8 -*-
"""
After cleaning the HES data: 
    - calls functions to produce table-one details;
    - selects episodes for inclusion in trajectories;
    - calls functions to compute statistics on these trajectories;
    - saves the trajectories and statistics into files which can be read by
    'last_step.py' (which constructs tables and figures using these files)

@author: Chris Hayward
"""

import pandas as pd
import pdb
import pprint
import pickle
import os

from pm4py.util import constants
from pm4py.statistics.traces.generic.pandas import case_statistics

from pipeline_hes.params import params
from pipeline_hes import trace_stats

from pipeline_hes import trace_plots


## Traces from HES

def get_subset_of_traces_for_figure(variants_controls_per_subject,
                                    variants_patients_per_subject):
    """Get the common trajectories, based on how often they appear in the 
    MI cohort."""
    
    n = variants_patients_per_subject.shape[0]
    
    # Rarity is always based on the ~IS_CONTROL group
    # The control/patient split might not be based on IS_CONTROL, so we
    # need to account for this and recompute the set of ~IS_CONTROL counts
    tmp_pats = pd.concat([
        variants_controls_per_subject.loc[
            ~variants_controls_per_subject['IS_CONTROL']],
        variants_patients_per_subject.loc[
            ~variants_patients_per_subject['IS_CONTROL']]])
    tmp_pats_vc = tmp_pats['variant'].value_counts()
    tmp_pats_vc = pd.DataFrame(tmp_pats_vc.values,
                               columns=['count'], index=tmp_pats_vc.index)
    
    # keep >=0.1%
    thresh_upper = n
    thresh_lower = n*params.AMI_TRACE_COUNTS_PRC_THRESHOLD/100 # 0.001    
    
    # # keep 0.1% - 0.05%
    # thresh_upper = n*0.1/100
    # thresh_lower = n*0.05/100
    
    # # keep 0.05% - 0.025%
    # thresh_upper = n*0.05/100
    # thresh_lower = n*0.025/100
    
    keep_traces = tmp_pats_vc.loc[
        pd.concat([thresh_upper > tmp_pats_vc['count'],
                   tmp_pats_vc['count'] >= thresh_lower],axis=1).all(axis=1)]

    
    #%%
    keep_traces = pd.DataFrame(keep_traces.values,
                                columns=['count'], index=keep_traces.index)

    # sort by index, then final sort by count (two traces could have the same count)
    return keep_traces.sort_index().sort_values('count',ascending=True)
    #%%
    

def sanity_check_trace_start_end(variants_per_subject):
    """Ensure that all trajectories start with the 'initial disease'
    placeholder, and end with the 'censor' placeholder."""
    non_censor_finish = variants_per_subject.loc[
        variants_per_subject['variant'].map(lambda x: x.split(',')[-1] != \
                                            params.CENSOR_CODE)]
    if non_censor_finish.shape[0]>0:
        print(non_censor_finish)
        raise Exception('Some traces do not terminate with CENSOR.')
    
    non_initial_start = variants_per_subject.loc[
        variants_per_subject['variant'].map(lambda x: x.split(',')[0] != \
                                            params.AMI_INIT)]
    if non_initial_start.shape[0]>0:
        print(non_initial_start)
        raise Exception('Some traces do not start with MI/initial.')


def get_variants_for_diag_col(df,diag_col):
    """Calls PM4PY package to discover trajectories, using case (subject ID),
    activity (disease), and timestamp (episode start date)."""
    parameters={constants.PARAMETER_CONSTANT_CASEID_KEY: 'case',
                constants.PARAMETER_CONSTANT_ACTIVITY_KEY: 'activity',
                constants.PARAMETER_CONSTANT_TIMESTAMP_KEY: 'timestamp'}

    df_alt = df.loc[~df['IGNORE'], ['case','timestamp',diag_col]].\
        rename(columns={diag_col:'activity'})    

    variants_per_subject = case_statistics.get_variants_df(
        df_alt[['case','activity','timestamp']],parameters=parameters)
    sanity_check_trace_start_end(variants_per_subject)
    
    # df_alt = df.loc[~df['IGNORE'], ['case','timestamp',diag_col]].\
    #     rename(columns={diag_col:'activity'}).sort_values(['timestamp'])
    # my_variants_per_subject = df_alt.groupby('case').\
    #     apply(lambda x: (','.join(x['activity'])))
    # my_variants_per_subject = pd.DataFrame(my_variants_per_subject, columns=['variant'])
    sanity_check_trace_start_end(variants_per_subject)
    
    return variants_per_subject


def get_variants_count(variants_per_subject):
    """Count the number of instances of each trajectory."""
    vc = variants_per_subject['variant'].value_counts()
    variants_count = pd.DataFrame(vc.values,columns=['count'],index=vc.index)

    # Minus two (AMI/initial and CENSOR are not event)
    variants_count['num_events'] = \
        variants_count.index.map(lambda x: len(x.split(','))) - 2
    variants_count['num_events_subs'] = \
        variants_count['num_events'] * variants_count['count']
    return variants_count


def get_variants_count_averages(variants_per_subject, variant_column):
    """Count the number of instances of each trajectory. Also compute mean
    age, sex, and deprivation."""
    vc = variants_per_subject[variant_column].value_counts()
    variants_count = pd.DataFrame({'count':vc.values, variant_column:vc.index})
    
    v_extended = variants_per_subject[[variant_column, 'INIT_AGE', 'SEX', 'IMD04']].copy()
    v_extended['FRACTION_FEMALE'] = v_extended['SEX']==2
    v_extended = v_extended.drop(columns=['SEX'])
    v_extended = v_extended.groupby(variant_column).mean().reset_index()
    v_extended = v_extended.merge(variants_count).sort_values('count', ascending=False)
    return v_extended

def get_variants(df):
    """Return trajectories, one per-subject, and a list of unqiue trajectories
    with the number of occurrences of each."""

    df.sort_values(['ENCRYPTED_HESID','ORDER'], inplace=True)

    # Rename
    df.rename(columns={'ENCRYPTED_HESID':'case', 
                       'ORDER':'timestamp'}, inplace=True)

    # low/high level chapter
    variants_per_subject = get_variants_for_diag_col(df,'DIAG_01')
    # low/high level chapter
    variants_per_subject_ALT1 = get_variants_for_diag_col(df,'DIAG_01_ALT1')
    # orig diagnosis (3 chars)
    variants_per_subject_ALT2 = get_variants_for_diag_col(df,'DIAG_01_ALT2')
        
    variants_per_subject = variants_per_subject.merge(
        variants_per_subject_ALT1[['variant']], left_index=True, right_index=True, suffixes=('','_ALT1'))
    variants_per_subject = variants_per_subject.merge(
        variants_per_subject_ALT2[['variant']], left_index=True, right_index=True, suffixes=('','_ALT2'))
    variants_per_subject = variants_per_subject.reset_index()
            
    # Minus two (AMI/initial and CENSOR are not event)
    variants_per_subject['num_events'] = variants_per_subject['variant'].\
        map(lambda x: len(x.split(','))) - 2
    
    hazard_column = 'IS_PATIENT'
    
    df_coefficients = df[['case',
                          hazard_column,
                          'IS_CONTROL',
                          'SEX',
                          'INIT_AGE',
                          'IMD04',
                          'MATCHED_DATE',
                          'DUR',
                          'Mortality',
                          'PROCODE']].drop_duplicates(subset=['case'])
    
    # These are needed for survival modelling later on
    variants_per_subject = variants_per_subject.merge(
        df_coefficients,on='case',how='left')
    
    # Calculate events per month based on the follow-up period
    # (not based on the range of dates in the sequence, which omits events which have
    # not survived filtering)
    variants_per_subject['events_per_month'] = \
        variants_per_subject['num_events'] / variants_per_subject['DUR']

    variants_count = get_variants_count(variants_per_subject)

    return variants_count, variants_per_subject 
        
def get_data_split(hes_data):
    """Split the dataframe into two, one for each cohort."""
        
    controls = hes_data.loc[hes_data['IS_CONTROL']].copy()
    patients = hes_data.loc[~hes_data['IS_CONTROL']].copy()

    hazard_column = 'IS_PATIENT'

    # Patients must be associated with a larger value (1, versus 0)
    # This ensures that the hazard ratio makes sense
    controls[hazard_column] = 0
    patients[hazard_column] = 1
    
    
    # If looking at males or females only:
    if params.ONLY_ONE_SEX == 'M':
        controls = controls.loc[controls['SEX']==params.SEX_MALE].copy()
        patients = patients.loc[patients['SEX']==params.SEX_MALE].copy()
    elif params.ONLY_ONE_SEX == 'F':
        controls = controls.loc[controls['SEX']==params.SEX_FEMALE].copy()
        patients = patients.loc[patients['SEX']==params.SEX_FEMALE].copy()

    return controls, patients



def get_traces():
    """After cleaning, produce a list of disease trajectories for the two
    cohorts."""

    hes_data = pd.read_parquet(os.path.join(params.DIR_CHECKPOINTS,
                                            'CLEAN_FILTERED_{}_.gzip'.\
                                                format(params.R)))

    # Split, and save number of controls and AMI subjects
    # ! This split MUST happen after the final filtering steps, because
    # we want the trace frequences to match the default split (Control/AMI)
    controls, patients = get_data_split(hes_data)    
    hes_data = None

    # Generate traces
    variants_controls_count, variants_controls_per_subject = get_variants(controls)
    variants_patients_count, variants_patients_per_subject = get_variants(patients)

    return variants_controls_count, variants_controls_per_subject, \
            variants_patients_count, variants_patients_per_subject



def get_df_rr_hr(variants_controls_count, variants_controls_per_subject,
                 variants_patients_count, variants_patients_per_subject):
    """Get the trajectories which commonly appear in the MI cohort. Append
    to this their statistics - relative risk and hazard ratio."""

    #%% Remove rare traces from patients
    display_these_traces = \
        get_subset_of_traces_for_figure(variants_controls_per_subject,
                                        variants_patients_per_subject)

    # Relative risk calc
    ctl_name = 'Controls'
    pat_name = 'MI'
    nC=variants_controls_per_subject.shape[0]
    nP=variants_patients_per_subject.shape[0]
    df_rr = trace_stats.calc_rr(variants_controls_count,
                                variants_patients_count,
                                display_these_traces,
                                nC, nP, ctl_name, pat_name)

    # Cox model
    df_hr = trace_stats.hazard_ratio_per_trace(variants_controls_per_subject,
                                               variants_patients_per_subject,
                                               display_these_traces)

    # %% Combine relative risk and hazard
    df_rr_hr = pd.concat([df_rr,df_hr], axis=1)
    
    return df_rr_hr
    #%%
        
def main(doPlots=False):
    """Entry function - discovers disease trajectories from the cleaned data,
    gets the most common trajectories in the MI cohort, plots the main
    trajectory figure, and saves the trajectory data-structures for use in
    further tables and figures (last_step.py)."""

    (variants_controls_count, variants_controls_per_subject, \
      variants_patients_count, variants_patients_per_subject) = get_traces()

    #%%
    df_rr_hr = get_df_rr_hr(variants_controls_count, variants_controls_per_subject,
                            variants_patients_count, variants_patients_per_subject)

    if doPlots:
        
        #%% PLOT
        import importlib
        from pipeline_hes import traces_hes
        importlib.reload(traces_hes)
        trace_plots.plot_rr_fig(df_rr_hr.sort_values('RR',na_position='first').copy())
    
        #%% save a list of the params used for these plots
        with  open(os.path.join(params.DIR_RESULTS,'params.txt'),'w') as outfile:
            pprint.pprint(vars(params), stream=outfile)
        with open(os.path.join(params.DIR_RESULTS,'params_pickled'),'wb') as outfile:
            pickle.dump(params,outfile,protocol=0)

        
        #%% SAVE in new folder
        
        df_rr_hr.to_parquet(
            os.path.join(
                params.DIR_RESULTS,'df_rr_hr_{}_.gzip'.format(
                    params.R)),compression='gzip')

        variants_controls_per_subject.to_parquet(
            os.path.join(
                params.DIR_RESULTS,'variants_controls_per_subject_{}_.gzip'.format(
            params.R)),compression='gzip')

        variants_patients_per_subject.to_parquet(
            os.path.join(
                params.DIR_RESULTS,'variants_patients_per_subject_{}_.gzip'.format(
            params.R)),compression='gzip')
        
        variants_controls_count.to_parquet(
            os.path.join(
                params.DIR_RESULTS,'variants_controls_count_{}_.gzip'.format(
            params.R)),compression='gzip')
        variants_patients_count.to_parquet(
            os.path.join(
                params.DIR_RESULTS,'variants_patients_count_{}_.gzip'.format(
            params.R)),compression='gzip')
        
    


