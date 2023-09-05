# -*- coding: utf-8 -*-
"""
After selecting the MI and control cohorts, clean the HES data - removing bad
episodes/subjects, and preparing the data for trajectory construction.

@author: Chris Hayward
"""

import pandas as pd
import numpy as np
import pdb
import time
import os

from pipeline_hes import append_counts
from pipeline_hes.params import params
from pipeline_hes import mask_hes
from pipeline_hes import clean_hes_deathdate
from pipeline_hes import clean_hes_prepare
from pipeline_hes import plot_utils


# ALLFIELDS = {'hesField': 'ENCRYPTED_HESID', \
#              'dateField': 'MYADMIDATE',\
#              'sexField': 'SEX',\
#              'ageField': 'STARTAGE',\
#              'dobField': 'MYDOB', \
#              'providerField': 'PROCODE',\
#              'deprivationField': 'IMD04',\
#              'epiorderField': 'EPIORDER',\
#              'epistatField': 'EPISTAT',\
#              'admimethField': 'ADMIMETH',\
#              'priDiag' : 'DIAG_01'}

    
    
def _set_init_field(df,field):
    """Set the value for the field for the whole subject."""
    init_mask = mask_hes.init_mask(df)
    init_field = df.loc[init_mask, ['ENCRYPTED_HESID',field]]
    init_field = init_field.loc[~init_field[field].isna()].\
        drop_duplicates(subset='ENCRYPTED_HESID')
    df = df.merge(init_field, how='left', on='ENCRYPTED_HESID', suffixes=('', '_INIT'))
    df[field] = df[field+'_INIT']
    df.drop(columns=[field+'_INIT'], inplace=True)
    return df


def set_init_deprivation(df):
    """Assign a single deprivation value for each individual."""
    return _set_init_field(df,'IMD04')

def set_init_procode(df):
    """Assign a single provider code for each individual."""
    return _set_init_field(df,'PROCODE')


def remove_subjects_nan_spell_id(df):
    """Remove subjects with NaN spell ID <PROVSPNOPS>."""
    #pdb.set_trace()
    bad_mask = df['PROVSPNOPS'].isna()
    # get ids
    invalid_ids = df.loc[bad_mask, 'ENCRYPTED_HESID'].drop_duplicates()
    ok_ids = pd.Index(df['ENCRYPTED_HESID'].drop_duplicates()).difference(invalid_ids, sort=False)
    df = df.merge(pd.Series(ok_ids), how='right', on='ENCRYPTED_HESID')
    # without nans, set to numeric type (int)
    df['PROVSPNOPS'] = pd.to_numeric(df['PROVSPNOPS'],downcast='unsigned')
    return df



def fix_amiid_group_ratio_remove_amis(group):
    """For cases where too many MI subjects exist for a matching-ID, restrict 
    the number of MI and the number of controls."""
    group_ctl = group.loc[group['IS_CONTROL']]
    group_ami = group.loc[~group['IS_CONTROL']]
    keep_n_amis = int(np.floor(group_ctl['ENCRYPTED_HESID'].\
                               drop_duplicates().shape[0]/params.CONTROL_CASE_RATIO))
    keep_n_ctls = keep_n_amis*params.CONTROL_CASE_RATIO

    # keep N controls (where N = numAMI*5)
    group_ctl = group_ctl.merge(
        group_ctl['ENCRYPTED_HESID'].drop_duplicates().iloc[:keep_n_ctls])
    group_ami = group_ami.merge(
        group_ami['ENCRYPTED_HESID'].drop_duplicates().iloc[:keep_n_amis])

    return pd.concat([group_ami,group_ctl]).reset_index(drop=True)
    

def fix_amiid_group_ratio_remove_controls(group):
    """For cases where too many controls exist for a matching-ID, restrict 
    the number of controls only."""
    group_ctl = group.loc[group['IS_CONTROL']]
    group_ami = group.loc[~group['IS_CONTROL']]
    keep_n_ctls = group_ami['ENCRYPTED_HESID'].drop_duplicates().shape[0]*\
        params.CONTROL_CASE_RATIO

    # keep N controls (where N = numAMI*5)
    group_ctl = group_ctl.merge(
        group_ctl['ENCRYPTED_HESID'].drop_duplicates().iloc[:keep_n_ctls])

    return pd.concat([group_ami,group_ctl]).reset_index(drop=True)  


def remove_using_amiID(hes_data):
    """Remove MI and controls using matching id (preserving 5:1 ratio)."""
    g = hes_data[['ENCRYPTED_HESID','IS_CONTROL','amiID']].drop_duplicates('ENCRYPTED_HESID').\
        groupby(['amiID'])

    # count the number of MI and control subjects associated with each matching-id
    tups = g['IS_CONTROL'].agg(lambda x: (sum(x), sum(~x)))

    # only keep matching-ids where both MI and controls exist
    tups = tups.loc[tups.map(lambda x: x[0] > 0 and x[1] > 0)]

    tups_ok = tups.loc[tups.map(lambda x: float(x[0])/x[1] == params.CONTROL_CASE_RATIO)]
    # strictly greater, as ratios under 5 cannot be fixed (too few controls)
    tups_too_many_ctls = tups.loc[tups.map(lambda x: float(x[0])/x[1] > \
                                           params.CONTROL_CASE_RATIO)]
    tups_too_many_amis = tups.loc[tups.map(lambda x: float(x[0])/x[1] < \
                                           params.CONTROL_CASE_RATIO)]

    df_too_many_ctls_fixed = hes_data.merge(pd.Series(tups_too_many_ctls.index)).\
        groupby('amiID').apply(fix_amiid_group_ratio_remove_controls)
    df_too_many_amis_fixed = hes_data.merge(pd.Series(tups_too_many_amis.index)).\
        groupby('amiID').apply(fix_amiid_group_ratio_remove_amis)

    df_ok = hes_data.merge(pd.Series(tups_ok.index))
    print('Remove using matching id (preserve ratio). DONE')

    return pd.concat([df_ok,
                      df_too_many_ctls_fixed,
                      df_too_many_amis_fixed]).reset_index(drop=True)
    


def check_single_option_within_subjects_sub(df, column):
    """Ensure consistency of values within subjects.
    Remove subjects which do not satisfy this."""
    # nan's are automatically excluded here
    pairs_count = df[['ENCRYPTED_HESID',column]].value_counts().reset_index()
    pairs_count = pairs_count.rename(columns={0:'count'})
    num_rows_per_sub = df[['ENCRYPTED_HESID']].value_counts().reset_index()
    num_rows_per_sub = num_rows_per_sub.rename(columns={0:'total'})
    pairs_count_prc = pairs_count.merge(num_rows_per_sub,how='left')
    pairs_count_prc['prc'] = 100* pairs_count_prc['count'] / pairs_count_prc['total']

    # only keep subjects with prc above threshold
    pairs_count_prc = pairs_count_prc.loc[pairs_count_prc['prc']>=\
                                          params.WITHIN_SUBJECT_SINGLE_VALUE_THRESHOLD]

    # AND replace field with the majority    
    df = df.drop(columns=column)

    return df.merge(pairs_count_prc[['ENCRYPTED_HESID',column]].\
                    drop_duplicates('ENCRYPTED_HESID'),
                    how='right', on='ENCRYPTED_HESID')
        

def check_single_option_within_subjects(df):
    """Check that fields are consistent within a subject.
    Do this for Mortality, DOB and Sex."""
    df = check_single_option_within_subjects_sub(df,'Mortality')
    df = check_single_option_within_subjects_sub(df,'MYDOB')
    df = check_single_option_within_subjects_sub(df,'SEX')
    df = df.astype({'Mortality':np.uint8,'SEX':np.uint8})
    return df


def remove_unfinished_episodes(df):
    """Remove unfinished episodes (EPISTAT=3 or nan)."""
    ok_mask = df['EPISTAT']==3
    df.loc[~ok_mask,'EPISTAT'] = np.nan
    df.dropna(subset=['EPISTAT'],inplace=True)
    df.drop(columns=['EPISTAT'],inplace=True)


def remove_subjects_with_dup_within_spell_events(hes_data):
    """Within Hospital/SpellID pairs, remove subjects with duplicate EPIORDERs."""
    dup = hes_data.duplicated(['ENCRYPTED_HESID','PROVSPNOPS','EPIORDER'])
    dup_ids = hes_data.loc[dup, 'ENCRYPTED_HESID'].drop_duplicates()
    good_ids = pd.Index(hes_data['ENCRYPTED_HESID'].drop_duplicates()).\
        difference(dup_ids, sort=False)
    return hes_data.merge(pd.Series(good_ids), on='ENCRYPTED_HESID', how='right')


def check_ami_rows_on_matched_date(df):
    """Check that all AMI subjects have matching EPISODE dates on matching
    date"""
    df_ami = df.loc[~df['IS_CONTROL'],
                    ['ENCRYPTED_HESID','AMI','MYEPISTART','MYADMIDATE',
                     'EPIORDER','MATCHED_DATE']]

    # get the earliest AMI
    epistart_first_ami = df_ami.loc[df_ami['AMI']].\
        sort_values(['MYEPISTART','EPIORDER']).drop_duplicates('ENCRYPTED_HESID')

    # ensure that episode dates do not overshoot admission date
    if (epistart_first_ami['MYEPISTART']!=epistart_first_ami['MATCHED_DATE']).any():
        raise Exception("""Not all MI subjects have matching MATCHED_DATE and 
EPISODE dates on initial AMI. This is a matching criteria.""")


def remove_subjects_with_no_event_on_matched_date(df):
    """Remove subjects which do not have an episode-start date on their
    matched date."""
    # matching MUST coinside with the EPISTART date;
    # matching on the admission date is not correct - spells can be months long
    df['INIT_ROW'] = False
    
    # Mark the subjects with episodes on their matched date
    df.loc[df['MYEPISTART']==df['MATCHED_DATE'],'INIT_ROW'] = True
    
    # Remove any subjects which do not have INIT_ROW set
    return df.merge(df.loc[df['INIT_ROW'], ['ENCRYPTED_HESID']].drop_duplicates(),
                    how='right', on='ENCRYPTED_HESID').reset_index(drop=True)


def remove_ami_subjects_where_first_ami_date_is_not_matched_date(df):
    """Ensure that MI subjects have an MI diagnosis on their matched date."""
    # Keep all controls
    # Keep non controls where first AMI occurred on MATCHED_DATE
    ok_mask = pd.concat([df['MATCHED_DATE']==df['MYEPISTART_FIRSTAMI'],
                         df['IS_CONTROL']], axis=1).any(axis=1)
    df.loc[~ok_mask,'MATCHED_DATE'] = np.nan
    df.dropna(subset=['MATCHED_DATE'],inplace=True)


def remove_controls_with_ami_matched_date_not_before_first_ami(df):
    """For controls with MI, remove those with a matched date not before their MI."""
    ok_mask = pd.concat([df['MYEPISTART_FIRSTAMI'].isna(),
                         df['MATCHED_DATE']<df['MYEPISTART_FIRSTAMI'],
                         ~df['IS_CONTROL']],axis=1).any(axis=1)
    df.loc[~ok_mask,'MATCHED_DATE'] = np.nan
    df.dropna(subset=['MATCHED_DATE'],inplace=True)


def preliminary_sanity_cleaning(df):
    """The cleaning steps which check the quality of the matching process."""


    count_subs0 = df['ENCRYPTED_HESID'].drop_duplicates().shape[0]
    count_epis0 = df.shape[0]
    print('subs:{}, epis:{}'.format(count_subs0, count_epis0))
    
    remove_unfinished_episodes(df)

    check_ami_rows_on_matched_date(df)

    count_subs0 = df['ENCRYPTED_HESID'].drop_duplicates().shape[0]
    count_epis0 = df.shape[0]
    print('subs:{}, epis:{}'.format(count_subs0, count_epis0))

    # #####
   # pdb.set_trace()
    df = remove_subjects_with_no_event_on_matched_date(df)
    # #####
    count_subs1 = df['ENCRYPTED_HESID'].drop_duplicates().shape[0]
    count_epis1 = df.shape[0]
    print('subs: {} ({}); epis: {} ({}) #episodes on matched'.\
          format(count_subs1, count_subs0-count_subs1,
                 count_epis1, count_epis0-count_epis1))
        
    #pdb.set_trace()

    # ######
    # Remove AMI subjects which do not have a MATCHED_DATE matching the first AMI date
    # ######
    remove_ami_subjects_where_first_ami_date_is_not_matched_date(df)
    count_subs2 = df['ENCRYPTED_HESID'].drop_duplicates().shape[0]
    count_epis2 = df.shape[0]
    print('subs: {} ({}); epis: {} ({}) #first ami is not on matched'.\
          format(count_subs2, count_subs1-count_subs2,
                 count_epis2, count_epis1-count_epis2))

    # ######
    # Removing controls with AMI where the matched date is not before their first AMI
    # ######
    remove_controls_with_ami_matched_date_not_before_first_ami(df)
    count_subs3 = df['ENCRYPTED_HESID'].drop_duplicates().shape[0]
    count_epis3 = df.shape[0]
    print('subs: {} ({}); epis: {} ({}) #controls with matched not before first ami'.\
          format(count_subs3, count_subs2-count_subs3,
                 count_epis3, count_epis2-count_epis3))

    # #########
    # NUKE all amiIDs for subjects that has been removed 
    # #########
    df = remove_using_amiID(df)
    count_subs4 = df['ENCRYPTED_HESID'].drop_duplicates().shape[0]
    count_epis4 = df.shape[0]
    print('subs: {} ({}); epis: {} ({}) #nuke'.format(count_subs4, count_subs3-count_subs4,
                                    count_epis4, count_epis3-count_epis4))
    return df


def print_counts(df):
    """Print the number of episodes and individuals."""
    tmp_all_subs = df[['ENCRYPTED_HESID','IS_CONTROL']].drop_duplicates()
    print('All episodes: Controls\n Subjects:{:,}\n Episodes:{:,}'.\
          format(tmp_all_subs.loc[tmp_all_subs['IS_CONTROL']].shape[0],
                 df.loc[df['IS_CONTROL']].shape[0]))
    print('All episodes: AMI\n Subjects:{:,}\n Episodes:{:,}'.\
          format(tmp_all_subs.loc[~tmp_all_subs['IS_CONTROL']].shape[0],
                 df.loc[~df['IS_CONTROL']].shape[0]))
    print('Ratio: {}'.format(tmp_all_subs.loc[tmp_all_subs['IS_CONTROL']].shape[0]/\
                             tmp_all_subs.loc[~tmp_all_subs['IS_CONTROL']].shape[0]))    


def main():
    """The entry function - cleans the HES data ready for 'filtering' later on
    where episodes are selected for inclusion in trajectories."""
    t = time.time()

    # # --------------
    # # load data
    # # --------------
    hes_data = pd.read_parquet(
        os.path.join(params.DIR_CHECKPOINTS,
                     'MATCHED_BEFORE_CLEANING_{}.gzip'.format(params.R)))

    #%%
    pdb.set_trace()
    #%%


    
    tmp = hes_data[['ENCRYPTED_HESID','IS_CONTROL','amiID',
                    'MATCHED_DATE','MYDOB','PROCODE','SEX']]\
        .drop_duplicates('ENCRYPTED_HESID').copy()
    
    tmp['INIT_AGE'] = np.ceil((tmp['MATCHED_DATE'] - tmp['MYDOB']).dt.days / 365.25)
    
    tmp_MI = tmp.loc[~tmp['IS_CONTROL']].copy()
    tmp_CTL = tmp.loc[tmp['IS_CONTROL']].copy()
    
    
    tmp_MI_main_fields = \
        tmp_MI[['amiID','MATCHED_DATE','PROCODE','SEX','INIT_AGE']].copy()
    
    #####################
    
    singular_amiID_MI = tmp_MI['amiID'].value_counts().loc[
        tmp_MI['amiID'].value_counts()==1].reset_index()
        
    tmp_MI_singular = tmp_MI.merge(singular_amiID_MI, left_on='amiID',
                                    right_on='index')
    
    tmp_merge = tmp_MI_singular[['INIT_AGE','amiID_x']].merge(
        tmp_CTL[['amiID', 'INIT_AGE']],
                                      left_on='amiID_x',
                                      right_on='amiID',
                                      how='left')
    
    tmp_merge['AGE_DIFF'] = np.abs(tmp_merge['INIT_AGE_x']-tmp_merge['INIT_AGE_y'])
    
    # singular bad example
    singular_bad_mi = tmp_merge.loc[(tmp_merge['AGE_DIFF'].round()==28)].copy()
    
    
    bad_mi = tmp_merge.loc[(tmp_merge['AGE_DIFF'].round()==10)].copy()
    

    #%%
    ### BRUTE FORCE - check age matching
    
    all_amiID = tmp_MI['amiID'].drop_duplicates()
    
    
    all_sing = 0
    all_plur = 0
    over_1y_age_singular = 0
    over_1y_age_plural_lower = 0
    over_1y_age_plural_upper = 0
    
    for i,iid in enumerate(all_amiID):
        if i % 100 == 0:
            print('{}/{}, s:{}/{}, p:{}:{}/{}'.format(i,all_amiID.shape[0],
                                             over_1y_age_singular,
                                             all_sing,
                                             over_1y_age_plural_upper,
                                             over_1y_age_plural_lower,
                                             all_plur))
        
        tmp_MI_part = tmp_MI.loc[tmp_MI['amiID'] == iid]
        tmp_CTL_part = tmp_CTL.loc[tmp_CTL['amiID'] == iid]
    
        if tmp_MI_part.shape[0] > 1:
            # multiple MI, need to find the minimum difference with controls
            for j in range(tmp_MI_part.shape[0]):
                diffs = tmp_MI_part.iloc[j]['INIT_AGE'] - \
                    tmp_CTL_part['INIT_AGE'].values
                if np.min(diffs) >= (1 + (1/11)):
                    over_1y_age_plural_lower = over_1y_age_plural_lower + 1
                if np.max(diffs) >= (1 + (1/11)):
                    over_1y_age_plural_upper = over_1y_age_plural_upper + 1
                all_plur = all_plur + 1
        else:
            diffs = tmp_MI_part['INIT_AGE'].values - tmp_CTL_part['INIT_AGE'].values
            if np.max(diffs) >= (1 + (1/11)):
                over_1y_age_singular = over_1y_age_singular + 1
            all_sing = all_sing + 1
            
        
        #320000/320087, s:9337/257123, p:18448:1213/144517
        # lower bound: 2.63%
        # upper bound: 6.92%

        # (9337+1213) / 399109 = 2.64%
    
    
    #%%

    # counts before cleaning
    print('Before cleaning:')
    print_counts(hes_data)

    #  Drop unused columns
    if 'DISMETH' in hes_data.columns:
        hes_data.drop(columns=['DISMETH'], inplace=True)
    if 'EPIDUR' in hes_data.columns:
        hes_data.drop(columns=['EPIDUR'], inplace=True)

    # ##########
    # Add/change data in the dataframe (no removing)
    # ##########
    hes_data = clean_hes_prepare.main(hes_data)

    # ###########    
    # Check matching quality - remove subjects accordingly
    # ###########
    hes_data = preliminary_sanity_cleaning(hes_data)

    #%%

    # counts after prelim cleaning
    print('After prelim. cleaning:')
    print_counts(hes_data)

    # ##########
    # Print initial counts
    # ##########
    counts = append_counts.init_counts(hes_data)
    msg = """Initial number of episodes and subjects."""
    counts = append_counts.append_counts(hes_data, msg, counts)

    # #########
    # Remove subjects with any nan spell IDs
    # #########
    hes_data = remove_subjects_nan_spell_id(hes_data)
    msg = """Removing subjects with unknown spell id."""
    counts = append_counts.append_counts(hes_data, msg, counts)

    # ######
    # Check single value within subject
    # E.g. mortality should be the same for all rows for a given subject
    # ######
    hes_data = check_single_option_within_subjects(hes_data)
    msg = """Removing subjects with inconsistent:
sex, date-of-birth, mortality indicator."""
    counts = append_counts.append_counts(hes_data, msg, counts)
    
    # ############
    # Remove subjects with duplicate EPIORDER within spells
    # ############
    hes_data = remove_subjects_with_dup_within_spell_events(hes_data)
    msg = """Removing subjects with duplicate episode-order
values within spells."""
    counts = append_counts.append_counts(hes_data, msg, counts)
    

    # #########
    # Remove subjects with ONLY NaN IMD04 in their MYADMIDATE_INIT event
    # #########
    hes_data = set_init_deprivation(hes_data)
    
    # #########
    # Remove subjects with ONLY NaN PROCODE in their MYADMIDATE_INIT event
    # #########
    hes_data = set_init_procode(hes_data)

    # ########
    # Create DEATHDATE
    # ########
    hes_data, counts = clean_hes_deathdate.main(hes_data, counts)

    # #########
    # Ensure 5:1 ratio of controls to MI subjects
    # #########
    hes_data = remove_using_amiID(hes_data)
    msg = """Removing subjects whose matched counterpart
has been removed during one of the cleaning steps."""
    counts = append_counts.append_counts(hes_data, msg, counts)

    # #######
    # Drop columns.
    # MYADMIDATE_INIT is used to set the AMI/Initial event and the end of the
    # filtering step. Do not drop.
    # #######
    hes_data.drop(columns=['MYDOB'], inplace=True)
        
    #%% Print flow
    plot_utils.plot_flowchart('cleaning',counts)
    

    print("""After cleaning, final Event counts:
          Controls: {:,}
          Patients: {:,}""".format(
          hes_data.loc[hes_data['IS_CONTROL']].shape[0],
          hes_data.loc[~hes_data['IS_CONTROL']].shape[0],))
    
    print("""After cleaning, final Subject counts:
          Controls: {:,}
          Patients: {:,}""".format(
          hes_data.loc[hes_data['IS_CONTROL'],'ENCRYPTED_HESID'].drop_duplicates().shape[0],
          hes_data.loc[~hes_data['IS_CONTROL'],'ENCRYPTED_HESID'].drop_duplicates().shape[0],))


    # ###########
    # Save data to 'checkpoint' folder
    # ###########
    hes_data.to_parquet(
        os.path.join(
            params.DIR_CHECKPOINTS,'CLEAN_{}_.gzip'.format(params.R)),
                        compression='gzip')

    print('clean_hes() time: {}'.format(time.time() - t))

