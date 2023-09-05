# -*- coding: utf-8 -*-
"""
After cleaning the data, select episodes to include in disease trajectories.
Episodes which are not to be included are marked as IGNORE=True.
E.g. episodes before the matched date; episodes containing 'rare' diagnoses.

@author: Chris Hayward
"""

import pandas as pd
import numpy as np
import time
import pdb
import os

from pipeline_hes import append_counts
from pipeline_hes import filter_hes_sort
from pipeline_hes import censoring_hes
from pipeline_hes import table_one_hes
from pipeline_hes import plot_utils
from pipeline_hes.params import params
from pipeline_hes import mask_hes
from pandas.api.types import CategoricalDtype

    
def ignore_invalid_primary_diag(df):
    """Exclude from trajectories episodes with primary diagnoses which are
    invalid/empty."""
    bad_mask_nan = mask_hes.invalid_primary_diag_mask_nan(df)
    bad_mask_r69 = mask_hes.invalid_primary_diag_mask_R69(df)
    df.loc[pd.concat([bad_mask_r69,
                      bad_mask_nan], axis=1).any(axis=1),'IGNORE'] = True


def ignore_invalid_epiorder(df):
    """Exclude from trajectories episodes with invalid epiorder."""
    df.loc[mask_hes.invalid_epiorder_mask(df), 'IGNORE'] = True


def ignore_invalid_epistart_epiend(df):
    """Exclude from trajectories episodes with invalid epistart/epiend
    dates."""
    bad_mask = pd.concat([mask_hes.invalid_epistart_mask(df),
                          mask_hes.invalid_epiend_mask(df)],axis=1).any(axis=1)
    df.loc[bad_mask,'IGNORE'] = True


def ignore_rare_diags_combined_simple(df):
    """Exclude from trajectories diagnoses which are deemed rare."""
    print('Removing rare diagnoses from post sequences...')
    df_reduced = df[['ENCRYPTED_HESID','IS_CONTROL']].drop_duplicates()
    nAMI = (~df_reduced['IS_CONTROL']).sum()
    nCTL = (df_reduced['IS_CONTROL']).sum()
    thresholdAMI = (params.RARE_EVENT_PRC_THRESH/100)*nAMI
    thresholdCTL = (params.RARE_EVENT_PRC_THRESH/100)*nCTL
    
    num_diag_cols = sum(map(lambda x: x.startswith('DIAG'), df.columns))
    df_parts = []
    for i in range(1,num_diag_cols+1):
        diagStr = 'DIAG_{:02d}'.format(i)
        df_tmp = df[['IS_CONTROL','ENCRYPTED_HESID',diagStr]].drop_duplicates()
        df_tmp = df_tmp.rename(columns={diagStr:'DIAG_01'})
        df_parts.append(df_tmp)
    
    df_flat = pd.concat(df_parts,ignore_index=True).drop_duplicates()
    
    c = df_flat[['IS_CONTROL','DIAG_01']].value_counts()
    c = c.reset_index().rename(columns={0:'diag_count'})
        
    # sum across DIAG_**
    c['RARE'] = False
    c.loc[c['IS_CONTROL'],'RARE'] = c.loc[c['IS_CONTROL'],'diag_count']<thresholdCTL
    c.loc[~c['IS_CONTROL'],'RARE'] = c.loc[~c['IS_CONTROL'],'diag_count']<thresholdAMI

    #pdb.set_trace()

    df = df.merge(c[['IS_CONTROL','DIAG_01','RARE']], on=('IS_CONTROL','DIAG_01'), how='left')
    
    # merge converts it to object, need to conv back to cat
    df['DIAG_01'] = df['DIAG_01'].astype('category')
    
    
    #%% counts of number of rare ignored episodes
    df_mi = df.loc[~df['IS_CONTROL'], ['IGNORE','RARE']]
    df_ctl = df.loc[df['IS_CONTROL'], ['IGNORE','RARE']]
    n_rare_mi = pd.concat([df_mi['IGNORE'], df_mi['RARE']], axis=1).\
        any(axis=1).sum() - df_mi['IGNORE'].sum()
    n_rare_ctl = pd.concat([df_ctl['IGNORE'], df_ctl['RARE']], axis=1).\
        any(axis=1).sum() - df_ctl['IGNORE'].sum()
    
    prc_rare_mi = 100*n_rare_mi/df_mi.shape[0]
    prc_rare_ctl = 100*n_rare_ctl/df_ctl.shape[0]
    
    print('Percentage rare MI ignored episodes: {}'.format(prc_rare_mi))
    print('Percentage rare CTL ignored episodes: {}'.format(prc_rare_ctl))

    #%%
    
    # update ignore
    df['IGNORE'] = pd.concat([df['IGNORE'], df['RARE']], axis=1).any(axis=1)

    # save memory - not used after this function
    df.drop(columns=['RARE'], inplace=True)

    return df


def controls_ignore_on_after_first_ami(df):
    """For controls with AMI, exclude episodes on/after the date of first MI."""
    ignore_mask = mask_hes.controls_ignore_on_after_first_ami_mask(df)
    df.loc[ignore_mask,'IGNORE'] = True


def ignore_before_index_date(df):
    """Exclude from trajectories episodes with start dates before the
    matching/index date."""
    ignore_mask = mask_hes.ignore_before_index_date_mask(df)
    df.loc[ignore_mask,'IGNORE'] = True


def flatten_sec_diags_into_pri(df):
    """Take the secondary diagnoses and convert them into primary. This takes
    place when both primary and secondary diagnoses are used to build
    trajectories."""
    
    diag_cols = np.array(df.columns)[list(map(lambda x: x.startswith('DIAG_'),
                                              df.columns))]
    acute_cols = np.array(df.columns)[list(map(lambda x: x.startswith('ACUTE_'),
                                               df.columns))]
    
    #%% get all diagnoses across all pri and sec columns
    df_parts = []
    for i in range(1,len(diag_cols)+1):
        diagStr = 'DIAG_{:02d}'.format(i)
        df_tmp = df[diagStr].drop_duplicates()
        df_tmp.name = 'DIAG_01'
        df_parts.append(df_tmp)
    df_flat = pd.concat(df_parts,ignore_index=True).drop_duplicates()

    allD = CategoricalDtype(df_flat.dropna())
    
    df_parts = []
    for i in range(2,len(diag_cols)+1):
        diagStr = 'DIAG_{:02d}'.format(i)
        acuteStr = 'ACUTE_{:02d}'.format(i)
        print('Flattening sec diags into pri: {}'.format(diagStr))

        # I need all the columns, apart from all other DIAG_NN,ACUTE_NN cols
        df_sec_part = df.copy()
        other_diag_cols = diag_cols[diag_cols!=diagStr]
        other_acute_cols = acute_cols[acute_cols!=acuteStr]
        
        df_sec_part = df_sec_part.drop(columns=np.append(other_diag_cols,
                                                         other_acute_cols))
        df_sec_part = df_sec_part.rename(columns={diagStr:'DIAG_01',
                                                  acuteStr:'ACUTE_01'})
        
        # This should allow us to always retain the DIAG_02->DIAG_20 order later on (when sorting)
        df_sec_part['COL_IDX'] = np.uint8(i-1)

        # Remove nan secondary
        df_sec_part = df_sec_part.dropna(subset=['DIAG_01'])
        df_sec_part['DIAG_01'] = df_sec_part['DIAG_01'].astype(allD)
        
        df_parts.append(df_sec_part)

    df_sec = pd.concat(df_parts,copy=False,ignore_index=True)
    
    # remove all sec
    df_pri = df.copy()
    sec_diag_cols = diag_cols[list(map(lambda x: x!='DIAG_01', diag_cols))]
    sec_acute_cols = acute_cols[list(map(lambda x: x!='ACUTE_01', acute_cols))]
    df_pri = df_pri.drop(columns=np.append(sec_diag_cols, sec_acute_cols))
    
    df_pri['COL_IDX'] = np.uint8(0)
    df_pri['DIAG_01'] = df_pri['DIAG_01'].astype(allD)
    
    df = pd.concat([df_pri,df_sec],copy=False,ignore_index=True)
    df['ACUTE_01'] = df['ACUTE_01'].astype('category')
        
    # re-run AMI check (some rows will now not hold AMI events)
    df['AMI'] = df['DIAG_01'].isin(params.AMI_RANGE)
        
    return df


def run_filtering(hes_data):
    """After cleaning, start to decide which episodes to include in the disease
    trajectories."""
    
    t = time.time()
    
    # #########
    # Add an 'IGNORE' column (this will be used to mark events to ignore when
    # building sequences)
    # #########
    hes_data['IGNORE'] = False
       
    ###########
    # Print initial counts
    ###########
    counts = append_counts.init_counts(hes_data)
    msg = """After cleaning... initial number of episodes."""
    counts = append_counts.append_counts(hes_data, msg, counts)

    #pdb.set_trace()

    # ######
    # Ignore rows with nan dates
    # ######
    ignore_invalid_epistart_epiend(hes_data)
    msg = """Excluding episodes with unknown/invalid episode start/end date."""
    counts = append_counts.append_counts(hes_data, msg, counts)

    # ######
    # Ignore rows with nan diags
    # ######
    ignore_invalid_primary_diag(hes_data)
    msg = """Excluding episodes with unknown or invalid primary diagnosis."""
    counts = append_counts.append_counts(hes_data, msg, counts)

    # ########
    # Ignore everything before first AMI, in:
    # 1) ami patients;
    # 2) controls, using the first ami date of the matched AMI subject        
    # ########
    ignore_before_index_date(hes_data)
    msg = """Excluding episodes before date of first AMI
(for controls without an AMI episode, excluding episodes
 before matched AMI date)."""
    counts = append_counts.append_counts(hes_data, msg, counts)
    
    # #########
    # For AMI Controls: ignore events on and after first AMI
    # #########
    controls_ignore_on_after_first_ami(hes_data)
    msg = """For controls with AMI, excluding episodes on/after first
AMI event."""
    counts = append_counts.append_counts(hes_data, msg, counts)

    # #########
    # If we want to treat secondary diags as primary, but always preserving the Diag02->Diag20 order
    # This will add cols:
    # COL_IDX (for keeping the order DIAG_01->DIAG_02-> ... etc)
    # #########
    if params.USE_SEC_DIAGS_IN_TRACES:
        print('Flattening secondary diagnoses into primary...')
        hes_data = flatten_sec_diags_into_pri(hes_data)
    else:
        # placeholders - still needed for sorting/amb-order later on
        hes_data['COL_IDX'] = np.uint8(0)

    # #########
    # Ignore RARE events
    # #########
    hes_data = ignore_rare_diags_combined_simple(hes_data.copy())
    msg = """Separately for controls and patients, excluding episodes
with a rare primary diagnosis (<{}% of subjects).""".\
    format(params.RARE_EVENT_PRC_THRESH)
    counts = append_counts.append_counts(hes_data, msg, counts)
    

    print('filter_hes() time: {}'.format(time.time() - t))

    return hes_data,counts


def main():
    """Read in the cleaned HES data, and subsequently call the functions
    which select diagnoses for inclusion in trajectories."""
    
    hes_data = pd.read_parquet(os.path.join(params.DIR_CHECKPOINTS,
                                            'CLEAN_{}_.gzip'.\
                                                format(params.R)))
    pdb.set_trace()
    
    
    #%%
    
    tmp = hes_data[['ENCRYPTED_HESID','IS_CONTROL','amiID','INIT_AGE']]\
        .drop_duplicates('ENCRYPTED_HESID').copy()
    
    tmp_MI = tmp.loc[~tmp['IS_CONTROL']].copy()
    tmp_CTL = tmp.loc[tmp['IS_CONTROL']].copy()
    
    
    singular_amiID_MI = tmp_MI['amiID'].value_counts().loc[
        tmp_MI['amiID'].value_counts()==1].reset_index()
        
    tmp_MI_singular = tmp_MI.merge(singular_amiID_MI, left_on='amiID',
                                   right_on='index')
    
    tmp_merge = tmp_MI_singular[['INIT_AGE','amiID_x']].merge(tmp_CTL,
                                      left_on='amiID_x',
                                      right_on='amiID',
                                      how='left')
    
    tmp_merge['AGE_DIFF'] = np.abs(tmp_merge['INIT_AGE_x']-tmp_merge['INIT_AGE_y'])
    
    # singular bad example
    singular_bad_mi = tmp_merge.loc[(tmp_merge['AGE_DIFF'].round()==28)].copy()
    
    
    bad_mi = tmp_merge.loc[(tmp_merge['AGE_DIFF'].round()==10)].copy()
    
#     tmp.loc[tmp['amiID']==3]
#          ENCRYPTED_HESID  IS_CONTROL  amiID   INIT_AGE
# 7797909          1294870        True      3  51.580002
# 7797937          1294872        True      3  61.250000
# 7797945          1294871        True      3  61.419998
# 7797952          2267609       False      3  61.419998
# 7797957          1294873        True      3  61.419998
# 7797959          1294874        True      3  61.080002
    
# 00009704A4745440FDE371152CFDEEF2	01/06/2013	3
    
    #%%
    from pipeline_hes import load_parquet
    import importlib
    importlib.reload(load_parquet)
    
    mi_ids = load_parquet.load_ami()
    ctl_ids = load_parquet.load_controls_matched()
    
    mi_ids = mi_ids.loc[mi_ids['amiID']<20].sort_values('amiID')
    ctl_ids = ctl_ids.loc[ctl_ids['amiID']<20].sort_values('amiID')
    
    mi_ids.to_csv(
        'n:/HES_CVEPI/chrish/AMI_data/ami_small.csv')
    ctl_ids.to_csv(
        'n:/HES_CVEPI/chrish/AMI_data/matchedControl_small.csv')
    #%%
    
#        unique_index                   ENCRYPTED_HESID MATCHED_DATE  amiID
# 258974        258974  00009704A4745440FDE371152CFDEEF2   2013-06-01      3
    
#              unique_index                   ENCRYPTED_HESID MATCHED_DATE  amiID
# 1294870       1294870  916C742EA41161C5933BFCBEC6DCA08C   2013-06-01      3
# 1294871       1294871  793A743FFF7E09DDBF9E5375B3D37F0A   2013-06-01      3
# 1294872       1294872  FD36321D76B206FA9FE5B15736A68A9A   2013-06-01      3
# 1294873       1294873  6C779F23529A77892BB668279084F4D6   2013-06-01      3
# 1294874       1294874  EF4630BAD5180ED0B43D4F40BCC102C6   2013-06-01      3
    
    #%% Table 1
    
    table_one_hes.table_one(hes_data.copy())
    
    # Also count the number of pri and sec diags in each category
    table_one_hes.diag_counts_per__is_control__is_pri__diag_chapter(hes_data.copy())
     
    
    #%%
    
    # Apply filters (decide which events to ignore for the variants)
    (hes_data,counts) = run_filtering(hes_data)

    # Add censor data (new censor event)
    hes_data = censoring_hes.set_censoring(hes_data)
    
    # Handle Ambiguous orderings
    # This will be repeated, and we store the relative risks and hazards
    (hes_data,counts) = filter_hes_sort.main(hes_data, counts)

    #%% value count on conditions AFTER init

    hes_data_part = hes_data[['ENCRYPTED_HESID','DIAG_01_ALT2','DIAG_01','IS_CONTROL']].copy()
    hes_data_part = hes_data_part.loc[~(hes_data_part['DIAG_01']=='Censor')]
    
    init_pos = hes_data_part.loc[hes_data_part['DIAG_01_ALT2']=='INIT'].copy().reset_index()
    hes_data_part = hes_data_part.reset_index()

    hes_data_part = hes_data_part.merge(init_pos[['ENCRYPTED_HESID','index']],
                                        how='left',on='ENCRYPTED_HESID')
    
    hes_data_part = hes_data_part.loc[hes_data_part['index_x']>hes_data_part['index_y']]
   # 
    hes_data_part_mi = hes_data_part.loc[~hes_data_part['IS_CONTROL']]
    hes_data_part_ctl = hes_data_part.loc[hes_data_part['IS_CONTROL']]
    
    print(hes_data_part_mi['DIAG_01_ALT2'].value_counts())
    print(100 * hes_data_part_mi['DIAG_01_ALT2'].value_counts() / \
          hes_data_part_mi.shape[0])
    
    
    print(hes_data_part_ctl['DIAG_01_ALT2'].value_counts())
    print(100 * hes_data_part_ctl['DIAG_01_ALT2'].value_counts() / \
          hes_data_part_ctl.shape[0])
    #print(hes_data_part_mi['DIAG_01_ALT2'].value_counts() / )

    #%%
    # Flow chart:
    plot_utils.plot_flowchart('filtering', counts)
    
    hes_data.to_parquet(os.path.join(params.DIR_CHECKPOINTS,
                                     'CLEAN_FILTERED_{}_.gzip'.\
                                         format(params.R)),compression='gzip')
    return hes_data



