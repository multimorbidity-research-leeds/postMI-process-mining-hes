# -*- coding: utf-8 -*-
"""
Functions concerning the calculation of dates of death, based on SURVIVALTIME
and discharge dates.

@author: Chris Hayward
"""

# Set the DEATHDATE

import pandas as pd
import numpy as np
import pdb

from pipeline_hes import mask_hes
from pipeline_hes import append_counts


NAN_FINAL_DISDATE = '1800-01-01'


def set_death_date(df):
    """Calculate the death date for those who have died.
    Use the final discharge date and the survival time in days."""
    
    # We dont want nan survivaltimes in the main df, as nans arent helpful when sorting later on...
    # This is why we take a copy()
    df_surv = df[['ENCRYPTED_HESID','DISDATE',
                  'SURVIVALTIME','Mortality','MYEPIEND']].copy()

    # Those who have died only
    df_surv = df_surv.loc[df_surv['Mortality']==1]
    
    # Exclude negative and excessive survivaltimes from the calc.
    df_surv.loc[df_surv['SURVIVALTIME']>4000, 'SURVIVALTIME'] = np.nan
    df_surv.loc[df_surv['SURVIVALTIME']<0, 'SURVIVALTIME'] = np.nan

    # where DISDATE is not available, use EPIEND
    df_surv['BASEDATE'] = df_surv['DISDATE']
    df_surv.loc[df_surv['BASEDATE'].isna(),'BASEDATE'] = \
        df_surv.loc[df_surv['BASEDATE'].isna(),'MYEPIEND']

    # get a latest disdate which has a non-nan survival time
    df_surv = \
        df_surv.\
        dropna(subset=['BASEDATE']).\
        dropna(subset=['SURVIVALTIME']).\
        sort_values(['BASEDATE'],ascending=True).\
        drop_duplicates(subset=['ENCRYPTED_HESID'],keep='last')

    # # Death date equals DISDATE + SURVIVIALTIME (in days)
    # print('Deathdate, DISDATE + SURVIVIALTIME...')
    df_surv['DEATHDATE'] = df_surv['BASEDATE'] + \
         pd.to_timedelta(df_surv['SURVIVALTIME'],'D')
             
    df_surv['DEATHDATE'] = df_surv['DEATHDATE']\
        .map(lambda x: x.strftime('%Y-%m-01'))
    df_surv['DEATHDATE'] = pd.to_datetime(df_surv['DEATHDATE'])
    
    df = df.merge(df_surv[['ENCRYPTED_HESID','DEATHDATE']],
                  how='left', on='ENCRYPTED_HESID')
    
    return df


def backfill_disdate(df):
    """Assign a discharge date to all episodes within a spell. This
    assists with sorting later on."""
    df_disdates = df[['ENCRYPTED_HESID','PROVSPNOPS','DISDATE']].\
        dropna(subset=['DISDATE']).\
        drop_duplicates(['ENCRYPTED_HESID','PROVSPNOPS'])

    # merge back into df
    df = df.drop(columns=['DISDATE']).merge(df_disdates, how='left',
                  on=['ENCRYPTED_HESID','PROVSPNOPS'])

    df.loc[mask_hes.invalid_disdate_mask(df),'DISDATE'] = \
        np.datetime64(None)

    return df


def remove_dead_with_nan_deathdate(df):
    """Remove any subjects with missing calculated deathdates."""
    df_bad = df.loc[pd.concat([
            df['DEATHDATE'].isna(),
            df['Mortality']==1],axis=1).all(axis=1), 'ENCRYPTED_HESID'].drop_duplicates()
    good_rows_hesids = pd.Index(df['ENCRYPTED_HESID'].drop_duplicates()).\
        difference(df_bad, sort=False)

    return df.merge(pd.Series(good_rows_hesids), on='ENCRYPTED_HESID', how='right')


def remove_subjects_with_early_deathdate(df):
    """Remove subjects with a calculated deathdate before any discharge date."""
    
    df_early = df['DEATHDATE']<df['DISDATE']
    
    df_bad = df.loc[pd.concat([
        df['Mortality']==1,
        df_early],axis=1).all(axis=1), 'ENCRYPTED_HESID'].drop_duplicates()
        
    good_rows_hesids = pd.Index(df['ENCRYPTED_HESID'].drop_duplicates()).\
        difference(df_bad, sort=False)
    return df.merge(pd.Series(good_rows_hesids), on='ENCRYPTED_HESID', how='right')


def remove_bad_deathdate(df):
    """Remove subjects according to quality of death date."""
    df = remove_dead_with_nan_deathdate(df)
    df = remove_subjects_with_early_deathdate(df)
    return df


def mark_controls_with_ami_as_alive(df):
    """Mark controls with MI as not dead. They are censored at first MI.
    These subjects were alive before their first MI."""
    df.loc[mask_hes.ctl_ami_mask(df), 'Mortality'] = 0
    return df


def set_negative_survivaltime_to_zero(df):
    """Fix negative survival time values (negative values are set to zero)."""
    df.loc[df['SURVIVALTIME']<0, 'SURVIVALTIME'] = 0


def main(hes_data, counts):
    """Main entry function returning a dataframe with a new DEATHDATE column.
    Subjects for which a death date could not be reliably estimated will be
    removed."""

    # ##############
    # Controls with AMI are always alive
    # this needs to happen before we set DEATHDATE
    # ##############
    hes_data = mark_controls_with_ami_as_alive(hes_data)

    # #########
    # Survivaltime < 0 to zero
    # This is important for keeping CENSOR as the final event in traces
    # #########
    set_negative_survivaltime_to_zero(hes_data)

    # ###########
    # Calculate the appropriate death date
    # takes mean of DISDATE+SURVIVALTIME
    # #######
    hes_data = set_death_date(hes_data)
        
    # ######
    # Backfill discharge date to the earlier episodes (within each spell)
    # ######
    hes_data = backfill_disdate(hes_data)

    # ########
    # Remove subjects where DISDATE+SURVIVALTIME was totally nan, for mortality==1
    # #######
    hes_data = remove_bad_deathdate(hes_data)
    msg = """Removing subjects who died but insufficient date information
meant that a date of death could not be approxmiated."""
    counts = append_counts.append_counts(hes_data, msg, counts)

    return hes_data, counts


