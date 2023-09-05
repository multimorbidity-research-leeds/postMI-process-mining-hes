# -*- coding: utf-8 -*-
"""
Deals with adding of CENSOR data to HES dataframe, one for each individual.
Also deals with calculating follow-up time for each individual and the
ignoring of episodes in the context of a landmrk study (e.g. episodes occurring
within the first 6 months).

@author: Chris Hayward
"""

import pandas as pd
import numpy as np
import pdb

from pipeline_hes.params import params
from pipeline_hes import mask_hes


def add_censor_col(df):
    """Add an additional censor column, containing the censor date or the
    date of death, if applicable, for each subject. The censor date for
    controls with MI is just before the date of first MI."""
    
    # those who haven't died, set to latest possible date (27th march 2017)
    df['CENSOR'] = pd.to_datetime(params.CENSOR_DATE)
    
    # those who HAVE died set to DEATHDATE
    df.loc[df['Mortality']==1, 'CENSOR'] = \
        df.loc[df['Mortality']==1, 'DEATHDATE'] + np.timedelta64(1,'s')

    # ######
    # For controls with MI, CENSOR = first MI date
    # ######
    ctlAMI_mask = mask_hes.ctl_ami_mask(df)
    df.loc[ctlAMI_mask, 'CENSOR'] = \
        df.loc[ctlAMI_mask, 'MYEPISTART_FIRSTAMI'] - np.timedelta64(1,'s')


def add_followup_duration_col(df):
    """Add an additional column containing the follow-up duration for each
    subject (time from matched date to censor date)."""
    
    # ########
    # Follow-up duration is the time from matched event to the CENSOR date
    # ########
    df['DUR'] = (df['CENSOR'] - df['MATCHED_DATE']) / np.timedelta64(1,'Y')


def add_censor_event(df):
    """Add a new row, representing a 'CENSOR' episode. This eases the
    constrution of trajectories."""
    
    censor_events = df[['ENCRYPTED_HESID',
                        'DUR',
                        'Mortality',
                        'INIT_AGE',
                        'IMD04',
                        'SEX',
                        'CENSOR',
                        'IS_CONTROL',
                        'MATCHED_DATE']].drop_duplicates(subset=['ENCRYPTED_HESID']).copy()

    censor_events.loc[censor_events['Mortality']==1,'SURVIVALTIME'] = 0

    # use CENSOR date for new event
    censor_events['MYEPISTART'] = censor_events['CENSOR']
    censor_events['MYEPIEND'] = censor_events['CENSOR']
    censor_events['DISDATE'] = censor_events['CENSOR']

    censor_events['IGNORE'] = False
    censor_events['EPIORDER'] = np.uint8(255)
    censor_events['COL_IDX'] = np.uint8(255)
    
    # Its own special SPELL (important)
    censor_events['PROVSPNOPS'] = df['PROVSPNOPS'].max()+1
    censor_events['PROCODE'] = df['PROCODE'].max()+1

    # Important - not an init row
    censor_events['INIT_ROW'] = False

    # Set additional stuff
    censor_events['AMI'] = False

    # Make sure that the categories match (for df and this new censor-df)
    censor_events['DIAG_01'] = params.CENSOR_CODE
    censor_events['DIAG_01'] = censor_events['DIAG_01'].astype('category')
    censor_events['DIAG_01'] = censor_events['DIAG_01'].cat.\
        add_categories(df['DIAG_01'].cat.categories)
    df['DIAG_01'] = df['DIAG_01'].cat.add_categories([params.CENSOR_CODE])

    return pd.concat([df,censor_events],copy=False).reset_index(drop=True)


def landmark_alteration(df):
    """If running a landmark study, ignore episodes within a time window."""

    if not (params.LIMIT_TO_TIME_IGNORE_THESE in [False,'<6m','>6m']):
        raise Exception('Unknown params.LIMIT_TO_TIME_IGNORE_THESE: {}'.format(
                        params.LIMIT_TO_TIME_IGNORE_THESE))

    # IGNORE BEFORE 6 months
    if params.LIMIT_TO_TIME_IGNORE_THESE=='<6m':

        window_limit = df['MATCHED_DATE'] + pd.to_timedelta(365.25/2,'D') + \
            pd.to_timedelta(1,'W')
        # set to first day of month
        window_limit = window_limit.map(lambda x: x.strftime('%Y-%m-01'))
        window_limit = pd.to_datetime(window_limit)

        # Ignore between initial event and N-month limit.
        ignore_portion_mask = df['MYADMIDATE'] < window_limit

        df.loc[ignore_portion_mask,'IGNORE'] = True

    # IGNORE AFTER 6 months
    # 1) Update mortality (some might not have died up to N-months)
    # 2) Update censor date (for all not dead)
    # 3) Ignore ALL future events
    if params.LIMIT_TO_TIME_IGNORE_THESE=='>6m':

        window_limit = df['MATCHED_DATE'] + pd.to_timedelta(365.25/2,'D') + \
            pd.to_timedelta(1,'W')
        # set to first day of month
        window_limit = window_limit.map(lambda x: x.strftime('%Y-%m-01'))

        before_censor_mask = window_limit < df['CENSOR']

        # not dead
        df.loc[before_censor_mask,'Mortality'] = 0
        # cap to new censor
        df.loc[before_censor_mask,'CENSOR'] = window_limit
        # ignore all events after CENSOR
        df.loc[window_limit <= df['MYADMIDATE'],'IGNORE'] = True



def set_censoring(hes_data):
    """Entry function for script."""

    # #############
    # Set Censor date
    # Either: DEATHDATE if mortality=1
    # Or: 01-04-2017 if mortality=0
    # #############
    add_censor_col(hes_data)

    # ##########
    # See if we need to change things based on time windows.
    # Its better to do it here, later in the pipeline as we need CENSOR col.
    # ##########
    landmark_alteration(hes_data)

    # ############
    # Also... add new column - follow up duration
    # DUR: Follow-up duration from MYADMIDATE_INIT to *****CENSOR date******
    # #############
    add_followup_duration_col(hes_data)

    # ############
    # Add CENSOR event to each subject
    # (each subject therefore has a minimum of two not-ignored events)
    # ############
    hes_data = add_censor_event(hes_data)
    
    return hes_data


