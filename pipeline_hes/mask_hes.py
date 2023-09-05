# -*- coding: utf-8 -*-
"""
Contains functions which return boolean masks that select rows of data for
the cleaning and filtering (episode exclusion/inclusion) steps.

@author: Chris Hayward
"""

import pandas as pd
import numpy as np
import pdb


def init_mask(df):
    """Masks for rows which are candidates for being used as the first
    episode/diagnosis for trajectories."""
    init_mask_ctl = pd.concat([df['INIT_ROW'],
                               df['IS_CONTROL']],axis=1).all(axis=1)
    # for patients, must also be an AMI event
    init_mask_pat = pd.concat([df['INIT_ROW'],
                               df['AMI'],
                               ~df['IS_CONTROL']], axis=1).all(axis=1)
    
    init_mask_conc = pd.concat([init_mask_ctl, init_mask_pat],axis=1).any(axis=1)
    return init_mask_conc


def ctl_ami_mask(df):
    """Select rows (episodes) for controls with an MI event."""
    return pd.concat([df['IS_CONTROL'],
                      ~(df['MYEPISTART_FIRSTAMI'].isna())],axis=1).all(axis=1)


def ignore_before_index_date_mask(df): 
    """Select rows (episodes) which occur before the index date (matched
    date)."""
    ignoreMask = df['MYEPISTART']<df['MATCHED_DATE']
    
    return ignoreMask


def controls_ignore_on_after_first_ami_mask(df):
    """Select rows which occur on and after the first MI event in controls."""
    ignoreMask = pd.concat([df['MYEPISTART']>=df['MYEPISTART_FIRSTAMI'],
                            ctl_ami_mask(df)], axis=1).all(axis=1)
    return ignoreMask


def ignore_diag_not_A_to_N_mask(df):
    """Select rows with a primary diagnosis which is not a disease (i.e.
    outside range A00-N99)."""
    # For category types, nans are preserved by map
    tmp_diag = df['DIAG_01'].astype(str)
    keep_mask = tmp_diag.map(lambda x: x[0]>='A' and x[0]<='N')
    return ~keep_mask


def invalid_epiorder_mask(df):
    """Select rows with an invalid/empty episode-order value."""
    tmpCol = df['EPIORDER'].astype(str)
    invalid_mask = pd.concat([tmpCol=='99',
                              tmpCol=='nan'],axis=1).any(axis=1)
    return invalid_mask


def invalid_primary_diag_mask_R69(df):
    """Select rows with an unknown/invalid primary diagnosis."""
    diag_col = df['DIAG_01']
    invalid_mask = diag_col.str[:3]=='R69'
    return invalid_mask


def invalid_primary_diag_mask_nan(df):
    """Select rows with an empty primary diagnosis."""
    diag_col = df['DIAG_01']
    invalid_mask = diag_col.astype(str)=='nan'
    return invalid_mask


def _invalid_date_mask_nan(df,field):
    """Select rows with invalid date value (in 'field')."""
    ## 2012/13 onwards:
    # 1800-01-01 - null
    # 1801-01-01 - invalid
    ## 89/90 onwards
    # 1600-01-01 - null
    # 1582-10-15 - invalid
    invalid_mask = pd.concat([
        df[field]<=np.datetime64('1801-01-01'),
        df[field].isna()
        ],axis=1).any(axis=1)
    return invalid_mask


def invalid_epistart_mask(df):
    """Select rows with an invalid episode start date."""
    return _invalid_date_mask_nan(df,'MYEPISTART')


def invalid_epiend_mask(df):
    """Select rows with an invalid episode end date."""
    return _invalid_date_mask_nan(df,'MYEPIEND')


def invalid_disdate_mask(df):
    """Select rows with an invalid discharge date."""
    ## 2012/13 onwards:
    # 1800-01-01 - null
    # 1801-01-01 - invalid
    ## 89/90 onwards
    # 1600-01-01 - null
    # 1582-10-15 - invalid
    # NaNs are ok (non-final episodes will have nan disdate as standard)
    invalid_mask = df['DISDATE']<=np.datetime64('1801-01-01')
    return invalid_mask

