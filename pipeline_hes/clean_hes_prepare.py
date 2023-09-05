# -*- coding: utf-8 -*-
"""
Without removing episodes or subjects, prepare the HES data for the next stage
of cleaning. This script will ultimately add columns that assist with the
cleaning steps and any additional pipeline steps (e.g. the minimum and
maximum episode start dates within spells, which are useful for sorting spells
within the same month).

@author: Chris Hayward
"""

# Initial preparation of data

import pandas as pd
import numpy as np
import pdb


def add_min_max_spell_times(df):
    """Set the min/max values per-spell. These are useful when sorting spells
    later on in the pipeline."""
    df_gb = df[['ENCRYPTED_HESID','PROVSPNOPS','SURVIVALTIME',
                'MYEPISTART','MYEPIEND']].\
        groupby(['ENCRYPTED_HESID','PROVSPNOPS'])
        
    df_gb_min_date = df_gb['MYEPISTART'].min()
    df_gb_max_date = df_gb['MYEPISTART'].max()

    df_gb_min_date.name = 'MYEPISTART_SPELLMIN'
    df_gb_max_date.name = 'MYEPISTART_SPELLMAX'

    df_gb_min_date_end = df_gb['MYEPIEND'].min()
    df_gb_max_date_end = df_gb['MYEPIEND'].max()

    df_gb_min_date_end.name = 'MYEPIEND_SPELLMIN'
    df_gb_max_date_end.name = 'MYEPIEND_SPELLMAX'

    df_gb_min_surv = df_gb['SURVIVALTIME'].min()
    df_gb_max_surv = df_gb['SURVIVALTIME'].max()
    
    df_gb_min_surv.name = 'SURVIVALTIME_SPELLMIN'
    df_gb_max_surv.name = 'SURVIVALTIME_SPELLMAX'
    
    df_gb_mean = pd.concat([df_gb_min_date,
                            df_gb_max_date,
                            df_gb_min_date_end,
                            df_gb_max_date_end,
                            df_gb_min_surv,
                            df_gb_max_surv],axis=1)
    
    return df.merge(df_gb_mean, how='left', on=['ENCRYPTED_HESID','PROVSPNOPS'])


def trim_all_diags_set_to_upper_replace_ami(df):
    """Restrict pri&sec diagnosis codes to 3 upper-case characters."""
    num_diag_cols = sum(map(lambda x: x.startswith('DIAG'), df.columns))
    for i in range(1,num_diag_cols+1):
        diagStr = 'DIAG_{:02d}'.format(i)
        df[diagStr] = df[diagStr].str.upper().str[:3]

def set_age(df):
    """Calculate subject age."""
    # Use as fine-grained as possible (days)
    df['INIT_AGE'] = df['MATCHED_DATE'] - df['MYDOB']
    df['INIT_AGE'] = df['INIT_AGE'].astype('timedelta64[D]') / 365.25
    df['INIT_AGE'] = df['INIT_AGE'].round(2)
    df['INIT_AGE'] = pd.to_numeric(df['INIT_AGE'],downcast='float')


def convert_all_diag_cols_to_category(df):
    """Convert pri&sec diagnosis columns to category data-type."""
    diag_cols = np.array(df.columns)[list(map(lambda x: x.startswith('DIAG_'), df.columns))]
    df = df.astype(dict(zip(diag_cols, np.repeat(['category'],len(diag_cols)))))
    return df

def prepare_df(hes_data):
    """Calculate additional information for further cleaning steps."""

    # #######
    # New column: average survivaltime within spell
    # #######
    hes_data = add_min_max_spell_times(hes_data)
    
    # ######
    # Set age at AMI/initial
    # ######
    set_age(hes_data)
    
    # ########
    # Trim diagnoses
    # Replace AMI codes with 'AMI'
    # This has to occur before searching for 'rare' events
    # This also has to happen before conversion to chapter headings
    # ########
    trim_all_diags_set_to_upper_replace_ami(hes_data)

    # ##########
    # Convert DIAG to category
    # After trimming and setting some to 'AMI'
    # ##########
    hes_data = convert_all_diag_cols_to_category(hes_data)

    return hes_data


def _check_all_patients_have_firstami_set(df):
    """Sanity check - ensure that all MI subjects have a date of first MI."""
    ami_without_firstami = pd.concat([~df['IS_CONTROL'], df['MYEPISTART_FIRSTAMI'].isna()], axis=1).all(axis=1)
    
    if (ami_without_firstami.any()):
        #pdb.set_trace()
        raise Exception("""
When setting MYEPISTART_FIRSTAMI, not all AMI subjects have _FIRSTAMI set.""")


def add_first_ami_date(df):
    """Calculate the date of first MI for the MI cohort."""
    # dates of first AMI    
    hes_ami_start_dates = df.loc[df['AMI'],['ENCRYPTED_HESID', 'MYEPISTART']].\
        sort_values(['MYEPISTART'], ascending=True).\
        drop_duplicates(subset='ENCRYPTED_HESID')
        
    # set the date of first AMI for entire subject (MYADMIDATE_FIRSTAMI)
    df = df.merge(hes_ami_start_dates, on='ENCRYPTED_HESID', how='left',
                  suffixes=('', '_FIRSTAMI'))
    _check_all_patients_have_firstami_set(df)
    return df


def main(hes_data):
    """Entry function for setting columns for downstream cleaning."""
    
    # #########
    # add new column = the FIRST AMI date
    # This is for where Pri/Sec diagnoses are AMI.
    # Just saves the date of first AMI event
    # #########
    hes_data = add_first_ami_date(hes_data)
    hes_data = prepare_df(hes_data)    

    return hes_data

