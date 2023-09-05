# -*- coding: utf-8 -*-
"""
After running csv_to_parquet, concatenates all the parquet files, after
merging on MI (patient) and matched control IDs.
Assigns acute/chronic indicator to every primary and secondary diagnosis.
Outputs the data for the two cohorts into a single parquet file.
This is read by clean_hes.py, and processed further by the pipeline.

@author: Chris Hayward
"""

import pandas as pd
import numpy as np
import glob
import pdb
import re
import os

from pipeline_hes.params import params
import concurrent.futures

# will check for different 'nan' representations, and convert to np.nan
CHECK_THESE_COLS_FOR_NANS=np.append(['ADMIMETH',
                                     'DISMETH',
                                     'PROVSPNOPS',
                                     'PROCODE'],
                                    params.DIAG_COLS)

NON_DIAG_ACUTE_COLS = \
     ['ENCRYPTED_HESID',
      'MYADMIDATE',
      'ADMIMETH',
      'EPIORDER',
      'EPISTAT',
      'Mortality',
      'SURVIVALTIME',
      'MYDOB',
      'SEX',
      'IMD04',
      'PROCODE',
      'DISDATE',
      'DISMETH',
      'PROVSPNOPS',
      'EPIDUR',
      'MYEPISTART',
      'MYEPIEND',
      'IS_CONTROL',
      'MATCHED_DATE',
      'amiID',]

STR_NAN='MYNAN'


def _load_ami_controls(csvFile):
    """Load the mi/control matching csv-file. Contains the subject IDs to
    go into each cohort."""
    df = pd.read_csv(csvFile)
    df = df.reset_index().rename(columns={'index':'unique_index'})
    df['myadmidate'] = pd.to_datetime(df['myadmidate'], format='%Y-%m-%d')
    df.rename(columns={'hesid': 'ENCRYPTED_HESID'}, inplace=True)
    df.rename(columns={'myadmidate': 'MATCHED_DATE'}, inplace=True)
    return df


def load_ami():
    """Load the subject IDs of the MI cohort."""
    return _load_ami_controls(params.FILE_HESID_PATIENT)

def load_controls_matched():
    """Load the subject IDs of the control cohort."""
    return _load_ami_controls(params.FILE_HESID_CONTROLS)


def split_into_controls_patients(hes_data,hes_data_control,hes_data_ami):
    """For the entirety of the HES data, pull out the rows (episodes) for
    the MI and control cohorts into a single dataframe."""
    # merge on hes-id
    patients = hes_data.merge(hes_data_ami, on='ENCRYPTED_HESID', how='inner')
    controls = hes_data.merge(hes_data_control, on='ENCRYPTED_HESID', how='inner')
    
    # Replace old string HESID with a numeric one (faster and slimmer)
    controls.drop(columns=['ENCRYPTED_HESID'],inplace=True)
    controls.rename(columns={'unique_index':'ENCRYPTED_HESID'},inplace=True)
    patients.drop(columns=['ENCRYPTED_HESID'],inplace=True)
    patients.rename(columns={'unique_index':'ENCRYPTED_HESID'},inplace=True)

    print('...Controls immediately following merge: subjects = {:,} | events = {:,}'.\
          format(controls['ENCRYPTED_HESID'].drop_duplicates().shape[0],controls.shape[0]))
    print('...AMI immediately following merge: subjects = {:,} | events = {:,}'.\
          format(patients['ENCRYPTED_HESID'].drop_duplicates().shape[0],patients.shape[0]))

    # Concat
    controls['IS_CONTROL'] = True
    patients['IS_CONTROL'] = False

    # Important to make a new array object
    cols = np.array(NON_DIAG_ACUTE_COLS)
    cols = np.append(cols,params.DIAG_COLS)

    print(controls.columns)
    print(patients.columns)

    # Include ALL diagnosis columns        
    return pd.concat([controls[cols],patients[cols]],ignore_index=True)


def _select_abc_unanimous(acute_codes_for_single_diag):
    """For grouping diagnoses and their acute/chronic indicators, return their
    indicator if all diagnoses (ICD-10 codes falling under this 3-char
    ICD-10 code) give the same indicator (unanimous decision). Otherwise,
    return special code 'X'."""
    codes = acute_codes_for_single_diag.drop_duplicates()
    # if a only one kind of code exists
    if codes.shape[0]==1:
        return codes.iloc[0]   
    # return a dummy value (no majority)
    return 'X'


def read_acute_chronic_csv():
    """Read the ICD-10 chronic indicator file, which marks each ICD-10 code as
    acute, chronic, both or n/a."""
    # file from: https://www.hcup-us.ahrq.gov/toolssoftware/chronic_icd10/chronic_icd10.jsp
    f = pd.read_csv(params.FILE_CSV_ACUTE_CHRONIC, encoding='raw_unicode_escape',
                    engine='python',
                    dtype=str,
                    skiprows=[0,1],
                    usecols=["'ICD-10-CM CODE'", "'CHRONIC INDICATOR'"],
                    sep=',')
    f = f.rename(columns={"'ICD-10-CM CODE'":'DIAG_01', "'CHRONIC INDICATOR'":'ACUTE'})
    # remove single quotes
    f['DIAG_01'] = f['DIAG_01'].map(lambda x: x.split("'")[1].upper())
    f['ACUTE'] = f['ACUTE'].map(lambda x: x.split("'")[1].upper())
    return f


def clean_diag(df):
    """Remove out-of-range characters from ICD-10 diagnosis codes."""
    for col in params.DIAG_COLS:
        print(' cleaning {}'.format(col))
        # clean up the diags a bit (remove dashes, etc)
        df[col] = df[col].map(lambda x: re.sub('[^a-zA-Z0-9]', '', x).upper()\
                              if type(x)==str else x)
        df = df.astype({col:'category'})
    return df


def flatten_unique_diags(df):
    """Get a dataframe of all unique diagnosis columns and flatten into a
    single list of unique diagnoses."""
    df_diag_all = pd.DataFrame(
        pd.concat([pd.Series(df[x].cat.categories) for x in params.DIAG_COLS]).\
            drop_duplicates(), columns=['DIAG_01']).reset_index(drop=True)
    return df_diag_all
    

def merge_with_acute_chronic_codes(df):
    """Assign acute/chronic/both/na/X indicators to all primary and secondary
    diagnoses."""
    df_flat = flatten_unique_diags(df)
    f = read_acute_chronic_csv()
    
    ## First pass (full diag code length) - creates ACUTE column
    df_flat = df_flat.merge(f, on='DIAG_01', how='left')
    print('No trim, remaining nans:\n{}'.format(df_flat['ACUTE'].isna().value_counts()))
    #pdb.set_trace()
    # Gradually find more matches at the diagnoses are trimmed further
    for i in [6,5,4,3]:
        print('merge_with_acute_chronic_codes() trim={}...'.format(i))
        ## additional pass (trim to N chars):
        df_flat['DIAG_01_N'] = df_flat['DIAG_01'].str[:i]
        f['DIAG_01'] = f['DIAG_01'].str[:i]
        # The same trimmed codes may have different acute/chronic codes...
        # Where different, combine into code = 'X'
        fg = f.groupby(by='DIAG_01').ACUTE.agg(_select_abc_unanimous)
        df_flat = df_flat.merge(fg, left_on='DIAG_01_N', right_index=True, how='left')
        
        ## NaNs from the first pass are replaced with the second pass
        df_flat.loc[df_flat['ACUTE_x'].isna(), 'ACUTE_x'] = df_flat.loc[df_flat['ACUTE_x'].isna(), 'ACUTE_y']

        # Clean-up
        df_flat.drop(columns=['ACUTE_y', 'DIAG_01_N'],inplace=True)
        df_flat = df_flat.rename(columns={'ACUTE_x':'ACUTE'})
        
        print('Remaining nans:\n{}'.format(df_flat['ACUTE'].isna().value_counts()))
    
    print('Mapping [DIAG_NN code]->[ACUTE_NN code]')
    # put back into main df
    df_flat = df_flat.set_index('DIAG_01')
    for i in range(1,21):
        # get the acute/chronic codes for the DIAGs in this column
        df['ACUTE_{:02d}'.format(i)] = df['DIAG_{:02d}'.format(i)].map(lambda x: df_flat.loc[x,'ACUTE'])
        
    return df


def mark_ami(df):
    """Create a new boolean column ('AMI') which indiciates whether an
    episode has an MI diagnosis in any of the primary and secondary fields.
    Default definition of MI = [I21,I22,I23]."""
    df['AMI'] = False
    for col in params.DIAG_COLS:
        # check primary and secondary diagnoses for any AMI codes
        df['AMI'] = df['AMI'] | df[col].str[:3].isin(params.AMI_RANGE)
        

def set_nans_as_str(df):
    """For a restricted set of columns, normalise 'empty' values with a string
    placeholder."""
    for col in CHECK_THESE_COLS_FOR_NANS:
        ## Set as nan (needed for replace() and trimming)
        if str(df[col].dtype)=='category':
            df[col] = df[col].cat.add_categories(STR_NAN)
        else:
            df = df.astype({col:str})
        df.loc[df[col].isna(),col] = STR_NAN
        df.loc[df[col].str.upper().isin(['NON','NAN','NULL','NA']), col] = STR_NAN
    return df


def set_nans_as_npnan(df):
    """Revert the string placeholders back to a standard np.nan value."""
    for col in CHECK_THESE_COLS_FOR_NANS:
        df.loc[df[col]==STR_NAN, col] = np.nan


def standardise_nans(df):
    """Ensure that marked/actual empty values are replaced with a standard
    np.nan value."""
    df = set_nans_as_str(df)
    set_nans_as_npnan(df)
    return df


def _load_parquet(pFile,hes_data_ami,hes_data_control):
    """For this thread, load one of the financial year HES parquet files, and
    ultimately obtain a restricted set of rows (for MI and controls), with
    diagnoses marked as acute/chronic."""
    
    print('Loading parquet: {}'.format(pFile))

    hes_data = pd.read_parquet(pFile)

    # #########
    # only keep patients and controls
    # #########
    print('Splitting into controls/patients...')
    hes_data = split_into_controls_patients(hes_data,hes_data_control,hes_data_ami)

    # #########
    # Normalising nan/null values
    # #########
    print('Normalise nans (categories are preserved)...')
    hes_data = standardise_nans(hes_data)

    # #########
    # Change to string, change to upper
    # #########
    print('Cleaning Diags...')
    hes_data = clean_diag(hes_data)

    # #########
    # mark rows with primary or secondary AMI
    # #########
    print('Marking AMI (AMI)')
    mark_ami(hes_data)

    # #########
    # Merge on the acute/chronic/both codes (A,C,B)
    # #########
    print('Setting acute/chronic/both/na/X')
    hes_data = merge_with_acute_chronic_codes(hes_data)


    # ##########
    # Which cols to use
    # ##########
    # Important to make a new array object
    final_cols = np.array(NON_DIAG_ACUTE_COLS)
    # specific to AMI/Control matching
    final_cols = np.append(final_cols,['AMI',])
    # Also save the primary and secondary diags
    final_cols = np.append(final_cols,params.DIAG_COLS)
    # Also save ACUTE codes for primary and sec diags
    final_cols = np.append(final_cols,params.ACUTE_COLS)

    return hes_data[final_cols]


def _replace_field_with_numeric(df,field):
    """For a given column/field, replace instances of values with integer
    values."""
    unique_vals = df[[field,]].copy()
    unique_vals = unique_vals.drop_duplicates()
    unique_vals = unique_vals.loc[~unique_vals[field].isna()]
    unique_vals['NEW_VAL'] = np.arange(unique_vals.shape[0])
    df = df.merge(unique_vals, on=field, how='left')
    df.drop(columns=[field], inplace=True)
    df.rename(columns={'NEW_VAL': field}, inplace=True)
    df[field] = pd.to_numeric(df[field],downcast='unsigned')
    return df
    

def replace_procode(df):
    """Replace provider codes with integer codes."""
    df['PROCODE'] = df['PROCODE'].str[:3]
    df = _replace_field_with_numeric(df,'PROCODE')
    return df
    
    
def replace_spellid(df):
    """Replace spell IDs with integer codes. Spell instances are unique
    within providers, so must be combined with provider code."""
    # make spell ids unique between providers
    pairs = df[['PROVSPNOPS','PROCODE']].copy()
    pairs = pairs.drop_duplicates()
    # get non-nan pairs
    pairs = pairs.loc[~pairs['PROVSPNOPS'].isna()]
    pairs = pairs.loc[~pairs['PROCODE'].isna()]
    pairs['NEW_VAL'] = np.arange(pairs.shape[0])
    df = df.merge(pairs, on=('PROVSPNOPS','PROCODE'), how='left')
    df.drop(columns=['PROVSPNOPS'], inplace=True)
    df.rename(columns={'NEW_VAL': 'PROVSPNOPS'}, inplace=True)
    df['PROVSPNOPS'] = pd.to_numeric(df['PROVSPNOPS'],downcast='unsigned')
    return df


def set_types_cat(df):
    """Set Admission method, Discharge method and diagnosis columns as
    type = category."""
    cat_cols = np.concatenate(
        [['ADMIMETH','DISMETH'],params.DIAG_COLS,params.ACUTE_COLS],axis=None)
    df = df.astype({v:'category' for v in cat_cols})
    return df


def set_types_numeric(df):
    """Set cohort matching id (amiID), subject ID, survival time, deprivation
    index and episode duration as numeric types."""
    df['amiID'] = pd.to_numeric(df['amiID'],downcast='unsigned')
    df['ENCRYPTED_HESID'] = pd.to_numeric(df['ENCRYPTED_HESID'], downcast='unsigned')
    df['SURVIVALTIME'] = pd.to_numeric(df['SURVIVALTIME'],downcast='float')
    df['IMD04'] = pd.to_numeric(df['IMD04'],downcast='float')
    df['EPIDUR'] = pd.to_numeric(df['EPIDUR'],downcast='unsigned')


def check_every_ami_has_ami(hes_data):
    """Sanity check - ensure that all MI individuals have at least one instance
    of MI."""
    print('Sanity check: do all AMI subjects have at least one AMI event? (numbers should be equal):')
    numAMI_with_ami = hes_data[['ENCRYPTED_HESID','IS_CONTROL','AMI']].\
        loc[pd.concat([~hes_data['IS_CONTROL'], hes_data['AMI']==True], axis=1).all(axis=1)].\
            drop_duplicates('ENCRYPTED_HESID')
    n1 = numAMI_with_ami.shape[0]
    n2 = hes_data.loc[~hes_data['IS_CONTROL']].\
        drop_duplicates('ENCRYPTED_HESID').shape[0]

    print(n1)
    print(n2)
    if n1 != n2:
        raise Exception('Not all loaded AMI subjects have an AMI event!')
    
    
def sanity_check_ids(hes_data_control,hes_data_ami):
    """Sanity check - ensure that the two cohorts do not have overlapping 
    subject IDs (ENCRYPTED_HESID)."""
    # get all control IDs not in AMI set
    n1 = pd.Index(hes_data_control['unique_index']).difference(hes_data_ami['unique_index']).shape[0]
    n2 = pd.Index(hes_data_ami['unique_index']).difference(hes_data_control['unique_index']).shape[0]
    if n1 != hes_data_control.shape[0] or n2 != hes_data_ami.shape[0]:
        raise Exception('Loaded control/AMI have overlapping HESIDs!')


def main():
    """Entry function. Load the yearly HES parquet files and write out a single
    parquet file containing the MI and control cohorts."""
    hes_data_ami = load_ami()
    hes_data_control = load_controls_matched()

    #pdb.set_trace()

    ## Ensure no overlapping IDs between cohorts
    hes_data_ami['unique_index'] = 1 + hes_data_ami['unique_index'] + \
        hes_data_control['unique_index'].max()
    sanity_check_ids(hes_data_control,hes_data_ami)
    
    # Split the process of loading each yearly HES parquet file across threads
    # Speeds things up a bit...
    parquetFiles_star = os.path.join(params.DIR_CHECKPOINTS,'NIC17649_APC_*.txt_{}_.gzip'.\
        format(params.R))
    parquetFiles = sorted(glob.glob(parquetFiles_star))
    hes_data = np.zeros(len(parquetFiles),dtype=object)
    with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
        for i,df in zip(range(len(parquetFiles)),executor.map(
                lambda x: _load_parquet(x,hes_data_ami,hes_data_control),parquetFiles)):
            hes_data[i] = df

    print('Big concat...')
    hes_data = pd.concat(hes_data,copy=False,ignore_index=True)
    
    #%% counts
    print(tmp.loc[tmp['IS_CONTROL'], ['ENCRYPTED_HESID']].drop_duplicates().shape)
    #Controls = (1992294, 1)
    print(tmp.loc[~tmp['IS_CONTROL'], ['ENCRYPTED_HESID']].drop_duplicates().shape)
    #MI = (399852, 1)

    
    #%%

    # ####
    # Replace PROCODE with a unique int
    # ####
    print('Replace PROCODE...')
    hes_data = replace_procode(hes_data)
    
    # ####
    # Replace PROVSPNOPS with a unique int
    # ####
    print('Replace Spell ID...')
    hes_data = replace_spellid(hes_data)
    print(hes_data['PROVSPNOPS'].isna().value_counts())
    print(hes_data['PROCODE'].isna().value_counts())

    # ############
    # After concat, set column types
    # ############
    print('Set types...')
    hes_data = set_types_cat(hes_data)
    set_types_numeric(hes_data)

    print(hes_data['PROVSPNOPS'].isna().value_counts())
    print(hes_data['PROCODE'].isna().value_counts())
    
    print('Check every ami subject has ami...')
    check_every_ami_has_ami(hes_data)

    print('Print dtypes...')
    print(hes_data.dtypes)

    # ######
    # Write to a single LOADED parquet file
    # ######
    print(hes_data['ENCRYPTED_HESID'].sum())
    hes_data.to_parquet(
        os.path.join(
            params.DIR_CHECKPOINTS,'MATCHED_BEFORE_CLEANING_{}.gzip'.\
                        format(params.R)),compression='gzip')

if __name__=='__main__':
    main()
