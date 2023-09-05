# -*- coding: utf-8 -*-
"""
After cleaning and an initial exclusion of episodes from trajectories, this
script excludes episodes based on a defined temporal ordering. Episodes
are ordered in a sensible chronological way, and then are excluded depending
on their position in that order.

@author: Chris Hayward
"""
import pdb

import pandas as pd
import numpy as np

from pipeline_hes.params import params
from pipeline_hes import append_counts
from pipeline_hes import parse_chapters


def strictly_order_events(df):
    """Sorts spells and their episodes into a fixed order, making use of
    survival, episode/discharge dates and episode-order information."""

    # STEP 1
    # Sort spells. Based on (per-spell):
    # 1. minimum/maximum survival time;
    # 2. minimum/maxmimum episode start time;
    # 3. minimum/maxmimum episode end time;
    # 4. discharge date.
    # If the order of spells within a single month is still unclear,
    # then order using a fixed random order (column=AMB_ORDER).

    # STEP 2
    # Sort episodes within spells, based on:
    # 1. Episode order.
    # If secondary diagnoses are included, then COL_IDX is relied upon to
    # retail 'as is' order of secondary diagnoses (DIAG_02 -> DIAG_20)
    # e.g: Episode order (=>1,1,2,2,3,3), COL_IDX (=>1,2,1,2,1,2)

    sort_dict = {'ENCRYPTED_HESID':True,
                 'SURVIVALTIME_SPELLMIN':False,
                 'SURVIVALTIME_SPELLMAX':False,
                 'MYEPISTART_SPELLMIN':True,
                 'MYEPISTART_SPELLMAX':True,
                 'MYEPIEND_SPELLMIN':True,
                 'MYEPIEND_SPELLMAX':True,
                 'DISDATE':True,
                 'AMB_ORDER':True,
                 'EPIORDER':True,
                 'COL_IDX':True}

    df.sort_values(list(sort_dict.keys()),
                        ascending=sort_dict.values(),
                        ignore_index=True,inplace=True)
    # save this order: 0->N-1 index as new column ('ORDER')
    df['ORDER'] = np.arange(df.shape[0])
    
    
    
def count_number_of_individuals_with_undetermined_spell_order(hes_data):
    """Count how many subjects have spells for which the true order is unknown."""
    
    num_individuals = hes_data['ENCRYPTED_HESID'].drop_duplicates().shape[0]
    
    #%% Count number of subjects with unclear spell order
    hes_part = hes_data[['ENCRYPTED_HESID','PROVSPNOPS',
                       'SURVIVALTIME_SPELLMIN','SURVIVALTIME_SPELLMAX',
                       'MYEPISTART_SPELLMIN','MYEPISTART_SPELLMAX',
                       'MYEPIEND_SPELLMIN','MYEPIEND_SPELLMAX','DISDATE']].copy()

    # one row per spell
    hes_part = hes_part.drop_duplicates(subset=['ENCRYPTED_HESID','PROVSPNOPS'])
    hes_part = hes_part.drop(columns='PROVSPNOPS')
    # individuals with duplicate sorting-information
    hes_part = hes_part.loc[hes_part.duplicated()].drop_duplicates('ENCRYPTED_HESID')
    return (hes_part.shape[0],num_individuals)


def shuffle_and_assign_final_order(hes_data):
    """Assigned a fixed order to spells and episodes for each individual.
    Also, assign a random order value to spells, to be used in cases where
    the true order of spells within the same month is unknown."""    

    # AMB ORDER must be applied at the SPELL level
    df_part = hes_data[['ENCRYPTED_HESID','PROVSPNOPS']].copy().drop_duplicates()

    amb_order = np.arange(df_part.shape[0])
    np.random.default_rng(params.SPELL_ORDER_RANDOM_SEED).shuffle(amb_order)
    df_part['AMB_ORDER'] = amb_order

    # Associate a single amb number with a SPELL ID.
    # Row Ids will be duplicated in the case where we are treating secondary
    # diags as primary (associating like this will keep the same AMB_ORDER per SPELL)
    hes_data = hes_data.merge(df_part, how='left',
                              on=['ENCRYPTED_HESID', 'PROVSPNOPS'])

    strictly_order_events(hes_data)

    return hes_data


def _get_init_first_ami_mask(df):
    """Get the boolean mask identifying the episodes which are candidates
    for selection as the 'initial' episode for trajectories. For MI, this
    is the subject's first MI."""
    ami_mask = pd.concat([df['AMI'],
                          ~df['IS_CONTROL'],
                          ~df.duplicated(subset=['ENCRYPTED_HESID','AMI']),
                          ],axis=1).all(axis=1)

    return ami_mask


# alternative (those which do not have any candidate initial events)
def _get_init_mask_ctl(df):
    """Get the boolean mask identifying the episodes which are candidates
    for selection as the 'initial' episode for trajectories. For controls,
    is any single episode existing on the matched date."""
    return pd.concat([
        df['INIT_ROW'],
        df['IS_CONTROL'],
        ],axis=1).all(axis=1)


def get_init_rows(df):
    """Combine the MI and control masks identifying the episodes which are
    candidates for selection as the 'initial' episode for trajectories."""
    inclusive_mask = pd.concat([
        _get_init_mask_ctl(df),
        _get_init_first_ami_mask(df)
        ],axis=1,ignore_index=True).any(axis=1)
    return df.loc[inclusive_mask, ['ENCRYPTED_HESID',]].copy()


def get_random_single_init_rows(df_init):
    """For each subject, pick (from the candidate 'initial' episodes) a
    random episode (for MI, this will always be the episode containing the
    first MI, while for controls this can be any episode existing on the
    matched date."""

    init_order = np.arange(df_init.shape[0])
    np.random.default_rng(params.INITIAL_EPISODE_RANDOM_SEED).shuffle(init_order)
    df_init['INIT_ORDER'] = init_order

    # * Just take a random one (when there are several candidate init events)
    df_init = df_init.sort_values(['INIT_ORDER'],ascending=(True))

    # After shuffling, take the first shuffled index for each subject
    df_init = df_init['ENCRYPTED_HESID'].drop_duplicates().reset_index()

    return df_init


def ignore_events_before_init_event(df, df_init):
    """Episodes which chonologically appear before the selected initial
    episode are marked as ignored (excluded from trajectories)."""
    # set the init-index for each subject (for ignoring previous events)
    df = df.merge(df_init[['ENCRYPTED_HESID','index']], on='ENCRYPTED_HESID', how='left')

    # all indices prior to this selected AMI/initial event are set to ignored
    df.loc[df.index<df['index'],'IGNORE'] = True
    df.drop(columns=['index'], inplace=True)
    return df


def set_initial_random(df):
    """Identify which episode to use as the 'initial' episode in trajectories."""
    df = df.sort_values(['ORDER']).reset_index(drop=True)

    # get the rows which are candidates for marking as 'initial'
    df_init = get_init_rows(df)

    #%%
    # Sanity check - all subjects are included
    if df['ENCRYPTED_HESID'].drop_duplicates().shape[0] != \
        df_init['ENCRYPTED_HESID'].drop_duplicates().shape[0]:
        pdb.set_trace()
        raise Exception('Error: Setting initial event - not all subjects have been included!')

    #%% choose random event
    df_init = get_random_single_init_rows(df_init)

    #%% add new column
    df['CHOSEN_INIT'] = False
    df.loc[df_init['index'],'CHOSEN_INIT'] = True
    
    #%% Print number of MI and Control episodes un-ignored:
    print('number of MI and Control episodes un-ignored:')
    print(pd.concat([~df['IS_CONTROL'], df['CHOSEN_INIT'], df['IGNORE']],
                    axis=1).all(axis=1).sum())
    print(pd.concat([df['IS_CONTROL'], df['CHOSEN_INIT'], df['IGNORE']],
                    axis=1).all(axis=1).sum())
    
    #%%
    df.loc[df_init['index'],'IGNORE'] = False

    #%% Sanity: check that first MI is selected as MI/initial for all MI subjects
    y = df.loc[pd.concat([~df['IS_CONTROL'], df['AMI']],axis=1).all(axis=1),
               ['ENCRYPTED_HESID', 'CHOSEN_INIT']].\
        drop_duplicates(subset=['ENCRYPTED_HESID'])
    if not y['CHOSEN_INIT'].all():
        raise Exception('Error: Setting initial event - not all MI subjects have their first AMI event chosen!')

    #%% ignore episodes ordered before the initial episode
    mi_num_ignore_before = pd.concat([~df['IS_CONTROL'], df['IGNORE']],
                    axis=1).all(axis=1).sum()
    ctl_num_ignore_before = pd.concat([df['IS_CONTROL'], df['IGNORE']],
                    axis=1).all(axis=1).sum()
    
    #%%
    df = ignore_events_before_init_event(df,df_init)

    #%% number of episodes ignored before marked initial episode
    print('Number of MI and Control episodes ignored before init episode.')
    print(pd.concat([~df['IS_CONTROL'], df['IGNORE']], axis=1).all(axis=1).sum() \
          - mi_num_ignore_before)
    print(pd.concat([df['IS_CONTROL'], df['IGNORE']], axis=1).all(axis=1).sum() \
          - ctl_num_ignore_before)
    
    #%%
    # Sanity check - all subjects have ONE init event
    if df['ENCRYPTED_HESID'].drop_duplicates().shape[0] != df['CHOSEN_INIT'].sum():
        raise Exception('Error: Setting initial event - mismatch with num of init events!')

    #%% Replace init event
    ## Add the MI/Initial category
    df['DIAG_01_ALT1'] = df['DIAG_01_ALT1'].cat.add_categories([params.AMI_INIT])
    df['DIAG_01_ALT2'] = df['DIAG_01_ALT2'].cat.add_categories([params.AMI_INIT])
    df['DIAG_01'] = df['DIAG_01'].cat.add_categories([params.AMI_INIT])

    df.loc[df['CHOSEN_INIT'], 'DIAG_01_ALT1'] = params.AMI_INIT
    df.loc[df['CHOSEN_INIT'], 'DIAG_01_ALT2'] = params.AMI_INIT
    df.loc[df['CHOSEN_INIT'], 'DIAG_01'] = params.AMI_INIT

    # Sanity check - check all first MI events for MI subjects are AMI/Initial
    tmpD = df.loc[pd.concat([~df['IS_CONTROL'],
                             df['AMI']],axis=1).all(axis=1),
                 ['ENCRYPTED_HESID','DIAG_01','AMI']].\
        drop_duplicates(subset=['ENCRYPTED_HESID'])['DIAG_01'].drop_duplicates()

    if (~df['IS_CONTROL']).any() and not (tmpD.shape[0]==1 and tmpD.iloc[0]==params.AMI_INIT):
        raise Exception('Not all first MI events have been set as initial.')

    return df



def _get_chronic_rows_pri_conc(df):
    """Select indices from dataframe for primary diagnoses marked with 'C'
    (Chronic)."""
    df_chronic_pri = df.loc[df['ACUTE_01']=='C',
                             ['ENCRYPTED_HESID','DIAG_01']]
    # Set index as column
    df_chronic_pri = df_chronic_pri.reset_index()
    df_chronic_pri = df_chronic_pri.rename(columns={'DIAG_01':'TMP_DIAG'})

    # Adding POS always ensures that sec diags appear AFTER pri when sorting
    df_chronic_pri['POS'] = 1

    return df_chronic_pri


def _get_chronic_rows_sec_conc_flatten(df):
    """Select indices from dataframe for secondary diagnoses marked with 'C'
    (Chronic)."""

    num_diag_cols = sum(map(lambda x: x.startswith('DIAG'), df.columns))
    secs = []
    print('Subsequent chronic check... (looking at secondary diags)')
    for i in range(2,num_diag_cols+1):
        diagStr = 'DIAG_{:02d}'.format(i)
        acuteStr = 'ACUTE_{:02d}'.format(i)
        
        print('Concatenating {} and {}'.format(diagStr,acuteStr))
        df_chronic_sec = df.loc[df[acuteStr]=='C',['ENCRYPTED_HESID',diagStr]]

        # repeated instances within a secondary acute column are useless - remove these
        df_chronic_sec = df_chronic_sec.drop_duplicates()
        df_chronic_sec = df_chronic_sec.rename(columns={diagStr:'TMP_DIAG'})

        df_chronic_sec['POS'] = i
        # only care about chronic events
        secs.append(df_chronic_sec)

    if len(secs)==0:
        return secs

    # Set index as column
    df_chronic_conc = pd.DataFrame(pd.concat(secs,copy=False)).reset_index()

    # Dropping dups here just saves memory
    # ! This has to happen after 'index' column added (run test.py to confirm)
    # - stops secondaries being removed which existed earlier, but in a different DIAG column
    # (a more RHS column after concat)
    return df_chronic_conc.drop_duplicates()

    
def only_keep_first_chronic_occurrence(df):
    """Exclude from trajectories episodes with diagnoses which are chronic,
    and have also appeared earlier in the subject's episodes (as a primary or
    secondary diagnosis)."""
    
    df = df.sort_values(['ORDER']).reset_index(drop=True)
    
    # PRIMARY - row indices of chronic primary diagnoses
    pri_conc = _get_chronic_rows_pri_conc(df)

    # SECONDARY - row indices of chronic secondary diagnoses
    sec_conc = _get_chronic_rows_sec_conc_flatten(df)
    
    # if there are no secondary fields (i.e. we flatten earlier in the pipeline)
    if len(sec_conc)==0:
        df_diag_flat = pri_conc
    else:
        # concat
        df_diag_flat = pd.concat([pri_conc,sec_conc],copy=False)

    # Sanity check - no dup index values for POS==1 rows
    if df_diag_flat.loc[df_diag_flat['POS']==1, 'index'].duplicated().any():
        raise Exception('Error: For ignoring chronic events, some POS==1 have duplicate indices.')
        
    # sort, first by index, then by POS (secondary DIAG rows ALWAYS after primary)
    df_diag_flat = df_diag_flat.sort_values(['index', 'POS'])
    
    # Only ignore rows with POS==1
    # i.e. rows should not be ignored because of a secondary following a primary
    dup_loc = pd.concat([df_diag_flat.duplicated(subset=['ENCRYPTED_HESID','TMP_DIAG']),
                         df_diag_flat['POS']==1],axis=1).all(axis=1)
    df.loc[df_diag_flat.loc[dup_loc, 'index'], 'IGNORE'] = True

    return df



def ignore_close_events(df):
    """Exclude from trajectories diagnoses which appear soon after the same
    diagnosis (based on a user-defined window of time)."""
    df = df.sort_values(['ORDER']).reset_index(drop=True)

    # get relevant columns
    df_close = df[['ENCRYPTED_HESID', 'MYEPISTART', 'MYEPIEND']].copy()
    df_close = pd.concat([df_close, df.filter(regex='^DIAG_*')], axis=1)
    
    unique_diags = df['DIAG_01'].drop_duplicates().values
    # for each unique pri diag, calculate the gaps (across pri and sec occurrences)
    for i,d in enumerate(unique_diags):
                
        print('ignore_close_events (2 month limit), diag={}, {}/{}'.\
              format(d, i+1, unique_diags.shape[0]))
        # get all rows containing diagnosis (pri and sec)
        df_rows_with_d = df_close.loc[
            (df_close.filter(regex='^DIAG_*')==d).any(axis=1)]
        
        # if first row for subject has a primary==d, then never ignore this row
        first_pri_mask = pd.concat([
                    ~df_rows_with_d.duplicated(subset=['ENCRYPTED_HESID']),
                    df_rows_with_d['DIAG_01']==d],axis=1).all(axis=1)
        
        # this EPISTART minus previous EPIEND
        previous_gap_days = np.append(
            np.timedelta64(0),
            df_rows_with_d.iloc[1:]['MYEPISTART'].values - \
            df_rows_with_d.iloc[:-1]['MYEPIEND'].values)
        previous_gap_days = previous_gap_days.astype('timedelta64[D]')

        too_close = previous_gap_days <= \
            np.timedelta64(params.MAX_EVENT_DIST_TOO_CLOSE_DAYS,'D')

        # get index for:
        # 1) those which are too close
        # 2) are not a first appearance (pri)
        # 3) the diag in question is a primary diag
        df_rows_with_d = df_rows_with_d.loc[
            np.logical_and(too_close,
               np.logical_and(~first_pri_mask,
                              df_rows_with_d['DIAG_01']==d))]

        df.loc[df_rows_with_d.index,'IGNORE'] = True

    return df


def convert_chapters_ignore_no_match(df):
    """Convert three-character ICD-10 diagnoses into their ICD-10 chapters."""
    # save orig diagnosis
    df['_DIAG_01'] = df['DIAG_01']
    
    # this will return a COARSE *and* a GRANULAR conversion
    parse_chapters.apply_diag_conversion_dict(df)

    # ALT2 is just the original DIAG_01
    df = df.rename(columns={'DIAG_01':'DIAG_01_ALT2'})
    
    # replace DIAG_01
    if params.CHAPTER_HEADINGS_USE_GRANULAR:
        df['DIAG_01'] = df['DIAG_01_CONV_LOW']
        df = df.drop(columns=['DIAG_01_CONV_LOW'])
        df = df.rename(columns={'DIAG_01_CONV_HIGH':'DIAG_01_ALT1'})
        
    else:
        df['DIAG_01'] = df['DIAG_01_CONV_HIGH']
        df = df.drop(columns=['DIAG_01_CONV_HIGH'])
        df = df.rename(columns={'DIAG_01_CONV_LOW':'DIAG_01_ALT1'})
    
    # Ignore diags which do not fall into any category ('nomatch')
    df.loc[df['DIAG_01']==params.CHAPTER_NO_MATCH, 'IGNORE'] = True
    
    return df



def main(hes_data, counts):
    """After initial filtering of some episodes for inclusion in trajectories,
    order the episodes and further exclude dependent on this ordering."""

    # Count number of individuals with spells for which the true order cannot
    # be determined
    
    frac_undetermined = \
        count_number_of_individuals_with_undetermined_spell_order(
            hes_data.loc[~hes_data['IGNORE']])
    print('No. of cases where true order of spells cannot be determined (all): {:,}/{:,}'.\
        format(frac_undetermined[0],frac_undetermined[1]))
    frac_undetermined = \
        count_number_of_individuals_with_undetermined_spell_order(
            hes_data.loc[pd.concat([~hes_data['IGNORE'],~hes_data['IS_CONTROL']],axis=1).all(axis=1)])
    print('No. of cases where true order of spells cannot be determined (MI): {:,}/{:,}'.\
        format(frac_undetermined[0],frac_undetermined[1]))
    frac_undetermined = \
        count_number_of_individuals_with_undetermined_spell_order(
            hes_data.loc[pd.concat([~hes_data['IGNORE'],hes_data['IS_CONTROL']],axis=1).all(axis=1)])
    print('No. of cases where true order of spells cannot be determined (Controls): {:,}/{:,}'.\
        format(frac_undetermined[0],frac_undetermined[1]))

    # Handle ambiguous dates (different diags in the same month)
    # Has to be done separately for patients and controls (merging on hes id)
    hes_data = shuffle_and_assign_final_order(hes_data)

    
    # Drop some columns - save memory
    hes_data.drop(columns=['MYADMIDATE',
                           'DISDATE',
                           'SURVIVALTIME_SPELLMIN',
                           'SURVIVALTIME_SPELLMAX',
                           'MYEPISTART_SPELLMIN',
                           'MYEPISTART_SPELLMAX',
                           'MYEPIEND_SPELLMIN',
                           'MYEPIEND_SPELLMAX',
                           'AMB_ORDER',
                           'EPIORDER',
                           'COL_IDX'], inplace=True)

    # #########
    # If CHRONIC (ACUTE=='C') Keep only the first occurence of a disease...
    # #########
    if params.IGNORE_REPEATED_CHRONIC:
        hes_data = only_keep_first_chronic_occurrence(hes_data)
        msg = """Using primary diagnosis, excluding subsequent chronic
episodes with matching diagnosis codes."""
        counts = append_counts.append_counts(hes_data, msg, counts)

    # save memory (drop all ACUTE columns)
    hes_data.drop(columns=[y for y in hes_data.columns if y.startswith('ACUTE')],
                  inplace=True)
    
    # #########
    # Ignore diags which are too close
    # #########
    if params.IGNORE_TOO_CLOSE:
        hes_data = ignore_close_events(hes_data)
        msg = """Excluding episodes with an acute primary diagnosis occuring
less than two months after the same diagnosis."""
        counts = append_counts.append_counts(hes_data, msg, counts)

    # save memory (episode start date is needed for table one)
    hes_data.drop(columns=['MYEPIEND'], inplace=True)

    # #########
    # Convert diagnoses to their chapters.
    # ignore any lines which havent matched
    # This happens BEFORE setting the init event
    # (as we want init events to be in range A-N)
    # #########
    hes_data = convert_chapters_ignore_no_match(hes_data)
    msg = """After converting diagnoses into chapter headings:
excluding episodes with diagnoses from non-disease ICD-10 chapter headings."""
    counts = append_counts.append_counts(hes_data, msg, counts)

    
    # #######
    # Set the episode to be used as the 'initial' diagnosis for trajectories.
    # For the MI cohort this will always be the first MI.
    # #######
    hes_data = set_initial_random(hes_data)
    msg = """Selecting and including the starting episode for each trajectory.
Earlier episodes are excluded."""
    counts = append_counts.append_counts(hes_data, msg, counts)

    return hes_data, counts

