# -*- coding: utf-8 -*-
"""
Functions which call the tableone package, counting instances from the HES
data.

@author: Chris Hayward
"""

import pandas as pd
import numpy as np

from pipeline_hes.params import params
from pipeline_hes import parse_chapters
from pipeline_hes import censoring_hes
from pipeline_hes import plot_utils

import pdb
import tableone
import re
from scipy.stats import chi2_contingency
import os


def _num_replace(str_n):
    """Format numbers as integers with commas."""
    return '{}{:,}{}'.format(str_n.group(1), int(str_n.group(2)), str_n.group(3),)


def format_table_numbers_with_commas(table_str):
    """Format numbers as integers with commas."""
    # number without a dash before or after (i.e. exclude 2008-2009)
    p = re.compile('([^-])([1-9][0-9][0-9][0-9]+)([^-])')
    return p.sub(_num_replace, table_str)
    

def format_table(table_str):
    """Format numbers as integers with commas and add border."""
    table_str = format_table_numbers_with_commas(table_str)
    
    # add border
    p = re.compile('<table>')
    table_str = p.sub('<table border=1>', table_str)
    return table_str


def _flatten_sec(df):
    """Flatten all secondary diagnoses into a dataframe holding subject ID,
    cohort indicator and diagnosis."""
    # 
    sec_all = []
    for i in range(2,21):
        diag_str = 'DIAG_{:02,}'.format(i)
        print('flatten, {}'.format(diag_str))
        sec_tmp = df[['ENCRYPTED_HESID', 'IS_CONTROL', diag_str]].copy()
        sec_tmp.rename(columns={diag_str:'DIAG_01'},inplace=True)
        sec_tmp = sec_tmp.dropna(subset=['DIAG_01'])
        sec_all.append(sec_tmp)
        #print(sec_tmp)
    return pd.concat(sec_all,ignore_index=True,copy=False)


def get_hesid_pri_sec_flat_df(df):
    """Get a dataframe holding the ICD-10 codes converted to chapters for
    primary and all secondary fields."""
    df_pri = df[['ENCRYPTED_HESID','IS_CONTROL','DIAG_01']].copy()

    df_sec = _flatten_sec(df)
    
    # concat
    df_pri['IS_PRI'] = True
    df_sec['IS_PRI'] = False
    print(df_pri)
    print(df_sec)
    df_pri_sec = pd.concat([df_pri,df_sec],ignore_index=True) 
    df_pri_sec['DIAG_01'] = df_pri_sec['DIAG_01'].astype('category')
    df_pri_sec = df_pri_sec.loc[~(df_pri_sec['DIAG_01']==params.CENSOR_CODE)]
    
    parse_chapters.apply_diag_conversion_dict(df_pri_sec)
    
    df_pri_sec['DESC'] = df_pri_sec['DIAG_01_CONV_HIGH'].map(
        parse_chapters.get_codes_description_short_desc())

    return df_pri_sec


def diag_counts_per__is_control__is_pri__diag_chapter(df):
    """Count number of ICD-10 chapters per cohort, and per pri/sec
    designation."""    
    df_pri_sec = get_hesid_pri_sec_flat_df(df)
    #%%
    
    xvc = df_pri_sec[['IS_CONTROL','DESC','IS_PRI','DIAG_01_CONV_HIGH']].\
        value_counts().reset_index().\
            sort_values(['DIAG_01_CONV_HIGH','IS_CONTROL','IS_PRI'],ascending=[True,True,False])
    xvc = xvc.rename(columns={0:'count'})

    # sum of 'count' within IS_PRI
    xsum = xvc.merge(xvc.groupby(['IS_PRI','IS_CONTROL'])['count'].sum(),
                      on=['IS_PRI','IS_CONTROL'])
    xsum = xsum.rename(columns={'count_x':'count','count_y':'sum_within_pri_control'})

    xsum['PRC'] = \
        (100*xsum['count'] / \
         xsum['sum_within_pri_control']).map(lambda x:np.round(x,1))
            
    # FORMAT (sums)
    xsum['count_STR'] = xsum['count'].astype(int).map(lambda x:'{:,}'.format(x))
    
    xsum = xsum.sort_values(['DESC','IS_PRI'],ascending=(True,False))
    
    plot_utils.create_fig_folder()
    
    saveHere = os.path.join(params.DIR_RESULTS,'diag_counts_per_cohort_per_pri.csv')
    print('Saving to: {}'.format(saveHere))
    with open(saveHere, 'w') as f:
        f.write(
            pd.concat([
                xsum.loc[~xsum['IS_CONTROL'].astype('bool'),
                   ['IS_CONTROL','DESC','IS_PRI','count_STR','PRC']],
                xsum.loc[xsum['IS_CONTROL'].astype('bool'),
                   ['IS_CONTROL','DESC','IS_PRI','count_STR','PRC']]
                ],
                ignore_index=True).to_csv())
    
    #%% chi sq test - between MI and controls
    x=pd.crosstab(df_pri_sec['DIAG_01_CONV_HIGH'],
                  [df_pri_sec['IS_CONTROL'],df_pri_sec['IS_PRI']])
    
    df_chi = pd.DataFrame(columns=['p-val'],
                          index=parse_chapters.get_chapter_mapping_high_level().values)
    
    for diag in df_chi.index:
        df_chi.loc[diag,['p-val']] = chi2_contingency([x.loc[diag][False], x.loc[diag][True]])[1]
        
    saveHere = os.path.join(params.DIR_RESULTS,'diag_counts_per_cohort_per_pri_chisq.csv')
    print('Saving to: {}'.format(saveHere))
    with open(saveHere, 'w') as f:
        f.write(df_chi.to_csv())
        
        
def group_dates(df,date_field,date_title):
    """Group dates into 1900-2007, 2008-2010, 2011-2013 and 2014-2017. Save
    grouping as new column."""
    df.loc[df[date_field].isna(), date_field] = pd.to_datetime('1800-01-01')
    yr = df[date_field].dt.year.astype(int)
    df.loc[yr<=1801,date_title] = np.nan

    mask1900 = yr >= 1900
    mask08 = yr >= 2008
    mask11 = yr >= 2011
    mask14 = yr >= 2014
    
    df[date_title] = np.nan
    df.loc[pd.concat([mask1900, ~mask08],axis=1).all(axis=1),
           date_title] = '1900-2007'
    df.loc[pd.concat([mask08, ~mask11],axis=1).all(axis=1),
           date_title] = '2008-2010'
    df.loc[pd.concat([mask11, ~mask14],axis=1).all(axis=1),
           date_title] = '2011-2013'
    df.loc[mask14, date_title] = '2014-2017'
    
    
def group_admission_methods(df):
    """Group admission methods into Elective, Emergency and Unknown."""
    
    elective_admission = ['11','12','13']
    emergency_admission = ['21','22','23','24','25','2A','2B','2C','2D','28']
    unknown_admission = ['99']
    
    m1 = df['ADMIMETH'].isin(elective_admission)
    m2 = df['ADMIMETH'].isin(emergency_admission)
    m3 = df['ADMIMETH'].isin(unknown_admission)
    
    df['ADMIMETH_GROUPED'] = 'Other'
    df['ADMIMETH_GROUPED'] = df['ADMIMETH_GROUPED'].astype(
        pd.CategoricalDtype(['Elective','Emergency','Other','Unknown']))
    
    df.loc[m1,'ADMIMETH_GROUPED'] = 'Elective'
    df.loc[m2,'ADMIMETH_GROUPED'] = 'Emergency'
    df.loc[m3,'ADMIMETH_GROUPED'] = 'Unknown'
    
    return df


def table_one(hes_data):
    """Main function - produces a table of count data."""

    # TABLE ONE
    relevant_cols = np.array(['ENCRYPTED_HESID',
                     'INIT_AGE',
                     'SEX',
                     'IMD04',
                     'IS_CONTROL',
                     'Mortality',
                     'MYEPISTART',
                     'MYEPISTART_FIRSTAMI',
                     'MATCHED_DATE',
                     'DEATHDATE',
                     'ADMIMETH'])
    relevant_cols = np.append(relevant_cols,params.DIAG_COLS)
    
    hes_data = hes_data[relevant_cols].copy()
    hes_data = hes_data.loc[hes_data['MYEPISTART'].dt.year>=2008].copy()

    hes_data_index = hes_data.copy()
    
    hes_data_index = hes_data_index.loc[
        hes_data_index['MATCHED_DATE']==hes_data_index['MYEPISTART']]
    hes_data_index = hes_data_index.drop_duplicates('ENCRYPTED_HESID')
    
    censoring_hes.add_censor_col(hes_data_index)
    censoring_hes.add_followup_duration_col(hes_data_index)

    # *********
    # FROM MATCHED DATE
    hes_data_index.rename(columns={'INIT_AGE':'Age'}, inplace=True)
    hes_data_index.rename(columns={'IMD04':'Index of multiple deprivation'}, inplace=True)
    
    # *********

    hes_data_index.rename(columns={'Mortality':'Died during study period'}, inplace=True)
    hes_data_index.rename(columns={'SEX':'Sex'}, inplace=True)
    hes_data_index.rename(columns={'DUR':'Follow-up duration (years)'}, inplace=True)

    hes_data_index = hes_data_index.astype({'Sex':'category'})
    hes_data_index['Sex'] = hes_data_index['Sex'].cat.rename_categories({1:'Male', 2:'Female', 0:'Missing_0', 9:'Missing_9'})
    
    hes_data_index = hes_data_index.astype({'IS_CONTROL':'category'})
    hes_data_index['IS_CONTROL'] = hes_data_index['IS_CONTROL'].cat.rename_categories({0:'MI',1:'zControl'})
    
    hes_data_index = hes_data_index.astype({'Died during study period':'category'})

    # ****************
    # also count the number of events
    vc = (hes_data['ENCRYPTED_HESID'].value_counts(sort=False)).astype(np.uint16)
    vc.name = 'No. of episodes'
    # *************
    
    hes_data_index = hes_data_index.merge(vc, left_on='ENCRYPTED_HESID', right_index=True)


    # Make table
    tbl = tableone.TableOne(hes_data_index,
                            columns=['Age',
                                     'Sex',
                                     'Index of multiple deprivation',
                                     'IS_CONTROL',
                                     'Follow-up duration (years)',
                                     'No. of episodes',
                                     'Died during study period'],
                            categorical=['Sex',
                                         'Died during study period'],
                            groupby='IS_CONTROL',
                            nonnormal=('Follow-up duration (years)',
                                       'No. of episodes',
                                       'Index of multiple deprivation'),
                            pval=True,
                            htest_name=True)
    
    print(format_table(tbl.tabulate(tablefmt="github")))
    
    # INDEX DATES
    group_dates(hes_data_index,'MYEPISTART','Episode start date')
    x = (hes_data_index[['IS_CONTROL','Episode start date']].value_counts())
    x = x.reset_index()
    print(x)
    
    print(chi2_contingency([x.loc[x['IS_CONTROL']=='zControl',0],
                            x.loc[x['IS_CONTROL']=='MI',0]]))
    
    # *********** ALL ROWS - separate table
    hes_data_dates = hes_data[['MYEPISTART','IS_CONTROL']].copy()
    
    group_dates(hes_data_dates,'MYEPISTART','Episode start date')
    tbl_dates = tableone.TableOne(hes_data_dates,
                               columns=['Episode start date',
                                        'IS_CONTROL'],
                               categorical=['Episode start date'],
                               groupby='IS_CONTROL',
                               pval=True,
                               htest_name=True)
    print(format_table(tbl_dates.tabulate(tablefmt="github")))


    # *********
    # Count admission methods
    hes_data_admimeth_grouped = \
        group_admission_methods(hes_data[['ADMIMETH','IS_CONTROL']].copy())
    print(hes_data_admimeth_grouped[['IS_CONTROL','ADMIMETH_GROUPED']].\
          value_counts().reset_index().sort_values(['IS_CONTROL',0]))


