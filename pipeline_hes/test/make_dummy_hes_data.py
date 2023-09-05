# -*- coding: utf-8 -*-
"""
Makes a HES data file (CSV) containing random data.
This data can be used as input into the pipeline (csv_to_parquet.py).

@author: Chris Hayward
"""

import pandas as pd
import numpy as np
import pdb
import os

from pipeline_hes.params import params


COL_NAMES = [
            'DIAG_01',
            'DIAG_02','DIAG_03',
            'DIAG_04','DIAG_05',
            'DIAG_06','DIAG_07',
            'DIAG_08','DIAG_09',
            'DIAG_10','DIAG_11',
            'DIAG_12','DIAG_13',
            'DIAG_14','DIAG_15',
            'DIAG_16','DIAG_17',
            'DIAG_18','DIAG_19',
            'DIAG_20',
            'ADMIMETH',
            'DISMETH',
            'PROCODE',
            'MYADMIDATE',
            'DISDATE',
            'MYDOB', 
            'MYEPIEND',
            'MYEPISTART',
            'EPIDUR',
            'IMD04',
            'SURVIVALTIME',
            'EPIORDER',
            'EPISTAT',
            'SEX',
            'Mortality',
            'PROVSPNOPS',
            'ENCRYPTED_HESID']


PROCODE_VALUES = [''.join([chr(np.random.randint(97,97+25)).upper() for _ in range(3)]) for i in range(50)]
PROVSPNOPS_VALUES = [''.join([chr(np.random.randint(97,97+25)).upper() for _ in range(3)]) for i in range(50)]
DATE_MONTH_VALUES = pd.Series(pd.date_range('2008-03-01','2017-01-01',freq='M')+\
                              pd.to_timedelta('1D')).dt.strftime('%m%Y')
DOB_VALUES = pd.Series(pd.date_range('1950-01-01','1990-01-01',freq='M')+\
                              pd.to_timedelta('1D')).dt.strftime('%m%Y')
EPIDUR_VALUES =  range(1,50)

VALID_VALUES = {'ADMIMETH':[11,12,13,21,22,23,24,25,'2A','2B','2C','2D',
                            28,31,32,82,83,81,84,89,98,99],
                'DISMETH':[1,2,3,4,5,8,9],
                'EPISTAT':[3],
                'SEX':[1,2,0,9],
                'Mortality':[0,1],
                'PROCODE':PROCODE_VALUES,
                'MYEPISTART':DATE_MONTH_VALUES,
                'MYEPIEND':DATE_MONTH_VALUES,
                'DISDATE':DATE_MONTH_VALUES,
                'MYADMIDATE':DATE_MONTH_VALUES,
                'MYDOB':DOB_VALUES,
                'SURVIVALTIME':range(1000),
                'IMD04':np.round(np.arange(0.1,20,.01),2),
                'EPIDUR':EPIDUR_VALUES}

MI_CODES = params.AMI_RANGE

def main(num_subs=500):
    df = pd.DataFrame(columns=COL_NAMES)
    
    subjectID = 1
    epiorder = 1
    spellID = 1
    rowId = -1
    while True:
        # if round(100*subjectID/num_subs) % 20 == 0:
        #     print('{}%'.format(100*subjectID/num_subs))
        
        if subjectID==num_subs:
            break
        rowId += 1
        
        allDiags = False
        
        # make a new subject
        if np.random.random() <.2:
            subjectID += 1
            epiorder = 1
            spellID = 1
    
        # make a new spell
        elif np.random.random() <.1:
            spellID += 1
            epiorder = 1
            
        # new episode within same spell
        else:
            epiorder += 1
            allDiags = np.random.random() <.01
            
        df.loc[rowId,'ENCRYPTED_HESID'] = str(subjectID) + 'X'
        df.loc[rowId,'PROVSPNOPS'] = str(spellID) + 'Y'
        df.loc[rowId,'EPIORDER'] = epiorder
        diag_stop = False
        
        for colName in COL_NAMES:
            #print(colName)
        
            if colName in VALID_VALUES.keys():
                # pick a random value
                
                if len(VALID_VALUES[colName])==1:
                    r = 0
                else:
                    r = np.random.randint(0,len(VALID_VALUES[colName])-1)
                df.loc[rowId,colName] = str(VALID_VALUES[colName][r])
            
            elif colName.startswith('DIAG'):
                diag_pos = int(colName.split('_')[1])
                # taper the diags
                if not allDiags and (diag_stop or (1.0/diag_pos < np.random.random()/6)):
                    diag_stop = True
                    df.loc[rowId,colName] = 'nan'
                    continue
                
                while True:
                    randDiag = chr(np.random.randint(97,97+25)).upper() + \
                        ''.join([str(np.random.randint(0,9)) for _ in range(2)])
                    # cannot be MI
                    if not (randDiag in MI_CODES):
                        df.loc[rowId,colName] = randDiag
                        break
            
            # random nan set:
            if not colName in ('MYEPISTART','EPIORDER','EPISTAT','ENCRYPTED_HESID'):
                if .01 > np.random.random():
                    df.loc[rowId,colName] = np.nan
                    
            # some unfinished
            if colName == 'EPISTAT':
                if .001 > np.random.random():
                    df.loc[rowId,colName] = 1
    
    #pdb.set_trace()

    # Sex, mortality, and DOB need to be per subject
    df = df.drop(columns=['SEX','MYDOB','Mortality'])
    
    subLevelVals = df[['ENCRYPTED_HESID']].drop_duplicates().reset_index(drop=True)
    subLevelVals['SEX'] = np.nan
    subLevelVals['MYDOB'] = np.nan
    subLevelVals['Mortality'] = np.nan
    
    for rowId in range(subLevelVals.shape[0]):
        for colName in ['SEX','MYDOB','Mortality']:
            r = np.random.randint(0,len(VALID_VALUES[colName])-1)
            subLevelVals.loc[rowId,colName] = str(VALID_VALUES[colName][r])
    
    df = df.merge(subLevelVals, on='ENCRYPTED_HESID')
    #pdb.set_trace()
    
    # %%
    # ############
    # ## Specify which are MI and which are matched controls...
    # ############
    
    subIds = df['ENCRYPTED_HESID'].drop_duplicates()
    
    
    # 5:1 ratio...
    lim = int(subIds.shape[0]/(params.CONTROL_CASE_RATIO+1))
    subIds_ami = subIds.iloc[:lim].reset_index()
    subIds_ctl = subIds.iloc[lim:lim*(params.CONTROL_CASE_RATIO+1)].reset_index()
    
    # only these subjects
    df = df.merge(pd.concat([subIds_ami['ENCRYPTED_HESID'],subIds_ctl['ENCRYPTED_HESID']]),how='right')
    

    #pdb.set_trace()
    # Also output MI/Control IDs for matching later on...
    
    
    #%% Create the matching files (5:1 ratio)
    df_ami = pd.DataFrame(index=range(subIds_ami.shape[0]),
                          columns=['hesid','myadmidate','amiID'])
    df_ctl = pd.DataFrame(index=range(subIds_ctl.shape[0]),
                          columns=['hesid','myadmidate','amiID'])
    df_ami['hesid'] = subIds_ami['ENCRYPTED_HESID']
    df_ctl['hesid'] = subIds_ctl['ENCRYPTED_HESID']
    
    # select matched date (first date after dropping duplicates)
    df_ami['myadmidate'] = df.loc[subIds_ami['index'],'MYEPISTART'].values
    df_ctl['myadmidate'] = np.repeat(df_ami['myadmidate'].values,params.CONTROL_CASE_RATIO)
    
    # matching ID
    df_ami['amiID'] = range(df_ami.shape[0])
    df_ctl['amiID'] = np.repeat(df_ami['amiID'].values,params.CONTROL_CASE_RATIO)

    #pdb.set_trace()
    
    # make sure the controls have an episode with the matched date...
    df.loc[subIds_ctl['index'],'MYEPISTART'] = df_ctl['myadmidate'].values
    
    # Make sure the MI subjects have an MI event on the matched date...
    df.loc[subIds_ami['index'],'DIAG_01'] = 'I21'
    
    
    #%% save csv files
    root = params.DIR_TMP
    
    df_ami['myadmidate'] = pd.to_datetime(df_ami['myadmidate'],format='%m%Y')
    df_ctl['myadmidate'] = pd.to_datetime(df_ctl['myadmidate'],format='%m%Y')
    
    df.to_csv(os.path.join(root,'raw_dummy.csv'),index=False,sep='|')
    df_ami.to_csv(os.path.join(root,'raw_dummy_amiIDs.csv'),index=False,sep=',')
    df_ctl.to_csv(os.path.join(root,'raw_dummy_ctlIDs.csv'),index=False,sep=',')


#%%

if __name__ == '__main__':
    main()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
