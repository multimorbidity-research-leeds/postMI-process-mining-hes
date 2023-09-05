# -*- coding: utf-8 -*-
"""
First step of the pipeline - converts CSV files to parquet files.
Loads all 9 raw HES data files (2008-2017), creating one parquet file for
each of these. Stores only the fields that we're interested in.
The parquet files are loaded by 'load_parquet.py'

@author: Chris Hayward
"""

import random
import pdb
import time
import concurrent.futures
import os

import numpy as np
import pandas as pd
from pipeline_hes.params import params

TYPE_PAIRS = {
            'DIAG_01':'category',
            'DIAG_02':'category','DIAG_03':'category',
            'DIAG_04':'category','DIAG_05':'category',
            'DIAG_06':'category','DIAG_07':'category',
            'DIAG_08':'category','DIAG_09':'category',
            'DIAG_10':'category','DIAG_11':'category',
            'DIAG_12':'category','DIAG_13':'category',
            'DIAG_14':'category','DIAG_15':'category',
            'DIAG_16':'category','DIAG_17':'category',
            'DIAG_18':'category','DIAG_19':'category',
            'DIAG_20':'category',
            'ADMIMETH':'category',
            'DISMETH':'category','PROCODE':'category',
            'MYADMIDATE':'object',
            'DISDATE':'object','MYDOB':'object', 
            'MYEPIEND':'object','MYEPISTART':'object',
            'EPIDUR':'float32',
            'IMD04':'float32',
            'SURVIVALTIME':'float32',
            'EPIORDER':'uint8',
            'EPISTAT':'uint8',
            'SEX':'uint8',
            'Mortality':'uint8',
            'PROVSPNOPS':'object',
            'ENCRYPTED_HESID':'object'
            }

DATE_FIELDS = ['MYADMIDATE','MYDOB','MYEPIEND','MYEPISTART','DISDATE']

DATE_FORMAT = '%m%Y'


def to_parquet(df,cFileBase,randStr):
    """Converts a dataframe (from a single HES CSV file) to a parquet file."""
    fname = os.path.join(params.DIR_CHECKPOINTS,cFileBase+'_{}_.gzip'.format(randStr))
    print('convert to parquet (rows={})... saving in: {}'.format(df.shape[0],fname))
    df.to_parquet(fname,compression='gzip')
    return fname


def load_hes(fullLoc):
    """Loads a HES CSV file and converts columns to smaller datatypes."""

    chunksize = 10 ** 5
        
    df = []
    print(fullLoc)
    t = time.time()
    for i,chunk in enumerate(pd.read_csv(fullLoc, \
                                         engine='c', \
                                         chunksize=chunksize, \
                                         dtype=str, \
                                         usecols=TYPE_PAIRS.keys(), \
                                         sep='|')):
        print(i*chunksize)
        print('chunk length: '+str(len(chunk)))
        df.append(chunk)
        print('Appended: '+str(len(chunk)))

        
    print('Read CSV elapsed time: {}'.format(time.time() - t))
    print('concat...')
    df = pd.concat(df)
    print('All chunks got. Total length: {}'.format(len(df)))
    m = df.memory_usage()
    print('Memory usage (STR):\n{}\nTotal:{}'.format(m,np.sum(m)))

    df = df.astype(TYPE_PAIRS)

    # Convert date:
    for dfield in DATE_FIELDS:
        df[dfield] = pd.to_datetime(df[dfield],format=DATE_FORMAT)
    
    m = df.memory_usage()
    print('Memory usage (after type conv.):\n{}\nTotal:{}'.format(m,np.sum(m)))
    
    return df


def output_head(in_name):
    """Just for debugging, saves the first few rows of a CSV file."""
    out_name = os.path.join(params.DIR_CHECKPOINTS,'head.csv')
    print("Simple head[%s] -> %s ..." % (in_name, out_name))
    with open(in_name, 'rt') as f_in, open(out_name, 'wt') as f_out:
        for _ in range(1000):
            f_out.write(f_in.readline().replace('|',','))


def main():
    """Entry function. Sets up a random string to be used in the filename
    for the saved parquet files."""

    # get all the csv files
    csvFiles = list(map(lambda x: x.strip('\n'), params.FILES_HES_CSV.split(',')))

    # random ID for identifying the output files
    randStr = str(round(random.random(),7)).split('.')[1]
    # Write out the head of the file
    output_head(os.path.join(params.DIR_RAW_DATA,csvFiles[0]))
    print(randStr)

    # In parallel - load each CSV file and save each as a smaller parquet file
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        for _ in executor.map(
                lambda x: to_parquet(load_hes(
                            os.path.join(params.DIR_RAW_DATA,x)),x,randStr),
                csvFiles):
            pass
        
    return randStr


if __name__=='__main__':
    main()

