# -*- coding: utf-8 -*-
"""
The initial script for the disease trajectory processing pipeline.

@author: Chris Hayward
"""

import os
import pandas as pd
import pdb
import datetime
from matplotlib import pyplot as plt

from pipeline_hes import load_parquet
from pipeline_hes import csv_to_parquet
from pipeline_hes import clean_hes
from pipeline_hes import filter_hes
from pipeline_hes import traces_hes
from pipeline_hes.params import params


def main():
    """Pipeline start."""
    # -----------
    # Extract columns of interest from raw HES csv files
    # -----------
   # runID = csv_to_parquet.main()
    
    # The unique ID for intermediate files
   # params.R = runID
    
    # -----------
    # Extract individuals of interest (MI and Control cohorts)
    # -----------
    #load_parquet.main()

    # --------------
    # Clean the hes data (removal of subjects and episodes which are not
    # useful in the creation of trajectories)
    # --------------
    clean_hes.main()
    
    # --------------    
    # Filter the data - select which episodes to include in trajectories.
    # --------------
    filter_hes.main()

    
    #%%
    
    # --------------
    # Generate the trajectories
    # Calculate relative risks, and hazard ratios for trajectories.
    # Continues towards plotting the figures.
    # --------------
    traces_hes.main(doPlots=True)


if __name__=='__main__':
    
    # Show full dataframes
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('mode.chained_assignment','raise')
    
    # folder to store files specific to selected params
    try:
        os.mkdir(params.DIR_CHECKPOINTS)
    except Exception:
        pass
    
    # Create a log file to store the counts of episodes and subjects during
    # the pipeline
    with open(params.FILE_LOG,'a') as f:
        f.write('\n{}\nexecute: main.py --- {}\n'.\
                format('$'*60,datetime.datetime.now()))
        
    print('main.py')
    plt.close('all')
    main()
    