# -*- coding: utf-8 -*-
"""
Set the global parameters.
Most parameter values are read in from pipeline.ini.

The parameters listed here tweak the processing performed by the pipeline.
e.g. changing the percentage threshold used to exclude rare diseases from
trajectories (RARE_EVENT_PRC_THRESH in pipeline.ini)

@author: Chris Hayward
"""

import os
import datetime
from configparser import ConfigParser, ExtendedInterpolation

class Params():
    """Pipeline parameters."""

    # Default values
    def __init__(self):
        """Set the parameter values (values of the attributes for this
        Params object)."""

        NOW_STR = datetime.datetime.now().strftime('%d-%m-%y_%H-%M-%S')
        
        # read in params from .ini
        config = ConfigParser(interpolation=ExtendedInterpolation())
        config.read(r'm:/medchaya/repos/GitHub/hes/src/pipeline_hes/pipeline.ini')
        
        self.SKIP_SAVING_COUNTS = \
            config['DEFAULT'].getboolean('SKIP_SAVING_COUNTS')
            
        self.AMI_RANGE = config['DEFAULT']['AMI_RANGE'].split(',')

        self.SPELL_ORDER_RANDOM_SEED = \
            int(config['DEFAULT']['SPELL_ORDER_RANDOM_SEED'])

        self.INITIAL_EPISODE_RANDOM_SEED = \
            int(config['DEFAULT']['INITIAL_EPISODE_RANDOM_SEED'])
                    
        if config['DEFAULT']['LIMIT_TO_TIME_IGNORE_THESE'] in ['<6m', '>6m']:
            self.LIMIT_TO_TIME_IGNORE_THESE = \
                config['DEFAULT']['LIMIT_TO_TIME_IGNORE_THESE']
        else:
            self.LIMIT_TO_TIME_IGNORE_THESE = False
            
        self.CHAPTER_HEADINGS_USE_GRANULAR = \
            config['DEFAULT'].getboolean('CHAPTER_HEADINGS_USE_GRANULAR')
            
        self.RARE_EVENT_PRC_THRESH = \
            float(config['DEFAULT']['RARE_EVENT_PRC_THRESH'])
            
        self.IGNORE_REPEATED_CHRONIC = \
            config['DEFAULT'].getboolean('IGNORE_REPEATED_CHRONIC')
            
        self.IGNORE_TOO_CLOSE = \
            config['DEFAULT'].getboolean('IGNORE_TOO_CLOSE')
            
        self.USE_SEC_DIAGS_IN_TRACES = \
            config['DEFAULT'].getboolean('USE_SEC_DIAGS_IN_TRACES')
            
        self.CTL_TRACE_COUNTS_LOWER_LIMIT = \
            int(config['DEFAULT']['CTL_TRACE_COUNTS_LOWER_LIMIT'])
            
        self.AMI_TRACE_COUNTS_PRC_THRESHOLD = \
            float(config['DEFAULT']['AMI_TRACE_COUNTS_PRC_THRESHOLD'])
            
        self.CHECK_PROP_HAZ_ASSUM = \
            config['DEFAULT'].getboolean('CHECK_PROP_HAZ_ASSUM')
            
        self.WITHIN_SUBJECT_SINGLE_VALUE_THRESHOLD = \
            int(config['DEFAULT']['WITHIN_SUBJECT_SINGLE_VALUE_THRESHOLD'])
            
        self.USE_CORRECTED_RR_HR = \
            config['DEFAULT'].getboolean('USE_CORRECTED_RR_HR')
            
        if config['DEFAULT']['ONLY_ONE_SEX'] in ['F', 'M']:
            self.ONLY_ONE_SEX = config['DEFAULT']['ONLY_ONE_SEX']
        else:
            self.ONLY_ONE_SEX = False
            
        ## Other params

        self.R = config['DEFAULT']['R']
   
        self.CENSOR_DATE = config['DEFAULT']['CENSOR_DATE']
        
        self.CONTROL_CASE_RATIO = \
            int(config['DEFAULT']['CONTROL_CASE_RATIO'])
            
        self.MAX_EVENT_DIST_TOO_CLOSE_DAYS = \
            int(config['DEFAULT']['MAX_EVENT_DIST_TOO_CLOSE_DAYS'])
            
        # ===========
        # DIRECTORIES
        # ===========
        self.DIR_RAW_DATA = config['DEFAULT']['DIR_RAW_DATA']
        self.DIR_CHECKPOINTS = config['DEFAULT']['DIR_CHECKPOINTS']
        self.DIR_RESULTS = os.path.join(config['DEFAULT']['DIR_RESULTS'],
                                        '{}_{}'.format(NOW_STR,self.R))
        self.DIR_TMP = config['DEFAULT']['DIR_TMP']
        
        # ===========
        # FILES
        # ===========
        self.FILE_CHAPTER_TEXT_HEADINGS = \
            config['DEFAULT']['FILE_CHAPTER_TEXT_HEADINGS']
        self.FILE_CHAPTER_TEXT_HEADINGS_GRANULAR = \
            config['DEFAULT']['FILE_CHAPTER_TEXT_HEADINGS_GRANULAR']        
        self.FILE_CSV_ACUTE_CHRONIC = config['DEFAULT']['FILE_CSV_ACUTE_CHRONIC']        
        self.FILE_HESID_PATIENT = config['DEFAULT']['FILE_HESID_PATIENT']
        self.FILE_HESID_CONTROLS = config['DEFAULT']['FILE_HESID_CONTROLS']
        self.FILE_LOG = config['DEFAULT']['FILE_LOG']
        self.FILES_HES_CSV = config['DEFAULT']['FILES_HES_CSV']
        
        # ===========
        # Additional non-.ini parameters
        # ===========
        self.AMI_CODE = 'MI'
        self.AMI_INIT = 'INIT'
        self.CHAPTER_NO_MATCH = 'nomatch'
        self.DUMMY_INIT_CODE = 'Dummy'
        # Primary and Secondary diagnosis columns
        self.DIAG_COLS = ['DIAG_{:02d}'.format(x) for x in range(1,21)]
        self.SEC_DIAG_COLS = ['DIAG_{:02d}'.format(x) for x in range(2,21)]        
        # Chronic/acute indicator columns for each diagnosis
        self.ACUTE_COLS = ['ACUTE_{:02d}'.format(x) for x in range(1,21)]
        self.SEC_ACUTE_COLS = ['ACUTE_{:02d}'.format(x) for x in range(2,21)]
        # Sex codes in HES csv
        self.SEX_MALE = 1
        self.SEX_FEMALE = 2
        # The placeholder name to use as the first diagnosis for trajectories
        self.AMI_INIT_PLOT = 'MI/Initial'
        # The placeholder name to use as the censor 'diagnosis' for trajectories
        self.CENSOR_CODE = 'Censor' 

        
    def check_params_valid_values(self):
        """A sanity check - ensure that the parameters are of the correct
        type."""
        
        if not isinstance(self.SKIP_SAVING_COUNTS,bool):
            raise Exception('Invalid param type SKIP_SAVING_COUNTS')
        
        if not isinstance(self.AMI_RANGE,list):
            raise Exception('Invalid param type AMI_RANGE')
        
        if not isinstance(self.SPELL_ORDER_RANDOM_SEED,int):
            raise Exception('Invalid param type SPELL_ORDER_RANDOM_SEED')
            
        if not isinstance(self.INITIAL_EPISODE_RANDOM_SEED,int):
            raise Exception('Invalid param type INITIAL_EPISODE_RANDOM_SEED')

        if self.LIMIT_TO_TIME_IGNORE_THESE not in (False,'<6m', '>6m'):
            raise Exception('Invalid param type LIMIT_TO_TIME_IGNORE_THESE')

        if not isinstance(self.CHAPTER_HEADINGS_USE_GRANULAR,bool):
            raise Exception('Invalid param type CHAPTER_HEADINGS_USE_GRANULAR')
        
        if not isinstance(self.RARE_EVENT_PRC_THRESH,float):
            raise Exception('Invalid param type RARE_EVENT_PRC_THRESH')
        
        if not isinstance(self.IGNORE_REPEATED_CHRONIC,bool):
            raise Exception('Invalid param type IGNORE_REPEATED_CHRONIC')

        if not isinstance(self.IGNORE_TOO_CLOSE,bool):
            raise Exception('Invalid param type IGNORE_TOO_CLOSE')

        if not isinstance(self.USE_SEC_DIAGS_IN_TRACES,bool):
            raise Exception('Invalid param type USE_SEC_DIAGS_IN_TRACES')
        
        if not isinstance(self.CTL_TRACE_COUNTS_LOWER_LIMIT,int):
            raise Exception('Invalid param type CTL_TRACE_COUNTS_LOWER_LIMIT')
            
        if not isinstance(self.AMI_TRACE_COUNTS_PRC_THRESHOLD,float):
            raise Exception('Invalid param type AMI_TRACE_COUNTS_PRC_THRESHOLD')

        if not isinstance(self.CHECK_PROP_HAZ_ASSUM,bool):
            raise Exception('Invalid param type CHECK_PROP_HAZ_ASSUM')
            
        if not isinstance(self.WITHIN_SUBJECT_SINGLE_VALUE_THRESHOLD,int):
            raise Exception('Invalid param type WITHIN_SUBJECT_SINGLE_VALUE_THRESHOLD')

        if not isinstance(self.USE_CORRECTED_RR_HR,bool):
            raise Exception('Invalid param type USE_CORRECTED_RR_HR')
            
        if not isinstance(self.CONTROL_CASE_RATIO,int):
            raise Exception('Invalid param type CONTROL_CASE_RATIO')

        if not isinstance(self.MAX_EVENT_DIST_TOO_CLOSE_DAYS,int):
            raise Exception('Invalid param type MAX_EVENT_DIST_TOO_CLOSE_DAYS')


# Params object
params = Params()

# Last step - check all params are the correct type
params.check_params_valid_values()

