# -*- coding: utf-8 -*-
"""
Keeps track of the number of individuals/episodes in the data as the data is
being pushed through the pipeline. Also constructs a network (networkx) which
depicts the individuals/episodes removed/ignored at each step.

@author: Chris Hayward
"""

import numpy as np
import pandas as pd
import pdb
import networkx
from pipeline_hes.params import params


def init_counts(hes_data, hesField='ENCRYPTED_HESID'):
    """Give an initial count of the number of individuals and episodes in the
    data."""

    print('Initialising counts...')
    
    patients = hes_data.loc[~hes_data['IS_CONTROL']]
    controls = hes_data.loc[hes_data['IS_CONTROL']]
    
    # exclude CENSOR events
    patients = patients.loc[patients['DIAG_01']!=params.CENSOR_CODE]
    controls = controls.loc[controls['DIAG_01']!=params.CENSOR_CODE]
    
    counts = pd.Series(dtype=float)
    counts['s_ami'] = patients[hesField].drop_duplicates().shape[0]
    counts['e_ami'] = patients.shape[0]
    
    counts['s_ctl'] = controls[hesField].drop_duplicates().shape[0]
    counts['e_ctl'] = controls.shape[0]
    
    counts['s_all'] = counts['s_ctl']+counts['s_ami']
    counts['e_all'] = counts['e_ctl']+counts['e_ami']
    
    # ####
    # Count the number of rows being left out of trajectories (ignored)
    # ####
    if 'IGNORE' in hes_data.columns:
    
        patients_notIgnored = patients.loc[~patients['IGNORE']]
        controls_notIgnored = controls.loc[~controls['IGNORE']]
        
        # Get data on not-ignored events
        counts['s_ami_ignore'] = patients_notIgnored[hesField].drop_duplicates().shape[0]
        counts['e_ami_ignore'] = patients_notIgnored.shape[0]
    
        counts['s_ctl_ignore'] = controls_notIgnored[hesField].drop_duplicates().shape[0]
        counts['e_ctl_ignore'] = controls_notIgnored.shape[0]

        counts['s_all_ignore'] = counts['s_ctl_ignore']+counts['s_ami_ignore']
        counts['e_all_ignore'] = counts['e_ctl_ignore']+counts['e_ami_ignore']
    
    print(counts)
    print('Done...')
    return counts


def my_round_prc(n):
    """Round numbers in a nice way."""
    x = np.format_float_positional(n, precision=2,
                                   fractional=False, trim='.')
    # If not a fraction of one, or rounded to 1DP, then just do a simple round
    if x.endswith('.') or str(x)[-2]=='.':
        x = str(np.round(n,2))
    return x


def _num_and_prc(nNew, nOld):
    """Get a string of a count change and its percentage change nicely
    rounded."""
    ch = nNew - nOld
    prefix=''
    if ch>0:
        prefix = '+'
    return '{0}{1:,} ({0}{2}%)'.format(prefix,ch,my_round_prc(100*ch/nOld))


def _num_and_prc_remaining_ignore(orig_count, notIgnored_now, notIgnored_before):
    """Get a string of the counts against the original data, and the previous
    step's data."""
    # Remaining is a fraction of the ORIGINAL amount
    xRemaining = '{:,} ({}%)'.format(
        notIgnored_now,
        my_round_prc(100*notIgnored_now/orig_count))
    # Change is the change since the last filter
    # (this matches the same approach as for the cleaning steps)
    xChange = _num_and_prc(notIgnored_now, notIgnored_before)
    return (xRemaining, xChange)


def append_to_network(msg,df_r,
                      oldCounts,counts,
                      countsSubField_ami,countsEventField_ami,
                      countsSubField_ctl,countsEventField_ctl):
    """Add a node to the network of cleaning/filtering steps. The nodes
    display counts of episodes and individuals remaining in the data as the data
    is pushed through the pipeline."""
    
    if not ('network' in oldCounts.index):
        G = networkx.DiGraph()
        pos = {}
        labels = {}

        # If First: (only showing the initial number of subs/events)
        pos[0] = (0,0)
        labels[0] ="""Initial patient and episode counts:
MI:      subjects {:,} | episodes {:,}
Control: subjects {:,} | episodes {:,}""".\
            format(oldCounts[countsSubField_ami],
                   oldCounts[countsEventField_ami],
                   oldCounts[countsSubField_ctl],
                   oldCounts[countsEventField_ctl]) 
        G.add_node(0)
        
    # If no change, dont add nodes
    elif ((counts[countsEventField_ami]-oldCounts[countsEventField_ami]) == 0) and \
        ((counts[countsEventField_ctl]-oldCounts[countsEventField_ctl]) == 0):
        G = oldCounts['network']
        labels = oldCounts['network_labels']
        pos = oldCounts['network_pos']
        
    else:
        G = oldCounts['network']
        labels = oldCounts['network_labels']
        pos = oldCounts['network_pos']

        nc = len(G.nodes)            

        # left alignment
        pos[nc] = (0,pos[nc-1][1]-1.1) # elbow
        pos[nc+1] = (.18,pos[nc][1])  # removed counts
        pos[nc+2] = (0,pos[nc][1]) # left boundary
        pos[nc+3] = (.4,pos[nc][1])  # right boundary
        pos[nc+4] = (0,pos[nc][1]-1) # resulting counts

        G.add_node(nc)
        G.add_node(nc+1)
        G.add_node(nc+2)
        G.add_node(nc+3)
        G.add_node(nc+4)
        
        #  elbow to change
        G.add_edge(nc,nc+1)
        
        # previous 'resulting counts' to this 'resulting counts'
        G.add_edge(nc-1,nc+4)

        # Label recording the CHANGE
        labels[nc+1] =\
"""Step {}
{}
MI:      subjects {} | episodes {}
Control: subjects {} | episodes {}""".\
            format(int(1+(nc-1)/5), msg,
                   df_r.loc['AMI','# Subjects change'],
                   df_r.loc['AMI','# Events change'],
                   df_r.loc['CONTROLS','# Subjects change'],
                   df_r.loc['CONTROLS','# Events change'])
            
        # Label recording the NEW AMOUNTS
        labels[nc+4] ="""MI:      subjects {:,} | episodes {:,}
Control: subjects {:,} | episodes {:,}""".\
            format(counts[countsSubField_ami],
                   counts[countsEventField_ami],
                   counts[countsSubField_ctl],
                   counts[countsEventField_ctl])  
        
    print('----------------------------------------------\n')
    print(G)

    counts['network'] = G
    counts['network_labels'] = labels
    counts['network_pos'] = pos


def append_counts(hes_data, msg, oldCounts=None, hesField='ENCRYPTED_HESID'):
    """Entry point - calls the relevant functions to count the number of
    episodes and individuals remaining in the data. Primarily for logging/
    debugging purposes."""
    

    if params.SKIP_SAVING_COUNTS:
        return oldCounts

    df_r = pd.DataFrame(data=[],columns=[
                            '# Subjects',
                            '# Subjects remaining',
                            '# Subjects change',
                            '# Events',
                            '# Events remaining',
                            '# Events change',
                            '# Events per subject',
                            '# Events per month'],
                        index=['AMI', 'CONTROLS', 'TOTAL'])    


    
    counts = init_counts(hes_data, hesField)

    # ####
    # Filtering steps (ignoring of rows)
    # ####
    if oldCounts is not None and 'e_ami_ignore' in oldCounts.index:
        
        # ############
        # MI
        # ############
        (n_rem_sub,n_ch_sub) = \
            _num_and_prc_remaining_ignore(counts['s_ami'],
                                          counts['s_ami_ignore'],
                                          oldCounts['s_ami_ignore'])
        (n_rem_ev,n_ch_ev) = \
            _num_and_prc_remaining_ignore(counts['e_ami'],
                                          counts['e_ami_ignore'],
                                          oldCounts['e_ami_ignore'])
        
        df_r.loc['AMI','# Subjects'] = '{:,}'.format(counts['s_ami'])
        df_r.loc['AMI','# Subjects remaining'] = n_rem_sub
        df_r.loc['AMI','# Subjects change'] = 0
        df_r.loc['AMI','# Events'] = '{:,}'.format(counts['e_ami'])
        df_r.loc['AMI','# Events remaining'] = n_rem_ev
        df_r.loc['AMI','# Events change'] = n_ch_ev


        # ############
        # Controls
        # ############
        (n_rem_sub,n_ch_sub) = \
            _num_and_prc_remaining_ignore(counts['s_ctl'],
                                          counts['s_ctl_ignore'],
                                          oldCounts['s_ctl_ignore'])
        (n_rem_ev,n_ch_ev) = \
            _num_and_prc_remaining_ignore(counts['e_ctl'],
                                          counts['e_ctl_ignore'],
                                          oldCounts['e_ctl_ignore'])
        
        df_r.loc['CONTROLS','# Subjects'] = '{:,}'.format(counts['s_ctl'])
        df_r.loc['CONTROLS','# Subjects remaining'] = n_rem_sub
        df_r.loc['CONTROLS','# Subjects change'] = 0
        df_r.loc['CONTROLS','# Events'] = '{:,}'.format(counts['e_ctl'])
        df_r.loc['CONTROLS','# Events remaining'] = n_rem_ev
        df_r.loc['CONTROLS','# Events change'] = n_ch_ev

        # ############
        # Total
        # ############
        (n_rem_sub,n_ch_sub) = \
            _num_and_prc_remaining_ignore(counts['s_all'],
                                          counts['s_all_ignore'],
                                          oldCounts['s_all_ignore'])
        (n_rem_ev,n_ch_ev) = \
            _num_and_prc_remaining_ignore(counts['e_all'],
                                          counts['e_all_ignore'],
                                          oldCounts['e_all_ignore'])

        df_r.loc['TOTAL','# Subjects'] = '{:,}'.format(counts['s_all'])
        df_r.loc['TOTAL','# Subjects remaining'] = n_rem_sub
        df_r.loc['TOTAL','# Subjects change'] = n_ch_sub
        df_r.loc['TOTAL','# Events'] = '{:,}'.format(counts['e_all'])
        df_r.loc['TOTAL','# Events remaining'] = n_rem_ev
        df_r.loc['TOTAL','# Events change'] = n_ch_ev

        append_to_network(msg,df_r,
                          oldCounts,counts,
                          's_ami','e_ami_ignore',
                          's_ctl','e_ctl_ignore')

    # ####
    # Cleaning steps (permanent removal of rows)
    # ####
    elif oldCounts is not None:
        # Cleaning steps
        df_r.loc['AMI','# Subjects'] = '{:,}'.format(counts['s_ami'])
        df_r.loc['AMI','# Events'] = '{:,}'.format(counts['e_ami'])
        df_r.loc['AMI','# Subjects change'] = \
            _num_and_prc(counts['s_ami'], oldCounts['s_ami'])
        df_r.loc['AMI','# Events change'] = \
            _num_and_prc(counts['e_ami'], oldCounts['e_ami'])

        df_r.loc['CONTROLS','# Subjects'] = '{:,}'.format(counts['s_ctl'])
        df_r.loc['CONTROLS','# Events'] = '{:,}'.format(counts['e_ctl'])
        df_r.loc['CONTROLS','# Subjects change'] = \
            _num_and_prc(counts['s_ctl'], oldCounts['s_ctl'])
        df_r.loc['CONTROLS','# Events change'] = \
            _num_and_prc(counts['e_ctl'], oldCounts['e_ctl'])
            
        df_r.loc['TOTAL','# Subjects'] = '{:,}'.format(counts['s_all'])
        df_r.loc['TOTAL','# Events'] = '{:,}'.format(counts['e_all'])
        df_r.loc['TOTAL', '# Subjects change'] = \
            _num_and_prc(counts['s_all'], oldCounts['s_all'])
        df_r.loc['TOTAL', '# Events change'] = \
            _num_and_prc(counts['e_all'], oldCounts['e_all'])

        append_to_network(msg,df_r,
                          oldCounts,counts,
                          's_ami','e_ami',
                          's_ctl','e_ctl')

    return counts
