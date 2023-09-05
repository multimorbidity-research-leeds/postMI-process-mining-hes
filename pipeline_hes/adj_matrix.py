# -*- coding: utf-8 -*-
"""
Functions which create adjacency matrices corresponding to transitions
between diseases extracted from trajectories. These matrices are used to
create graphs (spaghetti plots).

@author: Chris Hayward
"""

import numpy as np
import pandas as pd

from pipeline_hes.params import params


def generate_dfg_adj_from_traces(traces):
    """From a list of traces (strings), generate the corresponding
    adjacency matrix."""

    # get the diagnosis codes for these traces
    codes = np.unique(','.join(sorted(list(traces.index))).split(','))
    
    # Append MI/Initial and CENSOR (in proper order)
    if params.AMI_INIT in codes:
        codes = np.append([params.AMI_INIT], codes[codes!=params.AMI_INIT])
    codes = np.append(codes[codes!=params.CENSOR_CODE], [params.CENSOR_CODE])
    
    adj = np.zeros((codes.shape[0],codes.shape[0]))
    
    # for each unique trace, fill-in the position in the matrix
    for trace in traces.index:
        diag_idxs = np.array([np.flatnonzero(codes==diag)[0] for diag in trace.split(',')])
        from_idx = diag_idxs[:-1]
        to_idx = diag_idxs[1:]
        adj[from_idx,to_idx] = adj[from_idx,to_idx] + traces.loc[trace,'count']

    return pd.DataFrame(adj, index=codes, columns=codes)
    

def get_directly_follows_counts(df):
    """Get the number of times a disease preceeds another disease based on
    MYEPISTART."""
    
    # sort first
    df.sort_values(['ENCRYPTED_HESID','MYEPISTART'], inplace=True)
    # sequential events for the same subject
    same_subject = (df.iloc[:-1]['ENCRYPTED_HESID']).values == (df.iloc[1:]['ENCRYPTED_HESID']).values
    
    # HESID, DIAG_before, DIAG_after
    diag_pairs = np.array([df.iloc[:-1]['ENCRYPTED_HESID'],
                           df.iloc[:-1]['DIAG_01'],
                           df.iloc[1:]['DIAG_01']])
    
    diag_pairs_follows = pd.DataFrame(data=diag_pairs.T[same_subject],
                                      columns=['ENCRYPTED_HESID','From','To'])
    
    # Drop duplicates (so only measure one unique pair per subject)    
    return diag_pairs_follows.drop_duplicates()[['From','To']].value_counts()


def generate_dfg_adj_from_df(df):
    """Generate the adjacency matrix from a dataframe containing episode
    dates and diagnosis codes"""
    
    diag_pairs_follows_counts = get_directly_follows_counts(df)

    # get the diagnosis codes for these traces
    c1 = diag_pairs_follows_counts.reset_index()['From'].drop_duplicates()
    c2 = diag_pairs_follows_counts.reset_index()['To'].drop_duplicates()
    c = pd.concat([c1,c2]).drop_duplicates()
    
    # Append MI/Initial and CENSOR (in proper order)
    if params.AMI_INIT in c:
        c = np.append([params.AMI_INIT], c[c!=params.AMI_INIT])
    c = np.append(c[c!=params.CENSOR_CODE], [params.CENSOR_CODE])

    # # Build adjacency matrix for this directly-follows graph
    adj = np.zeros((c.shape[0],c.shape[0]))
    dfg_from_idx = np.array([np.flatnonzero(c==k[0])[0]
                             for k in diag_pairs_follows_counts.index.values])
    dfg_to_idx = np.array([np.flatnonzero(c==k[1])[0]
                           for k in diag_pairs_follows_counts.index.values])
    
    adj[dfg_from_idx,dfg_to_idx] = diag_pairs_follows_counts

    return pd.DataFrame(adj, index=c, columns=c)
