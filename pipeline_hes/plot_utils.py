# -*- coding: utf-8 -*-
"""
Utility functions for plotting manuscript figures.

@author: Chris Hayward
"""

import pathlib
import numpy as np
import pandas as pd
import pdb
import re
import os

from pipeline_hes import parse_chapters
from pipeline_hes.params import params

from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib import cm

import networkx


def create_fig_folder():
    """Creates the results directory (stores figures etc)."""
    pathlib.Path(params.DIR_RESULTS).mkdir(exist_ok=True)


def saveFig(figName):
    """Saves the current matplotlib figure in the results directory."""
    create_fig_folder()
    plt.savefig(os.path.join(params.DIR_RESULTS,'{}.png'.format(figName)),
                pad_inches=.1,bbox_inches='tight')


def saveFig_pdf(figName):
    """Saves the current matplotlib figure in the results directory (PDF)."""
    create_fig_folder()
    plt.savefig(os.path.join(params.DIR_RESULTS,'{}.pdf'.format(figName)),
                pad_inches=.1,bbox_inches='tight')


def get_chapters_low_level():
    """Get low-level chapter details for use in plots."""
    
    # get in the right order (for chapter display in rr/hr fig)
    chapter_mapping = pd.Series({params.AMI_INIT_PLOT: params.AMI_INIT_PLOT})
    chapter_mapping = pd.concat([chapter_mapping, parse_chapters.get_chapter_mapping_low_level()])
    chapter_mapping[params.CENSOR_CODE] = params.CENSOR_CODE
    
    # label to index
    labelsIdx = {chapter_name:i
                 for i,chapter_name in enumerate(chapter_mapping.drop_duplicates().values)}    
        
    # Set 3 has 12 unique colours
    colours1 = np.array(cm.get_cmap('Set2').colors)
    colours2 = np.array(cm.get_cmap('Dark2').colors)
    colours = np.concatenate([colours1,colours2],axis=0)[:len(labelsIdx)]
    
    
    # AMI/Initial and Censor
    colours[labelsIdx[params.AMI_INIT_PLOT],:] = .9
    colours[labelsIdx[params.CENSOR_CODE],:] = .9
    
    textCol = np.zeros((len(colours),3), dtype=object)    
    rC = colours[:,0] * .299
    gC = colours[:,1] * .587
    bC = colours[:,2] * .114
    useWhite = (rC + gC + bC) <= 150/255
    textCol[useWhite] = 1
    
    return chapter_mapping, labelsIdx, colours, textCol
        

def get_chapters_high_level():
    """Get high-level chapter details for use in plots."""
    
    # get in the right order (for chapter display in rr/hr fig)
    chapter_mapping = pd.Series({params.AMI_INIT_PLOT: params.AMI_INIT_PLOT})
    chapter_mapping = pd.concat([chapter_mapping, parse_chapters.get_chapter_mapping_high_level()])
    chapter_mapping[params.CENSOR_CODE] = params.CENSOR_CODE
    
    # label to index
    labelsIdx = {chapter_name:i
                 for i,chapter_name in enumerate(chapter_mapping.drop_duplicates().values)}    
    
    # Set 3 has 12 unique colours
    colours1 = np.array(cm.get_cmap('Set2').colors)
    colours2 = np.array(cm.get_cmap('Dark2').colors)
    colours = np.concatenate([colours1,colours2],axis=0)[:len(labelsIdx)]
    
    # AMI/Initial and Censor
    colours[labelsIdx[params.AMI_INIT_PLOT],:] = .9
    colours[labelsIdx[params.CENSOR_CODE],:] = .9
    
    # bar text colours (white or black)
    textCol = np.zeros((len(colours),3), dtype=object)
    
    rC = colours[:,0] * .299
    gC = colours[:,1] * .587
    bC = colours[:,2] * .114
    
    useWhite = (rC + gC + bC) <= 150/255
    textCol[useWhite] = 1
    
    return chapter_mapping, labelsIdx, colours, textCol


# INIT -> MI/non-MI
def convert_init_event(df_rr_hr):
    """Replace the 'initial diagnosis' placeholder with a name which will
    be shown in figures."""
    df_rr_hr.index = df_rr_hr.index.map(lambda x: re.sub(params.AMI_INIT, params.AMI_INIT_PLOT, x))


def plot_flowchart(name, counts):
    """Display the network of pipeline steps, with counts of the number of
    subjects and episodes removed/excluded at each step."""
    # if not recording counts, dont do anything
    if not ('network_pos' in counts.index):
        return
    
    figW = 350
    figH = max(20,len(counts['network_labels']) * 20)

    px = 1/plt.rcParams['figure.dpi']
    fig = plt.figure(tight_layout=True,figsize=(figW*px,figH*px),dpi=400)
    gs = gridspec.GridSpec(1,1)
    ax = fig.add_subplot(gs[0,0])
    pos = counts['network_pos']
    networkx.draw_networkx_nodes(counts['network'],
                                 pos,
                                 node_size=1,
                                 alpha=0,
                                 ax=ax,)

    node_sizes = np.ones(len(counts['network']))*200
    node_sizes[2::5] = 500 # bring the arrow back
    networkx.draw_networkx_edges(counts['network'],
                                 pos,
                                 arrowsize=5,
                                 node_size=node_sizes,
                                 width=.5,
                                 ax=ax,)
    
    labs = {k:v.expandtabs() for k,v in counts['network_labels'].items()}
    networkx.draw_networkx_labels(counts['network'],
                                  pos,
                                  labels=labs,
                                  font_size=3.5,
                                  font_family='monospace',
                                  horizontalalignment='left',
                                  ax=ax,
                                  bbox=dict(fc='white',lw=.3,mutation_aspect=.2))

    saveFig('flowchart_{}'.format(name))

    plt.show()
