# -*- coding: utf-8 -*-
"""
Plotting functions for the RMST trajectory figure (trajectories as blocks of
ICD-10 chapters, scaled by the RMST for each cohort).

@author: Chris Hayward
"""

import numpy as np
import pandas as pd
import pdb

from pipeline_hes import parse_chapters
from pipeline_hes import plot_utils
from pipeline_hes.params import params
from sigfig import round as sigfig_round

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

RR_FIG_TITLE_FONTSIZE = 9
RR_FIG_TITLE_FONTSIZE_LETTER = 11


def display_categories(fig,gs,labelsIdx,colours,textCol):
    """Display the ICD-10 chapter names and a brief title for each."""

    ax = fig.add_subplot(gs)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Plot chapter headings
    plt.xticks(ticks=[0],labels=[' '])
    ax.tick_params(width=0)
    plt.yticks(ticks=[],labels=[])
    xpos = -1
    numOnCol = 20
    
    code_desc_dict = parse_chapters.get_codes_description_short_desc()
    code_desc_dict['MI/Non-MI'] = code_desc_dict.pop('MI/Initial')

    ipos = 0
    for (lab,i) in labelsIdx.items():
        barCol = colours[i,:3]
        ypos = numOnCol - np.mod(ipos,numOnCol)
        ipos = ipos+1
        if ypos==numOnCol:
            xpos += 1
        ax.barh(ypos, 1, left=xpos, height=.9, color=barCol, zorder=3)
        
        ax.text(xpos+.5, ypos+.2, lab, color=textCol[i],
                zorder=3, ha='center', fontsize=9, va='center')
        
        ax.text(xpos+.5, ypos-.2, code_desc_dict[lab], color=textCol[i],
                zorder=3, ha='center', fontsize=6.5, va='center')

    #plt.xlabel('MI = ['+','.join(params.AMI_RANGE)+']', fontsize=8)
    plt.title('ICD-10\nchapters',fontsize=RR_FIG_TITLE_FONTSIZE)
    ax.margins(0.1,0)
    ax.tick_params(width=0, length=0, which='major')
    ax.tick_params(width=0, length=0, which='minor')
    

def display_traces_time_scaled(fig,gs,df_rmst,chapter_mapping,
                               labelsIdx,colours,textCol,contFlag):
    """Display the trajectories as horizontal bars, scaled in length by the
    RMST for each cohort."""
    # matrix of split traces
    ynames = df_rmst.index
    maxSeqLen = np.max([len(x.split(',')) for x in ynames])
    ynames_split = np.zeros((ynames.shape[0],maxSeqLen),dtype=object)
    ynames_split[:] = ''
    for i,seq in enumerate(ynames):
        seqSplit = seq.split(',')
        for j,s in enumerate(seqSplit):
            ynames_split[i,j] = s

    trace_fontsize = min(8, 8*np.log(1+(60/(ynames.shape[0]*2))))

    seqLens = [len(x.split(',')) for x in ynames]
    
    # draw the coloured bars
    ax = fig.add_subplot(gs)

    y = 0
    for i,seq in enumerate(ynames):

        timeScale1_perblock = (df_rmst.loc[seq,'RMST1'])/(seqLens[i])
        timeScale0_perblock = (df_rmst.loc[seq,'RMST0'])/(seqLens[i])

        if np.isnan(timeScale1_perblock) or np.isnan(timeScale0_perblock):
            # dummy bar
            ax.barh(y+.5, 1, left=0,
                    height=1.9, color='white', zorder=0)
        else:
            # TAU as a background shaded bar
            ax.barh(y+.5, df_rmst.loc[seq,'tau'], left=0,
                    height=1.9, color='#e6f2ff', zorder=0)#e6f2ff

            zorders = 1
            for x in range(maxSeqLen):
                if len(ynames_split[i,x]) > 0:
                    # low/high res chapter -> chapter mapping (to low res) -> index -> colour
                    cIdx = labelsIdx[chapter_mapping[ynames_split[i,x]]]
                    showName = ynames_split[i,x]

                    # CONTROLS (IS_PATIENT=0)
                    if ynames_split[i,x] == params.AMI_INIT_PLOT:
                        showName = 'Non-MI'
                    textPosX = (timeScale0_perblock*x)+\
                        max(timeScale0_perblock/2,.1*len(showName))
                        
                    ax.barh(y, timeScale0_perblock-.025, left=timeScale0_perblock*x,
                            height=.9, color=colours[cIdx,:3], zorder=zorders)
                    if showName==params.CENSOR_CODE:
                        ax.barh(y, timeScale0_perblock, left=timeScale0_perblock*x,
                                height=.9, color='#aaaaaa', zorder=zorders-1, linewidth=0)
                    ax.text(textPosX, y,
                            showName, color=textCol[cIdx], zorder=zorders,
                            ha='center',
                            fontsize=trace_fontsize, va='center')
                    
                    # AMI (IS_PATIENT=1)
                    if ynames_split[i,x] == params.AMI_INIT_PLOT:
                        showName = 'MI'                        
                    textPosX = (timeScale1_perblock*x)+\
                        max(timeScale1_perblock/2,.1*len(showName))

                    ax.barh(y+1, timeScale1_perblock-.025, left=timeScale1_perblock*x,
                            height=.9, color=colours[cIdx,:3], zorder=zorders)
                    if showName==params.CENSOR_CODE:
                        ax.barh(y+1, timeScale1_perblock, left=timeScale1_perblock*x,
                                height=.9, color='#aaaaaa', zorder=zorders-1, linewidth=0)
                    ax.text(textPosX, y+1,
                            showName, color=textCol[cIdx], zorder=zorders,
                            ha='center',
                            fontsize=trace_fontsize, va='center')
        
                ## keeps subsequent blocks on top of the previous
                zorders = zorders + 1
        y = y + 2.5

    ytickpos = np.arange(0,df_rmst.shape[0]*2.5,2.5)+1.75
    plt.yticks(ticks=ytickpos,
               labels=['' for _ in ytickpos])
    ax.tick_params(axis='y', which='major', left=False)
    ax.tick_params(axis='x', which='major', color='#999999')
    ax.yaxis.grid(which='major', color='#dddddd')
    ax.xaxis.grid(which='major', color='#eeeeee')
    # needed to put the grid lines behind everything else
    ax.set_axisbelow(True)
    plt.xlim([None,10])
    
    ax.spines['bottom'].set_color('#999999')
    ax.spines['left'].set_color('#999999')
    ax.spines['right'].set_color('#999999')
    ax.spines['top'].set_color('#999999')

    if contFlag:        
        plt.title('RMST-scaled disease trajectories (cont.)', fontsize=RR_FIG_TITLE_FONTSIZE)
    else:
        plt.title('RMST-scaled disease trajectories', fontsize=RR_FIG_TITLE_FONTSIZE)
    plt.xlabel('RMST (years)', fontsize=10)
    ax.set_ymargin(0.01)



def display_values(fig,gs_pos,df_rmst):
    """Display the RMST values alongside each trajectory, and the P-value
    from a comparison of RMST between cohorts."""
    
    ynames = df_rmst.index
    textSize = min(6, 6*np.log(1+(60/ynames.shape[0])))
    
    ax = fig.add_subplot(gs_pos)

    y = 0
    for i in range(len(ynames)):

        # RMST arm=0 (IS_PATIENT=0)
        ax.barh(y, 10, height=.9, zorder=0, color='white') # color=default_cmap(1)
        txtC_n = '{} ({}-{})'.format(np.round(df_rmst.iloc[i]['RMST0'],2),
                                     np.round(df_rmst.iloc[i]['RMST0_CIl'],2),
                                     np.round(df_rmst.iloc[i]['RMST0_CIu'],2)) \
            if ~df_rmst.isna().iloc[i]['RMST0'] else 'n/a'
        ax.text(.5, y, '{}'.format(txtC_n), color='black',
                zorder=3, ha='left', va='center', fontsize=textSize)
        
        
        # RMST arm=1 (IS_PATIENT=1)
        ax.barh(y+1, 10, height=.9, zorder=0, color='white') # color=default_cmap(1)
        txtC_n = '{} ({}-{})'.format(np.round(df_rmst.iloc[i]['RMST1'],2),
                                     np.round(df_rmst.iloc[i]['RMST1_CIl'],2),
                                     np.round(df_rmst.iloc[i]['RMST1_CIu'],2)) \
            if ~df_rmst.isna().iloc[i]['RMST1'] else 'n/a'
        ax.text(.5, y+1, '{}'.format(txtC_n), color='black',
                zorder=3, ha='left', va='center', fontsize=textSize)
        
        
        # # TAU (minimum cut-off)
        # txtC_n = 'Tau={}'.format(np.round(df_rmst.iloc[i]['tau'],2)) \
        #     if ~df_rmst.isna().iloc[i]['tau'] else 'n/a'
        # ax.text(10, y+1, '{}'.format(txtC_n), color='black',
        #         zorder=3, ha='right', va='center', fontsize=textSize)
        
        # P
        if ~df_rmst.isna().iloc[i]['p_diff_unadj']:
            p_rounded = sigfig_round(df_rmst.iloc[i]['p_diff_unadj'],sigfigs=2)
            p_rounded = 'P<0.001' if p_rounded<0.001 else 'P={}'.format(p_rounded)
            fweight = 'bold' if df_rmst.iloc[i]['p_diff_unadj']<=0.05 else 'normal'
        else:
            p_rounded = 'n/a'
            fweight = 'normal'
        ax.text(10, y, p_rounded, color='black',
                zorder=3, ha='right', va='center', fontsize=textSize, weight=fweight)
        
        y = y + 2.5

    ytickpos = np.arange(0,df_rmst.shape[0]*2.5,2.5)+1.75
    plt.yticks(ticks=ytickpos,
               labels=['' for _ in ytickpos])
    ax.tick_params(axis='y', which='major', left=False)
    ax.yaxis.grid(zorder=0,which='major', color='#dddddd')

    plt.xticks(ticks=[])
    plt.title('RMST (95% CI)', fontsize=RR_FIG_TITLE_FONTSIZE)

    ax.spines['bottom'].set_color('#999999')
    ax.spines['left'].set_color('#999999')
    ax.spines['right'].set_color('#999999')
    ax.spines['top'].set_color('#999999')
    ax.set_ymargin(0.01)


def plot_rr_fig_time_scaled(df_rr):
    """Main entry function for plotting the trajectories scaled in length by their RMST."""
    
    # convert INIT event to whatever should be shown
    plot_utils.convert_init_event(df_rr)
    
    half_cutoff = int(np.floor(df_rr.shape[0]/2))

    # POSTER (WIDER):
    fig = plt.figure(figsize=(15.5,7.5),dpi=350)
    gs = gridspec.GridSpec(1,9,width_ratios=[1.75,.0,.6,.1,1.75,.0,.6,.05,.75],wspace=0)

    chapter_mapping, labelsIdx, colours, textCol = plot_utils.get_chapters_high_level()

    # Trace sequences
    contFlag = False
    display_traces_time_scaled(fig,gs[0,0],
                                df_rr.iloc[half_cutoff:].copy(),
                                chapter_mapping,labelsIdx,colours,textCol,
                                contFlag)
    plt.title('A\n', loc='left',fontsize=RR_FIG_TITLE_FONTSIZE_LETTER)

    # quantities
    display_values(fig,gs[0,2],df_rr.iloc[half_cutoff:].copy())
    plt.title('B\n', loc='left', fontsize=RR_FIG_TITLE_FONTSIZE_LETTER)
    #ax.text(.55, len(ynames)+2.2, 'Number of subjects', ha='center', fontsize=RR_FIG_TITLE_FONTSIZE)

    # Trace sequences
    contFlag=True
    display_traces_time_scaled(fig,gs[0,4],
                                df_rr.iloc[:half_cutoff].copy(),
                                chapter_mapping,labelsIdx,colours,textCol,
                                contFlag)
    plt.title('C\n', loc='left',fontsize=RR_FIG_TITLE_FONTSIZE_LETTER)

    # quantities
    display_values(fig,gs[0,6],df_rr.iloc[:half_cutoff].copy())
    plt.title('D\n', loc='left', fontsize=RR_FIG_TITLE_FONTSIZE_LETTER)

    # Categories
    labelsIdx_edited = pd.Series(labelsIdx)
    labelsIdx_edited.index = np.append(['MI/Non-MI'],labelsIdx_edited.index.values[1:])
    display_categories(fig,gs[0,-1],labelsIdx_edited,colours,textCol)
    plt.title('E\n', loc='left',fontsize=RR_FIG_TITLE_FONTSIZE_LETTER)
    
    plot_utils.saveFig('rr_timescaled')
    plot_utils.saveFig_pdf('rr_timescaled')

    plt.show()

