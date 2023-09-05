# -*- coding: utf-8 -*-
"""
Plotting functions for the main trajectory figure (trajectories as blocks of
ICD-10 chapters, alongside counts, RR and HR values).

@author: Chris Hayward
"""

import numpy as np
import pandas as pd
import pdb

from pipeline_hes import parse_chapters
from pipeline_hes import plot_utils
from pipeline_hes.params import params

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator

RR_FIG_TITLE_FONTSIZE = 8


def get_fontsize(ynames):
    maxFont = 8*np.log(1+(60/49))
    actualFont = 8*np.log(1+(60/ynames.shape[0]))
    return min(maxFont,actualFont)


def display_categories(fig,gs,labelsIdx,colours,textCol):
    """Display the ICD-10 chapter names and a brief title for each."""

    ax = fig.add_subplot(gs)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Plot chapter headings
    plt.xticks(ticks=[0],labels=[' '])
    plt.yticks(ticks=[],labels=[])
    
    ax.tick_params(width=0, length=0, which='major')
    ax.tick_params(width=0, length=0, which='minor')
    
    ax.tick_params(width=0)
    xpos = -1
    numOnCol = 20
    
    code_desc_dict = parse_chapters.get_codes_description_short_desc()
    
    ipos = 0
    for (lab,i) in labelsIdx.items():
        
        barCol = colours[i,:3]
        ypos = numOnCol - np.mod(ipos,numOnCol)
        ipos = ipos+1
        if ypos==numOnCol:
            xpos += 1
        ax.barh(ypos, 1, left=xpos, height=.9, color=barCol, zorder=3)
        
        # ICD10 category
        ax.text(xpos+.5, ypos+.2, lab, color=textCol[i],
                zorder=3, ha='center', fontsize=9, va='center')
        
        # ICD10 title
        ax.text(xpos+.5, ypos-.2, code_desc_dict[lab], color=textCol[i],
                zorder=3, ha='center', fontsize=6.5, va='center')

    #lab_font = {'fontname': 'serif'}
    #plt.xlabel('MI = ['+','.join(params.AMI_RANGE)+']', fontsize=8, **lab_font)
    ax.margins(0.1,0)
    plt.xlim([0,1])


def display_traces(fig,gs,ynames,ydat,chapter_mapping,labelsIdx,colours,textCol):
    """Draw the trajectoires (aka traces) as horizontal bars."""
    
    ax = fig.add_subplot(gs)
    ax.axis('on')
    maxSeqLen = np.max([len(x.split(',')) for x in ynames])
    # matrix of labels
    labels = np.zeros((ynames.shape[0],maxSeqLen),dtype=object)
    labels[:] = ''
    for i,seq in enumerate(ynames):
        seqSplit = seq.split(',')
        for j,s in enumerate(seqSplit):
            labels[i,j] = s


    trace_fontsize = get_fontsize(ynames)

    # draw the coloured bars
    for y in range(len(ynames)):
        for x in range(maxSeqLen):
            if len(labels[y,x]) > 0:
                # low/high res chapter -> chapter mapping (to low res) -> index -> colour
                
                try:
                    cIdx = labelsIdx[chapter_mapping[labels[y,x]]]
                except KeyError:
                    # if using 3-char, then need to conv back to chapter first
                    tmp_df = pd.DataFrame.from_dict(
                        {'DIAG_01':\
                         np.concatenate([x.split(',')[1:-1] for x in ynames])}).drop_duplicates()
                    tmp_df['DIAG_01'] = tmp_df['DIAG_01'].astype('category')
                    parse_chapters.apply_diag_conversion_dict(tmp_df)                                        
                    cIdx = labelsIdx[chapter_mapping[tmp_df.loc[tmp_df['DIAG_01']==labels[y,x],
                                                                'DIAG_01_CONV_HIGH'].iloc[0]]]
                    
                c = colours[cIdx,:3]
                tc = textCol[cIdx]
                    
                ax.barh(y, 1, left=x, height=.9, color=c, zorder=3)
                ax.text(x+.5, y, labels[y,x], color=tc, zorder=3, ha='center',
                        fontsize=trace_fontsize, va='center')
    
    
    # number the trajectories (for ease of referencing)
    plt.yticks(ticks=ydat,labels=[i for i in range(len(ynames),0,-1)],fontsize=7)
    
    tmp_labs = np.append(['$\mathregular{1^{st}}$',
                          '$\mathregular{2^{nd}}$',
                          '$\mathregular{3^{rd}}$'],
                         ['$\mathregular{'+str(x)+'^{th}}$' for x in range(4,20)])
    tmp_labs = tmp_labs[:maxSeqLen]
    plt.xticks(ticks=np.arange(0,maxSeqLen)+.5,
               labels=tmp_labs, fontsize=8)

    # ###########
    # Sequences - minor ticks (arrows)
    # ###########
    global MAXSEQLEN
    MAXSEQLEN = maxSeqLen
    def minor_tick(x, pos):
        #from matplotlib import rc
        global MAXSEQLEN
        if x==0 or x % 1.0 or x==MAXSEQLEN:
            return ""
        return r"$\rightarrow$"
    plt.minorticks_on()
    ml = MultipleLocator(.5)

    ax.xaxis.set_minor_locator(ml)
    #ax.tick_params(axis='x', which='major', length=2)
    ax.xaxis.set_minor_formatter(minor_tick)
    #ax.tick_params(axis='y', which='minor', left=False)
    
    #ax.xaxis.grid(which='minor', visible=False)
    ax.tick_params(axis='y', width=0, length=0, which='major')
    ax.tick_params(axis='y', width=0, length=0, which='minor')
    ax.tick_params(axis='x', color='#555555', width=1, which='minor')
    ax.tick_params(axis='x', color='#ffffff', length=2, which='major')

    plt.xlabel('Position in disease trajectory', fontsize=8)
    ax.set_ymargin(0)
    plt.xlim([0-0.01,maxSeqLen+0.01])
    
    add_alternating_row_colours(ax)


def display_counts(fig,gs_pos,df_rrS,ynames,ydat,name_split,totalN):
    """Display the number of individuals following each trajectory."""
    
    textSize = get_fontsize(ynames)
    blankLabels = np.zeros(len(ynames), dtype=object)
    blankLabels[:] = ''
    
    ax = fig.add_subplot(gs_pos)

    ax.set_ymargin(0)
    ax.tick_params(width=0)
    plt.yticks(ticks=ydat+.5,labels=blankLabels)
    
    plt.xticks(ticks=[])
    
    ax.tick_params(width=0, length=0, which='major')
    ax.tick_params(width=0, length=0, which='minor')

    plt.xlabel('N={:,}'.format(totalN), fontsize=8)

    for j in range(len(ynames)):

        count = df_rrS.iloc[j]['#{}'.format(name_split)]
        txtC_n = '{:,}'.format(int(count)) if not np.isnan(count) else 'n/a'
        prc = df_rrS.iloc[j]['%{}'.format(name_split)]
        if prc < 0.05:
            txtC_prc = '<0.05'
        else:
            txtC_prc = '{:.1f}'.format(prc) if not np.isnan(prc) else 'n/a'
        
        ax.barh(j, .5, height=.9, zorder=0, color='white') # color=default_cmap(1)
        ax.text(.48, j, '{} ({}%)'.format(txtC_n,txtC_prc), color='black',
                zorder=3, ha='right', va='center', fontsize=textSize)

    plt.title(name_split, loc='center',
              fontsize=RR_FIG_TITLE_FONTSIZE)

    add_alternating_row_colours(ax)
    plt.xlim([0,.5])
    plt.ylim([-.5,(len(ydat)-1)+.5])


def _display_values_rr_hr(ax,df_rrS,ynames,ydat,valStr):
    """Display relative risk, or hazard ratio values."""
    
    textSize = get_fontsize(ynames)
    blankLabels = np.zeros(len(ynames), dtype=object)
    blankLabels[:] = ''
    
    ax.set_ymargin(0)
    ax.tick_params(width=0)
    plt.yticks(ticks=ydat+.5,labels=blankLabels)
    
    ax.tick_params(width=0, length=0, which='major')
    ax.tick_params(width=0, length=0, which='minor')
    ax.tick_params(color='#555555')
    
    plt.xticks(ticks=[])

    for j in range(len(ynames)):
        val = df_rrS.iloc[j]['{}'.format(valStr)]
        if np.isnan(val):
            val = 'n/a'
        else:        
            val = '{:.1f} [{:.1f}-{:.1f}]'.format(val,
                                                  df_rrS.iloc[j]['{}_CIl'.format(valStr)],
                                                  df_rrS.iloc[j]['{}_CIu'.format(valStr)])
        ax.barh(j, 1, height=.9, zorder=0, color='white') 
        ax.text(.5, j, val, color='black',
                zorder=3, ha='center', va='center', fontsize=textSize)
    add_alternating_row_colours(ax)
    plt.xlim([0,1])



def display_values_rr(ax,df_rrS,ynames,ydat):
    """Display relative risk values."""
    _display_values_rr_hr(ax,df_rrS,ynames,ydat,'RR')


def display_values_hr(ax,df_rrS,ynames,ydat):
    """Display hazard ratio values."""
    _display_values_rr_hr(ax,df_rrS,ynames,ydat,'HR')


def add_alternating_row_colours(ax):
    """For this axis, colour every-other row a different colour (shade of
    blue)."""
    ylocs = ax.yaxis.get_ticklocs()
    xinterval = ax.xaxis.get_view_interval()
    for j in range(len(ylocs))[::2]:
        ax.barh(j, max(xinterval)-min(xinterval), height=1, left=min(xinterval),
                zorder=0, color='#e4f0fc') # color=default_cmap(1) #e6f2ff, #F2FAFF
    ax.yaxis.grid(which='major', visible=False)
    
    ax.spines['bottom'].set_color('#555555')
    ax.spines['left'].set_color('#555555')
    ax.spines['right'].set_color('#555555')
    ax.spines['top'].set_color('#555555')


def plot_rr_fig(df_rr):
    """Entry function - create the main trajectory figure."""
    # Get the names of the two groups
    name_split_pat = ([x.split('TOTALpat_')[1] 
                       for x in df_rr.columns if x.startswith('TOTALpat_')])[0]
    name_split_ctl = ([x.split('TOTALctl_')[1]
                       for x in df_rr.columns if x.startswith('TOTALctl_')])[0]
    
    # convert INIT event to whatever should be shown
    plot_utils.convert_init_event(df_rr)
    
    ynames = df_rr.index
    ydat = np.arange(df_rr.shape[0])

    nPat = int(df_rr.dropna(subset=['TOTALpat_{}'.format(name_split_pat)],axis='rows')\
               .iloc[0]['TOTALpat_{}'.format(name_split_pat)])
    nCtl = int(df_rr.dropna(subset=['TOTALctl_{}'.format(name_split_ctl)],axis='rows')\
               .iloc[0]['TOTALctl_{}'.format(name_split_ctl)])

    fig = plt.figure(figsize=(14.5,8.5),dpi=350)
    
    gs = gridspec.\
        GridSpec(3,13,
                 height_ratios=[.04,.05,1],
                 width_ratios=[1.35,.05,.34,.0,.34,.05,.55,.34,.05,.55,.33,.05,.6],
                 wspace=0,hspace=0)

    chapter_mapping, labelsIdx, colours, textCol = plot_utils.get_chapters_high_level()

    # #######
    # Titles
    # #######
    ax=fig.add_subplot(gs[0,0])
    ax.text(.5,.8,'A: Disease trajectories',color='black',
            zorder=3, ha='center', va='center', fontsize=10)
    plt.xlim([0,1])
    plt.ylim([0,1])
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks(ticks=[],labels=[])
    plt.yticks(ticks=[],labels=[])
    
    ax=fig.add_subplot(gs[0,2:5])
    ax.text(.5,.8,'B: Number of individuals',color='black',
            zorder=3, ha='center', va='center', fontsize=10)
    plt.xlim([0,1])
    plt.ylim([0,1])
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks(ticks=[],labels=[])
    plt.yticks(ticks=[],labels=[])
    
    ax=fig.add_subplot(gs[0,6:11])
    ax.text(.5,.8,'C: MI vs. Matched controls',color='black',
            zorder=3, ha='center', va='center', fontsize=10)
    plt.xlim([0,1])
    plt.ylim([0,1])
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks(ticks=[],labels=[])
    plt.yticks(ticks=[],labels=[])
    
    ax=fig.add_subplot(gs[0,12])
    ax.text(.5,.8,'D: ICD-10 chapters',color='black',
            zorder=3, ha='center', va='center', fontsize=10)
    plt.xlim([0,1])
    plt.ylim([0,1])
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks(ticks=[],labels=[])
    plt.yticks(ticks=[],labels=[])
    
    # #######
    # SUB Titles
    # #######
    ax=fig.add_subplot(gs[1,6:8])
    ax.text(.5,.2,'Relative risk [95% CI]',color='black',
            zorder=3, ha='center', va='bottom', fontsize=RR_FIG_TITLE_FONTSIZE)
    plt.xlim([0,1])
    plt.ylim([0,1])
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks(ticks=[],labels=[])
    plt.yticks(ticks=[],labels=[])
    
    ax=fig.add_subplot(gs[1,9:11])
    ax.text(.5,.2,'All-cause mortality\nhazard ratio [95% CI]',color='black',
            zorder=3, ha='center', va='bottom', fontsize=RR_FIG_TITLE_FONTSIZE)
    plt.xlim([0,1])
    plt.ylim([0,1])
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks(ticks=[],labels=[])
    plt.yticks(ticks=[],labels=[])
    
    # ################
    # Trace sequences
    # ################
    display_traces(fig,gs[2,0],ynames,ydat,chapter_mapping,labelsIdx,colours,textCol)

    # ################
    # text: numbers and percentages of subjects
    # ################
    display_counts(fig,gs[2,2],df_rr,ynames,ydat,'MI',nPat)
    display_counts(fig,gs[2,4],df_rr,ynames,ydat,'Controls',nCtl)
    plt.title('Matched\nControls',fontsize=RR_FIG_TITLE_FONTSIZE)
    
    # ################
    # relative risk horiz: POINT + LINE
    # ################
    xdat = df_rr['RR']
    ax = fig.add_subplot(gs[2,6])

    ax.plot(xdat,ynames,'D',markersize=3)    
    # bold line at relative risk = 1
    ax.plot(np.repeat(1,df_rr.shape[0]+2),
            np.arange(-1,df_rr.shape[0]+1),markersize=0,color='#666666',linewidth=.6)

    if params.USE_CORRECTED_RR_HR:
        # mulitple comparisons corrected?
        xerr = np.array([xdat-df_rr['RR_CIl_corrected'], df_rr['RR_CIu_corrected']-xdat])
    else:
        # mulitple comparisons corrected?
        xerr = np.array([xdat-df_rr['RR_CIl'], df_rr['RR_CIu']-xdat])
    ax.errorbar(xdat, ydat, xerr=xerr, \
        linestyle='', color = "black", capsize = 2, capthick = .5, zorder=4, linewidth=.5)

    if np.max(xdat) > 10:
        ax.set_xscale('log')
        minD = int(np.floor(np.log10(np.min(xdat))))
        maxD = int(np.ceil(np.log10(np.max(xdat))))
        plt.xticks(np.logspace(minD,maxD,maxD-minD+1))
        
        plt.xlim([10**np.floor(np.log10(np.min(xdat))),
                  10**np.ceil(np.log10(np.max(xdat)))])
        plt.grid(True, which='major',ls='-', color='#dddddd')
    else:
        plt.grid(True, which='major',ls='-', color='#dddddd')
        if np.max(xdat)<3:
            plt.xticks(ticks=np.arange(0,np.ceil(np.max(xdat))+1,0.5))
        else:
            plt.xticks(ticks=np.arange(0,np.ceil(np.max(xdat))+1))
                
        plt.xlim([0,(np.max(xdat) + .1*(np.max(xdat)-np.min(xdat)))])

    plt.yticks(ticks=ydat+.5,labels='')
    ax.tick_params(which='both', labelsize=8)    
    
    ax.tick_params(axis='y',width=0, length=0, which='major')
    ax.tick_params(axis='y',width=0, length=0, which='minor')
    ax.tick_params(axis='x', color='#555555')

    ax.set_ymargin(0)

    add_alternating_row_colours(ax)

    plt.ylim([-.5,(len(ydat)-1)+.5])
    plt.xlabel('Relative risk', fontsize=8)


    # ###################
    # RR VALUES
    # ###################
    ax = fig.add_subplot(gs[2,7])
    display_values_rr(ax,df_rr,ynames,ydat)

    # ################
    # HAZARD: POINT + LINE
    # ################
    xdat = df_rr['HR']

    ax = fig.add_subplot(gs[2,9])

    ci_lower = df_rr['HR_CIl']
    ci_upper = df_rr['HR_CIu']
    xerr = np.array([xdat-ci_lower, ci_upper-xdat])
        
    ax.plot(xdat,ynames,'D',markersize=3)
    # bold line at HR = 1
    ax.plot(np.repeat(1,df_rr.shape[0]+2),
            np.arange(-1,df_rr.shape[0]+1),markersize=0,color='#666666',linewidth=.6)

    # mulitple comparisons corrected?
    if params.USE_CORRECTED_RR_HR:
        xerr = np.array([xdat-df_rr['HR_CIl_corrected'], df_rr['HR_CIu_corrected']-xdat])
    else:
        xerr = np.array([xdat-df_rr['HR_CIl'], df_rr['HR_CIu']-xdat])
        
    ax.errorbar(xdat, ydat, xerr=xerr, \
        linestyle='', color = "black", capsize = 2, capthick = .5, zorder=4, linewidth=.5)

    if np.isnan(np.max(xdat)):
        plt.xlim([0,2])
    else:            
        plt.xticks(ticks=np.arange(0,np.ceil(np.max(xdat))+1))
        plt.xlim([0,(np.max(xdat) + .1*(np.max(xdat)-np.min(xdat)))])

    plt.yticks(ticks=ydat+.5,labels='')
    
    ax.tick_params(axis='y',width=0, length=0, which='major')
    ax.tick_params(axis='y',width=0, length=0, which='minor')
    ax.tick_params(axis='x', color='#555555')

    ax.xaxis.grid(zorder=0,which='major', color='#dddddd')
    ax.tick_params(which='both', labelsize=8)
    ax.set_ymargin(0)

    add_alternating_row_colours(ax)    

    plt.ylim([-.5,(len(ydat)-1)+.5])
    plt.xlabel('Hazard ratio', fontsize=8)

    # ###########
    # HR VALUES
    # ############
    
    ax = fig.add_subplot(gs[2,10])
    display_values_hr(ax,df_rr,ynames,ydat)
    
    # #########
    # Categories
    # #########
    display_categories(fig,gs[2,-1],labelsIdx,colours,textCol)

    plot_utils.saveFig('rr')
    plot_utils.saveFig_pdf('rr_pdf')

    plt.show()

