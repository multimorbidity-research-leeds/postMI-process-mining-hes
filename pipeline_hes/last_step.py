# -*- coding: utf-8 -*-
"""
After running the pipeline, and saving the resulting 'trajectory dataframe'
(df_rr_hr) and the trajectory for each subject across both cohorts, this
script loads that saved data and draws figures and creates tables based on
this data. This is a 'final stage' script, containing blocks of code to
generate figures and tables for the manuscript.

@author: Chris Hayward
"""

from pipeline_hes import append_counts
from pipeline_hes.params import params
from pipeline_hes import parse_chapters
from pipeline_hes import adj_matrix
from pipeline_hes import plot_adj_graph
from pipeline_hes import traces_hes
from pipeline_hes import trace_plots
from pipeline_hes import trace_plots_time_scaled

import numpy as np
import pandas as pd
import re
import pdb
import pathlib
import os
import importlib

# When writing HTML tables to file, this is the header of the table, and
# styles the table.
HTML_TEMPLATE_START = """
<!DOCTYPE html>
<html>
<head>

<style>

table {
	border-collapse: collapse;
	/*width: 100%;*/
}
th {
	text-align: left;
	padding: 10px;
	background-color: #eeeeee;
}

tr{
	line-height:18px;
	padding:5px;
	margin:5px;
}

tbody > tr > th{
	line-height:18px;
	/*padding:10px;*/
	padding: 7px 5px 3px 5px;
}

td{
  	padding: 7px 5px 3px 5px;  
    }

tr:nth-child(odd){
	background-color: #ffffff;
}
tr:nth-child(even){
	/*background-color: #f3f3f3;*.
	background-color: #ffffff;
}

</style>

</head>
<body>

"""

HTML_TEMPLATE_END = """
</body>
"""



def attach_median_and_iqr_duration_per_num_events(variants_count,
                                                  variants_per_subject):
    """Calculates the median and IQR for the number of trajectory diagnoses
    per subject."""
    tmp = variants_count.copy()
    tmp = tmp[['num_events','count']].groupby('num_events').sum()
    variants_countDUR = tmp.merge(
        variants_per_subject.groupby('num_events')['DUR'].median(),
                    left_on='num_events',right_index=True)
    tmpq1 = variants_per_subject.groupby('num_events')[['DUR']].quantile(.25).rename(columns={'DUR':'Q1'})
    tmpq3 = variants_per_subject.groupby('num_events')[['DUR']].quantile(.75).rename(columns={'DUR':'Q3'})
    return pd.concat([variants_countDUR,tmpq1,tmpq3],axis=1)
    

def print_table_of_median_and_iqr_duration_per_num_events(variants_patients_count,
                                                          variants_controls_count,
                                                          variants_patients_per_subject,
                                                          variants_controls_per_subject):
    """For both MI and controls, calculate the median and IQR number of
    diagnoses appearing in trajectories."""

    # # need to do these steps to get the median DUR
    # variants_all_per_subject = pd.concat([variants_controls_per_subject,
    #                                       variants_patients_per_subject],ignore_index=True)
    # variants_all_count = variants_patients_count.merge(
    #     variants_controls_count,left_index=True,right_index=True,suffixes=('MI','CTL'),how='outer')
    # # set ALL count
    # variants_all_count.loc[variants_all_count['countMI'].isna(),'countMI'] = 0
    # variants_all_count.loc[variants_all_count['countCTL'].isna(),'countCTL'] = 0
    # variants_all_count['count'] = (variants_all_count['countMI'] + variants_all_count['countCTL']).astype(int)
    # # setl ALL num_events
    # variants_all_count['num_events'] = variants_all_count['num_eventsMI']
    # variants_all_count.loc[variants_all_count['num_events'].isna(),'num_events']=\
    #     variants_all_count.loc[variants_all_count['num_events'].isna(),'num_eventsCTL']
    # variants_all_count['num_events'] = variants_all_count['num_events'].astype(int)
    
    # # group-by, number of events
    # variants_all_count = variants_all_count[['num_events','count']].groupby('num_events').sum().reset_index()


    # median and IQR duration per 'num_events'
    variants_patients_countDUR = \
        attach_median_and_iqr_duration_per_num_events(variants_patients_count,
                                                      variants_patients_per_subject)
    variants_controls_countDUR = \
        attach_median_and_iqr_duration_per_num_events(variants_controls_count,
                                                      variants_controls_per_subject)
    # variants_all_countDUR = \
    #     attach_median_and_iqr_duration_per_num_events(variants_all_count,
    #                                                   variants_all_per_subject)


    counts_arr = []
    # For AMI, CTL, ALL: Print table of median/IQR duration per num_events
    for name,v in zip(('MI','CTL'),
                      [variants_patients_countDUR,
                       variants_controls_countDUR]):
        tmp = v.copy()
        tmp['prc'] = 100*tmp['count'] / tmp['count'].sum()
        tmp['cumprc'] = tmp['prc'].cumsum()
        
        smallprc = tmp['prc']<0.05
        tmp['prc'] = tmp['prc'].map(lambda x: '{}%'.format(np.round(x,1)))
        tmp.loc[smallprc,'prc'] = '< 0.05%'
        
        tmp[['DUR','Q1','Q3']] = tmp[['DUR','Q1','Q3']].applymap(lambda x: np.round(x,1))
        tmp['DUR'] = tmp[['DUR','Q1','Q3']].apply(lambda x: '{} [{},{}]'.format(x[0],x[1],x[2]),axis=1)
        tmp = tmp.drop(columns=['Q1','Q3'])

        print('--'*10+name+'--'*10)
                
        # suppress small values
        smallcount = tmp['count']<10
        tmp['count'] = tmp['count'].map(lambda x: '{:,}'.format(x))
        tmp.loc[smallcount, 'count'] = '< 10'
        
        tmp['count'] = tmp['count'] + " (" + tmp['prc'] + ")"
        tmp.rename(columns={'count':'Number of subjects',
                            'DUR':'Follow-up (years), median [Q1,Q3]',
                            'cumprc':'% (cumsum.)'},inplace=True)
        tmp.index.name='No. of diagnoses'
        counts_arr.append(tmp)

    var_countDUR = counts_arr[0].merge(counts_arr[1],how='outer',left_index=True,right_index=True)

    print(var_countDUR)
    saveHere = os.path.join(params.DIR_RESULTS,'follow_up_by_number_of_diagnoses.csv')
    
    print('Saving to: {}'.format(saveHere))
    with open(saveHere, 'w') as f:
        f.write(var_countDUR[['Number of subjects_x',
                              'Follow-up (years), median [Q1,Q3]_x',
                              'Number of subjects_y',
                              'Follow-up (years), median [Q1,Q3]_y']].to_csv(sep=';'))



def count_diagnosis_appearances_groupby_high(traces_high,traces_low):
    """For each high-level chapter, count how often 'low level' chapter
    appears.'Low level' typically refers to the three-character ICD-10 codes."""

    # get list of unique diagnoses/chapters from these traces
    allD_high = []
    allD_low = []
    for (h,l) in zip(traces_high,traces_low):
        allD_high.extend(h.split(',')[1:-1])
        allD_low.extend(l.split(',')[1:-1])
        
    allD = pd.DataFrame({'high':allD_high, 'low':allD_low})
    
    # count the number of low-level chapters per high-level chapter
    allD_g = allD.groupby('high').apply(lambda x: x['low'].value_counts())
    
    print(allD)
    print(allD_g)
    print(allD_g.index)
    
    
    # make the table
    all_tables = []
    # For each high-level chapter, count appearances of low-level
    for i in np.unique(allD_g.index.get_level_values(0)):
        tmp = allD_g.loc[(i,)].reset_index()
        print(tmp)
 
        print('{}, occurrences: {:,}'.format(i,tmp['low'].sum()))

        tmp['prc_withinChapter'] = 100 * tmp['low'] / tmp['low'].sum()
        tmp['prc'] = 100 * tmp['low'] / allD.shape[0]

        tmp['prcSTR_withinChapter'] = \
            tmp['prc_withinChapter'].map(lambda x: '{:.1f}%'.format(np.round(x,1)))
        tmp['prcSTR'] = \
            tmp['prc'].map(lambda x: '{:.1f}%'.format(np.round(x,1)))
            
            
        # tmp['prcAll'] = \
        #     tmp['prcAll'].map(lambda x: '{}%'.format(append_counts.my_round_prc(x)))
        


        # suppress small
        tmp.loc[tmp['prc']<0.05,'prcSTR'] = '< 0.05%'
        sup10 = tmp['low']<10
        tmp['Occurrences_raw'] = tmp['low']
        tmp['low'] = tmp['low'].map(lambda x: '{:,}'.format(x))
        tmp.loc[sup10,'low'] = '< 10'
        
        tmp['low'] = tmp['low'] + ' (' + tmp['prcSTR'] + ')'


        tmp = tmp.rename(columns={'index':'ICD-10 diagnosis',
                                  'low':'Occurrences'})
        all_tables.append(tmp[['ICD-10 diagnosis','Occurrences','prcSTR_withinChapter','Occurrences_raw']])
        #print(tmp)
    return all_tables

        
def count_diagnosis_appearances_groupby_high_merge(traces_high_MI,traces_low_MI,
                                                   traces_high_CTL,traces_low_CTL,
                                                   filename):
    """Combine into a single csv file, several csv-tables for each 'high-level'
    category, with MI and control counts for each csv-table. Each table
    shows the number of appearances of low-level chapters per-high-level
    chapter."""
    
    all_tables_MI = count_diagnosis_appearances_groupby_high(traces_high_MI,traces_low_MI)
    all_tables_CTL = count_diagnosis_appearances_groupby_high(traces_high_CTL,traces_low_CTL)

    table_str = ""
    for i in range(len(all_tables_MI)):
        tmp = all_tables_MI[i].merge(all_tables_CTL[i],how='outer',on='ICD-10 diagnosis')
        
        tmp.loc[tmp['Occurrences_x'].isna(),'Occurrences_x'] = '0 (0%)'
        tmp.loc[tmp['Occurrences_y'].isna(),'Occurrences_y'] = '0 (0%)'
        
        table_str = table_str + 'NEWLINE' + tmp.to_csv()
    
    table_str = table_str.replace('\n','').replace('NEWLINE','\n')
    
    #x = pd.concat(all_tables,ignore_index=True)
    #table_str = x.to_html(index=False).replace('NEWLINE','<br>')
    
    # table_str = table_str + '<h4>{}</h4>'.format(i) + \
    #     tmp.to_html(index=False).replace('NEWLINE','<br>')
    
    saveHere = os.path.join(params.DIR_RESULTS,'3char_counts_{}.csv'.format(filename))
    print('Saving to: {}'.format(saveHere))
    with open(saveHere, 'w') as f:
        #f.write(HTML_TEMPLATE_START)
        f.write(table_str)
        #f.write(HTML_TEMPLATE_END)
    
    # diag_counts = pd.Series(allD).value_counts().reset_index()
    # diag_counts['prc'] = 100 * diag_counts[0] /diag_counts[0].sum()
    # print('top 20:')
    # print(diag_counts.iloc[:20])
    
     
def count_chapter_appearances(traces_high):
    """Count the number of chapter appearances (not per-trajectory)."""
    
    # get list of unique diagnoses/chapters from these traces
    allD = []
    for h in traces_high:
        allD.extend(h.split(',')[1:-1])
    allD = pd.Series(allD)
                
    diag_counts = pd.Series(allD).value_counts().reset_index()
    diag_counts['prc'] = 100 * diag_counts[0] /diag_counts[0].sum()
    print(diag_counts)


def _count_diagnosis_appearances_of_traces(allTraces):
    """For each chapter appearing in the trajectories, count how many 
    trajectories it appears in."""

    # get list of unique diagnoses/chapters from these traces
    allD = []
    for i in allTraces:
        allD.extend(i.split(',')[1:-1])
    allD = pd.Series(allD).drop_duplicates()
    
    # Print how often diagnoses appear in traces
    res = pd.DataFrame([], columns=['count_all', 'prc_all',
                                    'count_immed', 'prc_immed'])
    for diag in allD.values:
        # Anywhere in trace
        regstr = '{},.*{},.*'.format(params.AMI_INIT,diag)
        print(regstr)
        count = 0
        for my_trace in allTraces:
            if not (re.search(regstr,my_trace) is None):
                count = count + 1        
        res.loc[diag, ['count_all','prc_all']] = [count,100* count / allTraces.shape[0]]
        print('{} / {}'.format(count,allTraces.shape[0]))
        # Immediate events only
        regstr = '{},{},.*'.format(params.AMI_INIT,diag)
        count = 0
        for my_trace in allTraces:
            if not (re.search(regstr,my_trace) is None):
                count = count + 1        
        res.loc[diag, ['count_immed','prc_immed']] = [count,100* count / allTraces.shape[0]]

    res = res.sort_values('prc_all',ascending=False)
    res['count_allSTR'] = res['count_all'].map(lambda x: '{:,}'.format(x))
    res['prc_allSTR'] = \
            res['prc_all'].map(lambda x: '{}%'.format(np.round(x,1)))
    
    res['count_prc'] = res['count_allSTR'] + ' (' + res['prc_allSTR'] + ')'
    
    #res = res[['count_all', 'prc_all']]
    res = res.rename(columns={'count_prc':'Number of trajectories',})
    
    res.index.name = 'ICD-10 chapter'
    res = res.reset_index()
    desc_dict = parse_chapters.get_codes_description_short_desc()
    res['ICD-10 chapter'] = res['ICD-10 chapter'].map(lambda x: '{}: {}'.format(x,desc_dict[x]))
    
    return res
    
def count_diagnosis_appearances_of_traces(miTraces, ctlTraces):
    """For MI and controls: For each chapter appearing in the trajectories,
    count how many trajectories it appears in. Merge the counts for these
    two cohorts into a single dataframe."""

    res_mi = _count_diagnosis_appearances_of_traces(miTraces)
    res_ctl = _count_diagnosis_appearances_of_traces(ctlTraces)
    
    print(res_mi)
    print(res_ctl)
    
    res = res_mi.merge(res_ctl,on='ICD-10 chapter',how='outer')
    res = res.sort_values('ICD-10 chapter')

#    saveHere = r'M:\medchaya\repos\GitHub\hes\doc\chapter_counts.html
    saveHere = os.path.join(params.DIR_RESULTS,'chapter_counts.html')
    print('Saving to: {}'.format(saveHere))
    with open(saveHere, 'w') as f:
        f.write(HTML_TEMPLATE_START)
        f.write(res[['ICD-10 chapter',
                      'Number of trajectories_x',
                      'Number of trajectories_y',]].to_html(index=False).replace('NEWLINE','<br>'))
        f.write(HTML_TEMPLATE_END)
    #pdb.set_trace()
    # print(res)
    # print(res.sum())
    

def table_of_alt_trace_frequences(variants_controls_per_subject, \
                                  variants_patients_per_subject,
                                  topN_traces):
    """For each high-level trajectory, give a breakdown of the lower-level
    trajectories, counting the number of individuals following these
    lower-level trajectories for the MI and control cohorts."""

    #%%
    #topN_traces = variants_patients_count.iloc[:5].index
    #topN_traces = variants_patients_count.index

    # Where to store tables of alt-variant proportions
    # dir_for_table = \
    #     'n:/HES_CVEPI/chrish/results/alt_variants_tables_{}/'.format(
    #         now_str)
    dir_for_table = os.path.join(
        params.DIR_RESULTS,'alt_variants_tables')

    pathlib.Path(dir_for_table).mkdir(exist_ok=True)

    arw = '\U00002192'
    variants_patients_per_subject['variant_ALT2'] = \
        variants_patients_per_subject['variant_ALT2'].map(
            lambda x: params.AMI_INIT_PLOT+arw+arw.join(x.split(',')[1:]))
    variants_controls_per_subject['variant_ALT2'] = \
        variants_controls_per_subject['variant_ALT2'].map(
            lambda x: params.AMI_INIT_PLOT+arw+arw.join(x.split(',')[1:]))

    
    for i,trace in enumerate(topN_traces):

        alt_traces_pat = variants_patients_per_subject.loc[
            variants_patients_per_subject['variant'] == trace, ['variant_ALT2']]
        alt_traces_ctl = variants_controls_per_subject.loc[
            variants_controls_per_subject['variant'] == trace, ['variant_ALT2']]

        alt_traces_pat = alt_traces_pat.rename(
            columns={'variant_ALT2':params.AMI_INIT_PLOT+arw+arw.join(trace.split(',')[1:])})
        alt_traces_ctl = alt_traces_ctl.rename(
            columns={'variant_ALT2':params.AMI_INIT_PLOT+arw+arw.join(trace.split(',')[1:])})


        #print(alt_traces_pat)

        # count the number of trace appearances
        df_alt_pat = alt_traces_pat.value_counts()
        df_alt_ctl = alt_traces_ctl.value_counts()
        

        pat_col_sum = '# MI (Total={})'.format(df_alt_pat.sum())
        ctl_col_sum = '# Controls (Total={})'.format(df_alt_ctl.sum())
        
        df_alt_new = pd.concat([df_alt_pat,
                                df_alt_ctl],axis=1)
        df_alt_new = df_alt_new.fillna(0).astype(int)
        
        df_alt_new = df_alt_new.rename(columns={0:pat_col_sum, 1:ctl_col_sum})
        
        df_alt_new = df_alt_new.sort_values(pat_col_sum,ascending=False)
    

        # #####
        # Top 1% of MI
        # #####
        num_pat = df_alt_new[pat_col_sum].sum()
        num_ctl = df_alt_new[ctl_col_sum].sum()
        df_alt_new = df_alt_new.loc[df_alt_new[pat_col_sum] >= max(10,num_pat*.01)] # num_pat*.01
    
        df_alt_new[pat_col_sum] = df_alt_new[pat_col_sum].map(
            lambda x: '{} ({}%)'.format(x, append_counts.my_round_prc(100*x/num_pat)))
        df_alt_new[ctl_col_sum] = df_alt_new[ctl_col_sum].map(
            lambda x: '{} ({}%)'.format(x, append_counts.my_round_prc(100*x/num_ctl)))
    
        # df_alt_new['% MI'] = 100 * df_alt_new[pat_col_sum] / \
        #     df_alt_new[pat_col_sum].sum()
        # df_alt_new['% Controls'] = 100 * df_alt_new[ctl_col_sum] / \
        #     df_alt_new[ctl_col_sum].sum()
        
        # # round
        # df_alt_new['% MI'] = df_alt_new['% MI'].map(
        #     lambda x: '({}%)'.format(np.round(x,1)))
        # df_alt_new['% Controls'] = df_alt_new['% Controls'].map(
        #     lambda x: '({}%)'.format(np.round(x,1)))

        # # append %
        # df_alt_new[pat_col_sum] = df_alt_new[pat_col_sum] + " " + df_alt_new['% MI']
        # df_alt_new[ctl_col_sum] = df_alt_new[ctl_col_sum] + " " + df_alt_new['% Controls']
        
        trace_file = str(i)  + " " + trace.replace(',','_').replace('/','-')
        saveTo = os.path.join(dir_for_table,'{}.html'.format(trace_file))
        print('Saving to: {}'.format(saveTo))
        with open(saveTo,'w',encoding='utf8') as f:
            f.write(HTML_TEMPLATE_START)
            print(trace_file)
            f.write("""<br><br>{}""".format(df_alt_new[[pat_col_sum, ctl_col_sum]].to_html(index=True)))
            f.write(HTML_TEMPLATE_END)
            

def for_shiny_disease_counts(variants_controls_per_subject, \
                             variants_patients_per_subject,
                             topN_traces,
                             filename):
    """For each high-level trajectory, count the number of 3-char appearances
    in each chapter within that trajectory - save as a CSV file for the
    shiny app."""

    df_csv = pd.DataFrame(columns=['trace','pos','3-char',
                                   'count_MI','count_CTL',
                                   'prc_MI','prc_CTL'])

    for i,trace in enumerate(topN_traces):
        print(trace)
        
        # get the 3-char traces for each high-level trace
        alt_traces_pat = variants_patients_per_subject.loc[
            variants_patients_per_subject['variant'] == trace, 'variant_ALT2']
        alt_traces_ctl = variants_controls_per_subject.loc[
            variants_controls_per_subject['variant'] == trace, 'variant_ALT2']

        mi_split = alt_traces_pat.map(lambda x: x.split(','))
        ctl_split = alt_traces_ctl.map(lambda x: x.split(','))
        
        trace_split = trace.split(',')
        trace_len = len(trace_split)
        # for each disease within the middle-portion of the trajectory
        for pos in range(1,trace_len-1):
            print(pos)
            
            # get the diseases for this position
            mi_all_for_pos = mi_split.map(lambda x: x[pos])
            mi_all_for_pos = mi_all_for_pos.value_counts().reset_index()
            ctl_all_for_pos = ctl_split.map(lambda x: x[pos])
            ctl_all_for_pos = ctl_all_for_pos.value_counts().reset_index()
            
            # percentages of those diseases at this position
            mi_all_for_pos['prc'] = 100 * mi_all_for_pos['variant_ALT2'] / \
                mi_all_for_pos['variant_ALT2'].sum()
            ctl_all_for_pos['prc'] = 100 * ctl_all_for_pos['variant_ALT2'] / \
                ctl_all_for_pos['variant_ALT2'].sum()
            
            # remove low counts
            mi_all_for_pos = mi_all_for_pos.loc[mi_all_for_pos['variant_ALT2']>=10]
            ctl_all_for_pos = ctl_all_for_pos.loc[ctl_all_for_pos['variant_ALT2']>=10]
            
            # merge counts into single dataframe
            all_for_pos = pd.DataFrame(mi_all_for_pos).merge(
                pd.DataFrame(ctl_all_for_pos),
                left_on='index',right_on='index',
                how='outer',suffixes=('_MI','_CTL'))
            
            all_for_pos = all_for_pos.sort_values('variant_ALT2_MI',ascending=False)
            
            all_for_pos.loc[all_for_pos['variant_ALT2_MI'].isna(),'variant_ALT2_MI'] = 0
            all_for_pos.loc[all_for_pos['variant_ALT2_CTL'].isna(),'variant_ALT2_CTL'] = 0
            all_for_pos.loc[all_for_pos['prc_MI'].isna(),'prc_MI'] = 0
            all_for_pos.loc[all_for_pos['prc_CTL'].isna(),'prc_CTL'] = 0
            
            all_for_pos['variant_ALT2_MI'] = all_for_pos['variant_ALT2_MI'].astype(int)
            all_for_pos['variant_ALT2_CTL'] = all_for_pos['variant_ALT2_CTL'].astype(int)
            
            # append to main dataframe (trajectory, position, disease, count, percentage)
            df_csv = pd.concat([df_csv,
                pd.DataFrame({'trace':np.repeat(trace,all_for_pos.shape[0]),
                              'pos':np.repeat(pos,all_for_pos.shape[0]),
                              '3-char':all_for_pos['index'],
                              'count_MI':all_for_pos['variant_ALT2_MI'],
                              'count_CTL':all_for_pos['variant_ALT2_CTL'],
                              'prc_MI':all_for_pos['prc_MI'],
                              'prc_CTL':all_for_pos['prc_CTL']})],ignore_index=True)
  
            # for mi_key in mi_all_for_pos.index:
            #     df_csv.loc[df_csv.shape[0]] = [trace, pos, True, mi_key, mi_all_for_pos[mi_key]]
            # for ctl_key in ctl_all_for_pos.index:
            #     df_csv.loc[df_csv.shape[0]] = [trace, pos, False, ctl_key, ctl_all_for_pos[ctl_key]]
                
            # print(mi_all_for_pos)
            # print([trace,pos,list(mi_all_for_pos.index),True])
            # df_csv[[trace,pos,list(mi_all_for_pos.index),True]] = mi_all_for_pos
            # df_csv[[trace,pos,list(ctl_all_for_pos.index),False]] = ctl_all_for_pos
            
            
        # if i==3:
        #     break
    
    
    print(df_csv)
    df_csv.to_csv(filename,index=False)
    

def main():
    pathlib.Path(params.DIR_RESULTS).mkdir(exist_ok=True)
    results_dir = 'n:/HES_CVEPI/chrish/results_paper/'    

    ### ---------------------------
    ### american chapters... WRONG.
    # datasets = {'main':os.path.join(results_dir,
    #                                 '15-05-22_10-43-09_7304891_main'),
    #             'pri_and_sec': os.path.join(results_dir,
    #                                         '14-05-22_19-21-10_7304891_inclSec'),
    #             'ignoring_6m':os.path.join(results_dir,
    #                                        '16-05-22_00-32-12_7304891_ignore6m'),
    #             'seed1':os.path.join(results_dir,
    #                                  '05-06-22_15-34-20_7304891_seed1')}

    # tmpPlotFolder = datasets['main']
    # tmpPlotFolder = datasets['pri_and_sec']
    # tmpPlotFolder = datasets['ignoring_6m']
    # tmpPlotFolder = datasets['seed1']

    ### ---------------------------
    ### with updated NHS chapters (non american)
    datasets = {'main':os.path.join(results_dir,
                                '17-03-23_08-26-11_7304891_main_changedChaps'),
                'pri_and_sec': os.path.join(results_dir,
                                            '18-03-23_19-35-43_7304891_priAndSec_changedChaps'),
                'ignoring_6m': os.path.join(results_dir,
                                            '17-03-23_15-48-55_7304891_ignore6m_changedChaps'),
                'seed1': os.path.join(results_dir,
                                            '18-03-23_10-04-48_7304891_seed1_changedChaps'),
                
                ## new:
                'main_female': os.path.join(results_dir,
                                            '20-06-23_16-57-10_7304891_female_only'),
                'main_male': os.path.join(results_dir,
                                            '20-06-23_21-28-30_7304891_male_only'),
                'pri_and_sec_female': os.path.join(results_dir,
                                            '21-06-23_10-39-14_7304891_pri_sec_female_only'),
                'pri_and_sec_male': os.path.join(results_dir,
                                            '21-06-23_11-56-57_7304891_pri_sec_male_only'),
                'pri_and_sec_seed1':os.path.join(results_dir,
                                                 '15-06-23_13-44-13_7304891_pri_and_sec_seed1')}

    tmpPlotFolder = datasets['main_female']
    

    #  # THESE TWO ARENT ACTUALLY NEEDED - can be calculated from the above

    # # ####### Trajectories with 3-char diagnoses instead
    # tmpPlotFolder = os.path.join(results_dir,
    #                              '31-05-22_20-40-30_7304891_3char')

    # # ####### Sub-chapter grouping
    # tmpPlotFolder = os.path.join(results_dir,
    #                              '21-09-22_15-47-14_7304891_subchapter_grouping')


    #%% Read in the files
    # - df_rr_hr: (for the common trajectories, counts, relative risks and hazard
    # ratios)
    # - variants_per_subject (patients = MI): for each individual, their
    # trajectory

    df_rr_hr = pd.read_parquet(
        os.path.join(tmpPlotFolder,'df_rr_hr_{}_.gzip'.format(
        params.R)))
    variants_controls_per_subject = pd.read_parquet(
        os.path.join(tmpPlotFolder,
                     'variants_controls_per_subject_{}_.gzip'.format(params.R)))
    variants_patients_per_subject = pd.read_parquet(
        os.path.join(tmpPlotFolder,
                     'variants_patients_per_subject_{}_.gzip'.format(params.R)))
    
    
    #%% swap if looking at 3-char, etc
    
    # look at granular chapters
    # variants_controls_per_subject['variant_TMP'] = variants_controls_per_subject['variant']
    # variants_controls_per_subject['variant'] = variants_controls_per_subject['variant_ALT1']
    # variants_controls_per_subject['variant_ALT1'] = variants_controls_per_subject['variant_TMP']
    # variants_controls_per_subject = variants_controls_per_subject.drop(columns='variant_TMP')
    
    # variants_patients_per_subject['variant_TMP'] = variants_patients_per_subject['variant']
    # variants_patients_per_subject['variant'] = variants_patients_per_subject['variant_ALT1']
    # variants_patients_per_subject['variant_ALT1'] = variants_patients_per_subject['variant_TMP']
    # variants_patients_per_subject = variants_patients_per_subject.drop(columns='variant_TMP')
    
    
    #%% RECALC variants_count, if needed (e.g. for 3char)
    print('Count how often each trajectory appears...')

    importlib.reload(traces_hes)
    variants_controls_count = traces_hes.get_variants_count(variants_controls_per_subject)
    variants_patients_count = traces_hes.get_variants_count(variants_patients_per_subject)



    #%% Pause here. Run blocks as needed.
    pdb.set_trace()
    
    
    #%%RECALC df_rr_hr, if needed (for different % thresholds for df_rr_hr)
    
    # import importlib
    # from pipeline_hes import traces_hes
    # importlib.reload(traces_hes)
    # df_rr_hr = traces_hes.get_df_rr_hr(variants_controls_count, variants_controls_per_subject,
    #                                    variants_patients_count, variants_patients_per_subject)

    
    #%% #################
    ### Main RR/HR plot
    # ###################
    importlib.reload(trace_plots)
    trace_plots.plot_rr_fig(df_rr_hr.sort_values('RR').copy())
    
    
    #%% Export the trajectories
    importlib.reload(traces_hes)
    # variants_patients_mean = traces_hes.get_variants_count_averages(
    #     variants_patients_per_subject, 'variant')
    # variants_controls_mean = traces_hes.get_variants_count_averages(
    #     variants_controls_per_subject, 'variant')
    
    # variants_patients_mean = traces_hes.get_variants_count_averages(
    #     variants_patients_per_subject, 'variant_ALT1')
    # variants_controls_mean = traces_hes.get_variants_count_averages(
    #     variants_controls_per_subject, 'variant_ALT1')
    
    variants_patients_mean = traces_hes.get_variants_count_averages(
        variants_patients_per_subject, 'variant_ALT2')
    variants_controls_mean = traces_hes.get_variants_count_averages(
        variants_controls_per_subject, 'variant_ALT2')
    
    for v,v_name in [(variants_patients_mean, 'MI'),
                     (variants_controls_mean, 'controls')]:
        v_EXPORT = v.copy().rename(columns={'count':'NUM_INDIVIDUALS',
                                            'variant':'ICD10_TRAJECTORY'})
        v_EXPORT = v_EXPORT.loc[v_EXPORT['NUM_INDIVIDUALS']>=10]
        v_EXPORT.to_csv(os.path.join(params.DIR_TMP,
                                     'trajectories_{}.csv'.format(v_name)),index=False)
    
    #%% RMST - TIME SCALED RR plot
    importlib.reload(trace_plots_time_scaled)
    trace_plots_time_scaled.plot_rr_fig_time_scaled(df_rr_hr.sort_values('RR').copy())
    
    
    #%% Percentage of subjects following the common trajectories
    print(df_rr_hr['#MI'].sum())
    print(100 * df_rr_hr['#MI'].sum() / df_rr_hr['TOTALpat_MI'][0])
    

    #%% view RR and HR
    print(df_rr_hr.sort_values('RR')[['RR','RR_CIl', 'RR_CIu',
                                      'HR','HR_CIl', 'HR_CIu']])
    
    #%%
    print(df_rr_hr.sort_values('HR_CIl')[['RR','RR_CIl', 'RR_CIu',
                                      'HR','HR_CIl', 'HR_CIu']])
    
    #%%
    print(df_rr_hr.sort_index()[['RR','RR_CIl', 'RR_CIu',
                                      'HR','HR_CIl', 'HR_CIu']])

    #%% RR multiple circ
    print(df_rr_hr.loc[[
        'INIT,I00-I99,I00-I99,I00-I99,Censor',
        'INIT,I00-I99,I00-I99,Censor'],
                       ['RR','RR_CIl', 'RR_CIu',
                        'HR','HR_CIl', 'HR_CIu']].apply(lambda x: np.round(x,2)))

    #%% RR circ...digestive
    print(df_rr_hr.loc[[
        'INIT,I00-I99,I00-I99,K00-K93,Censor',
        'INIT,I00-I99,K00-K93,Censor'],
                       ['RR','RR_CIl', 'RR_CIu']].apply(lambda x: np.round(x,2)))

    #%% RR circ...resp
    print(df_rr_hr.loc[[
        'INIT,I00-I99,I00-I99,J00-J99,Censor',
        'INIT,I00-I99,J00-J99,Censor'],
                       ['RR','RR_CIl', 'RR_CIu']].apply(lambda x: np.round(x,2)))

    #%% HR circ...digestive
    print(df_rr_hr.loc[[
        'INIT,I00-I99,K00-K93,Censor',
        'INIT,I00-I99,K00-K93,K00-K93,Censor',
        'INIT,I00-I99,I00-I99,K00-K93,Censor'],
                       ['HR','HR_CIl', 'HR_CIu']].apply(lambda x: np.round(x,2)))

    #%% RR/HR circ...eye
    print(df_rr_hr.loc[[
        'INIT,I00-I99,H00-H59,Censor',],
                       ['RR','RR_CIl', 'RR_CIu',
                        'HR','HR_CIl', 'HR_CIu']].apply(lambda x: np.round(x,2)))

    #%% RR/HR eye...circ
    print(df_rr_hr.loc[[
        'INIT,H00-H59,I00-I99,Censor',],
                       ['RR','RR_CIl', 'RR_CIu',
                        'HR','HR_CIl', 'HR_CIu']].apply(lambda x: np.round(x,2)))

    #%% RR/HR cancer
    print(df_rr_hr.loc[[
        'INIT,C00-D48,Censor',
        'INIT,C00-D48,C00-D48,Censor'],
                       ['RR','RR_CIl', 'RR_CIu',
                        'HR','HR_CIl', 'HR_CIu']].apply(lambda x: np.round(x,2)))


   #%% RR/HR 2x musculo.
    print(df_rr_hr.loc[[
        'INIT,M00-M99,M00-M99,Censor',
        'INIT,N00-N99,N00-N99,Censor'],
                       ['RR','RR_CIl', 'RR_CIu',
                        'HR','HR_CIl', 'HR_CIu']].apply(lambda x: np.round(x,2)))

   #%% RR/HR, single infectious or single blood.
    print(df_rr_hr.loc[[
        'INIT,A00-B99,Censor',
        'INIT,D50-D89,Censor'],
                       ['RR','RR_CIl', 'RR_CIu',
                        'HR','HR_CIl', 'HR_CIu']].apply(lambda x: np.round(x,2)))

   #%% RR/HR, MULTIPLE circ, + mental
    print(df_rr_hr.loc[[
        'INIT,I00-I99,F00-F99,Censor',
        'INIT,F00-F99,I00-I99,I00-I99,Censor',
        'INIT,I00-I99,F00-F99,I00-I99,Censor',
        'INIT,I00-I99,I00-I99,F00-F99,Censor',
        'INIT,I00-I99,I00-I99,I00-I99,F00-F99,Censor',
        'INIT,I00-I99,I00-I99,F00-F99,I00-I99,Censor'],
                       ['RR','RR_CIl', 'RR_CIu',
                        'HR','HR_CIl', 'HR_CIu']].apply(lambda x: np.round(x,2)))
   #%% RR/HR, MULTIPLE circ, + mental
    print(df_rr_hr.loc[[
        'INIT,I00-I99,I00-I99,F00-F99,Censor'],
                       ['RR','RR_CIl', 'RR_CIu',
                        'HR','HR_CIl', 'HR_CIu']].apply(lambda x: np.round(x,2)))


   #%% RR/HR, circ PUNCUTATED BY mental
    print(df_rr_hr.loc[[
        'INIT,I00-I99,F00-F99,Censor',
        'INIT,I00-I99,I00-I99,F00-F99,Censor',
        'INIT,I00-I99,I00-I99,I00-I99,F00-F99,Censor'],
                       ['RR','RR_CIl', 'RR_CIu',
                        'HR','HR_CIl', 'HR_CIu']].apply(lambda x: np.round(x,2)))


    #%% RR circ+resp (any order)
    print(df_rr_hr.loc[[
        'INIT,I00-I99,I00-I99,J00-J99,Censor',
        'INIT,I00-I99,J00-J99,Censor',
        'INIT,J00-J99,I00-I99,Censor'],
                       ['RR','RR_CIl', 'RR_CIu']].apply(lambda x: np.round(x,2)))



    #%% I want the RMST for ALL traces in AMI, and ALL traces in CTL
    
    from rpy2.robjects.conversion import localconverter
    from rpy2 import robjects
    from rpy2.robjects import pandas2ri
    
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_from_pd_df = robjects.conversion.py2rpy(
            pd.concat([variants_controls_per_subject[['DUR','Mortality','IS_PATIENT']],
                       variants_patients_per_subject[['DUR','Mortality','IS_PATIENT']]],ignore_index=True))

    robjects.globalenv['x'] = r_from_pd_df
    formula = """res<-rmst2(
        time=x$DUR,
        status=x$Mortality,
        arm=x${},)""".format('IS_PATIENT')
    # formula = """res<-rmst2(
    #     time=x$DUR,
    #     status=x$Mortality,
    #     arm=x${},
    #     covariates=x[,c(5,6,7,8)])""".format(hazard_column)
                                                                    
    #pdb.set_trace()
    try:
        robjects.r(formula)
    except Exception as ex:
        print(ex)

    robjects.r('print(res)')
    r_tmp = robjects.globalenv['res']
    r_dict = dict(zip(r_tmp.names, r_tmp))
    
    # Restricted Mean Survival Time (RMST) by arm 
    #           Est.    se lower .95 upper .95
    print(r_dict['RMST.arm0'][1][0]) # hazard col = 0 (arm=0)
    print(r_dict['RMST.arm1'][1][0]) # hazard col = 1 (arm=1)
    
    #%% Results: OVERALL stats
    
    # # I dont think this is very interesting (DUR)
    # # median DUR for AMI and ctl
    # print('AMI median dur: {}[{}-{}]\nCTL median dur: {}[{}-{}]'.\
    #       format(variants_patients_per_subject['DUR'].median(),
    #              variants_patients_per_subject['DUR'].quantile(.25),
    #              variants_patients_per_subject['DUR'].quantile(.75),
    #              variants_controls_per_subject['DUR'].median(),
    #              variants_controls_per_subject['DUR'].quantile(.25),
    #              variants_controls_per_subject['DUR'].quantile(.75)))
        
    print('AMI median length: {}[{}-{}]\nCTL median length: {}[{}-{}]'.\
          format(variants_patients_per_subject['num_events'].median(),
                 variants_patients_per_subject['num_events'].quantile(.25),
                 variants_patients_per_subject['num_events'].quantile(.75),
                 variants_controls_per_subject['num_events'].median(),
                 variants_controls_per_subject['num_events'].quantile(.25),
                 variants_controls_per_subject['num_events'].quantile(.75)))
        
        
    #%% number of common MI trajectories
    print(df_rr_hr['#MI'].sum())
    print(df_rr_hr['%MI'].sum())

    # controls
    print(df_rr_hr['#Controls'].sum())
    print(df_rr_hr['%Controls'].sum())
    
    #%% Paper OUTPUT: For shiny - counts of 3-char at each trace position
    
    # output all CSV files in one go
    from pipeline_hes import last_step
    importlib.reload(last_step)
    for (k,v) in datasets.items():
        
        df_rr_hr_TMP = pd.read_parquet(
            os.path.join(v,'df_rr_hr_{}_.gzip'.format(params.R)))
        variants_controls_per_subject_TMP = pd.read_parquet(
            os.path.join(v,'variants_controls_per_subject_{}_.gzip'.format(params.R)))
        variants_patients_per_subject_TMP = pd.read_parquet(
            os.path.join(v,'variants_patients_per_subject_{}_.gzip'.format(params.R)))

        variants_controls_count_TMP = \
            traces_hes.get_variants_count(variants_controls_per_subject_TMP)
        variants_patients_count_TMP = \
            traces_hes.get_variants_count(variants_patients_per_subject_TMP)

        filename = os.path.join(params.DIR_RESULTS,'shiny_3char_counts_{}.csv'.format(k))
        last_step.for_shiny_disease_counts(variants_controls_per_subject_TMP, \
                                           variants_patients_per_subject_TMP,
                                           list(df_rr_hr_TMP.index),filename)
        df_rr_hr_TMP.to_csv(os.path.join(params.DIR_RESULTS,
                                         'df_rr_hr_{}.csv'.format(k)), index=True)
        
    
    #%% Paper OUTPUT: TABLE - trace NUM EVENTS vs DURATION 
    print_table_of_median_and_iqr_duration_per_num_events(
        variants_patients_count.copy(),
        variants_controls_count.copy(),
        variants_patients_per_subject.copy(),
        variants_controls_per_subject.copy())
    
    #%% Paper OUTPUT: TABLE - counts of ICD-10 3-char per chapter
    
    # e.g.:
    # A00-B99
    # A01, 100 (MI), 0.1% (MI), 100 (Control), 0.1% (Control)
    # ...
    # C00-D49
    # C01, 100 (MI), 0.1% (MI), 100 (Control), 0.1% (Control)
    # ....
    
    count_diagnosis_appearances_groupby_high_merge(
        variants_patients_per_subject['variant'].copy(),
        variants_patients_per_subject['variant_ALT2'].copy(),
        variants_controls_per_subject['variant'].copy(),
        variants_controls_per_subject['variant_ALT2'].copy(),
        'all')
    
    #%% Paper OUTPUT: TABLE C - freq of chapter appearances (HTML)

    # e.g.:
    # A00-B99: Infectious Diseases, 	5,000 (19.0%), 	29,000 (22.1%)
    # C00-D49: Neoplasms,	8,000 (31.0%), 	52,000 (38.8%)

    # unique
    count_diagnosis_appearances_of_traces(
        variants_patients_per_subject['variant'].drop_duplicates().copy(),
        variants_controls_per_subject['variant'].drop_duplicates().copy())

    # # non-unique
    # count_diagnosis_appearances_of_traces(
    #     variants_patients_per_subject['variant'].copy(),
    #     variants_controls_per_subject['variant'].copy())

    
    #%% most frequently recorded pri diagnosis
    
    all_tables_MI = count_diagnosis_appearances_groupby_high(
        variants_patients_per_subject['variant'].copy(),
        variants_patients_per_subject['variant_ALT2'].copy())
    all_tables_CTL = count_diagnosis_appearances_groupby_high(
        variants_controls_per_subject['variant'].copy(),
        variants_controls_per_subject['variant_ALT2'].copy())


    all_tables_MI = pd.concat(all_tables_MI)
    all_tables_CTL = pd.concat(all_tables_CTL)
    
    print(all_tables_MI.sort_values('Occurrences_raw').tail())
    print(all_tables_CTL.sort_values('Occurrences_raw').tail())
    


    #%% EYE->CIRC
    
    tmpEye = variants_patients_per_subject.loc[
        variants_patients_per_subject['variant']=='INIT,H00-H59,I00-I99,Censor'].copy()
    x = count_diagnosis_appearances_groupby_high(tmpEye['variant'],
                                                 tmpEye['variant_ALT2'])
    
    #%% INIT->GENITO.
    
    # this BREAKS, but can put a break-point in if needing the counts
    tmpEye = variants_patients_per_subject.loc[
        variants_patients_per_subject['variant']=='INIT,N00-N99,Censor'].copy()
    x = count_diagnosis_appearances_groupby_high(tmpEye['variant'],
                                                 tmpEye['variant_ALT2'])
    

    #%% CIRC->CIRC, and CIRC->CIRC->CIRC (MI)

    tmp_mask= pd.concat([
        variants_patients_per_subject['variant']=='INIT,I00-I99,I00-I99,Censor',
        variants_patients_per_subject['variant']=='INIT,I00-I99,I00-I99,I00-I99,Censor'],axis=1).any(axis=1)
    tmpCirc = variants_patients_per_subject.loc[tmp_mask].copy()
    x = pd.Series(np.concatenate(tmpCirc['variant_ALT2'].\
                                 str.split(',').map(lambda x: x[1:-1]).values)).value_counts()
    x = pd.concat([x, 100*x/x.sum()],axis=1)
    
    #%% CIRC->CIRC, and CIRC->CIRC->CIRC (CONTROL)    
    
    tmp_mask= pd.concat([
        variants_controls_per_subject['variant']=='INIT,I00-I99,I00-I99,Censor',
        variants_controls_per_subject['variant']=='INIT,I00-I99,I00-I99,I00-I99,Censor'],axis=1).any(axis=1)
    tmpCirc = variants_controls_per_subject.loc[tmp_mask].copy()
    x = pd.Series(np.concatenate(tmpCirc['variant_ALT2'].\
                                 str.split(',').map(lambda x: x[1:-1]).values)).value_counts()
    x = pd.concat([x, 100*x/x.sum()],axis=1)


    #%% INCL sec, circ*->mental
    # tmp_mask= pd.concat([
    #     variants_patients_per_subject['variant']=='INIT,I00-I99,F00-F99,Censor',
    #     variants_patients_per_subject['variant']=='INIT,I00-I99,I00-I99,F00-F99,Censor',
    #     variants_patients_per_subject['variant']=='INIT,I00-I99,I00-I99,I00-I99,F00-F99,Censor'],axis=1).any(axis=1)
    tmp_mask= pd.concat([
        variants_patients_per_subject['variant']=='INIT,I00-I99,I00-I99,F00-F99,Censor'],axis=1).any(axis=1)
    tmpCirc = variants_patients_per_subject.loc[tmp_mask].copy()
    
    
    # -- circ
    x = pd.Series(np.concatenate(tmpCirc['variant_ALT2'].\
                                 str.split(',').map(lambda x: x[1:-2]).values)).value_counts()
    x = pd.concat([x, 100*x/x.sum()],axis=1)

    # -- last one (mental)
    x2 = pd.Series(tmpCirc['variant_ALT2'].\
                                 str.split(',').map(lambda x: x[-2]).values).value_counts()
    x2 = pd.concat([x2, 100*x2/x2.sum()],axis=1)



    #%%  INCL Sec, endocrine + circ range
    trace_mask_anyENDO = \
        df_rr_hr.index.map(
        lambda trace: not re.search('INIT,.*((E00-E90,.*I00-I99)|(I00-I99,.*E00-E90))+,.*',
                                    trace) is None).values.astype(bool)
    print(df_rr_hr.loc[trace_mask_anyENDO]['RR'].max())
    print(df_rr_hr.loc[trace_mask_anyENDO]['RR'].min())
    print(df_rr_hr.loc[trace_mask_anyENDO]['HR'].max())
    print(df_rr_hr.loc[trace_mask_anyENDO]['HR'].min())
    

    print(df_rr_hr.loc[trace_mask_anyENDO, ['RR', 'RR_CIl', 'RR_CIu', 'HR', 'HR_CIl', 'HR_CIu']].sort_values('RR'))

    #%%
    trace_mask_any_BEGIN_etc = \
        df_rr_hr.index.map(
        lambda trace: not re.search('INIT,((A00-B99)|(D50-D89)|(E00-E90)|(F00-F99))+,.*',
                                    trace) is None).values.astype(bool)
    print(df_rr_hr.loc[trace_mask_any_BEGIN_etc][['RR','RR_CIl','RR_CIu']])
    print(df_rr_hr.loc[trace_mask_any_BEGIN_etc][['HR','HR_CIl','HR_CIu']])
    
    
    #%% RR/HR for CIRC and MENTAL
    trace_mask_tmp = \
        df_rr_hr.index.map(
        lambda trace: not re.search('INIT,((F00-F99,)|(I00-I99,))*((I00-I99,F00-F99,)|(F00-F99,I00-I99,))((F00-F99,)|(I00-I99,))*Censor',
                                    trace) is None).values.astype(bool)
    print(df_rr_hr.loc[trace_mask_tmp][['RR','RR_CIl','RR_CIu','HR','HR_CIl','HR_CIu']].sort_values('HR'))    



    #%% TABLE (circ + digestive)
    
    print('In interesting traces, look at 3-char ICD-10 frequency')
    
    # MORE FREQUENT, DIGESTIVE
    interesting_traces = df_rr_hr.sort_values('RR').merge(
        variants_patients_per_subject,
        left_index=True, right_on='variant')[['variant','variant_ALT2']]
    allT = []
    for t in interesting_traces['variant'].drop_duplicates():
        # Only look at traces containing particular chapters for now...
        if not ('K00-K93' in t) or not ('I00-I99' in t):
            continue
        print('{} {} {}'.format('-'*20, t, '-'*20))
        tmp = variants_patients_per_subject.loc[
            variants_patients_per_subject['variant']==t].copy()
        allT.append(tmp)
    allT = pd.concat(allT,ignore_index=True)
    x = count_diagnosis_appearances_groupby_high(allT['variant'],
                                                 allT['variant_ALT2'])


    #%% TABLE (ALL Genitourinary)
    
    interesting_traces = df_rr_hr.merge(
        variants_patients_per_subject,
        left_index=True, right_on='variant')[['variant','variant_ALT2']]
    allT = []
    for t in interesting_traces['variant'].drop_duplicates():
        # Only look at traces containing particular chapters for now...
        if not ('N00-N99' in t):
            continue
        print('{} {} {}'.format('-'*20, t, '-'*20))
        tmp = variants_patients_per_subject.loc[
            variants_patients_per_subject['variant']==t].copy()
        allT.append(tmp)
    allT = pd.concat(allT,ignore_index=True)
    x = count_diagnosis_appearances_groupby_high(allT['variant'],
                                              allT['variant_ALT2'])

    #%% TABLE (ALL EYE)
    
    interesting_traces = df_rr_hr.merge(
        variants_patients_per_subject,
        left_index=True, right_on='variant')[['variant','variant_ALT2']]
    allT = []
    for t in interesting_traces['variant'].drop_duplicates():
        # Only look at traces containing particular chapters for now...
        if not ('H00-H59' in t):
            continue
        print('{} {} {}'.format('-'*20, t, '-'*20))
        tmp = variants_patients_per_subject.loc[
            variants_patients_per_subject['variant']==t].copy()
        allT.append(tmp)
    allT = pd.concat(allT,ignore_index=True)
    x = count_diagnosis_appearances_groupby_high(allT['variant'],
                                              allT['variant_ALT2'])
    
    #%% TABLE (ALL CIRC+RESP)

    interesting_traces = df_rr_hr.merge(
        variants_patients_per_subject,
        left_index=True, right_on='variant')[['variant','variant_ALT2']]
    allT = []
    for t in interesting_traces['variant'].drop_duplicates():
        # Only look at traces containing particular chapters for now...
        if not ('J00-J99' in t) or not ('I00-I99' in t):
            continue
        print('{} {} {}'.format('-'*20, t, '-'*20))
        tmp = variants_patients_per_subject.loc[
            variants_patients_per_subject['variant']==t].copy()
        allT.append(tmp)
    allT = pd.concat(allT,ignore_index=True)
    x = count_diagnosis_appearances_groupby_high(allT['variant'],
                                              allT['variant_ALT2'])
    
    
   #%% TABLE (ALL RESP)

    interesting_traces = df_rr_hr.merge(
        variants_patients_per_subject,
        left_index=True, right_on='variant')[['variant','variant_ALT2']]
    allT = []
    for t in interesting_traces['variant'].drop_duplicates():
        # Only look at traces containing particular chapters for now...
        if not ('J00-J99' in t):
            continue
        print('{} {} {}'.format('-'*20, t, '-'*20))
        tmp = variants_patients_per_subject.loc[
            variants_patients_per_subject['variant']==t].copy()
        allT.append(tmp)
    allT = pd.concat(allT,ignore_index=True)
    x = count_diagnosis_appearances_groupby_high(allT['variant'],
                                              allT['variant_ALT2'])
    
    #%% TABLE  - LESS FREQUENT, CANCER
    interesting_traces = df_rr_hr.sort_values('RR').iloc[:5].merge(
        variants_patients_per_subject, 
        left_index=True, right_on='variant')[['variant','variant_ALT2']]
    allT = []
    for t in interesting_traces['variant'].drop_duplicates():
        # Only look at traces containing particular chapters for now...
        if not ('C00-D49' in t):
            continue
        print('{} {} {}'.format('-'*20, t, '-'*20))
        tmp = variants_patients_per_subject.loc[
            variants_patients_per_subject['variant']==t].copy()
        allT.append(tmp)

    allT = pd.concat(allT,ignore_index=True)
    x = count_diagnosis_appearances_groupby_high(allT['variant'],
                                                 allT['variant_ALT2'])
        

    #%% Table of ALT traces
    # For each high-level trace, a breakdown of the low-level traces
    # = One html table per high-level trace

    table_of_alt_trace_frequences(variants_controls_per_subject.copy(),
                                  variants_patients_per_subject.copy(),
                                  df_rr_hr.sort_values('RR').index.copy())
    
    
    #%% number of unique traces
    print('Number of unique traces (COMBINED)')
    print(pd.concat([
        variants_patients_per_subject[['variant']],
        variants_controls_per_subject[['variant']]],ignore_index=True).\
            drop_duplicates().shape[0])
    
    print('Number of unique traces (MI)')
    print(variants_patients_per_subject[['variant']].drop_duplicates().shape[0])

    print('Number of unique traces (controls)')
    print(variants_controls_per_subject[['variant']].drop_duplicates().shape[0])


    #%% 3-char freq.

    # for t in interesting_traces['variant'].drop_duplicates():    
    #     print('{} {} {}'.format('-'*20, t, '-'*20))
    #     tmp = variants_patients_per_subject.loc[
    #         variants_patients_per_subject['variant']==t].copy()
    #     tmp.groupby('variant').apply(
    #         lambda x: count_diagnosis_appearances(x['variant_ALT2']))
      
 #   interesting_traces.groupby('variant').apply(lambda x: count_diagnosis_appearances(x['variant_ALT'],lowlevel=True))


    #%% number of traces, using 3-char diags
    print('Num unique trajectories (at 3-char level):')
    print(pd.concat([variants_patients_per_subject['variant_ALT2'],
                     variants_controls_per_subject['variant_ALT2']]).\
          drop_duplicates().shape)


    #%% Average HR for a group of trajectories
    trace_mask_anyI00 = \
        df_rr_hr.index.map(
        lambda trace: not re.search('INIT,{},.*'.\
                                    format('I00-I99'),
                                    trace) is None).values.astype(bool)
    print(df_rr_hr.loc[trace_mask_anyI00]['HR'].mean())
    print(df_rr_hr.loc[trace_mask_anyI00]['HR'].std())
    
    
    
    #%%
    # ######################
    # ######################
    # ######################
    
    #%% Adjacency matrices (for manuscript spaghetti)
    df_adj_c = adj_matrix.generate_dfg_adj_from_traces(variants_controls_count[['count']])
    df_adj_p = adj_matrix.generate_dfg_adj_from_traces(variants_patients_count[['count']])

    v_all = pd.concat([variants_patients_count,
                       variants_controls_count]).reset_index().\
        groupby('index').sum().sort_values('count')
    
    # the THREE adj matrices
    importlib.reload(plot_adj_graph)
    plot_adj_graph.plot_the_three_adj_matrices(df_adj_c, df_adj_p)


    #%% Spaghetti plot (manuscript)
    importlib.reload(plot_adj_graph)

    plot_adj_graph.plot_graph(df_adj_c.values,df_adj_c.index.values,
                              'Matched controls',variants_controls_count['count'].sum(),
                              'Initial')

    plot_adj_graph.plot_graph(df_adj_p.values,df_adj_p.index.values,
                              'MI subjects',variants_patients_count['count'].sum(),
                              'Initial MI')
    
    # empty for GIMP legend
    tmpA = df_adj_p.copy()
    tmpA[:] = 0
    plot_adj_graph.plot_graph(tmpA.values,df_adj_p.index.values,
                              'empty',variants_patients_count['count'].sum(),
                              'Initial MI')
    

    #%% For poster - simplified spaghetti plot
    importlib.reload(plot_adj_graph)

    df_adj_p = adj_matrix.generate_dfg_adj_from_traces(variants_patients_count[['count']])
    plot_adj_graph.plot_graph_poster(df_adj_p.values,df_adj_p.index.values,
                              'MI subjects',variants_patients_count['count'].sum(),
                              'Initial MI')
    
    
    #%%
    # ######################
    # ######################
    # ######################
    
    #%% Granular - adjacency matrices
    
    # Sub-chapters:
    variants_patients_count_alt = variants_patients_per_subject['variant_ALT1'].value_counts()
    # ICD-10 3 char:
    # variants_patients_count_alt = variants_patients_per_subject['variant_ALT2'].value_counts()
    variants_patients_count_alt = pd.DataFrame(variants_patients_count_alt.values,
                                               columns=['count'],index=variants_patients_count_alt.index)
    
    df_adj_p = adj_matrix.generate_dfg_adj_from_traces(variants_patients_count_alt[['count']])

    # drop nomatch
    df_adj_p = df_adj_p.loc[~(df_adj_p.index=='nomatch')]
    df_adj_p = df_adj_p.drop(columns=['nomatch'], errors='ignore')


    # Sub-chapters:
    variants_controls_count_alt = variants_controls_per_subject['variant_ALT1'].value_counts()
    # ICD-10 3 char:
    #variants_controls_count_alt = variants_controls_per_subject['variant_ALT2'].value_counts()
    variants_controls_count_alt = pd.DataFrame(variants_controls_count_alt.values,
                                               columns=['count'],index=variants_controls_count_alt.index)
    
    df_adj_c = adj_matrix.generate_dfg_adj_from_traces(variants_controls_count_alt[['count']])

    # drop nomatch
    df_adj_c = df_adj_c.loc[~(df_adj_c.index=='nomatch')]
    df_adj_c = df_adj_c.drop(columns=['nomatch'], errors='ignore')


    #%% Granular spaghetti plot - node for each sub-chapter, grouped
    # This as an attempt for the LIDA poster
    importlib.reload(plot_adj_graph)
    
    numNodes = 127#30# 127 max
    tmpA = df_adj_p.values[0:numNodes,0:numNodes]
    tmpA_index = np.array(df_adj_p.index)[0:numNodes]
    
    plot_adj_graph.plot_adj(tmpA, tmpA_index, 'MI_granular', isControl=False)
    
    plot_adj_graph.plot_graph_granular(tmpA,tmpA_index,
                                       'MI subjects',variants_patients_count_alt['count'].sum(),
                                       'Initial MI')

    #%% Granular for VIS.JS html spaghetti - saves DOT file for javascript
    importlib.reload(plot_adj_graph)
    plot_adj_graph.plot_graph_granular_for_visjs_javascript(
        df_adj_p.values,df_adj_p.index.values,'MI subjects',
        variants_patients_count_alt['count'].sum(),'Initial MI')
    plot_adj_graph.plot_graph_granular_for_visjs_javascript(
        df_adj_c.values,df_adj_c.index.values,'Controls',
        variants_controls_count_alt['count'].sum(),'Initial non-MI')


    #%%
if __name__ == '__main__':
    main()
    
