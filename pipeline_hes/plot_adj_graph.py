# -*- coding: utf-8 -*-
"""
Functions which call on matplotlib and pygraphviz to create visualisations
of the adjacency matrices and their graphical representation (spaghetti plots).
The adjacency matrices here refer to the transisions between pairs of
diseases appearing in disease trajectories.

@author: Chris Hayward
"""

import pathlib
import pandas as pd
import numpy as np
import pdb
import re
import os

from pipeline_hes.params import params
from pipeline_hes import plot_utils
from pipeline_hes import parse_chapters
import pygraphviz as pgv

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.colors import LogNorm


def plot_graph_granular_for_visjs_javascript(adj,ticks_renamed,title,nSubs,initial_label):
    """Creates a graphical reprsentation of the transitions between diseases
    appearing in trajectories (pair-wise). This version plots the sub-chapters
    as nodes, spatially grouped together according to the main ICD chapter."""    
    ticks_renamed[ticks_renamed==params.AMI_INIT] = initial_label
    
    chapter_mapping, labels_idx, colours, text_col = plot_utils.get_chapters_low_level()
    
    labels_idx.pop('MI/Initial')
    chapter_mapping['Initial MI'] = params.AMI_INIT
    chapter_mapping['Initial non-MI'] = params.AMI_INIT
    chapter_mapping['Initial'] = params.AMI_INIT
    chapter_mapping['Initial/MI'] = params.AMI_INIT

    sub_chapter_descs = parse_chapters.get_codes_description_short_desc_low_level()

    labels_idx[params.AMI_INIT] = np.max(list(labels_idx.values()))+1
    colours = np.append(colours,[[.9,.9,.9]],axis=0)
    text_col = np.append(text_col,[[0,0,0]],axis=0)

    nodeNames = ticks_renamed
    
#    edgeColours = plt.get_cmap('viridis',200)
#    edgeColours = np.flipud(np.array([edgeColours(i)[:3] for i in range(0,200)]))

    edgeColours = plt.get_cmap('magma',200)
    edgeColours = np.flipud(np.array([edgeColours(i)[:3] for i in range(0,200)]))

    
    G = _build_graph_granular_for_visjs_javascript(adj,nodeNames,ticks_renamed,labels_idx,
                colours,text_col,chapter_mapping,nSubs,edgeColours,sub_chapter_descs)
    
    pathlib.Path(params.DIR_RESULTS).mkdir(exist_ok=True)
    G.draw(os.path.join(params.DIR_RESULTS,'dfg_granular_visjs_{}.png'.format(title.replace(' ','_'))),
            prog='fdp',
            args='-Gbgcolor="#000000" -GK=3 -Gstart="random4" -Gmaxiter=1') # -Gmaxiter=100 
    
    saveTo = os.path.join(params.DIR_TMP,'graph_dot_out_granular_visjs_{}'.format(title.replace(' ','_')))
    print('Saving to {}'.format(saveTo))
    G.write(saveTo)

    
def _build_graph_granular_for_visjs_javascript(adj,nodeNames,ticks_renamed,labels_idx,
                colours,text_col,chapter_mapping,nSubs,edgeColours,sub_chapter_descs):
    """Calls pygraphviz to assemble a graph containing nodes and edges."""
    
    np.random.seed(0)    
    
    G = pgv.AGraph(directed=True,splines='',margin=.1,size=12,dpi=300)
    
    labelIds_no_dups = pd.Series(labels_idx.values()).drop_duplicates()
    clusterNames = pd.Series(['Infectious/\nParasitic',
              'Neoplasms','Blood/Immune','Endocrine/\nMetabolic',
              'Mental/\nBehavioural','Nervous System','Eye/Adnexa','Ear/Mastoid',
              'Circulatory','Respiratory','Digestive\nSystem','Skin/Subcut.','Musculoskeletal',
              'Genitourinary','Died/End of follow-up','Initial Heart Attack'], index=labelIds_no_dups)
    for i in labelIds_no_dups:
        print(i)
        print(clusterNames[i])
        
        if not (re.search('Initial', clusterNames[i]) is None):
            bgcolor=''
            style=''
            fontsize=50
        elif not (re.search('Died', clusterNames[i]) is None):
            bgcolor=''
            style=''
            fontsize=50
        else:
            bgcolor=''
            fontsize=50
            style=''

        G.add_subgraph(name='cluster_'+str(i),
                       label=clusterNames[i],
                       fontcolor='#ffffff',
                       fontsize=fontsize,
                       color=bgcolor,
                       style=style)
    
    numCats = np.max(list(labels_idx.values()))-1

    
    for i in range(len(ticks_renamed)):
        coarse_label_idx = labels_idx[chapter_mapping[ticks_renamed[i]]]
        print('{}, {}'.format(ticks_renamed[i],coarse_label_idx))
        
        colorRGB = colours[coarse_label_idx,:]*255
        text_col_RGB = text_col[coarse_label_idx,:]*255
        
        colorHex = '#{:02x}{:02x}{:02x}'.format(
            int(colorRGB[0]), int(colorRGB[1]), int(colorRGB[2]))
        textColorHex = '#{:02x}{:02x}{:02x}'.format(
            int(text_col_RGB[0]), int(text_col_RGB[1]), int(text_col_RGB[2]))
        
        
        if ticks_renamed[i] in ['Censor']:
            colorHex = '#666666'
        
        outgoing = adj[i,:].sum() / np.max(adj.sum(axis=0))
        incoming = adj[:,i].sum() / np.max(adj.sum(axis=1))
        node_size = .1 + np.log(1+(outgoing + incoming))/1.5
        
        if i==0 or i==(len(ticks_renamed)-1):
            nodeStyle='filled'
            nodePenWidth=.1
        else:
            nodeStyle='filled'
            nodePenWidth=.1
                    
        G_sub = G.get_subgraph('cluster_'+str(coarse_label_idx))
        
        if ticks_renamed[i] in ['Initial MI']:
            tmpLab = 0
            circPosX = 8*np.sin(2*np.pi*tmpLab/numCats)
            circPosY = 8*np.cos(2*np.pi*tmpLab/numCats)

            G_sub.add_node(nodeNames[i],
                           label=nodeNames[i],
                           labelloc='t',
                           color='#222222', fillcolor='#df2020',
                           style=nodeStyle,
                           penwidth=nodePenWidth,
                           fontsize=np.round(node_size*13),
                           fixedsize=True, height=node_size, width=node_size,
                           pos='{},{}!'.format(circPosX,circPosY))
        else:
            tmpLab = np.array(coarse_label_idx)
            
            circPosX = 8*np.sin(2*np.pi*tmpLab/numCats) + np.random.randn()/2 #*circPosY/12
            circPosY = 8*np.cos(2*np.pi*tmpLab/numCats) + np.random.randn()/2 #*circPosX/12
            
            G_sub.add_node(nodeNames[i],
                           label=nodeNames[i],
                           comment=sub_chapter_descs[nodeNames[i]],
                           labelloc='t',
                           color='#222222',
                           fillcolor=colorHex,
                           fontcolor=textColorHex,
                           style=nodeStyle,
                           penwidth=nodePenWidth,
                           fontsize=np.round(node_size*13),#np.round(node_size*13),
                           fixedsize=True, height=node_size, width=node_size,
                           pos='{},{}!'.format(circPosX-node_size*2,circPosY-node_size*2))
        

        maxPrc = np.max(100*adj/nSubs,axis=(0,1))
        
        for j in range(len(ticks_renamed)): 
            
            # skip if no edge
            if np.isnan(adj[i,j]) or adj[i,j]<=10:
                continue

            #edgeWidth = .1+(np.log10(1+999*adj[i,j]/np.max(adj)))
            constraint=True
            edgeStyle='solid'
            weight=adj[i,j]

            # if ticks_renamed[i] in ['Initial MI']:
            #     colorHex='red'
            
            prc_val = 100 * adj[i,j] / nSubs
            logIdx = int(np.ceil((edgeColours.shape[0]-1)*np.log(1+prc_val)/np.log(1+maxPrc)))
            edgeWidth = min(20,5+prc_val)
            x = 255*np.array(edgeColours[logIdx])
            edgeColourHex = '#{:02x}{:02x}{:02x}'.format(int(x[0]), int(x[1]), int(x[2]))
            
            v_count = int(adj[i,j])
            v_prc = np.round(100 * adj[i,j] / nSubs,1)
            if v_prc==0:
                v_prc = '<0.05'
            
            G.add_edge(nodeNames[i],nodeNames[j],
                       label='{:,} ({}%)'.format(v_count,v_prc),
                       penwidth=edgeWidth,
                       arrowsize=.1,
                       arrowhead='normal',
                       weight=weight,
                       constraint=constraint,
                       style=edgeStyle,
                       color=edgeColourHex)
    return G








def plot_graph_granular(adj,ticks_renamed,title,nSubs,initial_label):
    """Creates a graphical reprsentation of the transitions between diseases
    appearing in trajectories (pair-wise). This version plots the sub-chapters
    as nodes, spatially grouped together according to the main ICD chapter."""    
    ticks_renamed[ticks_renamed==params.AMI_INIT] = initial_label
    
    chapter_mapping, labels_idx, colours, text_col = plot_utils.get_chapters_low_level()
    
    labels_idx.pop('MI/Initial')
    
    chapter_mapping['Initial MI'] = params.AMI_INIT
    chapter_mapping['Initial non-MI'] = params.AMI_INIT
    chapter_mapping['Initial'] = params.AMI_INIT
    chapter_mapping['Initial/MI'] = params.AMI_INIT

    labels_idx[params.AMI_INIT] = np.max(list(labels_idx.values()))+1
    colours = np.append(colours,[[.9,.9,.9]],axis=0)
    text_col = np.append(text_col,[[0,0,0]],axis=0)

    nodeNames = ticks_renamed
    
    edgeColours = plt.get_cmap('viridis',200)
    edgeColours = np.flipud(np.array([edgeColours(i)[:3] for i in range(0,200)]))
    
    G = _build_graph_granular(adj,nodeNames,ticks_renamed,labels_idx,
                colours,text_col,chapter_mapping,nSubs,edgeColours)
    
    pathlib.Path(params.DIR_RESULTS).mkdir(exist_ok=True)
    G.draw(os.path.join(params.DIR_RESULTS,'dfg_granular_{}.png'.format(title.replace(' ','_'))),
            prog='fdp',
            args='-Gbgcolor="#000000" -GK=3 -Gstart="random4" -Gmaxiter=1') # -Gmaxiter=100 
    G.write(os.path.join(params.DIR_TMP,'graph_dot_out_granular_{}'.format(title.replace(' ','_'))))


    
def _build_graph_granular(adj,nodeNames,ticks_renamed,labels_idx,
                colours,text_col,chapter_mapping,nSubs,edgeColours):
    """Calls pygraphviz to assemble a graph containing nodes and edges."""
    
    np.random.seed(0)    
    
    G = pgv.AGraph(directed=True,splines='',margin=.1,size=12,dpi=300)
    
    labelIds_no_dups = pd.Series(labels_idx.values()).drop_duplicates()
    clusterNames = pd.Series(['Infectious/\nParasitic',
              'Neoplasms','Blood/Immune','Endocrine/\nMetabolic',
              'Mental/Behavioural','Nervous System','Eye/Adnexa','Ear/Mastoid',
              'Circulatory','Respiratory','Digestive\nSystem','Skin/Subcut.','Musculoskeletal',
              'Genitourinary','<<B>Died/</B><BR/><B>End of follow-up</B>>','<<B>Initial</B><BR/><B>Heart Attack</B>>'], index=labelIds_no_dups)
    for i in labelIds_no_dups:
        print(i)
        print(clusterNames[i])
        
        if not (re.search('Initial', clusterNames[i]) is None):
            bgcolor=''
            style=''
            fontsize=50
        elif not (re.search('Died', clusterNames[i]) is None):
            bgcolor=''
            style=''
            fontsize=50
        else:
            bgcolor=''
            fontsize=40
            style=''

        G.add_subgraph(name='cluster_'+str(i),
                       label=clusterNames[i],
                       fontcolor='#ffffff',
                       fontsize=fontsize,
                       color=bgcolor,
                       style=style)
    
    numCats = np.max(list(labels_idx.values()))-1

    
    for i in range(len(ticks_renamed)):           
        coarse_label_idx = labels_idx[chapter_mapping[ticks_renamed[i]]]
        print('{}, {}'.format(ticks_renamed[i],coarse_label_idx))
        
        colorRGB = colours[coarse_label_idx,:]*255
        text_col_RGB = text_col[coarse_label_idx,:]*255
        
        colorHex = '#{:02x}{:02x}{:02x}'.format(
            int(colorRGB[0]), int(colorRGB[1]), int(colorRGB[2]))
        textColorHex = '#{:02x}{:02x}{:02x}'.format(
            int(text_col_RGB[0]), int(text_col_RGB[1]), int(text_col_RGB[2]))
        
        
        if ticks_renamed[i] in ['Censor']:
            colorHex = '#666666'
        
        outgoing = adj[i,:].sum() / np.max(adj.sum(axis=0))
        incoming = adj[:,i].sum() / np.max(adj.sum(axis=1))
        node_size = .1 + np.log(1+(outgoing + incoming))/1.5
        
        if i==0 or i==(len(ticks_renamed)-1):
            nodeStyle='filled'
            nodePenWidth=.1
        else:
            nodeStyle='filled'
            nodePenWidth=.1
                    
        G_sub = G.get_subgraph('cluster_'+str(coarse_label_idx))
        
        if ticks_renamed[i] in ['Initial MI']:
            tmpLab = 0
            
            circPosX = 8*np.sin(2*np.pi*tmpLab/numCats)
            circPosY = 8*np.cos(2*np.pi*tmpLab/numCats)

            
            G_sub.add_node(nodeNames[i], label='',#nodeNames[i],
                       labelloc='t',
                       color='#222222', fillcolor='#df2020',
                       style=nodeStyle,
                       penwidth=nodePenWidth,
                       fontsize=np.round(node_size*13),
                       fixedsize=True, height=node_size, width=node_size,
                       pos='{},{}!'.format(circPosX,circPosY))
        else:
            tmpLab = np.array(coarse_label_idx)
            
            circPosX = 8*np.sin(2*np.pi*tmpLab/numCats) + np.random.randn()/2 #*circPosY/12
            circPosY = 8*np.cos(2*np.pi*tmpLab/numCats) + np.random.randn()/2 #*circPosX/12
            
            G_sub.add_node(nodeNames[i], label='',#nodeNames[i],
                       labelloc='t',
                       color='#222222', fillcolor=colorHex,
                       fontcolor=textColorHex,
                       style=nodeStyle,
                       penwidth=nodePenWidth,
                       fontsize=np.round(node_size*13),#np.round(node_size*13),
                       fixedsize=True, height=node_size, width=node_size,
                       pos='{},{}!'.format(circPosX-node_size*2,circPosY-node_size*2))
        
        
        for j in range(len(ticks_renamed)): 
            
            # skip if no edge
            if np.isnan(adj[i,j]) or adj[i,j]==0:
                continue

            edgeWidth = .1+(np.log10(1+99*adj[i,j]/np.max(adj)))
            constraint=True
            edgeStyle='solid'
            weight=adj[i,j]

            if ticks_renamed[i] in ['Initial MI']:
                colorHex='red'
                        
            G.add_edge(nodeNames[i],nodeNames[j],
                       label='',
                       penwidth=edgeWidth,
                       arrowsize=.1,
                       arrowhead='normal',
                       weight=weight,
                       constraint=constraint,
                       style=edgeStyle,
                       color=colorHex)
    return G



def plot_graph_poster(adj,ticks_renamed,title,nSubs,initial_label):
    """Creates a graphical reprsentation of the transitions between diseases
    appearing in trajectories (pair-wise). This version plots a simplified
    graph designed for a poster."""  
    
    ticks_renamed[ticks_renamed==params.AMI_INIT] = initial_label
    
    chapter_mapping, labels_idx, colours, text_col = plot_utils.get_chapters_high_level()
    
    chapter_mapping['Initial MI'] = params.AMI_INIT
    chapter_mapping['Initial non-MI'] = params.AMI_INIT
    chapter_mapping['Initial'] = params.AMI_INIT
    chapter_mapping['Initial/MI'] = params.AMI_INIT

    labels_idx[params.AMI_INIT] = len(labels_idx)
    colours = np.append(colours,[[.9,.9,.9]],axis=0)
    text_col = np.append(text_col,[[0,0,0]],axis=0)

    nodeNames = ['<<B>Initial<BR/>heart attack</B>>',
                 '<<B>Infectious and<BR/>Parasitic</B>>',
                  '<<B>Neoplasms</B>>',
                  '<<B>Blood and<BR/>Immune</B>>',
                  '<<B>Endocrine, Nutritional<BR/>and Metabolic</B>>',
                  '<<B>Mental and<BR/>Behavioural</B>>',
                  '<<B>Nervous System</B>>',
                  '<<B>Eye and Adnexa</B>>',
                  '<<B>Ear and Mastoid</B>>',
                  '<<B>Circulatory</B>>',
                  '<<B>Respiratory</B>>',
                  '<<B>Digestive<BR/>System</B>>',
                  '<<B>Skin and<BR/>Subcutaneous</B>>',
                  '<<B>Musculoskeletal</B>>',
                  '<<B>Genitourinary</B>>',
                  '<<B>End of follow-up<BR/>or death</B>>']
    
    edgeColours = plt.get_cmap('inferno',200)
    edgeColours = np.array([edgeColours(i)[:3] for i in range(100,180)])
    
    G = _build_graph_poster(adj,nodeNames,ticks_renamed,labels_idx,
                colours,text_col,chapter_mapping,nSubs,edgeColours)
    
    edgeFontSize = 100
    pathlib.Path(params.DIR_RESULTS).mkdir(exist_ok=True)
    G.draw(os.path.join(params.DIR_RESULTS,'dfg_poster_{}.png'.format(title.replace(' ','_'))),
           args='-Gratio=1.4 -Granksep=2 -Efontsize={} -Gbgcolor="#333333" -Gmaxiter=1'.\
               format(edgeFontSize), prog='dot')
    
    
def _build_graph_poster(adj,nodeNames,ticks_renamed,labels_idx,
                colours,text_col,chapter_mapping,nSubs,edgeColours):
    """Calls pygraphviz to assemble a graph containing nodes and edges."""
    G = pgv.AGraph(directed=True,splines='',margin=.1,size=12,dpi=300)

    for i in range(len(ticks_renamed)):   
                
        print(nodeNames[i])
        fontsize = 100
        nodeColor = '#333333'
        nodeStyle=''
        
        nodeSize_h = 5
        nodeSize_w = 10
        
        if ticks_renamed[i]=='I00-I99':
            nodeSize_h = 10
            nodeSize_w = 20
        if ticks_renamed[i]=='Initial MI':
            nodeSize_h = 10
            nodeSize_w = 10
            fontsize=150
            nodeStyle='bold'
            nodeColor='#993333'
        if ticks_renamed[i]=='Censor':
            nodeSize_h = 10
            nodeSize_w = 20
            nodeStyle='bold'
            nodeColor='#993333'
            fontsize=150
        
        G.add_node(nodeNames[i], label=nodeNames[i],
                   penwidth=0,
                   color=nodeColor,
                   fontcolor='white',
                   fontname='Arial',
                   style=nodeStyle,
                   fontsize=fontsize,
                   fixedsize=False, height=nodeSize_h, width=nodeSize_w)
        
        for j in range(len(ticks_renamed)): 
            
            # skip if no edge
            if np.isnan(adj[i,j]) or adj[i,j]<10:
                continue
            
            prc_val = 100 * adj[i,j] / nSubs
            
            label = ""
            weight=1
            constraint=False
            edgeStyle='solid'

            if prc_val >= .1:
                constraint=False
                weight=2
                
            if prc_val >= .5:
                constraint=True
                weight=3

            if prc_val==100*np.max(adj)/nSubs:
                label = '    '
                
            maxPrc = np.max(100*adj/nSubs,axis=(0,1))
            logIdx = int(np.ceil((edgeColours.shape[0]-1)*np.log(1+prc_val)/np.log(1+maxPrc)))

            edgeWidth = min(20,5+prc_val)

            x = 255*np.array(edgeColours[logIdx])
            edgeColourHex = '#{:02x}{:02x}{:02x}'.format(int(x[0]), int(x[1]), int(x[2]))
            G.add_edge(nodeNames[i],nodeNames[j],
                       xlabel=label,
                       labelfloat=False,
                       penwidth=edgeWidth,
                       arrowsize=5,#min(edgeWidth*2,10),
                       arrowhead='normal',
                       weight=weight,
                       constraint=constraint,
                       style=edgeStyle,
                       color=edgeColourHex,
                       fontcolor='white')
    return G


def plot_graph(adj,ticks_renamed,title,nSubs,initial_label):
    """Creates a graphical reprsentation of the transitions between diseases
    appearing in trajectories (pair-wise). This version creates a graph for
    use in the manuscript."""
    
    ticks_renamed[ticks_renamed==params.AMI_INIT] = initial_label
    
    chapter_mapping, labels_idx, colours, text_col = plot_utils.get_chapters_high_level()
    
    chapter_mapping['Initial MI'] = params.AMI_INIT
    chapter_mapping['Initial non-MI'] = params.AMI_INIT
    chapter_mapping['Initial'] = params.AMI_INIT
    chapter_mapping['Initial/MI'] = params.AMI_INIT

    labels_idx[params.AMI_INIT] = len(labels_idx)
    colours = np.append(colours,[[.9,.9,.9]],axis=0)
    text_col = np.append(text_col,[[0,0,0]],axis=0)

    nodeNames = [initial_label,'Infectious/\nParasitic',
              'Neoplasms','Blood/\nImmune','Endocrine/\nMetabolic',
              'Mental/\nBehavioural','Nervous\nSystem','Eye/\nAdnexa','Ear/\nMastoid',
              'Circulatory\nSystem','Respiratory\nSystem','Digestive\nSystem',
              'Skin/Subcut.\nTissue','Musculo-\nskeletal',
              'Genito-\nurinary','Censor']
    
    edgeColours = plt.get_cmap('magma',200)
    edgeColours = np.flipud(np.array([edgeColours(i)[:3] for i in range(0,200)]))
    
    G = _build_graph(adj,nodeNames,ticks_renamed,labels_idx,
                colours,text_col,chapter_mapping,nSubs,edgeColours)
    
    edgeFontSize = 100

    pathlib.Path(params.DIR_RESULTS).mkdir(exist_ok=True)
    G.draw(os.path.join(params.DIR_RESULTS,'dfg_{}.pdf'.format(title.replace(' ','_'))),
           args='-Gratio=2.5 -Granksep=1 -Efontsize={} -Gbgcolor="#ffffff" -Gcolor="#ffffff"'.\
               format(edgeFontSize), prog='dot')
    G.write(os.path.join(params.DIR_TMP,'graph_dot_out_{}'.format(title.replace(' ','_'))))
    
    
def _build_graph(adj,nodeNames,ticks_renamed,labels_idx,
                colours,text_col,chapter_mapping,nSubs,edgeColours):
    """Calls pygraphviz to assemble a graph containing nodes and edges."""

    G = pgv.AGraph(directed=True,splines='',margin=.1,size=12,dpi=300)

    for i in range(len(ticks_renamed)):
        
        coarse_label_idx = labels_idx[chapter_mapping[ticks_renamed[i]]]
        
        colorRGB = colours[coarse_label_idx,:]*255
        text_col_RGB = text_col[coarse_label_idx,:]*255
        
        colorHex = '#{:02x}{:02x}{:02x}'.format(
            int(colorRGB[0]), int(colorRGB[1]), int(colorRGB[2]))
        textColorHex = '#{:02x}{:02x}{:02x}'.format(
            int(text_col_RGB[0]), int(text_col_RGB[1]), int(text_col_RGB[2]))
        
        node_size = 10
        
        if i==0 or i==(len(ticks_renamed)-1):
            node_size = 10
            nodeStyle='filled,bold'
            nodePenWidth=20
        else:
            nodeStyle='filled'
            nodePenWidth=5
            
        print(nodeNames[i])
        
        G.add_node(nodeNames[i], label=nodeNames[i],
                   color='#222222', fillcolor=colorHex,
                   fontcolor=textColorHex,
                   style=nodeStyle,
                   penwidth=nodePenWidth,
                   fontsize=np.round(node_size*12),
                   fixedsize=True, height=node_size/2, width=node_size)
        
        for j in range(len(ticks_renamed)): 
            
            # skip if no edge
            if np.isnan(adj[i,j]) or adj[i,j]<10:
                continue
            
            prc_val = 100 * adj[i,j] / nSubs
            
            prc_thresh = 0
            
            if prc_val < prc_thresh:
                constraint=True
                label = ""
                weight=1
                edgeStyle='solid'
            
            if prc_val >= prc_thresh:
                constraint=True
                label = """<<TABLE BORDER="0">
                            <TR>
                            <TD BORDER="0">{:,}</TD>
                            </TR>
                            <TR>
                            <TD BORDER="0">({}%)</TD>
                            </TR>
                            </TABLE>>""".format(int(adj[i,j]), np.round(prc_val,2))
                weight=2
                edgeStyle='solid'

            maxPrc = np.max(100*adj/nSubs,axis=(0,1))
            logIdx = int(np.ceil((edgeColours.shape[0]-1)*np.log(1+prc_val)/np.log(1+maxPrc)))

            edgeWidth = min(20,5+prc_val)
                        
            x = 255*np.array(edgeColours[logIdx])
            edgeColourHex = '#{:02x}{:02x}{:02x}'.format(int(x[0]), int(x[1]), int(x[2]))
            G.add_edge(nodeNames[i],nodeNames[j],
                       label=label,
                       labelfloat=False,
                       penwidth=edgeWidth,
                       arrowsize=5,
                       arrowhead='normal',
                       weight=weight,
                       constraint=constraint,
                       style=edgeStyle,
                       color=edgeColourHex)

    return G



def plot_adj(adj, adj_ticks, title, isControl):
    """Sets the text information for the subsequent plotting of the
    disease-transition adjacency matrices."""
    
    ticks_renamed = adj_ticks.copy()
    
    if isControl==True:
        ticks_renamed[adj_ticks==params.AMI_INIT] = 'Initial non-MI'
        cbar_label = 'Number of subjects'
    elif isControl==False:
        ticks_renamed[adj_ticks==params.AMI_INIT] = 'Initial MI'
        cbar_label = 'Number of subjects'
    else:
        #cbar_label = '(AMI% - Controls%) / Controls%'
        ticks_renamed[adj_ticks==params.AMI_INIT] = 'Initial'
        cbar_label = 'MI percentage difference relative to controls'
        
    # Plot adj matrix
    _plot_adj(adj,'Succeeding disease category','Preceeding disease category',
            cbar_label, ticks_renamed, title, isControl)
    

def plot_the_three_adj_matrices(df_adj_c, df_adj_p):
    """Calls functions which ultimate plot three adjacency matrices: 1) for
    the MI cohort; 2) for the control cohort; 3) of the relative difference
    in percentages of subjects transitioning between diseases between the
    two cohorts."""
    adj_ticks = list(df_adj_c.index)
    plot_adj(df_adj_c, adj_ticks, 'Controls', isControl=True)
    plot_adj(df_adj_p, adj_ticks, 'MI', isControl=False)
    # Relative difference
    x = 100 * ((df_adj_p/(df_adj_p).sum().sum()) - \
               (df_adj_c/(df_adj_c).sum().sum())) / (df_adj_c/(df_adj_c).sum().sum())
    # get rid of 'inf'
    x[df_adj_c==0] = np.nan
    plot_adj(x, adj_ticks, 'MI versus Controls', isControl=np.NaN)    


def get_adj_cmap_diverging(minVal,maxVal):
    """Gets a reb-blue colourmap used to visualise the differences in
    numbers of individuals transitioning between diseases, between the MI
    and control cohorts."""
    
    # assumes minVal<0, maxVal>0
    maxOf = max(abs(minVal),abs(maxVal))
    rng = 512
    upper = int((abs(maxVal) / maxOf) * rng)
    lower = int((abs(minVal) / maxOf) * rng)
    print(upper)
    print(lower)
    c = cm.get_cmap('seismic',rng*2)
    newcolors = c(np.arange(rng-lower,upper+rng))

    newcmp = ListedColormap(newcolors)
    return newcmp


def get_adj_cmap():
    """For a single cohort, get the colour scheme for the adjacency matrix."""
    # Colors
    c = cm.get_cmap('viridis',1024) # coolwarm
    newcolors = c(np.arange(0,1024))
    
    newcmp = ListedColormap(newcolors)
    return newcmp


def _plot_adj(A,xl,yl,cl,tick_labels,title,isControl):
    """Sub-routine which calls upon matplotlib to plot the matrices."""
    plt.figure(figsize=(7.5,7.5),dpi=300)

    newcmp = get_adj_cmap()
    # x and y ticks - replace AMI/initial depending on cohort
    if np.isnan(isControl):
        # Plotting DIFF
        # use red/blue diverging colormap instead
        # Do not log-scale
        
        minVal = (A[~np.isnan(A)]).min().min()
        maxVal = np.abs(A[~np.isnan(A)]).max().max()
        newcmp = get_adj_cmap_diverging(minVal,maxVal)
        plt.imshow(A, cmap=newcmp, aspect='equal',
                   vmin=minVal,
                   vmax=maxVal)

        cbar=plt.colorbar()
        cbar.ax.set_ylabel(cl, rotation=90)
        cbar.ax.set_yticklabels(['{0:+}%'.format(int(x)) for x in cbar.ax.get_yticks()])

    else:
        plt.imshow(A, cmap=newcmp, aspect='equal', norm = LogNorm())
        cbar=plt.colorbar()
        cbar.ax.set_ylabel(cl, rotation=90)

    
    plt.gca().set_facecolor('k')
    plt.subplots_adjust(bottom=0.20)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.xticks(range(A.shape[0]), tick_labels, rotation=80)
    plt.yticks(range(A.shape[0]), tick_labels)
    
    n = A.shape[0]
    ax = plt.gca()
    ax.set_xticks(np.arange(0, n, 1)-.5, minor=True)
    ax.set_yticks(np.arange(0, n, 1)-.5, minor=True)
    ax.grid(which='minor', color='#888888', linestyle='-', linewidth=1)
    
    plt.title(title, fontsize=15)

    plot_utils.saveFig('adj_{}'.format(title.replace(' ','_')))
    plt.show()
    
    return tick_labels
    
