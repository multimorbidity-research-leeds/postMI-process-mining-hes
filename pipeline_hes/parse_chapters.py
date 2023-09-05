# -*- coding: utf-8 -*-
"""
Reads in the ICD-10 chapter text files.
ICD-10 chapters are of the form: ([A-N][0-9][0-9]-[A-N][0-9][0-9])
These chapters are used to group together ICD-10 3-character codes.

High-level chapters represent the main ICD-10 chapters (e.g. Neoplasms)
Low-level chapters represent the sub-chapters (e.g. Malignant neoplasms)

@author: Chris Hayward
"""

import numpy as np
import pdb
import pandas as pd

from pipeline_hes.params import params


def _read_chapter_file_highlevel():
    """Read in the high-level (main chapters) ICD-10 chapter file."""
    chapter_file = params.FILE_CHAPTER_TEXT_HEADINGS
    with open(chapter_file, 'r') as f:
        _chaptersRaw = f.readlines()
    return _chaptersRaw

def _read_chapter_file_lowlevel():
    """Read in the low-level (sub-chapters) ICD-10 chapter file."""
    chapter_file = params.FILE_CHAPTER_TEXT_HEADINGS_GRANULAR
    with open(chapter_file, 'r') as f:
        _chaptersRaw = f.readlines()
    return _chaptersRaw


def _split_chapter_lines(_chaptersRaw):
    """For a line in the chapter text file, obtain the left-hand-side and
    right-hand-side chapters. For the low-level chapter file, this corresponds
    to obtaining a mapping between the high- and low-level chapters (where
    one or more sub-chapter exists in a main chapter)."""
    chapters_mapping = pd.Series(dtype=object)
        
    for c in _chaptersRaw:
        chapters_mapping[c.split(' ')[0]] = c.split(' ')[-1].strip()

    # check for repeated pairs:
    if np.logical_and(chapters_mapping.index.duplicated(),
                      chapters_mapping.duplicated()).any():
        raise Exception('Repeated pairs in chapter mapping.')
    
    # drop repeated pairs
    return chapters_mapping


def _read_chapter_file_short_desc_high_level():
    """Read in the high-level (main chapters) ICD-10 chapter file (with
    shortened descriptions for each)."""
    chapter_file = '{}_short_desc.txt'.format(params.FILE_CHAPTER_TEXT_HEADINGS.split('.txt')[0])
    # Read
    with open(chapter_file, 'r') as f:
        _chaptersRaw = f.readlines()
    return _chaptersRaw

def _read_chapter_file_short_desc_low_level():
    """Read in the low-level (sub chapters) ICD-10 chapter file (with
    shortened descriptions for each)."""
    chapter_file = '{}_short_desc.txt'.format(params.FILE_CHAPTER_TEXT_HEADINGS_GRANULAR.split('.txt')[0])
    # Read
    with open(chapter_file, 'r') as f:
        _chaptersRaw = f.readlines()
    return _chaptersRaw


def get_codes_description_short_desc():
    """Get a mapping from high-level ICD-10 chapter to its shortened
    description."""
    _chaptersRaw = _read_chapter_file_short_desc_high_level()
    # get the prefix [A00-B45, ...]
    chaptersLHS = [c.split(' ')[0] for c in _chaptersRaw]
    chaptersDesc = [' '.join(c.split(' ')[1:-1]) for c in _chaptersRaw]
    
    desc = dict(zip(chaptersLHS, chaptersDesc))
    
    desc[params.AMI_CODE] = 'Acute Myocardial Infarction'
    desc[params.CENSOR_CODE] = 'Died/Right-censored'
    desc[params.AMI_INIT_PLOT] = 'Index Hospitalisation Event*'
    desc[params.AMI_INIT] = 'Index Hospitalisation Event*'
    desc[params.CHAPTER_NO_MATCH] = 'No match'
    
    return desc


def get_codes_description_short_desc_low_level():
    """Get a mapping from low-level ICD-10 chapter to its
    description."""
    _chaptersRaw = _read_chapter_file_short_desc_low_level()
    # get the prefix [A00-B45, ...]
    chaptersLHS = [c.split(' ')[0] for c in _chaptersRaw]
    chaptersDesc = [' '.join(c.split(' ')[1:-1]) for c in _chaptersRaw]
    
    desc = dict(zip(chaptersLHS, chaptersDesc))
    
    desc[params.AMI_CODE] = 'Acute Myocardial Infarction'
    desc[params.CENSOR_CODE] = 'Died/Right-censored'
    desc[params.AMI_INIT_PLOT] = 'Index Hospitalisation Event*'
    desc[params.AMI_INIT] = 'Index Hospitalisation Event*'
    desc[params.CHAPTER_NO_MATCH] = 'No match'
    
    # for granular network
    desc['Initial non-MI'] = 'Initial non-MI'
    desc['Initial MI'] = 'Initial MI'
    
    return desc


def get_chapter_mapping_high_level():
    """Get the dictionary mapping from high-level to high-level chapters.
    (placeholder)."""
    # Decide which chapter file to open
    _chaptersRaw = _read_chapter_file_highlevel()    
    return _split_chapter_lines(_chaptersRaw)

def get_chapter_mapping_low_level():
    """Get the dictionary mapping from high-level to low-level chapters."""
    _chaptersRaw = _read_chapter_file_lowlevel()
    return _split_chapter_lines(_chaptersRaw)


def parse_chapter_txt_file(_chaptersRaw):
    """Process the chapter file into a data-structure enabling the conversion
    of ICD-10 codes (limited to three-characters) to be converted into their
    respective chapters."""
  
    # get the prefix [A00-B45, ...]
    chapterNames = [c.split(' ')[0] for c in _chaptersRaw]
        
    chapterRanges = []
    chapterNames_extended = []
    for i,c in enumerate(chapterNames):
        
        # Special: if neuroendocrine (XXA-XXA) (3rd or 7th character)
        if (c[2] in ['A', 'B']) or (c[6] in ['A', 'B']):
            continue
        
        initLetter_start = c[0]
        initLetter_end = c[4]
        n1 = int(c[1:3])
        n2 = int(c[5:7])
        # if the initial alphabetic chars are the same
        if initLetter_start == initLetter_end:
            chapterRanges.append((initLetter_start, initLetter_end, n1, n2))
            chapterNames_extended.append(chapterNames[i])

        # if no intermediate 0-99 (one letter ahead)
        elif (ord(initLetter_start)+1) == ord(initLetter_end):
            chapterRanges.append((initLetter_start, initLetter_start, n1, 99))
            chapterRanges.append((initLetter_end, initLetter_end, 0, n2))
            chapterNames_extended.append(chapterNames[i])
            chapterNames_extended.append(chapterNames[i])

        # fill in the intermediate with 0-99. e.g. B00-B99 for A20-C49
        else:
            chapterRanges.append((initLetter_start, initLetter_start, n1, 99)) # A20 - A99
            chapterRanges.append((initLetter_end, initLetter_end, 0, n2)) # C00 - C49
            chapterNames_extended.append(chapterNames[i])
            chapterNames_extended.append(chapterNames[i])
            
            nextMinChar = chr(ord(initLetter_start)+1) # converts Y -> Z (e.g.)
            prevMaxChar = chr(ord(initLetter_end)-1) # converts Z -> Y (e.g.)
            chapterRanges.append((nextMinChar, prevMaxChar, 0, 99)) # B00 - B99
            chapterNames_extended.append(chapterNames[i])

    return chapterRanges, chapterNames_extended


def build_diag_conversion_dict(unique_diags, chapterRanges, chapterNames_extended):
    """Map 3-character ICD-10 diagnoses to one of the chapters obtained from
    the chapter text file."""
    tbl_conv = {params.AMI_CODE:params.AMI_CODE,
                params.CENSOR_CODE:params.CENSOR_CODE}
    for ud in unique_diags:
        found = False
        if ud == params.AMI_CODE or ud == params.CENSOR_CODE:
            found = True
        elif ud[2] in ['A', 'B']:
            # Neuroendrocrine tumors have a special structure:
            # C7A-C7A Malignant neuroendocrine tumors
            # C7B-C7B Secondary neuroendocrine tumors
            # D3A-D3A Benign neuroendocrine tumors
            tbl_conv[ud] = '{}-{}'.format(ud[:3],ud[:3])
            found = True
        else:
            initChar = ud[0]
            initNum = ud[1:3]
            # look for the conversion in the list of tuples
            for i,rng in enumerate(chapterRanges):
                if (initChar>=rng[0]) and (initChar<=rng[1]) and initNum.isnumeric() \
                    and (int(initNum)>=rng[2]) and (int(initNum)<=rng[3]):
                    tbl_conv[ud] = chapterNames_extended[i]
                    found = True
                    break
        if not found:
            tbl_conv[ud] = params.CHAPTER_NO_MATCH
    return tbl_conv

 
def apply_diag_conversion_dict(df):
    """Entry function - replaces 3-character ICD-10 codes to high level and
    low level chapters (CONV_HIGH and CONV_LOW columns respectively)."""

    _chaptersRaw_coarse = _read_chapter_file_highlevel()
    _chaptersRaw_granular = _read_chapter_file_lowlevel()
    _chaptersRaw_names = ('CONV_HIGH','CONV_LOW')

    # e.g. tbl[A01] -> A00-A99
    unique_diags = df['DIAG_01'].cat.categories.values
            
    for i,_chaptersRaw in enumerate([_chaptersRaw_coarse, _chaptersRaw_granular]):
    
        # get the tuples describing the range (A,A,n,n) and the titles
        chapterRanges, chapterNames_extended = parse_chapter_txt_file(_chaptersRaw)

        tbl_conv = build_diag_conversion_dict(unique_diags, chapterRanges, chapterNames_extended)
        #check_chapter_conversion(tbl_conv)
            
        print('No match:')
        print([(t,y) for t,y in tbl_conv.items() if y==params.CHAPTER_NO_MATCH])
        
        # change the DIAGNOSIS codes to chapter headings 
        df['DIAG_01_{}'.format(_chaptersRaw_names[i])] = \
            df['DIAG_01'].map(tbl_conv).astype('category')

