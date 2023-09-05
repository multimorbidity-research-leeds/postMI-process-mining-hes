# HES Process Mining pipeline

**Pipeline for the construction of disease trajectories from the HES NHS dataset (2008-2017).**
**Discovers trajectories for an MI cohort, and a matched control cohort.**


## Directories
/pipeline_hes
- contains the code for constructing disease trajectories from HES data

/pipeline_hes/test
- includes all unit-tests > run all by calling test.py
- pipeline_test.py will do a quick run-through of the whole pipeline (using a subset of the data)

## Parameter file
/pipeline_hes/pipeline.ini
- contains user-defined parameters for controlling pipeline settings

## /supporting_data
- ICD-10 chapters to group ICD-10 3-char codes

## Order of operations

### Executing main.py will run the following:

**1) csv_to_parquet.py**
- extracts relevant columns from the raw HES CSV files.
- saves as several smaller parquet files.

**2) load_parquet.py**
- obtain a single smaller parquet file, consisting of MI and control subjects
- uses an external list of MI and matched controls

**3) clean_hes.py**
- remove individuals and episodes which are deemed unusable for building disease trajectories

**4) fliter_hes.py
- without further removal of inidviduals, select the diseases to be included in disease trajectories

**5) traces_hes.py**
- extract disease trajectories from the cleaned and filtered HES data
- produces the following data-structures:
--- df_rr_hr (dataframe containing the common trajectories and the associated relative risks and hazard ratios)
--- variants_patients/controls_per_subject (all found trajectories, per cohort)


### As a final step, run last_step.py to produce plots/tables/graphs of the pipeline output.

**6) last_step.py**
- plotting tables, figures and spaghetti plots for the manuscript
- also makes lots of useful tables of counts (number of diagnosis appearances etc)

---------------

### Author: Chris Hayward
