# fMRIPrep-only FC extraction scripts

This folder contains scripts for atlas-based functional-connectivity extraction from fMRIPrep-derived resting-state fMRI outputs.

## Scripts

- `01_extract_fc_fmriprep_batch.py`: batch extraction for non-HC fMRIPrep derivatives.
- `02_extract_fc_fmriprep_hc.py`: extraction for HC fMRIPrep derivatives.
- `fmriprep_fc_extraction_notes.txt`: brief command-line notes.

## Required input policy

Eligible scans must have:

1. a fMRIPrep preprocessed BOLD file with `desc-preproc_bold` in the filename;
2. a matched fMRIPrep `desc-confounds_timeseries.tsv` file in the functional directory;
3. atlas files supplied by the user at runtime.

Scans without a matched fMRIPrep confounds file are marked as skipped and are not extracted by this GitHub-release version.

## Scope

These scripts perform post-fMRIPrep ROI time-series extraction and functional-connectivity construction. They do not perform image preprocessing outside fMRIPrep.
