# Resting-state functional connectivity analysis scripts

This repository contains analysis scripts used for fMRIPrep-derived resting-state functional connectivity analyses and manuscript supplementary-table generation.

The code is provided for transparency and reproducibility of the analysis workflow. Participant-level clinical data, imaging derivatives, and intermediate result tables are not included because of privacy, ethics, and data-sharing restrictions.

## Repository structure

- `fc_extraction/`: fMRIPrep-only atlas-based ROI time-series extraction and FC matrix construction.
- `analysis/`: main statistical analyses for whole-brain screening, locked candidate-edge moderation, longitudinal plasticity, sensitivity analyses, and main tables.
- `supplementary_tables/`: scripts used to build Supplementary Tables S1-S7.
- `s8/`: scripts used to build and verify Supplementary Table S8.

## Script order

### FC extraction

1. `fc_extraction/01_extract_fc_fmriprep_batch.py`
2. `fc_extraction/02_extract_fc_fmriprep_hc.py`

### Main analysis

1. `analysis/01_whole_brain_baseline_fc_screen.py`
2. `analysis/02_candidate_edge_lineage_audit.py`
3. `analysis/03_fixed_candidate_moderation_plasticity.py`
4. `analysis/04_prepost_fc_waitlist_hc_boundary.py`
5. `analysis/05_treatment_selection_plasticity_boundary.py`
6. `analysis/06_secondary_clinical_outcomes.py`
7. `analysis/07_tms_favoring_longitudinal_plasticity.py`
8. `analysis/08_design_risk_sensitivity_models.py`
9. `analysis/09_table2_interaction_confidence_intervals.py`
10. `analysis/10_table2_interaction_delta_r2.py`
11. `analysis/11_table1_baseline_characteristics.py`

### Supplementary tables

1. `supplementary_tables/01_build_supplementary_tables_s1_s2.py`
2. `supplementary_tables/02_integrate_supplementary_table_s3.py`
3. `supplementary_tables/03_generate_supplementary_tables_s4_s6.py`
4. `supplementary_tables/04_generate_supplementary_table_s7.py`
5. `s8/01_build_supplementary_table_s8.mjs`
6. `s8/02_verify_supplementary_table_s8.mjs`

## fMRIPrep-only preprocessing statement

The FC-extraction scripts in this upload folder are limited to fMRIPrep-derived inputs. Eligible scans must include fMRIPrep `desc-preproc_bold` files and matched `desc-confounds_timeseries.tsv` files. Scans without matched fMRIPrep confounds are skipped by this GitHub-release version.

ROI time-series extraction, nuisance-regressor application, Pearson correlation, and Fisher-z transformation are post-fMRIPrep feature-construction steps rather than a separate image-preprocessing pipeline.

## Requirements

Python scripts were written for Python 3 and use common scientific Python packages:

- `numpy`
- `pandas`
- `scipy`
- `statsmodels`
- `nilearn`
- `nibabel`
- `openpyxl`
- `python-docx`

The S8 scripts are JavaScript modules and require the spreadsheet tooling used in the original analysis environment.

## Data availability

This repository does not contain participant-level data or imaging derivatives. To run the scripts, users must supply appropriate fMRIPrep derivatives, atlas files, and de-identified analysis tables according to applicable ethics approvals and data-sharing agreements.
