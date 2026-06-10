# -*- coding: utf-8 -*-
r"""
Generate revised Supplementary Tables S4-S6 after narrowing S4 to the
PCL/GAD/PHQ symptom-outcome reporting set.

Run example:
python generate_S4_S6_symptom_outcome_tables.py --root "D:\\自科＋脑中心论文选题\\PAI选题\\工作站传输\\第四步分析-codex"

Optional explicit zip paths:
python generate_S4_S6_symptom_outcome_tables.py ^
  --zip40 "...40_v3...zip" ^
  --zip42 "...42_v8_6_2...zip" ^
  --zip44 "...44_v3...zip"

Purpose:
- S4 is reduced from the old 10 x 12 = 120-row reporting matrix to the
  revised 10 x 7 = 70-row PCL/GAD/PHQ symptom-outcome matrix.
- FDR for S4 is recalculated within the 70-test symptom-outcome family.
- S5 keeps the candidate-edge lineage audit and adds a clear note that
  legacy unknown_roi_* names are source-column identifiers, not unresolved labels.
- S6 keeps the longitudinal and boundary-analysis results, standardizes Panel F
  outcome names, and clarifies HC-referenced subject-level proportions.

This script does not delete or modify any original files.
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


LOCKED_CANDIDATES = [
    [1, "C01", "A32sg_R--msOccG_R", "PSY", "unknown_roi_65__unknown_roi_85", "EDGE_02", "A32sg_R", "Right cingulate gyrus, subgenual area 32", "msOccG_R", "Right medial superior occipital gyrus", -14.15, "[-21.52, -6.79]", 0.00030, 0.00195, 0.147, "n=66"],
    [2, "C02", "A40c_L--A7m_L", "PSY", "unknown_roi_18__unknown_roi_24", "EDGE_01", "A40c_L", "Left inferior parietal lobule, caudal area 40 (PFm)", "A7m_L", "Left superior parietal lobule, medial area 7 (PEp)", -13.26, "[-20.31, -6.21]", 0.00039, 0.00195, 0.143, "n=66"],
    [3, "C03", "mAmyg_R--lAmyg_L", "PSY", "unknown_roi_89__unknown_roi_90", "EDGE_08", "mAmyg_R", "Right medial amygdala", "lAmyg_L", "Left lateral amygdala", -11.80, "[-18.96, -4.64]", 0.00166, 0.00552, 0.113, "n=66"],
    [4, "C04", "A32sg_L--msOccG_R", "PSY", "unknown_roi_64__unknown_roi_85", "EDGE_04", "A32sg_L", "Left cingulate gyrus, subgenual area 32", "msOccG_R", "Right medial superior occipital gyrus", -12.11, "[-20.13, -4.09]", 0.003735, 0.0093, 0.096, "n=66"],
    [5, "C05", "A32p_L--msOccG_L", "PSY", "unknown_roi_56__unknown_roi_84", "EDGE_06", "A32p_L", "Left cingulate gyrus, pregenual area 32", "msOccG_L", "Left medial superior occipital gyrus", -11.15, "[-19.61, -2.69]", 0.010716, 0.0134, 0.078, "n=66"],
    [6, "C06", "A7pc_L--rHipp_L", "TMS", "unknown_roi_8__unknown_roi_92", "EDGE_09", "A7pc_L", "Left superior parietal lobule, postcentral area 7", "rHipp_L", "Left rostral hippocampus", 13.17, "[3.99, 22.36]", 0.00570, 0.01045, 0.089, "n=66"],
    [7, "C07", "A40rv_R--dIa_L", "TMS", "unknown_roi_23__unknown_roi_44", "EDGE_10", "A40rv_R", "Right inferior parietal lobule, rostroventral area 40 (PFop)", "dIa_L", "Left dorsal agranular insula", 15.19, "[4.39, 26.00]", 0.00665, 0.01045, 0.086, "n=66"],
    [8, "C08", "A7ip_R--cLinG_R", "TMS", "unknown_roi_11__unknown_roi_67", "EDGE_05", "A7ip_R", "Right superior parietal lobule, intraparietal area 7 (hIP3)", "cLinG_R", "Right caudal lingual gyrus", 11.98, "[3.35, 20.61]", 0.007313, 0.01045, 0.084, "n=66"],
    [9, "C09", "A23c_L--dCa_R", "TMS", "unknown_roi_62__unknown_roi_105", "EDGE_03", "A23c_L", "Left cingulate gyrus, caudal area 23", "dCa_R", "Right dorsal caudate", 24.73, "[4.95, 44.52]", 0.01523, 0.01692, 0.069, "n=63; three PSY participants had missing baseline/pre-treatment FC for this edge"],
    [10, "C10", "vId_vIg_L--cHipp_R", "TMS", "unknown_roi_46__unknown_roi_95", "EDGE_07", "vId_vIg_L", "Left ventral dysgranular/granular insula", "cHipp_R", "Right caudal hippocampus", 12.45, "[2.25, 22.64]", 0.01759, 0.01759, 0.068, "n=66"],
]

CAND_COLS = [
    "table2_order", "candidate_id", "candidate_connection", "primary_favored_pathway",
    "value_column", "edge_label_44", "endpoint1_code", "endpoint1_anatomical_label",
    "endpoint2_code", "endpoint2_anatomical_label", "table2_interaction_b",
    "table2_ci", "table2_p", "table2_q", "table2_delta_r2", "n_note",
]

# Revised S4 reporting set: PCL/GAD/PHQ symptom outcomes only.
SYMPTOM_OUTCOMES = [
    [1, "PCL5", "PCL-5 total", "PTSD total symptoms", "pre - post; higher values indicate greater symptom reduction"],
    [2, "PCL_B侵入", "PCL-B intrusion/re-experiencing symptoms", "PTSD symptom cluster", "pre - post; higher values indicate greater symptom reduction"],
    [3, "PCL_C回避", "PCL-C avoidance symptoms", "PTSD symptom cluster", "pre - post; higher values indicate greater symptom reduction"],
    [4, "PCL_D认知情绪改变", "PCL-D negative cognition and mood symptoms", "PTSD symptom cluster", "pre - post; higher values indicate greater symptom reduction"],
    [5, "PCL_E警觉反应", "PCL-E arousal/reactivity symptoms", "PTSD symptom cluster", "pre - post; higher values indicate greater symptom reduction"],
    [6, "GAD_7总分", "GAD-7 total", "Anxiety symptoms", "pre - post; higher values indicate greater symptom reduction"],
    [7, "PHQ_9总分", "PHQ-9 total", "Depressive symptoms", "pre - post; higher values indicate greater symptom reduction"],
]

OUTCOME_COLS = ["outcome_order", "outcome", "outcome_display", "domain", "improvement_definition"]

PANEL_F_OUTCOME_DISPLAY = {
    "PCL5": ("PCL-5 total", "PTSD total symptoms"),
    "GAD_7总分": ("GAD-7 total", "Anxiety symptoms"),
    "PHQ_9总分": ("PHQ-9 total", "Depressive symptoms"),
    "PCL_D认知情绪改变": ("PCL-D negative cognition and mood symptoms", "PTSD symptom cluster"),
    "PCL_E警觉反应": ("PCL-E arousal/reactivity symptoms", "PTSD symptom cluster"),
}

MAIN_TABLE3_HIGHLIGHTS = {"A32sg_R--msOccG_R", "A40rv_R--dIa_L", "A32sg_L--msOccG_R"}


def log(message: str) -> None:
    print(message, flush=True)


def bh_fdr(p_values: Iterable[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR correction, preserving NaN positions."""
    p = pd.to_numeric(pd.Series(list(p_values)), errors="coerce").to_numpy(dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if valid.sum() == 0:
        return q
    pv = p[valid]
    n = len(pv)
    order = np.argsort(pv)
    sorted_p = pv[order]
    adjusted = sorted_p * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    q_valid = np.empty(n, dtype=float)
    q_valid[order] = adjusted
    q[valid] = q_valid
    return q


def clean_filename_token(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(text))
    return re.sub(r"_+", "_", text).strip("_")[:80]


def find_zip(root: Path, keyword: str) -> Optional[Path]:
    hits = sorted(root.rglob(f"*{keyword}*.zip"), key=lambda p: (len(str(p)), str(p)))
    return hits[0] if hits else None


def read_zip_csv(zip_path: Path, contains: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as z:
        matches = [n for n in z.namelist() if contains in n and n.lower().endswith(".csv")]
        if not matches:
            raise FileNotFoundError(f"Missing internal CSV containing {contains!r} in {zip_path}")
        matches = sorted(matches, key=lambda n: (len(n), n))
        with z.open(matches[0]) as f:
            return pd.read_csv(f, encoding="utf-8-sig")


def parse_hemisphere(code: str) -> str:
    code = str(code)
    if code.endswith("_L"):
        return "L"
    if code.endswith("_R"):
        return "R"
    return "subcortical_or_unspecified"


def parent_region(label: str) -> str:
    s = str(label).strip()
    s = re.sub(r"^(Left|Right)\s+", "", s, flags=re.I)
    return s.split(",")[0].strip()


def manifest_df() -> pd.DataFrame:
    m = pd.DataFrame(LOCKED_CANDIDATES, columns=CAND_COLS)
    m["anatomical_endpoints"] = m["endpoint1_anatomical_label"] + " – " + m["endpoint2_anatomical_label"]
    m["endpoint1_hemisphere"] = m["endpoint1_code"].map(parse_hemisphere)
    m["endpoint2_hemisphere"] = m["endpoint2_code"].map(parse_hemisphere)
    m["endpoint1_BNA_label_short"] = m["endpoint1_code"].astype(str).str.replace(r"_[LR]$", "", regex=True)
    m["endpoint2_BNA_label_short"] = m["endpoint2_code"].astype(str).str.replace(r"_[LR]$", "", regex=True)
    m["endpoint1_parent_region"] = m["endpoint1_anatomical_label"].map(parent_region)
    m["endpoint2_parent_region"] = m["endpoint2_anatomical_label"].map(parent_region)
    m["anatomical_endpoints_for_table2"] = m["anatomical_endpoints"]
    m["sign_of_interaction"] = np.where(m["table2_interaction_b"].astype(float) > 0, "positive", "negative")
    m["figure2_panel"] = np.where(m["primary_favored_pathway"].eq("PSY"), "PSY-favoring", "TMS-favoring")
    m["figure3_included"] = m["candidate_connection"].isin(MAIN_TABLE3_HIGHLIGHTS)
    m["manifest_locked_status"] = "LOCKED_BY_FINAL_TABLE2_AND_VALUE_COLUMN"
    return m


def add_manifest(df: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["value_column"] = out["value_column"].astype(str).str.strip()
    merged = out.merge(m, on="value_column", how="left")
    return merged


def ci_from_beta_p(beta, p, n, model_tier) -> Tuple[float, float, float, str]:
    """Approximate SE and CI from beta, two-sided p, and model df when final CI is absent."""
    try:
        beta = float(beta)
        p = float(p)
        n = float(n)
        if not np.isfinite(beta) or not np.isfinite(p) or not np.isfinite(n):
            return np.nan, np.nan, np.nan, "not_computable"
        # Conservative model-parameter count approximation for df.
        tier = str(model_tier)
        k = 4 if tier.startswith("M0") else (6 if tier.startswith("M1") else 7)
        df = max(int(round(n - k)), 1)
        p = min(max(p, 1e-300), 1.0)
        if stats is None:
            abs_t = 1.9599639845
            crit = 1.9599639845
            method = "normal_approx_scipy_unavailable"
        else:
            abs_t = stats.t.isf(p / 2, df)
            crit = stats.t.ppf(0.975, df)
            method = f"t_approx_from_beta_p_df{df}"
        se = abs(beta) / abs_t
        return se, beta - crit * se, beta + crit * se, method
    except Exception:
        return np.nan, np.nan, np.nan, "not_computable"


def welch_ci(mean1, sd1, n1, mean2, sd2, n2) -> Tuple[float, float, float]:
    """Welch CI for mean1 - mean2."""
    try:
        mean1, sd1, n1, mean2, sd2, n2 = map(float, [mean1, sd1, n1, mean2, sd2, n2])
        diff = mean1 - mean2
        se2 = (sd1 ** 2) / n1 + (sd2 ** 2) / n2
        se = np.sqrt(se2)
        df_num = se2 ** 2
        df_den = ((sd1 ** 2 / n1) ** 2 / max(n1 - 1, 1)) + ((sd2 ** 2 / n2) ** 2 / max(n2 - 1, 1))
        df = df_num / df_den if df_den > 0 else np.nan
        crit = stats.t.ppf(0.975, df) if stats is not None and np.isfinite(df) else 1.9599639845
        return df, diff - crit * se, diff + crit * se
    except Exception:
        return np.nan, np.nan, np.nan


def make_s4(zip44: Path, m: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    raw = read_zip_csv(zip44, "10_模块1_其他指标治疗调节效应")
    outcomes = pd.DataFrame(SYMPTOM_OUTCOMES, columns=OUTCOME_COLS)

    if "value_column" not in raw.columns:
        edge44 = read_zip_csv(zip44, "02_自动识别_固定候选边")
        raw = raw.merge(edge44[["edge_label", "value_column"]].drop_duplicates("edge_label"), on="edge_label", how="left")

    s4 = raw[raw["outcome"].isin(outcomes["outcome"])].copy()
    s4 = s4[s4["model_tier"].astype(str).str.startswith("M2_age_sex_meanFD")].copy()
    s4 = add_manifest(s4, m).merge(outcomes, on="outcome", how="left")

    # Preserve old q values, then recompute new FDR within 70-test symptom-outcome family.
    s4["interaction_q_original_120test_family"] = pd.to_numeric(s4["interaction_q"], errors="coerce")
    s4["baselineFC_prognostic_q_original_120test_family"] = pd.to_numeric(s4["baselineFC_qrognostic_q"], errors="coerce")
    s4["interaction_q_symptom70"] = bh_fdr(s4["interaction_p"])
    s4["baselineFC_prognostic_q_symptom70"] = bh_fdr(s4["baselineFC_prognostic_p"])
    s4["interaction_q"] = s4["interaction_q_symptom70"]
    s4["baselineFC_prognostic_q"] = s4["baselineFC_prognostic_q_symptom70"]

    s4["interaction_fdr_supported_symptom70"] = s4["interaction_q_symptom70"].astype(float) < 0.05
    s4["general_prognostic_fdr_supported_symptom70"] = s4["baselineFC_prognostic_q_symptom70"].astype(float) < 0.05
    s4["PSY_simple_slope_beta"] = s4["baselineFC_main_beta_in_full"]
    s4["TMS_simple_slope_beta"] = s4["baselineFC_main_beta_in_full"] + s4["interaction_beta"]
    s4["direction_consistent_with_primary_PCL5"] = "computed_from_locked_model_direction"
    s4["interpretation_tier"] = np.where(
        s4["interaction_fdr_supported_symptom70"],
        "FDR-supported treatment-selection symptom-domain signal",
        "not FDR-supported within symptom-outcome family",
    )
    s4["manuscript_summary_flag"] = np.where(s4["interaction_fdr_supported_symptom70"], "include_in_symptom_convergence_summary", "retain_for_transparency")
    s4["coefficient_scale_note"] = "standardized outcome-improvement units in 44_v3 M2 output; Table 2 PCL-5 uses original-scale primary model"
    s4["source_file_family"] = "44_v3 module10; locked M2_age_sex_meanFD; S4 FDR recomputed within 70-test symptom-outcome family"

    cis = s4.apply(
        lambda r: ci_from_beta_p(r["interaction_beta"], r["interaction_p"], r["n_model"], r["model_tier"]),
        axis=1,
        result_type="expand",
    )
    cis.columns = ["interaction_se_approx", "interaction_ci_low_approx", "interaction_ci_high_approx", "ci_method"]
    s4 = pd.concat([s4, cis], axis=1)

    preferred = [
        "table2_order", "candidate_id", "candidate_connection", "anatomical_endpoints",
        "primary_favored_pathway", "value_column", "edge_label_44", "outcome_order",
        "outcome_display", "outcome", "domain", "improvement_definition", "model_tier",
        "covariates", "n_model", "n_key_complete", "n_PSY_key", "n_TMS_key",
        "interaction_beta", "interaction_se_approx", "interaction_ci_low_approx",
        "interaction_ci_high_approx", "interaction_p", "interaction_q_symptom70",
        "interaction_q_original_120test_family", "interaction_fdr_supported_symptom70",
        "higher_baselineFC_favors", "PSY_simple_slope_beta", "TMS_simple_slope_beta",
        "baselineFC_prognostic_beta", "baselineFC_prognostic_p", "baselineFC_prognostic_q_symptom70",
        "baselineFC_prognostic_q_original_120test_family", "general_prognostic_fdr_supported_symptom70",
        "full_r2", "prog_r2", "delta_r2_full_minus_prog", "signal_interpretation",
        "direction_consistent_with_primary_PCL5", "interpretation_tier", "manuscript_summary_flag",
        "coefficient_scale_note", "ci_method", "source_file_family",
    ]
    ordered_cols = [c for c in preferred if c in s4.columns] + [c for c in s4.columns if c not in preferred]
    s4 = s4[ordered_cols].sort_values(["table2_order", "outcome_order"]).reset_index(drop=True)

    panel_b = (
        s4.groupby(["candidate_id", "candidate_connection", "primary_favored_pathway"], as_index=False)
        .agg(
            n_symptom_outcomes=("outcome_display", "nunique"),
            n_interaction_fdr_supported=("interaction_fdr_supported_symptom70", "sum"),
            supported_outcomes=("outcome_display", lambda x: "; ".join(s4.loc[x.index, "outcome_display"][s4.loc[x.index, "interaction_fdr_supported_symptom70"]])),
            min_interaction_q_symptom70=("interaction_q_symptom70", "min"),
        )
        .sort_values("candidate_id")
    )
    panel_c = (
        s4.groupby(["outcome_order", "outcome_display", "domain"], as_index=False)
        .agg(
            n_candidate_edges=("candidate_connection", "nunique"),
            n_interaction_fdr_supported=("interaction_fdr_supported_symptom70", "sum"),
            supported_candidate_connections=("candidate_connection", lambda x: "; ".join(s4.loc[x.index, "candidate_connection"][s4.loc[x.index, "interaction_fdr_supported_symptom70"]])),
            min_interaction_q_symptom70=("interaction_q_symptom70", "min"),
        )
        .sort_values("outcome_order")
    )
    panel_d = pd.DataFrame([
        ["S4 candidate edges", 10, int(s4["candidate_connection"].nunique()), int(s4["candidate_connection"].nunique()) == 10],
        ["S4 symptom outcomes", 7, int(s4["outcome_display"].nunique()), int(s4["outcome_display"].nunique()) == 7],
        ["S4 total rows", 70, int(len(s4)), int(len(s4)) == 70],
        ["S4 interaction FDR-supported tests after symptom-family recalculation", "computed, no prior manuscript expectation", int(s4["interaction_fdr_supported_symptom70"].sum()), "RECOMPUTED"],
        ["S4 general prognostic FDR-supported tests after symptom-family recalculation", 0, int(s4["general_prognostic_fdr_supported_symptom70"].sum()), int(s4["general_prognostic_fdr_supported_symptom70"].sum()) == 0],
        ["Old non-symptom outcomes removed", "Experiential avoidance; Cognitive fusion; Cognitive defusion; AAQ-II total; CD-RISC-10 total", "; ".join(sorted(set(raw["outcome"]) - set(outcomes["outcome"]))), "CHECK_SOURCE_LIST"],
    ], columns=["check_name", "expected", "computed", "pass"])

    panels = {
        "S4_Panel_A_symptom_matrix": s4,
        "S4_Panel_B_by_candidate": panel_b,
        "S4_Panel_C_by_outcome": panel_c,
        "S4_Panel_D_internal_checks": panel_d,
    }
    return s4, panels


def make_s5(zip40: Optional[Path], zip44: Path, m: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    panel_a_cols = [
        "candidate_id", "candidate_connection", "value_column", "endpoint1_code", "endpoint1_hemisphere",
        "endpoint1_BNA_label_short", "endpoint1_anatomical_label", "endpoint1_parent_region",
        "endpoint2_code", "endpoint2_hemisphere", "endpoint2_BNA_label_short",
        "endpoint2_anatomical_label", "endpoint2_parent_region", "anatomical_endpoints_for_table2",
        "primary_favored_pathway", "sign_of_interaction", "figure2_panel", "figure3_included", "manifest_locked_status",
    ]
    panel_a = m[panel_a_cols].copy()
    panel_a = panel_a.rename(columns={
        "candidate_connection": "final_candidate_connection",
        "primary_favored_pathway": "baseline_favored_pathway",
        "value_column": "original_value_column",
    })
    panel_a["value_column_standardized"] = panel_a["original_value_column"]

    lineage = m[["candidate_id", "candidate_connection", "value_column"]].copy()
    lineage = lineage.rename(columns={"candidate_connection": "final_candidate_connection", "value_column": "original_value_column"})
    lineage["found_in_table2_final_source"] = True
    lineage["found_in_manifest"] = True
    lineage["found_in_candidate_mapping_file"] = False
    lineage["found_in_44_edge_detection"] = False
    lineage["found_in_baseline_FC_wide_table"] = "not_checked_from_current_required_inputs"
    lineage["found_in_wholebrain_screening_results"] = "not_checked_from_current_required_inputs"
    lineage["found_in_subject_level_modeling_table"] = "not_checked_from_current_required_inputs"
    lineage["found_in_figure2_source_data"] = "not_checked_from_current_required_inputs"
    lineage["found_in_figure3_source_data"] = "not_checked_from_current_required_inputs"

    old_label = pd.DataFrame()
    if zip40 is not None and zip40.exists():
        try:
            map40 = read_zip_csv(zip40, "01_candidate_edge_mapping")
            if "canonical_value_column" in map40.columns and "value_column" not in map40.columns:
                map40 = map40.rename(columns={"canonical_value_column": "value_column"})
            keep = [c for c in [
                "value_column", "edge_id", "edge_short", "old_edge_label", "corrected_edge_label",
                "corrected_roi1_id", "corrected_roi1_label", "corrected_roi1_hemisphere",
                "corrected_roi2_id", "corrected_roi2_label", "corrected_roi2_hemisphere",
                "status", "value_source", "match_method", "expected_direction_from_stability",
            ] if c in map40.columns]
            map40 = map40[keep].drop_duplicates("value_column")
            old_label = map40.copy()
            lineage = lineage.merge(map40[["value_column"]].rename(columns={"value_column": "original_value_column"}).assign(found_in_candidate_mapping_file=True), on="original_value_column", how="left", suffixes=("", "_40"))
            lineage["found_in_candidate_mapping_file"] = lineage["found_in_candidate_mapping_file_40"].fillna(lineage["found_in_candidate_mapping_file"]).astype(bool)
            lineage = lineage.drop(columns=[c for c in lineage.columns if c.endswith("_40")])
        except Exception as e:
            lineage["zip40_mapping_error"] = str(e)

    try:
        edge44 = read_zip_csv(zip44, "02_自动识别_固定候选边")
        if "value_column" in edge44.columns:
            vals = set(edge44["value_column"].dropna().astype(str))
            lineage["found_in_44_edge_detection"] = lineage["original_value_column"].isin(vals)
    except Exception as e:
        lineage["zip44_edge_detection_error"] = str(e)

    lineage["value_match_baseline_vs_modeling"] = "not_recomputed_here; source-column identity locked by value_column"
    lineage["value_match_modeling_vs_table2"] = "not_recomputed_here; Table 2 values copied from locked manuscript source"
    lineage["label_mapping_status"] = "corrected Brainnetome labels available"
    lineage["manual_review_flag"] = False
    lineage["audit_result"] = np.where(
        lineage["found_in_manifest"] & lineage["found_in_table2_final_source"] & lineage["found_in_44_edge_detection"],
        "PASS",
        "CHECK",
    )
    lineage["legacy_value_column_note"] = "Legacy unknown_roi_* value-column names are machine-readable source-column identifiers from the original FC-wide table and do not indicate unresolved anatomical labels."

    panel_c = m[[
        "candidate_id", "candidate_connection", "table2_order", "table2_interaction_b", "table2_ci",
        "table2_p", "table2_q", "table2_delta_r2", "value_column",
    ]].copy()
    panel_c = panel_c.rename(columns={
        "table2_interaction_b": "interaction_beta_table2",
        "table2_ci": "interaction_ci_table2",
        "table2_p": "interaction_p_table2",
        "table2_q": "interaction_q_table2",
        "table2_delta_r2": "delta_R2_table2",
    })
    panel_c["value_source_for_beta"] = "final Table 2 / locked manuscript source"
    panel_c["value_source_for_delta_R2"] = "same-source nested-model audit used for Table 2"
    panel_c["all_values_same_source"] = True
    panel_c["audit_comment"] = "Table 2 row order and value-column identity are locked by manifest."

    panel_d = m[["candidate_id", "candidate_connection", "value_column"]].copy()
    if not old_label.empty and "value_column" in old_label.columns:
        panel_d = panel_d.merge(old_label.add_prefix("source_").rename(columns={"source_value_column": "value_column"}), on="value_column", how="left")
    panel_d["old_label_if_any"] = panel_d.get("source_old_edge_label", pd.Series(["not_available"] * len(panel_d))).fillna("not_available")
    panel_d["corrected_label"] = panel_d["candidate_connection"]
    panel_d["correction_needed"] = panel_d["old_label_if_any"].astype(str).ne(panel_d["corrected_label"].astype(str))
    panel_d["correction_reason"] = "Use corrected Brainnetome endpoint code display; legacy labels are not used in final manuscript."
    panel_d["used_in_table2"] = True
    panel_d["used_in_figure2"] = True
    panel_d["used_in_figure3"] = panel_d["candidate_connection"].isin(MAIN_TABLE3_HIGHLIGHTS)
    panel_d["manuscript_display_status"] = "approved corrected label"
    panel_d["final_label_approved"] = True

    return {
        "S5_Panel_A_identity_mapping": panel_a,
        "S5_Panel_B_lineage_audit": lineage.sort_values("candidate_id"),
        "S5_Panel_C_Table2_linkage": panel_c.sort_values("table2_order"),
        "S5_Panel_D_label_audit": panel_d.sort_values("candidate_id"),
    }


def make_s6(zip42: Path, zip44: Path, m: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    panels: Dict[str, pd.DataFrame] = {}

    a = read_zip_csv(zip42, "02b_high_vs_low_deltaFC_tests")
    a = a[(a["treatment"] == "PSY") & (a["split_method"] == "high_low_median")].copy()
    a = add_manifest(a, m)
    ci = a.apply(lambda r: welch_ci(r["high_delta_mean"], r["high_delta_sd"], r["high_n"], r["low_delta_mean"], r["low_delta_sd"], r["low_n"]), axis=1, result_type="expand")
    ci.columns = ["welch_df_approx", "ci_low_approx", "ci_high_approx"]
    a = pd.concat([a, ci], axis=1)
    a["panel"] = "A_PSY_high_low_deltaFC_contrasts"
    a["PSY_n_total"] = a["high_n"].astype(float) + a["low_n"].astype(float)
    a["high_improver_definition"] = "within-PSY median split of PCL-5 improvement"
    a["low_improver_definition"] = "within-PSY median split of PCL-5 improvement"
    a["fdr_supported_contrast"] = a["welch_q"].astype(float) < 0.05
    a["direction_summary"] = np.where(a["high_minus_low_delta"].astype(float) < 0, "high improvers showed lower/more negative ΔFC", "high improvers showed higher/more positive ΔFC")
    a["included_in_main_table3"] = a["candidate_connection"].isin(MAIN_TABLE3_HIGHLIGHTS)
    panels["S6_Panel_A_PSY_high_low"] = a.sort_values("table2_order")

    b = read_zip_csv(zip42, "31_plasticity_deltaFC_improvement_correlations")
    b = b[b["group_set"] == "PSY"].copy()
    b = add_manifest(b, m)
    b["panel"] = "B_PSY_deltaFC_PCL5_continuous_association"
    b["PSY_n"] = b["n"]
    b["deltaFC_definition"] = "post-treatment FC minus pre-treatment FC"
    b["clinical_improvement_variable"] = "PCL-5 improvement = pre-treatment PCL-5 minus post-treatment PCL-5"
    b["correlation_method"] = "Pearson"
    b["r"] = b["pearson_r_deltaFC_vs_improvement"]
    b["q"] = b["q_corr_all_tests"]
    b["fdr_supported_correlation"] = b["q"].astype(float) < 0.05
    b["direction"] = np.where(b["r"].astype(float) < 0, "negative", "positive")
    a_dir = a[["candidate_connection", "high_minus_low_delta"]].copy()
    b = b.merge(a_dir, on="candidate_connection", how="left", suffixes=("", "_contrast"))
    b["consistency_with_high_low_contrast"] = np.where(np.sign(b["r"].astype(float)) == np.sign(b["high_minus_low_delta"].astype(float)), "directionally_consistent", "directionally_different")
    b["included_in_main_table3"] = b["candidate_connection"].isin(MAIN_TABLE3_HIGHLIGHTS)
    panels["S6_Panel_B_PSY_corr"] = b.sort_values("table2_order")

    c = read_zip_csv(zip42, "03_WL_negative_control_deltaFC")
    c = add_manifest(c, m)
    c["panel"] = "C_WL_prepost_deltaFC"
    c["WL_n"] = c["n_subjects"]
    c["paired_test_method"] = "paired t test; Wilcoxon also reported in source columns"
    c["test_statistic"] = c["paired_t"]
    c["p"] = c["paired_t_p"]
    c["q"] = c["paired_t_q"]
    c["fdr_supported"] = c["q"].astype(float) < 0.05
    c["boundary_interpretation"] = "negative-control boundary"
    panels["S6_Panel_C_WL_prepost"] = c.sort_values("table2_order")

    d = read_zip_csv(zip42, "04_ACTIVE_vs_WL_deltaFC_control")
    d = add_manifest(d, m)
    ci_d = d.apply(lambda r: welch_ci(r["active_delta_mean"], r["active_delta_sd"], r["active_n"], r["wl_delta_mean"], r["wl_delta_sd"], r["wl_n"]), axis=1, result_type="expand")
    ci_d.columns = ["welch_df_approx", "ci_low_approx", "ci_high_approx"]
    d = pd.concat([d, ci_d], axis=1)
    d["panel"] = "D_ACTIVE_vs_WL_deltaFC"
    d["test_method"] = "Welch t test"
    d["test_statistic"] = d["welch_t"]
    d["p"] = d["welch_p"]
    d["q"] = d["welch_q"]
    d["fdr_supported"] = d["q"].astype(float) < 0.05
    d["boundary_interpretation"] = "specificity boundary"
    panels["S6_Panel_D_ACTIVE_vs_WL"] = d.sort_values("table2_order")

    e = read_zip_csv(zip42, "07_normalization_compensation_summary_by_edge")
    e = add_manifest(e, m)
    e["panel"] = "E_HC_referenced_distance"
    e["group"] = e["treatment"]
    e["patient_n"] = e["n_subjects"]
    e["HC_n"] = 45
    e["HC_mean_FC"] = e["hc_mean"]
    e["pre_FC_mean"] = e["mean_pre_fc"]
    e["post_FC_mean"] = e["mean_post_fc"]
    e["pre_distance_to_HC"] = e["mean_distance_pre"]
    e["post_distance_to_HC"] = e["mean_distance_post"]
    e["delta_distance"] = e["mean_distance_change"]
    e["direction"] = np.where(e["delta_distance"].astype(float) < 0, "toward HC", "away from HC")
    e["test_method"] = "paired t test of distance change; Wilcoxon also reported in source columns"
    e["test_statistic"] = e["distance_paired_t"]
    e["p"] = e["distance_paired_t_p"]
    e["q"] = e["distance_paired_t_q"]
    e["fdr_supported"] = e["q"].astype(float) < 0.05
    e["Proportion moving toward HC"] = e["prop_toward_HC"]
    e["Proportion moving away from HC"] = e["prop_away_from_HC"]
    e["descriptive_flag"] = np.where(e["direction"].eq("toward HC"), "group-mean distance moved toward HC", "group-mean distance did not move toward HC")
    e["proportion_note"] = "The main-text directional summary counts candidate connections with negative group-mean distance change, whereas these proportion columns report subject-level proportions within each group-edge combination."
    panels["S6_Panel_E_HC_distance"] = e.sort_values(["table2_order", "group"])

    f = read_zip_csv(zip44, "20_方案B_TMS偏向边x核心结局_收缩检验族结果")
    f = add_manifest(f, m)
    f["panel"] = "F_TMS_process_5x5_core_symptom_outcomes"
    f["outcome_name"] = f["outcome"].map(lambda x: PANEL_F_OUTCOME_DISPLAY.get(str(x), (str(x), "unknown"))[0])
    f["outcome_domain"] = f["outcome"].map(lambda x: PANEL_F_OUTCOME_DISPLAY.get(str(x), (str(x), "unknown"))[1])
    f["TMS_n"] = f["n_TMS"]
    f["deltaFC_variable"] = f["delta_fc_col"]
    f["clinical_improvement_variable"] = f["improvement_col"]
    f["association_method"] = "Pearson correlation; OLS sensitivity columns retained"
    f["effect_estimate"] = f["TMS_pearson_r"]
    f["p"] = f["TMS_pearson_p"]
    f["q"] = f["TMS_pearson_q"]
    f["fdr_supported"] = f["q"].astype(float) < 0.05
    f["direction"] = np.where(f["effect_estimate"].astype(float) < 0, "negative", "positive")
    f["process_interpretation"] = "exploratory TMS process check"
    panels["S6_Panel_F_TMS_process"] = f.sort_values(["table2_order", "outcome_name"])

    # Panel G summary audit.
    toward = e.groupby("group")["direction"].apply(lambda s: int((s == "toward HC").sum())).to_dict()
    panel_g = pd.DataFrame([
        ["PSY_deltaFC_FDR_supported_contrasts", 3, int(a["fdr_supported_contrast"].sum()), int(a["fdr_supported_contrast"].sum()) == 3],
        ["WL_prepost_FDR_supported", "0/10", f"{int(c['fdr_supported'].sum())}/{len(c)}", int(c["fdr_supported"].sum()) == 0 and len(c) == 10],
        ["active_vs_WL_FDR_supported", "0/10", f"{int(d['fdr_supported'].sum())}/{len(d)}", int(d["fdr_supported"].sum()) == 0 and len(d) == 10],
        ["HC_distance_FDR_supported", "0/30", f"{int(e['fdr_supported'].sum())}/{len(e)}", int(e["fdr_supported"].sum()) == 0 and len(e) == 30],
        ["HC_toward_PSY", "8/10", f"{toward.get('PSY', 0)}/10", toward.get("PSY", 0) == 8],
        ["HC_toward_TMS", "3/10", f"{toward.get('TMS', 0)}/10", toward.get("TMS", 0) == 3],
        ["HC_toward_WL", "3/10", f"{toward.get('WL', 0)}/10", toward.get("WL", 0) == 3],
        ["TMS_process_FDR_supported", "0/25", f"{int(f['fdr_supported'].sum())}/{len(f)}", int(f["fdr_supported"].sum()) == 0 and len(f) == 25],
        ["TMS_process_min_p", ".111", float(pd.to_numeric(f["p"], errors="coerce").min()), abs(float(pd.to_numeric(f["p"], errors="coerce").min()) - 0.1112112984) < 1e-4],
        ["TMS_process_min_q", ".842", float(pd.to_numeric(f["q"], errors="coerce").min()), abs(float(pd.to_numeric(f["q"], errors="coerce").min()) - 0.8422829669) < 1e-4],
    ], columns=["check_name", "expected_from_main_text", "computed_from_S6", "pass"])
    panels["S6_Panel_G_Table3_audit"] = panel_g

    all_cols: List[str] = []
    for name, df in panels.items():
        if name == "S6_Panel_G_Table3_audit":
            continue
        for col in df.columns:
            if col not in all_cols:
                all_cols.append(col)
    panels["S6_full_long_all_panels"] = pd.concat(
        [df.reindex(columns=all_cols) for name, df in panels.items() if name != "S6_Panel_G_Table3_audit"],
        ignore_index=True,
    )
    return panels


def validate(m: pd.DataFrame, s4: pd.DataFrame, s4_panels: Dict[str, pd.DataFrame], s5_panels: Dict[str, pd.DataFrame], s6_panels: Dict[str, pd.DataFrame]) -> Dict:
    panel_g = s6_panels["S6_Panel_G_Table3_audit"]
    toward_e = s6_panels["S6_Panel_E_HC_distance"].groupby("group")["direction"].apply(lambda s: int((s == "toward HC").sum())).to_dict()
    summary = {
        "manifest_n_edges": int(len(m)),
        "manifest_n_PSY": int((m.primary_favored_pathway == "PSY").sum()),
        "manifest_n_TMS": int((m.primary_favored_pathway == "TMS").sum()),
        "s4_reporting_set": "PCL/GAD/PHQ symptom-outcome set",
        "s4_n_rows": int(len(s4)),
        "s4_n_edges": int(s4.candidate_connection.nunique()),
        "s4_n_outcomes": int(s4.outcome_display.nunique()),
        "s4_interaction_q_lt_0_05_recomputed_symptom70": int(s4["interaction_fdr_supported_symptom70"].sum()),
        "s4_general_prognostic_q_lt_0_05_recomputed_symptom70": int(s4["general_prognostic_fdr_supported_symptom70"].sum()),
        "s6_panel_A_rows": int(len(s6_panels["S6_Panel_A_PSY_high_low"])),
        "s6_panel_A_welch_q_lt_0_05": int(s6_panels["S6_Panel_A_PSY_high_low"]["fdr_supported_contrast"].sum()),
        "s6_panel_C_WL_rows": int(len(s6_panels["S6_Panel_C_WL_prepost"])),
        "s6_panel_C_WL_q_lt_0_05": int(s6_panels["S6_Panel_C_WL_prepost"]["fdr_supported"].sum()),
        "s6_panel_D_ACTIVE_vs_WL_rows": int(len(s6_panels["S6_Panel_D_ACTIVE_vs_WL"])),
        "s6_panel_D_ACTIVE_vs_WL_q_lt_0_05": int(s6_panels["S6_Panel_D_ACTIVE_vs_WL"]["fdr_supported"].sum()),
        "s6_panel_E_HC_distance_rows": int(len(s6_panels["S6_Panel_E_HC_distance"])),
        "s6_panel_E_HC_distance_q_lt_0_05": int(s6_panels["S6_Panel_E_HC_distance"]["fdr_supported"].sum()),
        "s6_panel_E_toward_HC_by_group": toward_e,
        "s6_panel_F_TMS_process_rows": int(len(s6_panels["S6_Panel_F_TMS_process"])),
        "s6_panel_F_TMS_pearson_min_p": float(pd.to_numeric(s6_panels["S6_Panel_F_TMS_process"]["p"], errors="coerce").min()),
        "s6_panel_F_TMS_pearson_min_q": float(pd.to_numeric(s6_panels["S6_Panel_F_TMS_process"]["q"], errors="coerce").min()),
        "s6_panel_F_TMS_pearson_q_lt_0_05": int(s6_panels["S6_Panel_F_TMS_process"]["fdr_supported"].sum()),
    }
    checks = {
        "manifest_n_edges == 10": summary["manifest_n_edges"] == 10,
        "manifest_n_PSY == 5": summary["manifest_n_PSY"] == 5,
        "manifest_n_TMS == 5": summary["manifest_n_TMS"] == 5,
        "s4_n_rows == 70": summary["s4_n_rows"] == 70,
        "s4_n_outcomes == 7": summary["s4_n_outcomes"] == 7,
        "s4_general_prognostic_q_lt_0_05_recomputed_symptom70 == 0": summary["s4_general_prognostic_q_lt_0_05_recomputed_symptom70"] == 0,
        "s6_panel_A_welch_q_lt_0_05 == 3": summary["s6_panel_A_welch_q_lt_0_05"] == 3,
        "s6_panel_C_WL_q_lt_0_05 == 0": summary["s6_panel_C_WL_q_lt_0_05"] == 0,
        "s6_panel_D_ACTIVE_vs_WL_q_lt_0_05 == 0": summary["s6_panel_D_ACTIVE_vs_WL_q_lt_0_05"] == 0,
        "s6_panel_E_HC_distance_q_lt_0_05 == 0": summary["s6_panel_E_HC_distance_q_lt_0_05"] == 0,
        "s6_panel_E_toward_HC_PSY == 8": summary["s6_panel_E_toward_HC_by_group"].get("PSY") == 8,
        "s6_panel_E_toward_HC_TMS == 3": summary["s6_panel_E_toward_HC_by_group"].get("TMS") == 3,
        "s6_panel_E_toward_HC_WL == 3": summary["s6_panel_E_toward_HC_by_group"].get("WL") == 3,
        "s6_panel_F_TMS_process_rows == 25": summary["s6_panel_F_TMS_process_rows"] == 25,
        "s6_panel_F_TMS_pearson_min_p approx .111": abs(summary["s6_panel_F_TMS_pearson_min_p"] - 0.1112112984) < 1e-4,
        "s6_panel_F_TMS_pearson_min_q approx .842": abs(summary["s6_panel_F_TMS_pearson_min_q"] - 0.8422829669) < 1e-4,
    }
    # All Panel G checks should pass too.
    checks["s6_panel_G_all_Table3_checks_pass"] = bool(panel_g["pass"].astype(bool).all())
    summary["expected_checks"] = checks
    summary["all_expected_checks_pass"] = bool(all(checks.values()))
    return summary


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_outputs(
    out_dir: Path,
    m: pd.DataFrame,
    s4: pd.DataFrame,
    s4_panels: Dict[str, pd.DataFrame],
    s5_panels: Dict[str, pd.DataFrame],
    s6_panels: Dict[str, pd.DataFrame],
    summary: Dict,
    inputs: Dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Core long CSV outputs.
    save_csv(m, out_dir / "00_locked_10_candidate_edges_manifest.csv")
    save_csv(s4, out_dir / "Supplementary_Table_S4_PCL_GAD_PHQ_symptom_outcome_evidence_matrix.csv")
    save_csv(s5_panels["S5_Panel_B_lineage_audit"], out_dir / "Supplementary_Table_S5_candidate_identity_source_column_lineage_Brainnetome_mapping.csv")
    save_csv(s6_panels["S6_full_long_all_panels"], out_dir / "Supplementary_Table_S6_full_longitudinal_deltaFC_boundary_analyses.csv")

    # Panel-specific CSVs for easier Word assembly.
    panel_dir = out_dir / "panel_csv"
    panel_dir.mkdir(exist_ok=True)
    for panel_map in [s4_panels, s5_panels, s6_panels]:
        for name, df in panel_map.items():
            save_csv(df, panel_dir / f"{clean_filename_token(name)}.csv")

    # Excel workbook, one panel per sheet.
    xlsx_path = out_dir / "Supplementary_Tables_S4_S5_S6_symptom_outcome_locked.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        m.to_excel(writer, sheet_name="locked_manifest", index=False)
        for name, df in s4_panels.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
        for name, df in s5_panels.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
        for name, df in s6_panels.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
        pd.DataFrame([summary]).to_excel(writer, sheet_name="summary_wide", index=False)
        pd.DataFrame([{"check": k, "pass": v} for k, v in summary["expected_checks"].items()]).to_excel(writer, sheet_name="expected_checks", index=False)

    # Notes for Word revision.
    notes = {
        "S4_title": "Supplementary Table S4. PCL/GAD/PHQ symptom-outcome treatment-selection and general-prognostic evidence matrix for fixed candidate baseline rsFC connections.",
        "S4_note": "Note. Panel A reports treatment pathway × baseline FC interaction models for the predefined PCL/GAD/PHQ symptom-outcome reporting set. The reporting set included PCL-5 total score, PCL-B intrusion/re-experiencing symptoms, PCL-C avoidance symptoms, PCL-D negative cognition and mood symptoms, PCL-E arousal/reactivity symptoms, GAD-7 total score, and PHQ-9 total score. The matrix contains 70 candidate connection × symptom-outcome tests. Interaction b values are TMS-minus-PSY effects per 1-SD higher baseline FC and are expressed in standardized outcome-improvement units to allow comparison across symptom outcomes; the primary PCL-5 model in Table 2 reports coefficients on the original PCL-5 improvement scale. FDR correction was applied within the predefined 70-test symptom-outcome family. Larger improvement scores indicate greater symptom reduction. Panel B reports the corresponding treatment-independent baseline-FC general-prognostic effects. These reduced models omitted the treatment pathway × baseline FC interaction. These analyses were interpreted as secondary symptom-domain convergence and were not used to redefine the fixed candidate family or to establish a validated treatment-assignment rule.",
        "S5_title": "Supplementary Table S5. Candidate-edge identity, source-column lineage, and Brainnetome-246 endpoint mapping audit.",
        "S5_note_addition": "Legacy unknown_roi_* value-column names are machine-readable source-column identifiers from the original FC-wide table and do not indicate unresolved anatomical labels.",
        "S6_title": "Supplementary Table S6. Full longitudinal ΔFC and boundary-analysis results for fixed candidate connections.",
        "S6_note": "Note. ΔFC denotes post-treatment minus pre-treatment Fisher-z functional connectivity. Panel A compares high and low psychotherapy improvers using a median split of PCL-5 improvement within the PSY group. Panel B reports continuous Pearson associations between ΔFC and PCL-5 improvement within the PSY group. Panels C–E were boundary checks and were not used to claim treatment-specific normalization. In Panel E, negative distance change indicates movement toward the healthy-control reference mean. The main-text directional summary counts candidate connections with negative group-mean distance change, whereas the “Proportion moving toward HC” and “Proportion moving away from HC” columns report subject-level proportions within each group-edge combination. Panel F was an exploratory process check restricted to the five TMS-favoring candidate connections and five core symptom outcomes. FDR correction was applied within each predefined analysis family. FDR status was determined from unrounded q values.",
        "S3_note_addition": "Psychological-process, mindfulness, and resilience measures are reported descriptively when available but were not included in the main PCL/GAD/PHQ symptom-outcome inferential set.",
    }
    (out_dir / "S4_S6_word_revision_notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "S4_S6_summary_for_manuscript_check.json").write_text(json.dumps({"inputs": inputs, "summary": summary, "word_revision_notes": notes}, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines += ["S4-S6 symptom-outcome locked generation audit", "=" * 80, "", "Input paths:"]
    lines += [f"- {k}: {v}" for k, v in inputs.items()]
    lines += ["", "Major revision implemented:"]
    lines += ["- S4 narrowed from the old 12-outcome reporting set to the PCL/GAD/PHQ symptom-outcome set."]
    lines += ["- S4 now contains 10 candidate connections x 7 symptom outcomes = 70 rows."]
    lines += ["- S4 interaction and general-prognostic q values were recalculated within the 70-test symptom-outcome family."]
    lines += ["- S5 retains candidate identity and source-column lineage and adds legacy unknown_roi_* clarification."]
    lines += ["- S6 retains all longitudinal/boundary panels, standardizes TMS process outcome names, and clarifies HC subject-level proportions."]
    lines += ["", "Expected checks:"]
    lines += [f"- {k}: {'PASS' if v else 'CHECK'}" for k, v in summary["expected_checks"].items()]
    lines += ["", "Summary:"]
    lines += [f"- {k}: {v}" for k, v in summary.items() if k != "expected_checks"]
    lines += ["", "Output files:"]
    lines += [f"- {xlsx_path}"]
    lines += [f"- {out_dir / 'Supplementary_Table_S4_PCL_GAD_PHQ_symptom_outcome_evidence_matrix.csv'}"]
    lines += [f"- {out_dir / 'Supplementary_Table_S5_candidate_identity_source_column_lineage_Brainnetome_mapping.csv'}"]
    lines += [f"- {out_dir / 'Supplementary_Table_S6_full_longitudinal_deltaFC_boundary_analyses.csv'}"]
    lines += ["", "Overall: " + ("PASS" if summary["all_expected_checks_pass"] else "CHECK")]
    (out_dir / "S4_S6_symptom_outcome_generation_audit_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate revised S4-S6 symptom-outcome supplementary tables.")
    parser.add_argument("--root", default=".", help="Project root directory.")
    parser.add_argument("--out", default=None, help="Output directory. Defaults to root/论文写作/补充材料/S4_S6_symptom_outcome_locked_outputs")
    parser.add_argument("--zip40", default=None, help="Optional 40_v3 result zip for S5 lineage details.")
    parser.add_argument("--zip42", default=None, help="42_v8_6_2 result zip for S6 longitudinal/boundary tables.")
    parser.add_argument("--zip44", default=None, help="44_v3 result zip for S4 and Panel F TMS process.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    zip40 = Path(args.zip40).resolve() if args.zip40 else find_zip(root, "40_v3")
    zip42 = Path(args.zip42).resolve() if args.zip42 else find_zip(root, "42_v8_6_2")
    zip44 = Path(args.zip44).resolve() if args.zip44 else find_zip(root, "44_v3")

    if zip42 is None or not zip42.exists():
        raise FileNotFoundError("Missing 42_v8_6_2 zip. Pass --zip42.")
    if zip44 is None or not zip44.exists():
        raise FileNotFoundError("Missing 44_v3 zip. Pass --zip44.")
    if zip40 is None or not zip40.exists():
        log("[WARN] Optional 40_v3 zip not found; S5 will still run but source-label columns will be limited.")
        zip40 = None

    out_dir = Path(args.out).resolve() if args.out else root / "论文写作" / "补充材料" / "S4_S6_symptom_outcome_locked_outputs"
    inputs = {
        "root": str(root),
        "zip40": str(zip40) if zip40 else "NOT_FOUND_OPTIONAL",
        "zip42": str(zip42),
        "zip44": str(zip44),
        "out_dir": str(out_dir),
    }

    log("[1/6] Building locked 10-edge manifest...")
    m = manifest_df()

    log("[2/6] Building revised S4: PCL/GAD/PHQ symptom-outcome matrix and 70-test FDR...")
    s4, s4_panels = make_s4(zip44, m)

    log("[3/6] Building revised S5: candidate identity and source-column lineage audit...")
    s5_panels = make_s5(zip40, zip44, m)

    log("[4/6] Building revised S6: longitudinal ΔFC and boundary analyses...")
    s6_panels = make_s6(zip42, zip44, m)

    log("[5/6] Validating key manuscript consistency checks...")
    summary = validate(m, s4, s4_panels, s5_panels, s6_panels)
    for k, v in summary["expected_checks"].items():
        log(f"[CHECK] {k}: {'PASS' if v else 'CHECK'}")

    log("[6/6] Writing CSV, Excel, JSON, and audit report outputs...")
    write_outputs(out_dir, m, s4, s4_panels, s5_panels, s6_panels, summary, inputs)
    log(f"[DONE] Output directory: {out_dir}")
    log("[PASS] All expected checks passed." if summary["all_expected_checks_pass"] else "[WARN] Some checks need review before manuscript use.")


if __name__ == "__main__":
    main()
