# -*- coding: utf-8 -*-
"""
generate_S7_clinical_incremental_value_sensitivity.py

Exploratory clinical incremental-value sensitivity analysis for locked baseline rsFC
candidate connections.

Purpose
-------
This script fits three nested OLS models for each locked candidate rsFC edge:

Model 0: clinical/QC-only
    PCL5_improvement ~ treatment_pathway + baseline_PCL5 + age + sex + pre_meanFD

Model 1: clinical/QC + FC main effect
    PCL5_improvement ~ treatment_pathway + baseline_FC + baseline_PCL5 + age + sex + pre_meanFD

Model 2: clinical/QC + FC interaction
    PCL5_improvement ~ treatment_pathway + baseline_FC + treatment_pathway × baseline_FC
                       + baseline_PCL5 + age + sex + pre_meanFD

The analysis is intentionally exploratory. It quantifies incremental explanatory value
of locked rsFC candidate features beyond baseline clinical/QC covariates. It does not
validate a deployable individualized treatment rule, establish clinical utility, or
support causal comparative efficacy between treatment pathways.

Typical command
---------------
python generate_S7_clinical_incremental_value_sensitivity.py --root "D:\\自科＋脑中心论文选题\\PAI选题\\工作站传输\\第四步分析-codex"

Optional faster smoke test:
python generate_S7_clinical_incremental_value_sensitivity.py --root "..." --bootstrap_n 200

Skip bootstrap:
python generate_S7_clinical_incremental_value_sensitivity.py --root "..." --skip_bootstrap
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "statsmodels is required. Please install it in the current Python environment. "
        "Example: pip install statsmodels"
    ) from exc


SCRIPT_NAME = "generate_S7_clinical_incremental_value_sensitivity"
OUT_DIR_NAME = "S7_clinical_incremental_value_outputs"

# Expected active-treatment sample structure from the current manuscript.
EXPECTED_ACTIVE_N = 66
EXPECTED_PSY_N = 23
EXPECTED_TMS_N = 43
EXPECTED_EDGES = 10
EXPECTED_PSY_EDGES = 5
EXPECTED_TMS_EDGES = 5


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_float(x) -> float:
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def find_files(root: Path, patterns: Iterable[str]) -> List[Path]:
    results: List[Path] = []
    if not root.exists():
        return results
    for pattern in patterns:
        results.extend(root.rglob(pattern))
    # De-duplicate while keeping deterministic ordering.
    uniq = sorted(set(results), key=lambda p: (len(str(p)), str(p)))
    return uniq


def find_first_existing(paths: Iterable[Optional[Path]]) -> Optional[Path]:
    for p in paths:
        if p is not None and p.exists():
            return p
    return None


def read_csv_auto(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def read_csv_from_zip(zip_path: Path, suffix: str) -> Tuple[pd.DataFrame, str]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        matches = [name for name in zf.namelist() if name.endswith(suffix)]
        if not matches:
            raise FileNotFoundError(f"Cannot find file ending with {suffix!r} inside {zip_path}")
        # Prefer shortest path if duplicates exist.
        name = sorted(matches, key=lambda s: (len(s), s))[0]
        with zf.open(name) as f:
            df = pd.read_csv(f)
    return df, name


def bh_fdr(p_values: Iterable[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR q values; NaN p values remain NaN."""
    p = np.asarray(list(p_values), dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if valid.sum() == 0:
        return q
    pv = p[valid]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty(n)
    out[order] = adjusted
    q[valid] = out
    return q


def fmt_p(x: float) -> str:
    if not np.isfinite(x):
        return "NA"
    if x < 0.001:
        return f"{x:.2e}"
    return f"{x:.3f}".lstrip("0")


def fmt_num(x: float, digits: int = 3) -> str:
    if not np.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def standardize(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mean = s.mean(skipna=True)
    sd = s.std(skipna=True, ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=series.index)
    return (s - mean) / sd


@dataclass
class ModelFit:
    model: object
    robust: object
    r2: float
    adj_r2: float
    aic: float
    bic: float
    n: int
    df_model: float
    df_resid: float


# -----------------------------------------------------------------------------
# Data loading and validation
# -----------------------------------------------------------------------------

def resolve_inputs(args) -> Dict[str, Path]:
    cwd = Path.cwd()
    root = Path(args.root).expanduser().resolve() if args.root else cwd

    if args.zip40:
        zip40 = Path(args.zip40).expanduser().resolve()
    else:
        candidates = []
        for base in [cwd, root]:
            candidates.extend(find_files(base, ["*40_v3*.zip", "*固定候选边*正式固定候选治疗调节*.zip"]))
        candidates = sorted(set(candidates), key=lambda p: (len(str(p)), str(p)))
        zip40 = candidates[0] if candidates else None

    if args.manifest:
        manifest = Path(args.manifest).expanduser().resolve()
    else:
        candidates = []
        for base in [cwd, root]:
            candidates.extend(find_files(base, ["00_locked_10_candidate_edges_manifest.csv"]))
        candidates = sorted(set(candidates), key=lambda p: (len(str(p)), str(p)))
        manifest = candidates[0] if candidates else None

    if args.subject_table:
        subject_table = Path(args.subject_table).expanduser().resolve()
    else:
        subject_table = None

    if zip40 is None or not zip40.exists():
        raise FileNotFoundError(
            "Cannot locate 40_v3 result zip. Please pass --zip40 explicitly."
        )

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (cwd / OUT_DIR_NAME).resolve()
    return {"root": root, "zip40": zip40, "manifest": manifest, "subject_table": subject_table, "out_dir": out_dir}


def load_manifest(manifest_path: Optional[Path], zip40: Path) -> Tuple[pd.DataFrame, str]:
    if manifest_path is not None and manifest_path.exists():
        m = read_csv_auto(manifest_path)
        source = str(manifest_path)
    else:
        # Fallback: construct a minimal manifest from the 40_v3 candidate mapping.
        mapping, internal_name = read_csv_from_zip(zip40, "01_candidate_edge_mapping.csv")
        label_col = "corrected_edge_label" if "corrected_edge_label" in mapping.columns else "edge_short"
        if label_col not in mapping.columns:
            raise ValueError(
                "No manifest was found, and 01_candidate_edge_mapping.csv does not contain "
                "corrected_edge_label or edge_short. Please pass --manifest explicitly."
            )
        direction_col = "expected_direction_from_stability"
        value_col = "canonical_value_column" if "canonical_value_column" in mapping.columns else "selected_unknown_col"
        rows = []
        for i, row in mapping.iterrows():
            direction = str(row.get(direction_col, "")).upper()
            if "PSY" in direction:
                fav = "PSY"
            elif "TMS" in direction:
                fav = "TMS"
            else:
                fav = "UNKNOWN"
            rows.append(
                {
                    "table2_order": int(i + 1),
                    "candidate_id": f"C{i+1:02d}",
                    "candidate_connection": row[label_col],
                    "primary_favored_pathway": fav,
                    "value_column": row.get(value_col, ""),
                    "manifest_locked_status": "FALLBACK_FROM_40V3_MAPPING_REVIEW_BEFORE_USE",
                }
            )
        m = pd.DataFrame(rows)
        source = f"{zip40}::{internal_name} (fallback manifest)"
    return normalize_manifest(m), source


def normalize_manifest(m: pd.DataFrame) -> pd.DataFrame:
    m = m.copy()
    rename_candidates = {
        "edge_short": "candidate_connection",
        "corrected_edge_label": "candidate_connection",
        "expected_direction_from_stability": "primary_favored_pathway",
        "canonical_value_column": "value_column",
    }
    for old, new in rename_candidates.items():
        if new not in m.columns and old in m.columns:
            m[new] = m[old]

    required = ["candidate_connection", "primary_favored_pathway"]
    missing = [c for c in required if c not in m.columns]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")

    if "table2_order" not in m.columns:
        m["table2_order"] = np.arange(1, len(m) + 1)
    if "candidate_id" not in m.columns:
        m["candidate_id"] = [f"C{i:02d}" for i in range(1, len(m) + 1)]
    if "value_column" not in m.columns:
        m["value_column"] = ""

    m["candidate_connection"] = m["candidate_connection"].astype(str).str.strip()
    m["primary_favored_pathway"] = (
        m["primary_favored_pathway"].astype(str).str.upper().str.extract(r"(PSY|TMS)", expand=False).fillna("UNKNOWN")
    )
    m = m.sort_values("table2_order").reset_index(drop=True)
    return m


def load_subject_table(subject_table_path: Optional[Path], zip40: Path) -> Tuple[pd.DataFrame, str]:
    if subject_table_path is not None and subject_table_path.exists():
        return read_csv_auto(subject_table_path), str(subject_table_path)
    df, internal_name = read_csv_from_zip(zip40, "02_subject_level_core_edges.csv")
    return df, f"{zip40}::{internal_name}"


def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    cols = set(df.columns)

    def first_of(candidates: List[str], required_name: str) -> str:
        for c in candidates:
            if c in cols:
                return c
        raise ValueError(f"Cannot find required column for {required_name}. Tried: {candidates}")

    detected = {
        "subject": first_of(["subject_key", "subject_id", "subject", "id"], "subject id"),
        "treatment": first_of(["treatment_TMS", "treatment", "TMS", "treat_TMS"], "treatment code"),
        "pathway": first_of(["group_tms_psy", "treatment_pathway", "pathway", "group"], "pathway label"),
        "age": first_of(["age", "Age"], "age"),
        "sex": first_of(["sex", "Sex"], "sex"),
        "fd": first_of(["pre_meanFD", "meanFD_pre", "baseline_meanFD", "mean_FD", "meanFD"], "pre-treatment mean FD"),
    }

    if "pcl_improvement" in cols:
        detected["outcome"] = "pcl_improvement"
    else:
        detected["pcl_pre"] = first_of(["PCL_pre", "PCL5_pre", "pcl_pre", "baseline_PCL5"], "baseline PCL-5")
        detected["pcl_post"] = first_of(["PCL_post", "PCL5_post", "pcl_post", "post_PCL5"], "post-treatment PCL-5")
    detected["pcl_pre"] = detected.get("pcl_pre", first_of(["PCL_pre", "PCL5_pre", "pcl_pre", "baseline_PCL5"], "baseline PCL-5"))
    return detected


def edge_fc_column(edge: str, df: pd.DataFrame) -> Tuple[str, str]:
    """Return column used for baseline FC and a source description."""
    candidates = [
        f"{edge}__pre",
        f"{edge}__baseline",
        f"{edge}__baseline_z",
        edge,
    ]
    for c in candidates:
        if c in df.columns:
            return c, "standardized_in_script"
    raise ValueError(
        f"Cannot find baseline FC column for {edge}. Tried: {candidates}"
    )


def validate_inputs(manifest: pd.DataFrame, df: pd.DataFrame, colmap: Dict[str, str]) -> List[str]:
    issues: List[str] = []
    if len(manifest) != EXPECTED_EDGES:
        issues.append(f"Manifest edge count is {len(manifest)}, expected {EXPECTED_EDGES}.")
    if manifest["candidate_connection"].duplicated().any():
        issues.append("Manifest contains duplicated candidate_connection values.")
    psy_edges = int((manifest["primary_favored_pathway"] == "PSY").sum())
    tms_edges = int((manifest["primary_favored_pathway"] == "TMS").sum())
    if psy_edges != EXPECTED_PSY_EDGES:
        issues.append(f"Manifest PSY edge count is {psy_edges}, expected {EXPECTED_PSY_EDGES}.")
    if tms_edges != EXPECTED_TMS_EDGES:
        issues.append(f"Manifest TMS edge count is {tms_edges}, expected {EXPECTED_TMS_EDGES}.")

    for edge in manifest["candidate_connection"]:
        try:
            edge_fc_column(edge, df)
        except Exception as exc:
            issues.append(str(exc))

    # Active-treatment check.
    treatment = pd.to_numeric(df[colmap["treatment"]], errors="coerce")
    active = df[treatment.isin([0, 1])].copy()
    n_active = len(active)
    n_psy = int((active[colmap["treatment"]] == 0).sum())
    n_tms = int((active[colmap["treatment"]] == 1).sum())
    if n_active != EXPECTED_ACTIVE_N:
        issues.append(f"Active-treatment n is {n_active}, expected {EXPECTED_ACTIVE_N}.")
    if n_psy != EXPECTED_PSY_N:
        issues.append(f"PSY n is {n_psy}, expected {EXPECTED_PSY_N}.")
    if n_tms != EXPECTED_TMS_N:
        issues.append(f"TMS n is {n_tms}, expected {EXPECTED_TMS_N}.")

    for c in [colmap["treatment"], colmap["pcl_pre"], colmap["age"], colmap["sex"], colmap["fd"]]:
        if active[c].isna().any():
            issues.append(f"Missing values detected in required column {c!r} within active-treatment sample.")

    return issues


# -----------------------------------------------------------------------------
# Modeling
# -----------------------------------------------------------------------------

def fit_ols(y: pd.Series, X: pd.DataFrame) -> ModelFit:
    Xc = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y.astype(float), Xc.astype(float), missing="raise").fit()
    robust = model.get_robustcov_results(cov_type="HC3")
    return ModelFit(
        model=model,
        robust=robust,
        r2=float(model.rsquared),
        adj_r2=float(model.rsquared_adj),
        aic=float(model.aic),
        bic=float(model.bic),
        n=int(model.nobs),
        df_model=float(model.df_model),
        df_resid=float(model.df_resid),
    )


def robust_param(fit: ModelFit, param_name: str) -> Dict[str, float]:
    names = list(fit.model.params.index)
    if param_name not in names:
        return {"coef": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan, "t": np.nan, "p": np.nan}
    idx = names.index(param_name)
    params = np.asarray(fit.robust.params, dtype=float)
    bse = np.asarray(fit.robust.bse, dtype=float)
    tvals = np.asarray(fit.robust.tvalues, dtype=float)
    pvals = np.asarray(fit.robust.pvalues, dtype=float)
    ci = np.asarray(fit.robust.conf_int(alpha=0.05), dtype=float)
    return {
        "coef": float(params[idx]),
        "se": float(bse[idx]),
        "ci_low": float(ci[idx, 0]),
        "ci_high": float(ci[idx, 1]),
        "t": float(tvals[idx]),
        "p": float(pvals[idx]),
    }


def compare_nested(full_fit: ModelFit, reduced_fit: ModelFit) -> Dict[str, float]:
    try:
        f_stat, p_val, df_diff = full_fit.model.compare_f_test(reduced_fit.model)
        return {"F": float(f_stat), "p": float(p_val), "df_diff": float(df_diff)}
    except Exception:
        return {"F": np.nan, "p": np.nan, "df_diff": np.nan}


def prepare_edge_dataframe(df: pd.DataFrame, colmap: Dict[str, str], edge: str, fc_col: str) -> pd.DataFrame:
    work = pd.DataFrame({
        "subject_id": df[colmap["subject"]],
        "pathway_label": df[colmap["pathway"]],
        "treatment_TMS": pd.to_numeric(df[colmap["treatment"]], errors="coerce"),
        "baseline_PCL5": pd.to_numeric(df[colmap["pcl_pre"]], errors="coerce"),
        "age": pd.to_numeric(df[colmap["age"]], errors="coerce"),
        "sex": pd.to_numeric(df[colmap["sex"]], errors="coerce"),
        "pre_meanFD": pd.to_numeric(df[colmap["fd"]], errors="coerce"),
        "baseline_FC_raw": pd.to_numeric(df[fc_col], errors="coerce"),
    })
    if "outcome" in colmap:
        work["PCL5_improvement"] = pd.to_numeric(df[colmap["outcome"]], errors="coerce")
    else:
        post = pd.to_numeric(df[colmap["pcl_post"]], errors="coerce")
        work["PCL5_improvement"] = work["baseline_PCL5"] - post

    work = work[work["treatment_TMS"].isin([0, 1])].copy()
    work = work.dropna(subset=[
        "treatment_TMS", "baseline_PCL5", "age", "sex", "pre_meanFD", "baseline_FC_raw", "PCL5_improvement"
    ]).copy()
    work["baseline_FC_zstd"] = standardize(work["baseline_FC_raw"])
    work = work.dropna(subset=["baseline_FC_zstd"]).copy()
    work["treatment_x_FC"] = work["treatment_TMS"] * work["baseline_FC_zstd"]
    work["edge"] = edge
    return work


def fit_edge_models(edge_df: pd.DataFrame) -> Tuple[ModelFit, ModelFit, ModelFit]:
    y = edge_df["PCL5_improvement"]
    X0 = edge_df[["treatment_TMS", "baseline_PCL5", "age", "sex", "pre_meanFD"]]
    X1 = edge_df[["treatment_TMS", "baseline_FC_zstd", "baseline_PCL5", "age", "sex", "pre_meanFD"]]
    X2 = edge_df[["treatment_TMS", "baseline_FC_zstd", "treatment_x_FC", "baseline_PCL5", "age", "sex", "pre_meanFD"]]
    return fit_ols(y, X0), fit_ols(y, X1), fit_ols(y, X2)


def analyze_edges(manifest: pd.DataFrame, df: pd.DataFrame, colmap: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    edge_rows: List[Dict] = []
    model_rows: List[Dict] = []
    edge_data: Dict[str, pd.DataFrame] = {}

    for i, row in manifest.iterrows():
        edge = row["candidate_connection"]
        fc_col, fc_source = edge_fc_column(edge, df)
        log(f"    Fitting edge {i+1}/{len(manifest)}: {edge}")
        edf = prepare_edge_dataframe(df, colmap, edge, fc_col)
        edge_data[edge] = edf.copy()

        fit0, fit1, fit2 = fit_edge_models(edf)
        interaction = robust_param(fit2, "treatment_x_FC")
        fc_main = robust_param(fit1, "baseline_FC_zstd")

        cmp_10 = compare_nested(fit1, fit0)
        cmp_21 = compare_nested(fit2, fit1)
        cmp_20 = compare_nested(fit2, fit0)

        d_r2_10 = fit1.r2 - fit0.r2
        d_r2_20 = fit2.r2 - fit0.r2
        d_r2_21 = fit2.r2 - fit1.r2
        d_adj_10 = fit1.adj_r2 - fit0.adj_r2
        d_adj_20 = fit2.adj_r2 - fit0.adj_r2
        d_adj_21 = fit2.adj_r2 - fit1.adj_r2

        base = {
            "table2_order": row.get("table2_order", i + 1),
            "candidate_id": row.get("candidate_id", f"C{i+1:02d}"),
            "candidate_connection": edge,
            "favored_pathway_from_Table2": row.get("primary_favored_pathway", ""),
            "fc_source_column": fc_col,
            "fc_standardization": fc_source,
            "n_model_complete_case": len(edf),
            "n_PSY": int((edf["treatment_TMS"] == 0).sum()),
            "n_TMS": int((edf["treatment_TMS"] == 1).sum()),
        }

        edge_rows.append({
            **base,
            "R2_model0_clinical_QC_only": fit0.r2,
            "R2_model1_clinical_QC_plus_FC_main": fit1.r2,
            "R2_model2_clinical_QC_plus_FC_interaction": fit2.r2,
            "adjR2_model0_clinical_QC_only": fit0.adj_r2,
            "adjR2_model1_clinical_QC_plus_FC_main": fit1.adj_r2,
            "adjR2_model2_clinical_QC_plus_FC_interaction": fit2.adj_r2,
            "delta_R2_FC_main_over_clinical": d_r2_10,
            "delta_R2_full_over_clinical": d_r2_20,
            "delta_R2_interaction_over_FC_main": d_r2_21,
            "delta_adjR2_FC_main_over_clinical": d_adj_10,
            "delta_adjR2_full_over_clinical": d_adj_20,
            "delta_adjR2_interaction_over_FC_main": d_adj_21,
            "interaction_beta_TMS_minus_PSY_per1SD_FC": interaction["coef"],
            "interaction_HC3_SE": interaction["se"],
            "interaction_HC3_95CI_low": interaction["ci_low"],
            "interaction_HC3_95CI_high": interaction["ci_high"],
            "interaction_HC3_t": interaction["t"],
            "interaction_HC3_p": interaction["p"],
            "FC_main_beta_in_model1": fc_main["coef"],
            "FC_main_HC3_p_in_model1": fc_main["p"],
            "partial_F_model1_vs_model0": cmp_10["F"],
            "partial_F_p_model1_vs_model0": cmp_10["p"],
            "partial_F_model2_vs_model1": cmp_21["F"],
            "partial_F_p_model2_vs_model1": cmp_21["p"],
            "partial_F_model2_vs_model0": cmp_20["F"],
            "partial_F_p_model2_vs_model0": cmp_20["p"],
        })

        for model_name, fit in [
            ("Model0_clinical_QC_only", fit0),
            ("Model1_clinical_QC_plus_FC_main", fit1),
            ("Model2_clinical_QC_plus_FC_interaction", fit2),
        ]:
            model_rows.append({
                **base,
                "model": model_name,
                "n": fit.n,
                "df_model": fit.df_model,
                "df_resid": fit.df_resid,
                "R2": fit.r2,
                "adjusted_R2": fit.adj_r2,
                "AIC": fit.aic,
                "BIC": fit.bic,
            })

    edge_df = pd.DataFrame(edge_rows)
    edge_df["interaction_HC3_q_FDR_10edges"] = bh_fdr(edge_df["interaction_HC3_p"])
    edge_df["partial_F_q_model2_vs_model1_FDR_10edges"] = bh_fdr(edge_df["partial_F_p_model2_vs_model1"])
    edge_df["interaction_FDR_supported_q_lt_0_05"] = edge_df["interaction_HC3_q_FDR_10edges"] < 0.05
    edge_df["partial_F_model2_vs_model1_FDR_supported_q_lt_0_05"] = edge_df["partial_F_q_model2_vs_model1_FDR_10edges"] < 0.05

    return edge_df, pd.DataFrame(model_rows), edge_data


# -----------------------------------------------------------------------------
# Bootstrap
# -----------------------------------------------------------------------------

def bootstrap_edge(edge: str, edf: pd.DataFrame, n_boot: int, rng: np.random.Generator, progress_every: int) -> Dict[str, float]:
    vals = {
        "delta_R2_FC_main_over_clinical": [],
        "delta_R2_full_over_clinical": [],
        "delta_R2_interaction_over_FC_main": [],
    }
    failed = 0
    n = len(edf)
    idx = np.arange(n)
    for b in range(1, n_boot + 1):
        sample_idx = rng.choice(idx, size=n, replace=True)
        boot = edf.iloc[sample_idx].copy()
        try:
            # If a bootstrap sample has no treatment variation, models are not interpretable.
            if boot["treatment_TMS"].nunique(dropna=True) < 2:
                failed += 1
                continue
            fit0, fit1, fit2 = fit_edge_models(boot)
            vals["delta_R2_FC_main_over_clinical"].append(fit1.r2 - fit0.r2)
            vals["delta_R2_full_over_clinical"].append(fit2.r2 - fit0.r2)
            vals["delta_R2_interaction_over_FC_main"].append(fit2.r2 - fit1.r2)
        except Exception:
            failed += 1
        if progress_every and b % progress_every == 0:
            log(f"        {edge}: bootstrap {b}/{n_boot} complete")

    out = {"candidate_connection": edge, "bootstrap_n_requested": n_boot, "bootstrap_n_failed": failed}
    ok = n_boot - failed
    out["bootstrap_n_success"] = ok
    out["bootstrap_failure_rate"] = failed / n_boot if n_boot else np.nan
    for key, arr in vals.items():
        a = np.asarray(arr, dtype=float)
        out[f"{key}_bootstrap_mean"] = float(np.nanmean(a)) if len(a) else np.nan
        out[f"{key}_bootstrap_95CI_low"] = float(np.nanpercentile(a, 2.5)) if len(a) else np.nan
        out[f"{key}_bootstrap_95CI_high"] = float(np.nanpercentile(a, 97.5)) if len(a) else np.nan
    return out


def run_bootstrap(manifest: pd.DataFrame, edge_data: Dict[str, pd.DataFrame], n_boot: int, seed: int, progress_every: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []
    for i, row in manifest.iterrows():
        edge = row["candidate_connection"]
        log(f"    Bootstrapping edge {i+1}/{len(manifest)}: {edge}")
        res = bootstrap_edge(edge, edge_data[edge], n_boot, rng, progress_every)
        rows.append({
            "table2_order": row.get("table2_order", i + 1),
            "candidate_id": row.get("candidate_id", f"C{i+1:02d}"),
            "favored_pathway_from_Table2": row.get("primary_favored_pathway", ""),
            **res,
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Output and audit
# -----------------------------------------------------------------------------

def build_s7_panels(edge_results: pd.DataFrame, model_indices: pd.DataFrame, boot: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel_a_cols = [
        "table2_order", "candidate_id", "candidate_connection", "favored_pathway_from_Table2",
        "n_model_complete_case", "n_PSY", "n_TMS",
        "R2_model0_clinical_QC_only", "R2_model1_clinical_QC_plus_FC_main", "R2_model2_clinical_QC_plus_FC_interaction",
        "adjR2_model0_clinical_QC_only", "adjR2_model1_clinical_QC_plus_FC_main", "adjR2_model2_clinical_QC_plus_FC_interaction",
        "delta_R2_FC_main_over_clinical", "delta_R2_full_over_clinical", "delta_R2_interaction_over_FC_main",
        "partial_F_p_model1_vs_model0", "partial_F_p_model2_vs_model1", "partial_F_p_model2_vs_model0",
    ]
    panel_b_cols = [
        "table2_order", "candidate_id", "candidate_connection", "favored_pathway_from_Table2",
        "n_model_complete_case", "interaction_beta_TMS_minus_PSY_per1SD_FC", "interaction_HC3_SE",
        "interaction_HC3_95CI_low", "interaction_HC3_95CI_high", "interaction_HC3_p", "interaction_HC3_q_FDR_10edges",
        "interaction_FDR_supported_q_lt_0_05", "partial_F_p_model2_vs_model1", "partial_F_q_model2_vs_model1_FDR_10edges",
        "partial_F_model2_vs_model1_FDR_supported_q_lt_0_05",
    ]
    panel_a = edge_results[panel_a_cols].copy()
    panel_b = edge_results[panel_b_cols].copy()
    if boot is None:
        panel_c = pd.DataFrame({"note": ["Bootstrap was skipped for this run."]})
    else:
        panel_c = boot.copy()
    return panel_a, panel_b, panel_c


def summarize(edge_results: pd.DataFrame, boot: Optional[pd.DataFrame], inputs: Dict[str, Path]) -> Dict:
    summary = {
        "script": SCRIPT_NAME,
        "run_time": now_str(),
        "inputs": {k: str(v) if v is not None else None for k, v in inputs.items()},
        "summary": {
            "n_edges": int(edge_results["candidate_connection"].nunique()),
            "n_PSY_edges": int((edge_results["favored_pathway_from_Table2"] == "PSY").sum()),
            "n_TMS_edges": int((edge_results["favored_pathway_from_Table2"] == "TMS").sum()),
            "n_min_complete_case": int(edge_results["n_model_complete_case"].min()),
            "n_max_complete_case": int(edge_results["n_model_complete_case"].max()),
            "delta_R2_full_over_clinical_min": float(edge_results["delta_R2_full_over_clinical"].min()),
            "delta_R2_full_over_clinical_max": float(edge_results["delta_R2_full_over_clinical"].max()),
            "delta_R2_interaction_over_FC_main_min": float(edge_results["delta_R2_interaction_over_FC_main"].min()),
            "delta_R2_interaction_over_FC_main_max": float(edge_results["delta_R2_interaction_over_FC_main"].max()),
            "interaction_FDR_supported_q_lt_0_05": int(edge_results["interaction_FDR_supported_q_lt_0_05"].sum()),
            "partial_F_model2_vs_model1_FDR_supported_q_lt_0_05": int(edge_results["partial_F_model2_vs_model1_FDR_supported_q_lt_0_05"].sum()),
        },
        "interpretation_boundary": (
            "Exploratory clinical incremental-value sensitivity analysis only. "
            "Not a clinical utility analysis, not a formal PAI analysis, not a decision-curve analysis, "
            "and not validation of a deployable treatment-assignment rule."
        ),
    }
    if boot is not None:
        summary["summary"]["bootstrap_n_requested"] = int(boot["bootstrap_n_requested"].max())
        summary["summary"]["bootstrap_failure_rate_max"] = float(boot["bootstrap_failure_rate"].max())
    return summary


def write_outputs(out_dir: Path, manifest: pd.DataFrame, edge_results: pd.DataFrame, model_indices: pd.DataFrame,
                  boot: Optional[pd.DataFrame], summary: Dict, audit_lines: List[str]) -> None:
    ensure_dir(out_dir)
    panel_a, panel_b, panel_c = build_s7_panels(edge_results, model_indices, boot)

    manifest.to_csv(out_dir / "00_locked_10_candidate_edges_manifest_used.csv", index=False, encoding="utf-8-sig")
    edge_results.to_csv(out_dir / "01_clinical_incremental_value_edge_level_results.csv", index=False, encoding="utf-8-sig")
    model_indices.to_csv(out_dir / "02_clinical_incremental_value_nested_model_fit_indices.csv", index=False, encoding="utf-8-sig")
    if boot is not None:
        boot.to_csv(out_dir / "03_clinical_incremental_value_bootstrap_deltaR2_CI.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame({"note": ["Bootstrap skipped."]}).to_csv(
            out_dir / "03_clinical_incremental_value_bootstrap_deltaR2_CI.csv", index=False, encoding="utf-8-sig"
        )

    with open(out_dir / "04_clinical_incremental_value_summary_for_manuscript.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    audit_text = "\n".join(audit_lines)
    with open(out_dir / "05_clinical_incremental_value_audit_report.txt", "w", encoding="utf-8") as f:
        f.write(audit_text)

    readme = pd.DataFrame({
        "item": [
            "analysis_name",
            "purpose",
            "model0",
            "model1",
            "model2",
            "main_boundary",
        ],
        "description": [
            "Exploratory clinical incremental-value sensitivity analysis",
            "Quantifies whether locked baseline rsFC candidates add explanatory information beyond baseline clinical/QC covariates.",
            "PCL-5 improvement ~ treatment pathway + baseline PCL-5 + age + sex + pre-treatment mean FD",
            "Model 0 + standardized baseline FC",
            "Model 1 + treatment pathway × standardized baseline FC",
            "Not a deployable individualized treatment rule; not clinical utility validation; not causal comparative efficacy.",
        ],
    })

    xlsx = out_dir / "Supplementary_Table_S7_clinical_incremental_value_sensitivity_analysis.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        panel_a.to_excel(writer, sheet_name="Panel_A_nested_R2", index=False)
        panel_b.to_excel(writer, sheet_name="Panel_B_interaction", index=False)
        panel_c.to_excel(writer, sheet_name="Panel_C_bootstrap", index=False)
        model_indices.to_excel(writer, sheet_name="Model_fit_indices_long", index=False)
        edge_results.to_excel(writer, sheet_name="Full_edge_level_results", index=False)
        manifest.to_excel(writer, sheet_name="Manifest_used", index=False)
        pd.DataFrame({"audit_report": audit_lines}).to_excel(writer, sheet_name="Audit", index=False)


def check_results(edge_results: pd.DataFrame, boot: Optional[pd.DataFrame]) -> List[Tuple[str, bool, str]]:
    checks: List[Tuple[str, bool, str]] = []
    checks.append(("n_edges == 10", edge_results["candidate_connection"].nunique() == EXPECTED_EDGES, str(edge_results["candidate_connection"].nunique())))
    checks.append(("n_PSY_edges == 5", int((edge_results["favored_pathway_from_Table2"] == "PSY").sum()) == EXPECTED_PSY_EDGES, str(int((edge_results["favored_pathway_from_Table2"] == "PSY").sum()))))
    checks.append(("n_TMS_edges == 5", int((edge_results["favored_pathway_from_Table2"] == "TMS").sum()) == EXPECTED_TMS_EDGES, str(int((edge_results["favored_pathway_from_Table2"] == "TMS").sum()))))
    checks.append(("all complete-case n >= 60", bool((edge_results["n_model_complete_case"] >= 60).all()), f"min={edge_results['n_model_complete_case'].min()}"))
    checks.append(("all delta_R2_full_over_clinical finite", bool(np.isfinite(edge_results["delta_R2_full_over_clinical"]).all()), ""))
    checks.append(("all interaction p finite", bool(np.isfinite(edge_results["interaction_HC3_p"]).all()), ""))
    if boot is not None:
        checks.append(("bootstrap max failure rate < 0.10", bool((boot["bootstrap_failure_rate"] < 0.10).all()), f"max={boot['bootstrap_failure_rate'].max():.3f}"))
    return checks


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Supplementary Table S7: exploratory clinical incremental-value sensitivity analysis."
    )
    parser.add_argument("--root", default=None, help="Project root used for automatic file discovery.")
    parser.add_argument("--zip40", default=None, help="Path to 40_v3 result zip.")
    parser.add_argument("--manifest", default=None, help="Path to 00_locked_10_candidate_edges_manifest.csv.")
    parser.add_argument("--subject_table", default=None, help="Optional direct path to 02_subject_level_core_edges.csv.")
    parser.add_argument("--out_dir", default=None, help="Output directory. Default: ./S7_clinical_incremental_value_outputs")
    parser.add_argument("--bootstrap_n", type=int, default=2000, help="Number of bootstrap resamples per edge. Default: 2000")
    parser.add_argument("--skip_bootstrap", action="store_true", help="Skip bootstrap ΔR² confidence intervals.")
    parser.add_argument("--seed", type=int, default=20260531, help="Random seed for bootstrap.")
    parser.add_argument("--progress_every", type=int, default=200, help="Bootstrap progress print interval per edge.")
    parser.add_argument("--allow_manifest_fallback", action="store_true", help="Allow fallback manifest construction from 40_v3 mapping if manifest is absent.")
    return parser.parse_args()


def main() -> int:
    start = time.time()
    args = parse_args()
    audit: List[str] = []

    try:
        log("[1/6] Resolving inputs...")
        inputs = resolve_inputs(args)
        out_dir = inputs["out_dir"]
        ensure_dir(out_dir)
        audit.append(f"{SCRIPT_NAME} audit report")
        audit.append("=" * 90)
        audit.append(f"Run time: {now_str()}")
        for k, v in inputs.items():
            audit.append(f"- {k}: {v}")

        log("[2/6] Loading manifest and subject-level table...")
        if inputs["manifest"] is None and not args.allow_manifest_fallback:
            raise FileNotFoundError(
                "No locked manifest found. For safety, please pass --manifest or place "
                "00_locked_10_candidate_edges_manifest.csv in the working directory/root. "
                "Use --allow_manifest_fallback only if you intentionally want to construct a fallback manifest from 40_v3 mapping."
            )
        manifest, manifest_source = load_manifest(inputs["manifest"], inputs["zip40"])
        subject_df, subject_source = load_subject_table(inputs["subject_table"], inputs["zip40"])
        audit.append(f"Manifest source: {manifest_source}")
        audit.append(f"Subject table source: {subject_source}")
        audit.append(f"Subject table shape: {subject_df.shape}")

        log("[3/6] Validating locked candidate edges and required columns...")
        colmap = detect_columns(subject_df)
        audit.append("Detected columns:")
        for k, v in colmap.items():
            audit.append(f"  {k}: {v}")
        issues = validate_inputs(manifest, subject_df, colmap)
        if issues:
            audit.append("Validation issues:")
            for issue in issues:
                audit.append(f"  - {issue}")
            with open(out_dir / "05_clinical_incremental_value_audit_report_FAILED.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(audit))
            raise RuntimeError("Input validation failed. See audit report in output directory.")
        audit.append("Validation: PASS")

        log("[4/6] Fitting nested OLS models for 10 locked candidate edges...")
        edge_results, model_indices, edge_data = analyze_edges(manifest, subject_df, colmap)

        boot = None
        if args.skip_bootstrap:
            log("[5/6] Bootstrap skipped by user option.")
            audit.append("Bootstrap: skipped")
        else:
            log(f"[5/6] Running bootstrap ΔR² confidence intervals: B={args.bootstrap_n} per edge...")
            boot = run_bootstrap(manifest, edge_data, args.bootstrap_n, args.seed, args.progress_every)
            audit.append(f"Bootstrap: completed, B={args.bootstrap_n} per edge")

        log("[6/6] Validating and writing outputs...")
        checks = check_results(edge_results, boot)
        all_pass = True
        audit.append("Expected checks:")
        for name, ok, detail in checks:
            all_pass = all_pass and bool(ok)
            status = "PASS" if ok else "FAIL"
            line = f"- {name}: {status} {detail}".rstrip()
            audit.append(line)
            log(f"[CHECK] {name}: {status} {detail}".rstrip())

        summary = summarize(edge_results, boot, inputs)
        summary["summary"]["all_expected_checks_pass"] = bool(all_pass)
        audit.append("Summary:")
        for k, v in summary["summary"].items():
            audit.append(f"- {k}: {v}")
        audit.append("Overall: " + ("PASS" if all_pass else "CHECK_WARNINGS"))
        audit.append("")
        audit.append("Interpretation boundary:")
        audit.append(summary["interpretation_boundary"])

        write_outputs(out_dir, manifest, edge_results, model_indices, boot, summary, audit)

        elapsed = time.time() - start
        log(f"[DONE] Output directory: {out_dir}")
        log(f"[DONE] Elapsed seconds: {elapsed:.1f}")
        if all_pass:
            log("[PASS] All expected checks passed.")
        else:
            log("[WARN] Outputs written, but at least one expected check did not pass. Review audit report.")
        return 0

    except Exception as exc:
        log("[ERROR] " + str(exc))
        traceback.print_exc()
        try:
            out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (Path.cwd() / OUT_DIR_NAME).resolve()
            ensure_dir(out_dir)
            audit.append("")
            audit.append("ERROR:")
            audit.append(str(exc))
            audit.append(traceback.format_exc())
            with open(out_dir / "05_clinical_incremental_value_audit_report_FAILED.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(audit))
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
