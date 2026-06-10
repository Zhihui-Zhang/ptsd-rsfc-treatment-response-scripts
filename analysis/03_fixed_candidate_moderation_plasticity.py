# -*- coding: utf-8 -*-
r"""
40_v3_稳定性选择候选边_正式固定候选治疗调节和可塑性分析_修正BNA标签.py

目的
====
本脚本不再做全脑筛边，也不再继续追踪旧候选边 A11m–A32p / A37dl–OPC。
它只针对已经锁定的 10 条 stability-selected candidate edges【稳定性选择候选边】
进行 fixed-candidate moderation analysis【固定候选治疗调节分析】和
responder-linked plasticity analysis【疗效相关可塑性分析】。

固定候选边
==========
本 v3 版不在脚本中手动定义或替换候选边解剖名称。
固定候选边从 39_v3 生成的 canonical value_column 谱系表读取：
03_canonical_fixed_10_candidate_edges_v3.csv

核心原则：
- 建模取值列使用已验证的 value_column【例如 unknown_roi_*__unknown_roi_*】；
- corrected BNA labels【修正 Brainnetome 标签】仅作为输出标签和审计字段；
- 不改变统计模型、不改变取值列、不重跑筛边。

核心模型
========
PCL improvement = treatment + baseline FC + treatment × baseline FC

其中：
- treatment = 1 表示 TMS【经颅磁刺激】
- treatment = 0 表示 PSY【心理治疗，ACT+MIN】
- PCL improvement = PCL_pre - PCL_post【治疗前PCL减去治疗后PCL，越大表示改善越多】
- baseline FC 使用治疗前 pre/baseline 功能连接，并进行 z-score 标准化
- 重点检验 treatment × baseline FC【治疗方式 × 基线功能连接】交互项

输出
====
00_run_audit.json
00_pre_meanFD_recovery_audit.csv
01_candidate_edge_mapping.csv
02_subject_level_core_edges.csv
10_fixed_candidate_moderation_no_covariates.csv
11_fixed_candidate_moderation_covariates.csv
12_simple_slopes_by_treatment.csv
20_plasticity_high_low_tests.csv
21_plasticity_delta_improvement_correlations.csv
22_plasticity_summary_best_per_edge.csv
30_interaction_plot_data_long.csv
31_plasticity_plot_data_long.csv
40_final_core_edge_decision_table.csv
99_fixed_candidate_moderation_report.txt

默认运行
========
cd /d "E:\E_zhangzhihui\从yv那边提取\脚本\第四步分析：探索"
python -u 40_稳定性选择候选边_正式固定候选治疗调节和可塑性分析.py

显式指定输入
============
python -u 40_稳定性选择候选边_正式固定候选治疗调节和可塑性分析.py --input_file "E:\E_zhangzhihui\从yv那边提取\脚本\第二步分析\06_Brainnetome246_纵向FC宽表_合并clinical_postmeanFD_重跑合并版.csv"

说明
====
本脚本用于 characterization【效应刻画】，不是独立外部验证。
候选边来自同一项目数据中的全脑稳定性选择结果，因此论文中应表述为
fixed-candidate characterization of stability-selected edges【稳定性选择候选边的固定候选效应刻画】，
而不是 independent validation【独立验证】。

重要修正
========
本 v3 版使用 39_v3 谱系审计通过的 canonical candidate-edge table。
候选边身份以 value_column 为准，避免旧 anatomical label 映射错误进入
BrainNet 或后续报告。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


DEFAULT_INPUT = str(Path(__file__).resolve().parent / "论文分析过程文件备份" / "06_Brainnetome246_纵向FC宽表_合并clinical_postmeanFD_重跑合并版.csv")
DEFAULT_OUT_DIR = str(Path(__file__).resolve().parent / "40_v3_稳定性选择候选边_正式固定候选治疗调节和可塑性分析结果_修正BNA标签")
DEFAULT_CANONICAL_EDGES = str(Path(__file__).resolve().parent / "39_v3_稳定性选择候选边_value_column谱系审计_修正BNA标签结果" / "03_canonical_fixed_10_candidate_edges_v3.csv")
DEFAULT_V2_SUBJECT_TABLE = str(Path(__file__).resolve().parent / "40_稳定性选择候选边_正式固定候选治疗调节和可塑性分析结果_v2_A3536c_dIa映射修正版" / "02_subject_level_core_edges.csv")


# ============================================================
# 0. 固定候选边
# ============================================================

CORE_EDGE_SPECS = []


def _truthy(v: Any) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes", "y"}


def load_core_edge_specs_from_canonical(path: str, available_columns: Optional[set] = None) -> List[Dict[str, Any]]:
    """Load fixed candidate edges from the 39_v3 value_column lineage table.

    v3 uses `value_column` as the identity of each edge. Corrected BNA labels are
    attached only after the 39->40->BrainNet lineage has passed QC upstream.
    """
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"canonical candidate-edge table not found: {p}")
    canon = pd.read_csv(p, encoding="utf-8-sig")
    required = [
        "edge_id",
        "value_column",
        "old_edge_label",
        "corrected_edge_label",
        "corrected_roi1_id",
        "corrected_roi1_label",
        "corrected_roi1_hemisphere",
        "corrected_roi2_id",
        "corrected_roi2_label",
        "corrected_roi2_hemisphere",
        "favoring_direction",
        "value_column_verified_in_39",
        "value_column_verified_in_40_v2",
        "value_column_verified_in_wide_table",
        "safe_for_script40_v3",
    ]
    missing = [c for c in required if c not in canon.columns]
    if missing:
        raise RuntimeError(f"canonical candidate-edge table missing required columns: {missing}")
    if len(canon) != 10:
        raise RuntimeError(f"canonical candidate-edge table must contain 10 rows, found {len(canon)}")
    for col in [
        "value_column_verified_in_39",
        "value_column_verified_in_40_v2",
        "value_column_verified_in_wide_table",
        "safe_for_script40_v3",
    ]:
        if not canon[col].map(_truthy).all():
            bad = canon.loc[~canon[col].map(_truthy), ["edge_id", "value_column", col]]
            raise RuntimeError(f"canonical QC column {col} contains failed rows:\n{bad.to_string(index=False)}")
    if canon["value_column"].isna().any() or canon["value_column"].astype(str).str.strip().eq("").any():
        raise RuntimeError("canonical candidate-edge table contains missing value_column")
    for col in ["corrected_edge_label", "corrected_roi1_label", "corrected_roi2_label"]:
        if canon[col].isna().any() or canon[col].astype(str).str.strip().eq("").any():
            bad = canon.loc[canon[col].isna() | canon[col].astype(str).str.strip().eq(""), ["edge_id", "value_column", col]]
            raise RuntimeError(f"canonical candidate-edge table contains missing {col}:\n{bad.to_string(index=False)}")
    if available_columns is not None:
        missing_value_cols = sorted(set(canon["value_column"].astype(str)) - set(available_columns))
        if missing_value_cols:
            raise RuntimeError(f"canonical value_column not found in input wide table: {missing_value_cols}")

    specs: List[Dict[str, Any]] = []
    for _, row in canon.sort_values("edge_id").iterrows():
        direction = str(row["favoring_direction"]).strip().upper()
        if direction not in {"PSY", "TMS"}:
            raise RuntimeError(f"unexpected favoring_direction for edge_id={row['edge_id']}: {row['favoring_direction']}")
        specs.append({
            "edge_id": int(row["edge_id"]),
            "edge_short": str(row["corrected_edge_label"]),
            "old_edge_label": str(row["old_edge_label"]),
            "corrected_edge_label": str(row["corrected_edge_label"]),
            "preferred_unknown_col": str(row["value_column"]),
            "value_column": str(row["value_column"]),
            "corrected_roi1_id": int(row["corrected_roi1_id"]),
            "corrected_roi1_label": str(row["corrected_roi1_label"]),
            "corrected_roi1_hemisphere": str(row["corrected_roi1_hemisphere"]),
            "corrected_roi2_id": int(row["corrected_roi2_id"]),
            "corrected_roi2_label": str(row["corrected_roi2_label"]),
            "corrected_roi2_hemisphere": str(row["corrected_roi2_hemisphere"]),
            "system": "corrected_BNA_label",
            "system_cn": "修正BNA标签",
            "expected_direction_from_stability": f"{direction}-favoring",
            "rationale": "Loaded from 39_v3 canonical value_column lineage table; corrected BNA labels attached after lineage QC.",
        })
    return specs


# ============================================================
# 1. 基础工具
# ============================================================

def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str):
    print(f"[{now()}] {msg}", flush=True)


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_table(path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"输入文件不存在：{p}")
    if p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    return pd.read_csv(p, encoding="utf-8-sig", low_memory=False)


def write_csv(df: pd.DataFrame, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, encoding="utf-8-sig")


def safe_json_dump(obj: Dict[str, Any], path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def num(s) -> pd.Series:
    return pd.to_numeric(pd.Series(s), errors="coerce")


def safe_float(x):
    try:
        if pd.isna(x) or not np.isfinite(float(x)):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def zscore(s) -> pd.Series:
    x = num(s)
    sd = x.std(skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    return (x - x.mean(skipna=True)) / sd


def fdr_bh(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    if mask.sum() == 0:
        return out
    pv = p[mask]
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    tmp = np.empty(n, dtype=float)
    tmp[order] = q
    out[mask] = tmp
    return out


def hedges_g(x, y) -> float:
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    y = np.asarray(pd.Series(y).dropna(), dtype=float)
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return np.nan
    v1, v2 = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / max(n1 + n2 - 2, 1)
    if pooled <= 0 or not np.isfinite(pooled):
        return np.nan
    d = (np.mean(x) - np.mean(y)) / math.sqrt(pooled)
    correction = 1 - (3 / (4 * (n1 + n2) - 9)) if (n1 + n2) > 2 else 1
    return float(d * correction)


def fit_ols(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X = X[mask]
    y = y[mask]
    n, k = X.shape
    if n <= k + 2:
        return None
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        fitted = X @ beta
        resid = y - fitted
        dof = n - k
        s2 = np.sum(resid ** 2) / max(dof, 1)
        cov = s2 * np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        with np.errstate(divide="ignore", invalid="ignore"):
            tvals = beta / se
        pvals = 2 * stats.t.sf(np.abs(tvals), df=dof)
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {
            "beta": beta,
            "se": se,
            "t": tvals,
            "p": pvals,
            "n": int(n),
            "dof": int(dof),
            "r2": float(r2) if np.isfinite(r2) else np.nan,
            "fitted": fitted,
            "resid": resid,
            "mask": mask,
        }
    except Exception:
        return None


def pearson_pair(x, y):
    tmp = pd.DataFrame({"x": num(x), "y": num(y)}).dropna()
    if len(tmp) < 6 or tmp["x"].nunique() <= 2 or tmp["y"].nunique() <= 2:
        return np.nan, np.nan, int(len(tmp))
    try:
        r, p = stats.pearsonr(tmp["x"], tmp["y"])
        return float(r), float(p), int(len(tmp))
    except Exception:
        return np.nan, np.nan, int(len(tmp))


def spearman_pair(x, y):
    tmp = pd.DataFrame({"x": num(x), "y": num(y)}).dropna()
    if len(tmp) < 6 or tmp["x"].nunique() <= 2 or tmp["y"].nunique() <= 2:
        return np.nan, np.nan, int(len(tmp))
    try:
        r, p = stats.spearmanr(tmp["x"], tmp["y"])
        return float(r), float(p), int(len(tmp))
    except Exception:
        return np.nan, np.nan, int(len(tmp))


def norm(s) -> str:
    return re.sub(r"[^a-z0-9/]+", "", str(s).lower().replace("—", "_").replace("-", "_").replace(",", "_"))


# ============================================================
# 2. 输入列识别
# ============================================================

def canon_group(x) -> str:
    s = str(x).upper().strip()
    if "TMS" in s or "SEAT" in s:
        return "TMS"
    if "ACT" in s:
        return "ACT"
    if "MIN" in s:
        return "MIN"
    if "PSY" in s or "心理" in s:
        return "PSY"
    if "WL" in s or "WAIT" in s or "等待" in s:
        return "WL"
    if "HC" in s or "CONTROL" in s or "健康" in s:
        return "HC"
    return s


def infer_group_col(df: pd.DataFrame) -> str:
    for c in ["group_final", "group", "treatment", "arm", "组别"]:
        if c in df.columns:
            return c
    best, best_score = None, -1
    for c in df.columns:
        vals = df[c].dropna().astype(str).str.upper().head(300)
        score = sum(any(g in v for g in ["TMS", "ACT", "MIN", "WL", "HC", "SEAT", "PSY"]) for v in vals)
        if score > best_score:
            best, best_score = c, score
    if best is None:
        raise RuntimeError("未识别到组别列。")
    return best


def extract_timepoint(x) -> str:
    s = str(x).lower().strip()
    if s in ["post", "后", "后测"]:
        return "post"
    if s in ["pre", "baseline", "前", "前测", "基线"]:
        return "pre"
    if re.search(r"(^|[-_/])post($|[-_/])", s) or "后测" in s:
        return "post"
    if re.search(r"(^|[-_/])pre($|[-_/])", s) or "baseline" in s or "前测" in s or "基线" in s:
        return "pre"
    return "unknown"


def infer_timepoint(df: pd.DataFrame) -> Tuple[pd.Series, Optional[str]]:
    for c in ["time", "timepoint", "session", "scan_time"]:
        if c in df.columns:
            return df[c].apply(extract_timepoint), c
    if "scan_id" in df.columns:
        return df["scan_id"].apply(extract_timepoint), "scan_id"
    return pd.Series(["unknown"] * len(df), index=df.index), None


def subj_from_scan(x) -> str:
    s = str(x)
    m = re.search(r"scan[-_](?:pre|post)[-_]([A-Za-z]+)[-_]?([0-9A-Za-z]+)", s, flags=re.I)
    if m:
        return m.group(2)
    m = re.search(r"sub[-_]?([A-Za-z0-9]+)", s, flags=re.I)
    if m:
        return m.group(1)
    nums = re.findall(r"\d+", s)
    if nums:
        return nums[-1]
    return s


def standardize_subject_raw(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    return s.apply(lambda x: x.zfill(4) if re.fullmatch(r"\d+", str(x)) else str(x))


def infer_subject_raw(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    for c in ["subject_id4", "subject_digits", "subject_id", "subject", "participant_id", "被试编号"]:
        if c in df.columns:
            return standardize_subject_raw(df[c]), c
    if "scan_id" in df.columns:
        return standardize_subject_raw(df["scan_id"].apply(subj_from_scan)), "scan_id"
    raise RuntimeError("未找到 subject_id4/subject_digits/subject_id/scan_id，无法构建被试编号。")


def first_nonmissing(s):
    s = pd.Series(s).dropna()
    return s.iloc[0] if len(s) else np.nan


def infer_pcl_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    for c in ["PCL_improvement_pre_minus_post", "pcl_improvement_pre_minus_post", "PCL_improvement", "pcl_improvement", "PCL改善"]:
        if c in df.columns:
            return c, None, None
    pre, post = None, None
    for c in ["PCL_pre", "pcl_pre", "PCL_B", "PCL_baseline", "PCL前测", "PCL_前测"]:
        if c in df.columns:
            pre = c
            break
    for c in ["PCL_post", "pcl_post", "PCL后测", "PCL_后测"]:
        if c in df.columns:
            post = c
            break
    return None, pre, post



# ============================================================
# 2b. pre_meanFD 标准化恢复与等价审计
# ============================================================

PRE_MEANFD_TOL = 1e-10


def normalize_col_for_pre_meanfd(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(c).lower())


def pre_meanfd_requested(covariates_requested: List[str]) -> bool:
    return any(normalize_col_for_pre_meanfd(c) == "premeanfd" for c in covariates_requested)


def candidate_pre_meanfd_columns(df: pd.DataFrame) -> List[str]:
    """Find plausible pre-treatment mean FD columns in the formal wide table.

    This function does not use any anatomical labels and does not touch candidate FC values.
    It only searches metadata/covariate columns for pre_meanFD recovery.
    """
    priority_exact = [
        "pre_meanFD",
        "pre_mean_fd",
        "meanFD_pre",
        "mean_fd_pre",
        "preFD",
        "pre_FD",
        "baseline_meanFD",
        "baseline_mean_fd",
        "meanFD",
        "mean_FD",
    ]
    out: List[str] = []
    cols = list(df.columns)

    for c in priority_exact:
        if c in df.columns and c not in out:
            out.append(c)

    for c in cols:
        nc = normalize_col_for_pre_meanfd(c)
        if c in out:
            continue
        has_fd = "fd" in nc
        has_mean = "mean" in nc or nc == "fd"
        has_pre = "pre" in nc or "baseline" in nc or "base" in nc
        if has_fd and has_mean and has_pre:
            out.append(c)

    # meanFD is allowed as final fallback because old v2 used the same subject-level table logic.
    for c in cols:
        nc = normalize_col_for_pre_meanfd(c)
        if c in out:
            continue
        if nc in {"meanfd", "meanfdvalue", "fdmean"}:
            out.append(c)

    return out


def read_v2_pre_meanfd_table(v2_subject_table: str) -> Optional[pd.DataFrame]:
    p = Path(v2_subject_table) if v2_subject_table else Path("")
    if not p.exists():
        return None
    v2 = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
    if "subject_key" not in v2.columns or "pre_meanFD" not in v2.columns:
        return None
    out = v2[["subject_key", "pre_meanFD"]].copy()
    out["v2_pre_meanFD"] = num(out["pre_meanFD"]).values
    out = out.drop(columns=["pre_meanFD"])
    return out


def recover_pre_meanfd(
    df: pd.DataFrame,
    first: pd.DataFrame,
    subj: pd.DataFrame,
    out_dir: Path,
    covariates_requested: List[str],
    v2_subject_table: str = "",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Recover pre_meanFD in a standardized and auditable way.

    Priority:
    1. Use a valid source column from the formal wide table if it exactly matches v2 by subject_key.
    2. If no formal-wide-table source matches but v2 subject table is available, use v2 pre_meanFD as
       an explicit equivalence-preserving recovery source.
    3. If pre_meanFD is requested but cannot be recovered and validated, hard-stop.

    This function does not change sample inclusion, treatment coding, FC value columns, labels, model
    formulas, or FDR correction.
    """
    requested = pre_meanfd_requested(covariates_requested)
    meta: Dict[str, Any] = {
        "pre_meanFD_requested": bool(requested),
        "pre_meanFD_recovery_attempted": False,
        "pre_meanFD_source_column": "",
        "pre_meanFD_n_nonmissing_after_recovery": 0,
        "pre_meanFD_matched_to_v2": "not_checked",
        "pre_meanFD_v2_subject_table": str(v2_subject_table) if v2_subject_table else "",
    }

    before = num(subj["pre_meanFD"]).values if "pre_meanFD" in subj.columns else np.full(len(subj), np.nan)

    # Build candidate sources from the formal wide table aggregation already used for all other covariates.
    source_cols = candidate_pre_meanfd_columns(df)
    candidate_series: Dict[str, pd.Series] = {}
    if "pre_meanFD" in subj.columns:
        candidate_series["existing_subj_pre_meanFD_before_fix"] = pd.Series(before, index=subj.index)
    for c in source_cols:
        if c in first.columns:
            candidate_series[f"formal_wide::{c}"] = num(first[c]).reset_index(drop=True)

    v2_ref = read_v2_pre_meanfd_table(v2_subject_table)
    v2_map = None
    if v2_ref is not None:
        v2_map = subj[["subject_key"]].merge(v2_ref, on="subject_key", how="left")["v2_pre_meanFD"]

    chosen_name = ""
    chosen = None
    chosen_match_to_v2 = False

    if v2_map is not None and v2_map.notna().sum() > 0:
        best_name = ""
        best_series = None
        best_score = (-1, -np.inf)  # exact_matches, -maxdiff
        for name, s in candidate_series.items():
            ss = num(s).reset_index(drop=True)
            ok = v2_map.notna()
            if ok.sum() == 0:
                continue
            diff = (ss[ok] - v2_map[ok]).abs()
            exact = int((diff <= PRE_MEANFD_TOL).sum())
            maxdiff = float(diff.max(skipna=True)) if diff.notna().any() else np.inf
            score = (exact, -maxdiff if np.isfinite(maxdiff) else -np.inf)
            if score > best_score:
                best_score = score
                best_name = name
                best_series = ss
        if best_series is not None:
            ok = v2_map.notna()
            diff = (best_series[ok] - v2_map[ok]).abs()
            if int((diff <= PRE_MEANFD_TOL).sum()) == int(ok.sum()):
                chosen_name = best_name
                chosen = best_series
                chosen_match_to_v2 = True

        # Explicit equivalence-preserving fallback: v2 is the previous validated source of this covariate.
        if chosen is None:
            chosen_name = "v2_subject_table::pre_meanFD"
            chosen = v2_map.reset_index(drop=True)
            chosen_match_to_v2 = True
    else:
        # No v2 reference. Use the best formal-wide-table source.
        best_name = ""
        best_series = None
        best_n = -1
        for name, s in candidate_series.items():
            ss = num(s).reset_index(drop=True)
            n = int(ss.notna().sum())
            if n > best_n:
                best_n = n
                best_name = name
                best_series = ss
        if best_series is not None and best_n >= 20:
            chosen_name = best_name
            chosen = best_series
            chosen_match_to_v2 = False

    if chosen is None:
        audit_df = pd.DataFrame({
            "subject_key": subj["subject_key"],
            "v2_pre_meanFD": v2_map if v2_map is not None else np.nan,
            "v3_pre_meanFD_before_fix": before,
            "v3_pre_meanFD_after_fix": np.nan,
            "source_column_used": "",
            "abs_diff_v2_v3": np.nan,
            "match_to_v2": False,
        })
        write_csv(audit_df, out_dir / "00_pre_meanFD_recovery_audit.csv")
        if requested:
            raise RuntimeError(
                "pre_meanFD was requested as a covariate but could not be recovered. "
                "No covariate-adjusted model was run to avoid silent method changes."
            )
        return subj, meta

    meta["pre_meanFD_recovery_attempted"] = True
    meta["pre_meanFD_source_column"] = chosen_name

    subj = subj.copy()
    subj["pre_meanFD"] = num(chosen).values

    after = num(subj["pre_meanFD"])
    if v2_map is not None:
        diff = (after - v2_map).abs()
        match = (diff <= PRE_MEANFD_TOL) | (after.isna() & v2_map.isna())
        # Hard-stop only for v2 nonmissing values; v2 is the equivalence target.
        ok_v2 = v2_map.notna()
        all_v2_recovered = bool((after[ok_v2].notna()).all())
        all_v2_matched = bool(((diff[ok_v2] <= PRE_MEANFD_TOL)).all())
        meta["pre_meanFD_matched_to_v2"] = bool(all_v2_recovered and all_v2_matched)
    else:
        diff = pd.Series([np.nan] * len(subj), index=subj.index)
        match = pd.Series([False] * len(subj), index=subj.index)
        meta["pre_meanFD_matched_to_v2"] = "v2_reference_unavailable"

    audit_df = pd.DataFrame({
        "subject_key": subj["subject_key"],
        "v2_pre_meanFD": v2_map if v2_map is not None else np.nan,
        "v3_pre_meanFD_before_fix": before,
        "v3_pre_meanFD_after_fix": after,
        "source_column_used": chosen_name,
        "abs_diff_v2_v3": diff,
        "match_to_v2": match,
    })
    write_csv(audit_df, out_dir / "00_pre_meanFD_recovery_audit.csv")

    meta["pre_meanFD_n_nonmissing_after_recovery"] = int(after.notna().sum())
    meta["pre_meanFD_recovery_audit_file"] = str((out_dir / "00_pre_meanFD_recovery_audit.csv").resolve())

    log(f"pre_meanFD恢复：source={chosen_name}; nonmissing={int(after.notna().sum())}; match_to_v2={meta['pre_meanFD_matched_to_v2']}")

    if requested:
        if after.notna().sum() < 20 or after.nunique(dropna=True) <= 1:
            raise RuntimeError("pre_meanFD was requested but recovered values are insufficient for covariate modeling.")
        if v2_map is not None and not bool(meta["pre_meanFD_matched_to_v2"]):
            bad = audit_df.loc[v2_map.notna() & (~audit_df["match_to_v2"]), ["subject_key", "v2_pre_meanFD", "v3_pre_meanFD_after_fix", "abs_diff_v2_v3"]]
            raise RuntimeError(
                "pre_meanFD recovery failed v2 equivalence check. First mismatches:\n"
                + bad.head(20).to_string(index=False)
            )

    return subj, meta


# ============================================================
# 3. FC边识别与候选边映射
# ============================================================

def is_named_edge(c: str) -> bool:
    c = str(c)
    if "__" not in c:
        return False
    if "unknown_roi_" in c or "brainnetome_roi_" in c:
        return False
    cn = norm(c)
    bad = ["pcl", "meanfd", "subject", "scan", "group", "clinical", "merge", "time", "age", "sex"]
    if any(b in cn for b in bad):
        return False
    try:
        a, b = c.split("__", 1)
    except Exception:
        return False
    return len(a) > 1 and len(b) > 1


def is_unknown_edge(c: str) -> bool:
    return bool(re.fullmatch(r"unknown_roi_\d+__unknown_roi_\d+", str(c)))


def is_generic_edge(c: str) -> bool:
    return bool(re.fullmatch(r"brainnetome_roi_\d+__brainnetome_roi_\d+", str(c)))


def build_value_column_presence_table(df: pd.DataFrame, specs: List[Dict[str, Any]]) -> pd.DataFrame:
    """Audit canonical value_column availability without named-node remapping."""
    rows = []
    cols = set(df.columns)
    for spec in specs:
        value_col = str(spec.get("value_column", ""))
        rows.append({
            "edge_id": spec.get("edge_id", ""),
            "edge_short": spec.get("edge_short", ""),
            "old_edge_label": spec.get("old_edge_label", ""),
            "corrected_edge_label": spec.get("corrected_edge_label", spec.get("edge_short", "")),
            "canonical_value_column": value_col,
            "value_column_exists_in_input": value_col in cols,
            "corrected_roi1_id": spec.get("corrected_roi1_id", ""),
            "corrected_roi1_label": spec.get("corrected_roi1_label", ""),
            "corrected_roi1_hemisphere": spec.get("corrected_roi1_hemisphere", ""),
            "corrected_roi2_id": spec.get("corrected_roi2_id", ""),
            "corrected_roi2_label": spec.get("corrected_roi2_label", ""),
            "corrected_roi2_hemisphere": spec.get("corrected_roi2_hemisphere", ""),
        })
    return pd.DataFrame(rows)


def node_matches(node: str, tokens: List[str]) -> bool:
    nn = norm(node)
    for tok in tokens:
        tt = norm(tok)
        if not tt:
            continue
        if tt in nn:
            return True
    return False


# ============================================================
# 4. 构建被试级表
# ============================================================

def subject_pre_post_values(scan_v: pd.Series, subject_key: pd.Series, timepoint: pd.Series, subjects: List[str]) -> Tuple[pd.Series, pd.Series, pd.Series]:
    tmp = pd.DataFrame({"subject_key": subject_key, "timepoint": timepoint, "v": num(scan_v)})
    pre = tmp[tmp["timepoint"] == "pre"].groupby("subject_key")["v"].apply(first_nonmissing).reindex(subjects)
    post = tmp[tmp["timepoint"] == "post"].groupby("subject_key")["v"].apply(first_nonmissing).reindex(subjects)
    pre = pre.reset_index(drop=True)
    post = post.reset_index(drop=True)
    delta = post - pre
    return pre, post, delta


def build_subject_table(df: pd.DataFrame, out_dir: Path, covariates_requested: List[str], v2_subject_table: str = "") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    log("构建被试级表：TMS vs PSY，baseline FC + delta FC + PCL improvement")
    group_col = infer_group_col(df)
    group = df[group_col].apply(canon_group)

    timepoint, time_col = infer_timepoint(df)
    raw_subject, subject_col = infer_subject_raw(df)
    subject_key = group.astype(str) + "_" + raw_subject.astype(str)

    pairing = pd.DataFrame({
        "row": np.arange(len(df)),
        "subject_raw": raw_subject,
        "subject_key": subject_key,
        "group": group,
        "timepoint": timepoint,
        "scan_id": df["scan_id"].astype(str) if "scan_id" in df.columns else "",
    })
    write_csv(pairing, out_dir / "00_scan_to_subject_pairing_audit.csv")

    pre_ids = set(pairing.loc[pairing["timepoint"] == "pre", "subject_key"])
    post_ids = set(pairing.loc[pairing["timepoint"] == "post", "subject_key"])
    paired_subjects_all = sorted(pre_ids & post_ids)
    if not paired_subjects_all:
        raise RuntimeError("没有识别到pre/post成对被试，请检查timepoint或scan_id。")

    mapping_df = build_value_column_presence_table(df, CORE_EDGE_SPECS)
    write_csv(mapping_df, out_dir / "00_all_edge_name_mapping.csv")

    first = df.assign(__subject_key__=subject_key).groupby("__subject_key__").agg(first_nonmissing).reindex(paired_subjects_all)

    subj = pd.DataFrame({"subject_key": paired_subjects_all})
    subj["subject_raw"] = subj["subject_key"].str.replace(r"^[A-Z]+_", "", regex=True)
    subj["group_raw"] = first[group_col].apply(canon_group).values
    subj["group_tms_psy"] = subj["group_raw"].replace({"ACT": "PSY", "MIN": "PSY"})
    subj = subj[subj["group_tms_psy"].isin(["TMS", "PSY"])].reset_index(drop=True)
    subjects = subj["subject_key"].tolist()
    first = first.reindex(subjects)

    subj["treatment_TMS"] = (subj["group_tms_psy"] == "TMS").astype(int)

    # 常用临床/协变量
    for c in ["site", "age", "sex", "pre_meanFD", "meanFD", "PCL_pre", "PCL_post"]:
        if c in df.columns:
            subj[c] = first[c].values

    # 若sex为字符串，编码
    if "sex" in subj.columns and not np.issubdtype(pd.Series(subj["sex"]).dropna().dtype, np.number):
        subj["sex"] = pd.Categorical(subj["sex"].astype(str)).codes

    improve_col, pcl_pre_col, pcl_post_col = infer_pcl_cols(df)
    if pcl_pre_col and pcl_pre_col in df.columns:
        subj["PCL_pre"] = num(first[pcl_pre_col]).values
    if pcl_post_col and pcl_post_col in df.columns:
        subj["PCL_post"] = num(first[pcl_post_col]).values
    if improve_col:
        subj["pcl_improvement"] = num(first[improve_col]).values
    elif "PCL_pre" in subj.columns and "PCL_post" in subj.columns:
        subj["pcl_improvement"] = num(subj["PCL_pre"]) - num(subj["PCL_post"])
    else:
        raise RuntimeError("无法识别PCL improvement或PCL_pre/PCL_post。")

    # 标准化恢复 pre_meanFD：只修复协变量来源，不改变候选边取值、标签、样本或模型公式。
    subj, pre_meanfd_meta = recover_pre_meanfd(
        df=df,
        first=first,
        subj=subj,
        out_dir=out_dir,
        covariates_requested=covariates_requested,
        v2_subject_table=v2_subject_table,
    )

    # 映射10条候选边
    edge_mapping_rows = []
    for spec in CORE_EDGE_SPECS:
        value_col = str(spec.get("value_column", ""))
        if not value_col or value_col not in df.columns:
            edge_mapping_rows.append({
                "edge_id": spec.get("edge_id", ""),
                "edge_short": spec["edge_short"],
                "old_edge_label": spec.get("old_edge_label", ""),
                "corrected_edge_label": spec.get("corrected_edge_label", spec["edge_short"]),
                "canonical_value_column": spec.get("value_column", ""),
                "corrected_roi1_id": spec.get("corrected_roi1_id", ""),
                "corrected_roi1_label": spec.get("corrected_roi1_label", ""),
                "corrected_roi1_hemisphere": spec.get("corrected_roi1_hemisphere", ""),
                "corrected_roi2_id": spec.get("corrected_roi2_id", ""),
                "corrected_roi2_label": spec.get("corrected_roi2_label", ""),
                "corrected_roi2_hemisphere": spec.get("corrected_roi2_hemisphere", ""),
                "status": "not_found",
                "selected_named_col": "",
                "selected_unknown_col": value_col,
                "selected_generic_col": "",
                "value_source": "not_found",
                "match_method": "canonical_value_column_missing",
                "n_pre_nonmissing": 0,
                "n_post_nonmissing": 0,
                "n_delta_nonmissing": 0,
                "system": spec["system"],
                "system_cn": spec["system_cn"],
                "expected_direction_from_stability": spec["expected_direction_from_stability"],
                "rationale": spec["rationale"],
            })
            for suffix in ["pre", "post", "delta", "baseline_z"]:
                subj[f"{spec['edge_short']}__{suffix}"] = np.nan
            continue

        scan_v = num(df[value_col])
        value_source = "canonical_value_column"
        pre, post, delta = subject_pre_post_values(scan_v, subject_key, timepoint, subjects)

        base_col = f"{spec['edge_short']}__pre"
        post_col = f"{spec['edge_short']}__post"
        delta_col = f"{spec['edge_short']}__delta"
        z_col = f"{spec['edge_short']}__baseline_z"

        subj[base_col] = pre.values
        subj[post_col] = post.values
        subj[delta_col] = delta.values
        subj[z_col] = zscore(subj[base_col]).values

        edge_mapping_rows.append({
            "edge_id": spec.get("edge_id", ""),
            "edge_short": spec["edge_short"],
            "old_edge_label": spec.get("old_edge_label", ""),
            "corrected_edge_label": spec.get("corrected_edge_label", spec["edge_short"]),
            "canonical_value_column": spec.get("value_column", ""),
            "corrected_roi1_id": spec.get("corrected_roi1_id", ""),
            "corrected_roi1_label": spec.get("corrected_roi1_label", ""),
            "corrected_roi1_hemisphere": spec.get("corrected_roi1_hemisphere", ""),
            "corrected_roi2_id": spec.get("corrected_roi2_id", ""),
            "corrected_roi2_label": spec.get("corrected_roi2_label", ""),
            "corrected_roi2_hemisphere": spec.get("corrected_roi2_hemisphere", ""),
            "status": "matched",
            "selected_named_col": "",
            "selected_unknown_col": value_col,
            "selected_generic_col": "",
            "value_source": value_source,
            "match_method": "canonical_value_column_direct",
            "node_a": "",
            "node_b": "",
            "idx_a": "",
            "idx_b": "",
            "n_pre_nonmissing": int(pd.Series(pre).notna().sum()),
            "n_post_nonmissing": int(pd.Series(post).notna().sum()),
            "n_delta_nonmissing": int(pd.Series(delta).notna().sum()),
            "system": spec["system"],
            "system_cn": spec["system_cn"],
            "expected_direction_from_stability": spec["expected_direction_from_stability"],
            "rationale": spec["rationale"],
        })

    edge_map_df = pd.DataFrame(edge_mapping_rows)
    write_csv(edge_map_df, out_dir / "01_candidate_edge_mapping.csv")
    write_csv(subj, out_dir / "02_subject_level_core_edges.csv")

    # 实际可用协变量
    covariates_used = []
    for c in covariates_requested:
        c = c.strip()
        if not c:
            continue
        if c in subj.columns and num(subj[c]).notna().sum() >= 20 and num(subj[c]).nunique(dropna=True) > 1:
            covariates_used.append(c)

    audit = {
        "n_rows_input": int(len(df)),
        "n_cols_input": int(df.shape[1]),
        "group_col": group_col,
        "subject_col": subject_col,
        "timepoint_col": time_col,
        "outcome_col": improve_col if improve_col else "PCL_pre_minus_PCL_post",
        "n_paired_subjects_all_groups": int(len(paired_subjects_all)),
        "n_tms_psy_subjects": int(len(subj)),
        "n_TMS": int((subj["group_tms_psy"] == "TMS").sum()),
        "n_PSY": int((subj["group_tms_psy"] == "PSY").sum()),
        "n_candidate_edges_requested": int(len(CORE_EDGE_SPECS)),
        "n_candidate_edges_matched": int((edge_map_df["status"] == "matched").sum()),
        "candidate_edge_source": "39_v3 canonical value_column lineage table",
        "candidate_value_columns": [spec.get("value_column", "") for spec in CORE_EDGE_SPECS],
        "corrected_edge_labels": [spec.get("corrected_edge_label", spec["edge_short"]) for spec in CORE_EDGE_SPECS],
        "covariates_requested": covariates_requested,
        "covariates_used": covariates_used,
        "n_named_edges_mapping": int(len(mapping_df)),
    }
    audit.update(pre_meanfd_meta)
    return subj, audit


# ============================================================
# 5. 固定候选治疗调节模型
# ============================================================

def get_covariate_matrix(df: pd.DataFrame, covariates: List[str]) -> Tuple[List[np.ndarray], List[str]]:
    cols = []
    names = []
    for c in covariates:
        if c not in df.columns:
            continue
        x = num(df[c])
        if x.notna().sum() < 10 or x.nunique(dropna=True) <= 1:
            continue
        x = x.fillna(x.median(skipna=True))
        cols.append(zscore(x).values)
        names.append(c)
    return cols, names


def run_moderation_models(subj: pd.DataFrame, covariates: List[str], out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log("运行固定候选治疗调节模型：无协变量 + 协变量敏感性")
    base_rows = []
    cov_rows = []
    slope_rows = []
    plot_rows = []

    for spec in CORE_EDGE_SPECS:
        edge = spec["edge_short"]
        z_col = f"{edge}__baseline_z"
        base_col = f"{edge}__pre"
        if z_col not in subj.columns:
            continue

        d = subj.dropna(subset=["pcl_improvement", z_col, "group_tms_psy"]).copy().reset_index(drop=True)
        if len(d) < 30 or d["group_tms_psy"].nunique() < 2:
            continue

        y = num(d["pcl_improvement"]).values
        t = (d["group_tms_psy"] == "TMS").astype(float).values
        x = num(d[z_col]).values
        interaction = t * x

        # 无协变量模型
        X = np.column_stack([np.ones(len(d)), t, x, interaction])
        names = ["intercept", "treatment_TMS", "baseline_FC_z", "TMS_x_baseline_FC_z"]
        fit = fit_ols(X, y)

        if fit is not None:
            idx = names.index("TMS_x_baseline_FC_z")
            beta_inter = fit["beta"][idx]
            row = {
                "edge_short": edge,
                "system": spec["system"],
                "system_cn": spec["system_cn"],
                "expected_direction_from_stability": spec["expected_direction_from_stability"],
                "model": "no_covariates",
                "n": fit["n"],
                "n_TMS": int((d["group_tms_psy"] == "TMS").sum()),
                "n_PSY": int((d["group_tms_psy"] == "PSY").sum()),
                "beta_interaction_TMS_minus_PSY_per1SD": safe_float(beta_inter),
                "se_interaction": safe_float(fit["se"][idx]),
                "t_interaction": safe_float(fit["t"][idx]),
                "p_interaction": safe_float(fit["p"][idx]),
                "r2": safe_float(fit["r2"]),
                "direction_from_model": "TMS_favoring_positive_beta" if beta_inter > 0 else "PSY_favoring_negative_beta",
                "interpretation": "beta>0: baseline FC越高越偏向TMS获益；beta<0: baseline FC越高越偏向PSY获益。",
                "rationale": spec["rationale"],
            }
            base_rows.append(row)

        # 协变量模型
        cov_cols, cov_names = get_covariate_matrix(d, covariates)
        if cov_cols:
            X_cov = np.column_stack([np.ones(len(d)), t, x, interaction] + cov_cols)
            names_cov = ["intercept", "treatment_TMS", "baseline_FC_z", "TMS_x_baseline_FC_z"] + cov_names
            fit_cov = fit_ols(X_cov, y)
            if fit_cov is not None:
                idx = names_cov.index("TMS_x_baseline_FC_z")
                beta_inter = fit_cov["beta"][idx]
                cov_rows.append({
                    "edge_short": edge,
                    "system": spec["system"],
                    "system_cn": spec["system_cn"],
                    "expected_direction_from_stability": spec["expected_direction_from_stability"],
                    "model": "covariate_adjusted",
                    "covariates_used": ";".join(cov_names),
                    "n": fit_cov["n"],
                    "n_TMS": int((d["group_tms_psy"] == "TMS").sum()),
                    "n_PSY": int((d["group_tms_psy"] == "PSY").sum()),
                    "beta_interaction_TMS_minus_PSY_per1SD": safe_float(beta_inter),
                    "se_interaction": safe_float(fit_cov["se"][idx]),
                    "t_interaction": safe_float(fit_cov["t"][idx]),
                    "p_interaction": safe_float(fit_cov["p"][idx]),
                    "r2": safe_float(fit_cov["r2"]),
                    "direction_from_model": "TMS_favoring_positive_beta" if beta_inter > 0 else "PSY_favoring_negative_beta",
                    "interpretation": "beta>0: baseline FC越高越偏向TMS获益；beta<0: baseline FC越高越偏向PSY获益。",
                    "rationale": spec["rationale"],
                })

        # simple slopes by treatment
        for g in ["PSY", "TMS"]:
            dg = d[d["group_tms_psy"] == g].copy().reset_index(drop=True)
            if len(dg) < 8:
                continue
            yg = num(dg["pcl_improvement"]).values
            xg = num(dg[z_col]).values
            Xg = np.column_stack([np.ones(len(dg)), xg])
            fitg = fit_ols(Xg, yg)
            r_p, p_p, n_p = pearson_pair(xg, yg)
            r_s, p_s, n_s = spearman_pair(xg, yg)
            if fitg is not None:
                slope_rows.append({
                    "edge_short": edge,
                    "group": g,
                    "n": fitg["n"],
                    "slope_baselineFC_to_improvement": safe_float(fitg["beta"][1]),
                    "se_slope": safe_float(fitg["se"][1]),
                    "t_slope": safe_float(fitg["t"][1]),
                    "p_slope": safe_float(fitg["p"][1]),
                    "pearson_r": safe_float(r_p),
                    "pearson_p": safe_float(p_p),
                    "spearman_r": safe_float(r_s),
                    "spearman_p": safe_float(p_s),
                    "expected_direction_from_stability": spec["expected_direction_from_stability"],
                })

        # plot data
        for _, row in d.iterrows():
            plot_rows.append({
                "edge_short": edge,
                "system": spec["system"],
                "system_cn": spec["system_cn"],
                "subject_key": row.get("subject_key", ""),
                "group_tms_psy": row["group_tms_psy"],
                "treatment_TMS": int(row["group_tms_psy"] == "TMS"),
                "baseline_FC": safe_float(row.get(base_col, np.nan)),
                "baseline_FC_z": safe_float(row.get(z_col, np.nan)),
                "pcl_improvement": safe_float(row.get("pcl_improvement", np.nan)),
                "expected_direction_from_stability": spec["expected_direction_from_stability"],
            })

    base_df = pd.DataFrame(base_rows)
    if len(base_df):
        base_df["q_interaction_fdr_10edges"] = fdr_bh(base_df["p_interaction"])
        base_df = base_df.sort_values(["q_interaction_fdr_10edges", "p_interaction", "edge_short"], ascending=[True, True, True])

    cov_df = pd.DataFrame(cov_rows)
    if len(cov_df):
        cov_df["q_interaction_fdr_10edges"] = fdr_bh(cov_df["p_interaction"])
        cov_df = cov_df.sort_values(["q_interaction_fdr_10edges", "p_interaction", "edge_short"], ascending=[True, True, True])

    slope_df = pd.DataFrame(slope_rows)
    if len(slope_df):
        slope_df["q_slope_fdr"] = fdr_bh(slope_df["p_slope"])

    plot_df = pd.DataFrame(plot_rows)

    write_csv(base_df, out_dir / "10_fixed_candidate_moderation_no_covariates.csv")
    write_csv(cov_df, out_dir / "11_fixed_candidate_moderation_covariates.csv")
    write_csv(slope_df, out_dir / "12_simple_slopes_by_treatment.csv")
    write_csv(plot_df, out_dir / "30_interaction_plot_data_long.csv")

    return base_df, cov_df, slope_df, plot_df


# ============================================================
# 6. 高低改善者可塑性分析
# ============================================================

def add_improvement_splits(d: pd.DataFrame, improvement_col: str = "pcl_improvement") -> pd.DataFrame:
    out = d.copy()
    imp = num(out[improvement_col])

    # median split within provided dataframe
    med = imp.median(skipna=True)
    out["split_median"] = np.where(imp >= med, "high", "low")
    out.loc[imp.isna(), "split_median"] = np.nan

    # tertile split within provided dataframe
    q33, q67 = imp.quantile([1/3, 2/3])
    out["split_tertile"] = np.where(imp >= q67, "high", np.where(imp <= q33, "low", "middle"))
    out.loc[imp.isna(), "split_tertile"] = np.nan

    # clinical cutoff used in previous scripts: high >=20, low <=10
    out["split_08_cutoff"] = np.where(imp >= 20, "high", np.where(imp <= 10, "low", "middle"))
    out.loc[imp.isna(), "split_08_cutoff"] = np.nan
    return out


def high_low_test(d: pd.DataFrame, edge: str, group_label: str, split_col: str) -> Dict[str, Any]:
    delta_col = f"{edge}__delta"
    if group_label == "ACTIVE":
        dg = d[d["group_tms_psy"].isin(["TMS", "PSY"])].copy()
    else:
        dg = d[d["group_tms_psy"] == group_label].copy()
    dg = add_improvement_splits(dg)

    hl = dg[dg[split_col].isin(["high", "low"])].dropna(subset=[delta_col, "pcl_improvement"]).copy()
    high = num(hl.loc[hl[split_col] == "high", delta_col]).dropna()
    low = num(hl.loc[hl[split_col] == "low", delta_col]).dropna()

    res = {
        "edge_short": edge,
        "group": group_label,
        "split": split_col.replace("split_", ""),
        "n_total_group_with_delta": int(dg[delta_col].notna().sum()) if delta_col in dg.columns else 0,
        "n_high": int(len(high)),
        "n_low": int(len(low)),
        "high_mean_delta": safe_float(high.mean()),
        "low_mean_delta": safe_float(low.mean()),
        "mean_diff_high_minus_low": safe_float(high.mean() - low.mean()) if len(high) and len(low) else np.nan,
        "high_median_delta": safe_float(high.median()),
        "low_median_delta": safe_float(low.median()),
        "hedges_g_high_minus_low": hedges_g(high, low),
    }

    if len(high) >= 3 and len(low) >= 3:
        try:
            res["welch_p"] = safe_float(stats.ttest_ind(high, low, equal_var=False).pvalue)
        except Exception:
            res["welch_p"] = np.nan
        try:
            res["mannwhitney_p"] = safe_float(stats.mannwhitneyu(high, low, alternative="two-sided").pvalue)
        except Exception:
            res["mannwhitney_p"] = np.nan
    else:
        res["welch_p"] = np.nan
        res["mannwhitney_p"] = np.nan

    return res


def run_plasticity_analysis(subj: pd.DataFrame, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log("运行疗效相关可塑性分析：高低改善者delta FC + 连续相关")
    test_rows = []
    corr_rows = []
    plot_rows = []

    for spec in CORE_EDGE_SPECS:
        edge = spec["edge_short"]
        delta_col = f"{edge}__delta"
        if delta_col not in subj.columns:
            continue

        # high-low tests
        for group in ["PSY", "TMS", "ACTIVE"]:
            for split_col in ["split_median", "split_tertile", "split_08_cutoff"]:
                res = high_low_test(subj, edge, group, split_col)
                res.update({
                    "system": spec["system"],
                    "system_cn": spec["system_cn"],
                    "expected_direction_from_stability": spec["expected_direction_from_stability"],
                    "rationale": spec["rationale"],
                })
                test_rows.append(res)

        # continuous correlations
        for group in ["PSY", "TMS", "ACTIVE"]:
            if group == "ACTIVE":
                dg = subj[subj["group_tms_psy"].isin(["TMS", "PSY"])].copy()
            else:
                dg = subj[subj["group_tms_psy"] == group].copy()
            r_p, p_p, n_p = pearson_pair(dg[delta_col], dg["pcl_improvement"])
            r_s, p_s, n_s = spearman_pair(dg[delta_col], dg["pcl_improvement"])
            corr_rows.append({
                "edge_short": edge,
                "group": group,
                "n": n_p,
                "pearson_r_delta_vs_improvement": safe_float(r_p),
                "pearson_p": safe_float(p_p),
                "spearman_r_delta_vs_improvement": safe_float(r_s),
                "spearman_p": safe_float(p_s),
                "system": spec["system"],
                "system_cn": spec["system_cn"],
                "expected_direction_from_stability": spec["expected_direction_from_stability"],
            })

        # plot data
        dtmp = add_improvement_splits(subj.copy())
        for _, row in dtmp.iterrows():
            plot_rows.append({
                "edge_short": edge,
                "system": spec["system"],
                "system_cn": spec["system_cn"],
                "subject_key": row.get("subject_key", ""),
                "group_tms_psy": row.get("group_tms_psy", ""),
                "pcl_improvement": safe_float(row.get("pcl_improvement", np.nan)),
                "delta_FC": safe_float(row.get(delta_col, np.nan)),
                "split_median": row.get("split_median", ""),
                "split_tertile": row.get("split_tertile", ""),
                "split_08_cutoff": row.get("split_08_cutoff", ""),
                "expected_direction_from_stability": spec["expected_direction_from_stability"],
            })

    test_df = pd.DataFrame(test_rows)
    if len(test_df):
        test_df["welch_q_fdr_all_tests"] = fdr_bh(test_df["welch_p"])
        test_df["mannwhitney_q_fdr_all_tests"] = fdr_bh(test_df["mannwhitney_p"])

    corr_df = pd.DataFrame(corr_rows)
    if len(corr_df):
        corr_df["pearson_q_fdr_all_tests"] = fdr_bh(corr_df["pearson_p"])
        corr_df["spearman_q_fdr_all_tests"] = fdr_bh(corr_df["spearman_p"])

    plot_df = pd.DataFrame(plot_rows)

    # best summary per edge: smallest p across high-low/correlation
    summary_rows = []
    for spec in CORE_EDGE_SPECS:
        edge = spec["edge_short"]
        sub_test = test_df[test_df["edge_short"] == edge].copy() if len(test_df) else pd.DataFrame()
        sub_corr = corr_df[corr_df["edge_short"] == edge].copy() if len(corr_df) else pd.DataFrame()

        min_highlow_p = np.nan
        best_highlow_desc = ""
        if len(sub_test) and sub_test["welch_p"].notna().any():
            idx = sub_test["welch_p"].astype(float).idxmin()
            r = sub_test.loc[idx]
            min_highlow_p = safe_float(r["welch_p"])
            best_highlow_desc = f"{r['group']} / {r['split']} / diff={safe_float(r['mean_diff_high_minus_low']):.4f}"

        min_corr_p = np.nan
        best_corr_desc = ""
        if len(sub_corr) and sub_corr["pearson_p"].notna().any():
            idx = sub_corr["pearson_p"].astype(float).idxmin()
            r = sub_corr.loc[idx]
            min_corr_p = safe_float(r["pearson_p"])
            best_corr_desc = f"{r['group']} / r={safe_float(r['pearson_r_delta_vs_improvement']):.4f}"

        min_any = np.nanmin([p for p in [min_highlow_p, min_corr_p] if np.isfinite(p)]) if any(np.isfinite(p) for p in [min_highlow_p, min_corr_p]) else np.nan
        summary_rows.append({
            "edge_short": edge,
            "system": spec["system"],
            "system_cn": spec["system_cn"],
            "expected_direction_from_stability": spec["expected_direction_from_stability"],
            "min_highlow_welch_p": min_highlow_p,
            "best_highlow_desc": best_highlow_desc,
            "min_delta_improvement_corr_p": min_corr_p,
            "best_corr_desc": best_corr_desc,
            "min_any_plasticity_p": min_any,
            "plasticity_support_label": "strong_p_lt_0.01" if np.isfinite(min_any) and min_any < 0.01 else ("nominal_p_lt_0.05" if np.isfinite(min_any) and min_any < 0.05 else "weak_or_null"),
            "rationale": spec["rationale"],
        })

    summary_df = pd.DataFrame(summary_rows)
    if len(summary_df):
        summary_df["min_any_plasticity_q_fdr_10edges"] = fdr_bh(summary_df["min_any_plasticity_p"])

    write_csv(test_df, out_dir / "20_plasticity_high_low_tests.csv")
    write_csv(corr_df, out_dir / "21_plasticity_delta_improvement_correlations.csv")
    write_csv(summary_df, out_dir / "22_plasticity_summary_best_per_edge.csv")
    write_csv(plot_df, out_dir / "31_plasticity_plot_data_long.csv")
    return test_df, corr_df, summary_df, plot_df


# ============================================================
# 7. 最终综合决策表与报告
# ============================================================

def build_final_decision_table(base_df: pd.DataFrame, cov_df: pd.DataFrame, plasticity_summary: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for spec in CORE_EDGE_SPECS:
        edge = spec["edge_short"]
        b = base_df[base_df["edge_short"] == edge].iloc[0].to_dict() if len(base_df[base_df["edge_short"] == edge]) else {}
        c = cov_df[cov_df["edge_short"] == edge].iloc[0].to_dict() if len(cov_df[cov_df["edge_short"] == edge]) else {}
        p = plasticity_summary[plasticity_summary["edge_short"] == edge].iloc[0].to_dict() if len(plasticity_summary[plasticity_summary["edge_short"] == edge]) else {}

        no_cov_p = safe_float(b.get("p_interaction", np.nan))
        no_cov_q = safe_float(b.get("q_interaction_fdr_10edges", np.nan))
        cov_p = safe_float(c.get("p_interaction", np.nan))
        cov_q = safe_float(c.get("q_interaction_fdr_10edges", np.nan))
        plast_p = safe_float(p.get("min_any_plasticity_p", np.nan))
        plast_q = safe_float(p.get("min_any_plasticity_q_fdr_10edges", np.nan))

        moderation_label = "weak_or_null"
        if np.isfinite(no_cov_q) and no_cov_q < 0.05:
            moderation_label = "fixed_candidate_FDR_q_lt_0.05"
        elif np.isfinite(no_cov_p) and no_cov_p < 0.05:
            moderation_label = "nominal_fixed_candidate_p_lt_0.05"
        elif np.isfinite(no_cov_p) and no_cov_p < 0.10:
            moderation_label = "trend_p_lt_0.10"

        cov_label = "not_available_or_weak"
        if np.isfinite(cov_q) and cov_q < 0.05:
            cov_label = "covariate_adjusted_FDR_q_lt_0.05"
        elif np.isfinite(cov_p) and cov_p < 0.05:
            cov_label = "covariate_adjusted_nominal_p_lt_0.05"
        elif np.isfinite(cov_p) and cov_p < 0.10:
            cov_label = "covariate_adjusted_trend_p_lt_0.10"

        plast_label = p.get("plasticity_support_label", "weak_or_null")

        # 建议用途
        if moderation_label.startswith("fixed_candidate_FDR") or (
            moderation_label.startswith("nominal") and plast_label in ["strong_p_lt_0.01", "nominal_p_lt_0.05"]
        ):
            recommended_use = "main_text_candidate"
        elif moderation_label != "weak_or_null" or plast_label in ["strong_p_lt_0.01", "nominal_p_lt_0.05"]:
            recommended_use = "supplement_or_mechanistic_support"
        else:
            recommended_use = "report_in_full_table_only"

        rows.append({
            "edge_short": edge,
            "system": spec["system"],
            "system_cn": spec["system_cn"],
            "expected_direction_from_stability": spec["expected_direction_from_stability"],
            "no_cov_beta_interaction": safe_float(b.get("beta_interaction_TMS_minus_PSY_per1SD", np.nan)),
            "no_cov_p_interaction": no_cov_p,
            "no_cov_q_interaction_10edges": no_cov_q,
            "no_cov_direction_from_model": b.get("direction_from_model", ""),
            "cov_beta_interaction": safe_float(c.get("beta_interaction_TMS_minus_PSY_per1SD", np.nan)),
            "cov_p_interaction": cov_p,
            "cov_q_interaction_10edges": cov_q,
            "cov_direction_from_model": c.get("direction_from_model", ""),
            "min_plasticity_p": plast_p,
            "min_plasticity_q_10edges": plast_q,
            "best_highlow_desc": p.get("best_highlow_desc", ""),
            "best_corr_desc": p.get("best_corr_desc", ""),
            "moderation_support_label": moderation_label,
            "covariate_support_label": cov_label,
            "plasticity_support_label": plast_label,
            "recommended_use": recommended_use,
            "rationale_from_selection_stage": spec["rationale"],
        })

    final_df = pd.DataFrame(rows)
    # 排序：正式调节支持 > 可塑性支持 > p值
    label_rank = {
        "main_text_candidate": 0,
        "supplement_or_mechanistic_support": 1,
        "report_in_full_table_only": 2,
    }
    final_df["recommended_use_rank"] = final_df["recommended_use"].map(label_rank).fillna(9)
    final_df = final_df.sort_values(["recommended_use_rank", "no_cov_q_interaction_10edges", "min_plasticity_p", "edge_short"], ascending=[True, True, True, True])
    write_csv(final_df.drop(columns=["recommended_use_rank"]), out_dir / "40_final_core_edge_decision_table.csv")
    return final_df.drop(columns=["recommended_use_rank"])


def write_report(audit: Dict[str, Any], base_df: pd.DataFrame, cov_df: pd.DataFrame, slope_df: pd.DataFrame,
                 plasticity_summary: pd.DataFrame, final_df: pd.DataFrame, out_dir: Path):
    lines = []
    lines.append("40号：稳定性选择候选边_正式固定候选治疗调节和可塑性分析报告")
    lines.append("=" * 90)
    lines.append(f"生成时间: {now()}")
    lines.append("")
    lines.append("一、输入与样本")
    lines.append("-" * 90)
    lines.append(f"输入行数: {audit.get('n_rows_input')}")
    lines.append(f"输入列数: {audit.get('n_cols_input')}")
    lines.append(f"组别列: {audit.get('group_col')}")
    lines.append(f"被试列: {audit.get('subject_col')}")
    lines.append(f"时间点列: {audit.get('timepoint_col')}")
    lines.append(f"结局变量: {audit.get('outcome_col')}")
    lines.append(f"TMS vs PSY 被试数: {audit.get('n_tms_psy_subjects')}")
    lines.append(f"TMS: {audit.get('n_TMS')}")
    lines.append(f"PSY: {audit.get('n_PSY')}")
    lines.append(f"候选边请求数: {audit.get('n_candidate_edges_requested')}")
    lines.append(f"候选边成功匹配数: {audit.get('n_candidate_edges_matched')}")
    lines.append(f"协变量请求: {audit.get('covariates_requested')}")
    lines.append(f"协变量实际使用: {audit.get('covariates_used')}")
    lines.append("")
    lines.append("二、核心模型")
    lines.append("-" * 90)
    lines.append("PCL improvement = treatment + baseline FC + treatment × baseline FC")
    lines.append("treatment = 1 for TMS, 0 for PSY")
    lines.append("beta_interaction > 0 表示基线FC越高越偏向TMS获益；beta_interaction < 0 表示越偏向PSY获益。")
    lines.append("")
    lines.append("三、无协变量固定候选调节结果 Top")
    lines.append("-" * 90)
    if len(base_df):
        show = base_df.sort_values(["q_interaction_fdr_10edges", "p_interaction"]).head(10)
        lines.append(show[[
            "edge_short", "beta_interaction_TMS_minus_PSY_per1SD", "p_interaction", "q_interaction_fdr_10edges", "direction_from_model"
        ]].to_string(index=False))
    else:
        lines.append("无可用无协变量结果。")
    lines.append("")
    lines.append("四、协变量敏感性结果 Top")
    lines.append("-" * 90)
    if len(cov_df):
        show = cov_df.sort_values(["q_interaction_fdr_10edges", "p_interaction"]).head(10)
        lines.append(show[[
            "edge_short", "beta_interaction_TMS_minus_PSY_per1SD", "p_interaction", "q_interaction_fdr_10edges", "direction_from_model", "covariates_used"
        ]].to_string(index=False))
    else:
        lines.append("无可用协变量模型结果，可能是协变量缺失或未满足纳入条件。")
    lines.append("")
    lines.append("五、可塑性支持摘要")
    lines.append("-" * 90)
    if len(plasticity_summary):
        show = plasticity_summary.sort_values(["min_any_plasticity_p"]).head(10)
        lines.append(show[[
            "edge_short", "min_any_plasticity_p", "min_any_plasticity_q_fdr_10edges", "plasticity_support_label", "best_highlow_desc", "best_corr_desc"
        ]].to_string(index=False))
    else:
        lines.append("无可塑性结果。")
    lines.append("")
    lines.append("六、最终建议用途")
    lines.append("-" * 90)
    if len(final_df):
        show = final_df[[
            "edge_short", "expected_direction_from_stability", "moderation_support_label", "covariate_support_label", "plasticity_support_label", "recommended_use"
        ]]
        lines.append(show.to_string(index=False))
    else:
        lines.append("无最终决策表。")
    lines.append("")
    lines.append("七、论文表述提醒")
    lines.append("-" * 90)
    lines.append("1. 这一步是固定候选效应刻画，不是独立外部验证。")
    lines.append("2. 候选边来自前一步全脑稳定性选择；本脚本不再重新筛边。")
    lines.append("3. 如果10条边内FDR显著，可以写为固定候选集内校正显著；不能写为全脑FDR显著。")
    lines.append("4. 高低改善者可塑性分析用于机制支持，不应夸大为独立验证。")
    lines.append("")
    report_path = out_dir / "99_fixed_candidate_moderation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# 8. 可选绘图
# ============================================================

def make_basic_plots(interaction_data: pd.DataFrame, plasticity_data: pd.DataFrame, out_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        log("未安装 matplotlib，跳过绘图。")
        return

    fig_dir = ensure_dir(out_dir / "figures")
    # 只画前4条，避免图过大
    edges_to_plot = [spec["edge_short"] for spec in CORE_EDGE_SPECS[:4]]

    for edge in edges_to_plot:
        d = interaction_data[interaction_data["edge_short"] == edge].dropna(subset=["baseline_FC_z", "pcl_improvement"])
        if len(d) < 10:
            continue
        plt.figure(figsize=(6, 4))
        for group in ["PSY", "TMS"]:
            dg = d[d["group_tms_psy"] == group]
            if len(dg) < 3:
                continue
            plt.scatter(dg["baseline_FC_z"], dg["pcl_improvement"], label=group, alpha=0.75)
            try:
                coef = np.polyfit(dg["baseline_FC_z"], dg["pcl_improvement"], deg=1)
                xs = np.linspace(dg["baseline_FC_z"].min(), dg["baseline_FC_z"].max(), 50)
                ys = coef[0] * xs + coef[1]
                plt.plot(xs, ys)
            except Exception:
                pass
        plt.xlabel("Baseline FC (z)")
        plt.ylabel("PCL improvement")
        plt.title(f"Baseline FC × treatment: {edge}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"interaction_{safe_filename(edge)}.png", dpi=200)
        plt.close()

    for edge in edges_to_plot:
        d = plasticity_data[plasticity_data["edge_short"] == edge].dropna(subset=["delta_FC", "split_median"])
        if len(d) < 10:
            continue
        plt.figure(figsize=(6, 4))
        labels = []
        values = []
        for group in ["PSY", "TMS"]:
            for split in ["low", "high"]:
                vals = d[(d["group_tms_psy"] == group) & (d["split_median"] == split)]["delta_FC"].dropna().values
                if len(vals):
                    labels.append(f"{group}-{split}")
                    values.append(vals)
        if values:
            plt.boxplot(values, labels=labels, showfliers=True)
            plt.axhline(0, linestyle="--", linewidth=1)
            plt.ylabel("Delta FC (post - pre)")
            plt.title(f"Responder-linked plasticity: {edge}")
            plt.tight_layout()
            plt.savefig(fig_dir / f"plasticity_{safe_filename(edge)}.png", dpi=200)
            plt.close()


def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(s).replace("–", "_"))


# ============================================================
# 9. 主函数
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", default=DEFAULT_INPUT, help="纵向FC+临床宽表")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="输出目录")
    ap.add_argument("--canonical_edges", default=DEFAULT_CANONICAL_EDGES, help="39_v3生成的固定候选边value_column谱系表")
    ap.add_argument("--v2_subject_table", default=DEFAULT_V2_SUBJECT_TABLE, help="40_v2被试级表；用于pre_meanFD等价恢复审计")
    ap.add_argument("--covariates", default="PCL_pre,age,sex,pre_meanFD", help="协变量逗号分隔；不可用者会自动跳过")
    ap.add_argument("--make_plots", action="store_true", help="是否生成基础交互图和可塑性图")
    return ap.parse_args()


def main():
    global CORE_EDGE_SPECS
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    covariates_requested = [x.strip() for x in str(args.covariates).split(",") if x.strip()]

    log("=" * 100)
    log("40号：稳定性选择候选边_正式固定候选治疗调节和可塑性分析 启动")
    log(f"input_file = {args.input_file}")
    log(f"out_dir = {out_dir}")
    log(f"canonical_edges = {args.canonical_edges}")
    log(f"v2_subject_table = {args.v2_subject_table}")
    log(f"covariates = {covariates_requested}")
    log("=" * 100)

    t0 = time.time()
    df = read_table(args.input_file)
    log(f"读取完成：rows={len(df)}, cols={df.shape[1]}")

    CORE_EDGE_SPECS = load_core_edge_specs_from_canonical(args.canonical_edges, available_columns=set(df.columns))
    log("已从39_v3 canonical表读取固定候选边；使用value_column取值，输出修正BNA标签")

    subj, audit = build_subject_table(df, out_dir, covariates_requested, args.v2_subject_table)
    audit["canonical_edges_file"] = str(Path(args.canonical_edges).resolve())
    safe_json_dump(audit, out_dir / "00_run_audit.json")

    base_df, cov_df, slope_df, interaction_plot_data = run_moderation_models(subj, audit["covariates_used"], out_dir)
    highlow_df, corr_df, plasticity_summary, plasticity_plot_data = run_plasticity_analysis(subj, out_dir)

    final_df = build_final_decision_table(base_df, cov_df, plasticity_summary, out_dir)

    if args.make_plots:
        log("生成基础可视化图...")
        make_basic_plots(interaction_plot_data, plasticity_plot_data, out_dir)

    audit["runtime_minutes"] = round((time.time() - t0) / 60, 3)
    audit["finish_time"] = now()
    safe_json_dump(audit, out_dir / "00_run_audit.json")

    write_report(audit, base_df, cov_df, slope_df, plasticity_summary, final_df, out_dir)

    log(f"完成。runtime={audit['runtime_minutes']} min")
    log(f"结果目录：{out_dir}")


if __name__ == "__main__":
    main()
