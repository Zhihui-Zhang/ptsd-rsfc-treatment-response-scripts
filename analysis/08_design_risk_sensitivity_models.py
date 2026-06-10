# -*- coding: utf-8 -*-
"""
46_v6_value_column_only_PSY_TMS非头对头设计风险审计_敏感性模型_无旧标签正式版.py

目的：
1) 继续完成 46 号原任务：审计 PSY vs TMS 非头对头设计风险、site/trial 混杂、候选边 treatment × baseline FC 敏感性模型。
2) 修复“半个大脑/旧标签传播”风险：正式输出只使用 EDGE_01–EDGE_10 + value_column【取值列】身份。
3) 保持 v5 数值等价：模型、协变量、IPW、FDR、site 推断逻辑均不改，只改安全输入选择和输出身份。

推荐运行：
    python -u 46_v6_value_column_only_PSY_TMS非头对头设计风险审计_敏感性模型_无旧标签正式版.py

可手动指定：
    python -u 46_v6_value_column_only_PSY_TMS非头对头设计风险审计_敏感性模型_无旧标签正式版.py --input_file "43_v4_2结果\\03_subject_level_core_edges_TMS_PSY_WL_v4_2_value_column_only.csv"

正式输入优先级：
    43_v4_2_value_column_only_...\\03_subject_level_core_edges_TMS_PSY_WL_v4_2_value_column_only.csv

注意：
    - 本脚本不输出旧 Brainnetome 解剖标签。
    - 如需解剖名或 BrainNet 图，必须使用后续单独审计通过的 ROI 映射。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
except Exception as e:  # pragma: no cover
    sm = None
    STATSMODELS_IMPORT_ERROR = e
else:
    STATSMODELS_IMPORT_ERROR = None

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
except Exception:
    LogisticRegression = None
    roc_auc_score = None

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None


SCRIPT_NAME = "46_v6_value_column_only_PSY_TMS非头对头设计风险审计_敏感性模型_无旧标签正式版"
OUT_DIR_NAME = "46_v6_value_column_only_PSY_TMS非头对头设计风险审计_敏感性模型结果_无旧标签正式版"

EDGE_VALUE_COLUMNS = {
    "EDGE_01": "unknown_roi_18__unknown_roi_24",
    "EDGE_02": "unknown_roi_65__unknown_roi_85",
    "EDGE_03": "unknown_roi_62__unknown_roi_105",
    "EDGE_04": "unknown_roi_64__unknown_roi_85",
    "EDGE_05": "unknown_roi_11__unknown_roi_67",
    "EDGE_06": "unknown_roi_56__unknown_roi_84",
    "EDGE_07": "unknown_roi_46__unknown_roi_95",
    "EDGE_08": "unknown_roi_89__unknown_roi_90",
    "EDGE_09": "unknown_roi_8__unknown_roi_92",
    "EDGE_10": "unknown_roi_23__unknown_roi_44",
}

LEGACY_LABEL_REGEX = {
    "A45r": r"(?<![A-Za-z0-9])A45r(?![A-Za-z0-9])",
    "A11m": r"(?<![A-Za-z0-9])A11m(?![A-Za-z0-9])",
    "A5l": r"(?<![A-Za-z0-9])A5l(?![A-Za-z0-9])",
    "vId": r"(?<![A-Za-z0-9])vId(?![A-Za-z0-9])",
    "vIg": r"(?<![A-Za-z0-9])vIg(?![A-Za-z0-9])",
    "cpSTS": r"(?<![A-Za-z0-9])cpSTS(?![A-Za-z0-9])",
    "lsOccG": r"(?<![A-Za-z0-9])lsOccG(?![A-Za-z0-9])",
    "A7c": r"(?<![A-Za-z0-9])A7c(?![A-Za-z0-9])",
    "A9/46v": r"A9/46v",
    "A7ip": r"(?<![A-Za-z0-9])A7ip(?![A-Za-z0-9])",
    "A35/36c": r"A35/36c",
    "dIa": r"(?<![A-Za-z0-9])dIa(?![A-Za-z0-9])",
    "A37elv": r"(?<![A-Za-z0-9])A37elv(?![A-Za-z0-9])",
    "cLinG": r"(?<![A-Za-z0-9])cLinG(?![A-Za-z0-9])",
    "A24rv": r"(?<![A-Za-z0-9])A24rv(?![A-Za-z0-9])",
    "A32p": r"(?<![A-Za-z0-9])A32p(?![A-Za-z0-9])",
    "A9/46d": r"A9/46d",
    "A24cd": r"(?<![A-Za-z0-9])A24cd(?![A-Za-z0-9])",
    "A11l": r"(?<![A-Za-z0-9])A11l(?![A-Za-z0-9])",
    "aSTS": r"(?<![A-Za-z0-9])aSTS(?![A-Za-z0-9])",
    "A37dl": r"(?<![A-Za-z0-9])A37dl(?![A-Za-z0-9])",
    "OPC": r"(?<![A-Za-z0-9])OPC(?![A-Za-z0-9])",
    "A12/47o": r"A12/47o",
}

COMPRESSED_LEGACY_TERMS = [
    "a45ra11m", "a5lvidvig", "cpstslsoccg", "a7cvidvig", "a946va7ip",
    "a3536cdia", "a37elvcling", "a24rva32p", "a946da24cd", "a11lasts",
    "a37dlopc", "a1247oa946v",
]


@dataclass
class ModelResult:
    edge: str
    value_column: str
    feature_col: str
    model_type: str
    n: int
    n_psy: int
    n_tms: int
    beta_interaction: float
    se_interaction: float
    t_or_z: float
    p_interaction: float
    ci95_low: float
    ci95_high: float
    r2: float
    covariates_used: str
    estimable: bool
    warning: str


def log(msg: str) -> None:
    print(msg, flush=True)


def safe_read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def normalize_name(x: object) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", str(x).lower())


def find_project_root() -> Path:
    cwd = Path.cwd()
    candidates = [cwd] + list(cwd.parents)
    for c in candidates:
        if "第四步分析" in str(c.name):
            return c
    return cwd


def iter_search_roots() -> List[Path]:
    roots = []
    cwd = Path.cwd()
    roots.extend([cwd, cwd.parent, cwd.parent.parent if cwd.parent else cwd])
    roots.extend([
        Path(r"E:\E_zhangzhihui\从yv那边提取\脚本\第四步分析：探索"),
        Path(r"D:\自科＋脑中心论文选题\PAI选题\工作站传输\第四步分析"),
        Path(r"D:\自科＋脑中心论文选题\PAI选题\工作站传输\第四步分析-codex"),
    ])
    out = []
    seen = set()
    for r in roots:
        try:
            key = str(r.resolve()) if r.exists() else str(r)
        except Exception:
            key = str(r)
        if key not in seen:
            out.append(r)
            seen.add(key)
    return out


def auto_find_safe_input_file() -> Path:
    """优先寻找 43_v4_2 value_column-only 被试级表，避免误读旧 43/40 结果。"""
    candidates: List[Tuple[int, Path, str]] = []
    target_name_tokens = ["03_subject_level_core_edges", "v4_2", "value_column_only"]
    for root in iter_search_roots():
        if not root.exists():
            continue
        try:
            files = list(root.rglob("*.csv"))
        except Exception:
            continue
        for p in files:
            name = p.name.lower()
            path_str = str(p).lower()
            score = 0
            reason = []
            if all(t in name for t in target_name_tokens):
                score += 200; reason.append("文件名精确匹配43_v4_2 value_column-only subject-level core edges")
            if "43_v4_2" in path_str:
                score += 100; reason.append("路径包含43_v4_2")
            if "value_column_only" in path_str:
                score += 80; reason.append("路径/文件名含value_column_only")
            if "subject_level_core_edges" in name:
                score += 50; reason.append("subject-level core edges")
            if "43_固定候选边" in path_str and "v4_2" not in path_str:
                score -= 120; reason.append("旧43结果扣分")
            if "40_稳定性选择" in path_str:
                score -= 120; reason.append("旧40结果扣分")
            if score <= 0:
                continue
            try:
                header = list(pd.read_csv(p, encoding="utf-8-sig", nrows=0).columns)
            except Exception:
                try:
                    header = list(pd.read_csv(p, nrows=0).columns)
                except Exception:
                    continue
            safe_cols = [f"{eid}__pre" for eid in EDGE_VALUE_COLUMNS]
            value_cols = [f"{eid}__value_column" for eid in EDGE_VALUE_COLUMNS]
            if all(c in header for c in safe_cols) and all(c in header for c in value_cols):
                score += 200; reason.append("10条EDGE pre/value_column列完整")
            else:
                score -= 200; reason.append("EDGE/value_column列不完整")
            candidates.append((score, p, "; ".join(reason)))
    if not candidates:
        raise FileNotFoundError("未找到 43_v4_2 value_column-only 被试级核心边表。请手动使用 --input_file 指定。")
    candidates.sort(key=lambda x: x[0], reverse=True)
    log("[自动搜索] 安全输入表候选前5个：")
    for score, p, reason in candidates[:5]:
        log(f"  score={score:>4} | {p}")
        log(f"        {reason}")
    return candidates[0][1]


def standardize_treatment_value(x: object, psy_values: Sequence[str], tms_values: Sequence[str]) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip().upper().replace(" ", "")
    psy_set = {v.upper().replace(" ", "") for v in psy_values if v}
    tms_set = {v.upper().replace(" ", "") for v in tms_values if v}
    if s in tms_set or "TMS" in s or "RTMS" in s:
        return "TMS"
    if s in psy_set or any(token in s for token in ["PSY", "PSYCH", "PSYCHOTHERAPY", "ACT", "MIN", "CBT", "PE", "CPT"]):
        return "PSY"
    if any(token in s for token in ["WL", "WAIT", "HC", "CONTROL", "CTRL", "健康", "等待"]):
        return None
    return None


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    sd = x.std(skipna=True, ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (x - x.mean(skipna=True)) / sd


def clean_categorical(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def make_dummies(series: pd.Series, prefix: str) -> pd.DataFrame:
    s = clean_categorical(series)
    d = pd.get_dummies(s, prefix=prefix, dummy_na=False, drop_first=True)
    return d.astype(float)


def remove_zero_variance_columns(X: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for c in X.columns:
        vals = pd.to_numeric(X[c], errors="coerce")
        if vals.notna().sum() == 0:
            continue
        if c == "intercept":
            keep.append(c)
        elif vals.nunique(dropna=True) > 1:
            keep.append(c)
    return X[keep]


def design_rank_ok(X: pd.DataFrame) -> Tuple[bool, int, int]:
    arr = X.to_numpy(dtype=float)
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        return False, 0, arr.shape[1]
    rank = int(np.linalg.matrix_rank(arr))
    return rank == arr.shape[1], rank, arr.shape[1]


def two_sided_p_from_t(t_value: float, df_resid: int) -> float:
    if not np.isfinite(t_value):
        return np.nan
    if scipy_stats is not None and df_resid > 0:
        return float(2.0 * scipy_stats.t.sf(abs(t_value), df=df_resid))
    return float(math.erfc(abs(t_value) / math.sqrt(2.0)))


def t_crit_975(df_resid: int) -> float:
    if scipy_stats is not None and df_resid > 0:
        return float(scipy_stats.t.ppf(0.975, df=df_resid))
    return 1.96


def fit_linear_hc3_numpy(y: pd.Series, X: pd.DataFrame, weights: Optional[pd.Series] = None) -> Dict[str, object]:
    y_arr = y.to_numpy(dtype=float)
    X_arr = X.to_numpy(dtype=float)
    names = list(X.columns)
    n, k = X_arr.shape
    if n <= k:
        raise ValueError(f"有效样本量不足以估计模型：n={n}, k={k}")

    if weights is not None:
        w_arr = pd.to_numeric(weights.loc[y.index], errors="coerce").to_numpy(dtype=float)
        if np.any(~np.isfinite(w_arr)) or np.any(w_arr <= 0):
            raise ValueError("权重包含缺失、非有限值或非正数。")
    else:
        w_arr = np.ones(n, dtype=float)

    sw = np.sqrt(w_arr)
    Xw = X_arr * sw[:, None]
    yw = y_arr * sw
    xtx_inv = np.linalg.inv(Xw.T @ Xw)
    beta = xtx_inv @ (Xw.T @ yw)

    resid = y_arr - X_arr @ beta
    h = np.sum((Xw @ xtx_inv) * Xw, axis=1)
    h = np.clip(h, 0.0, 0.999999)
    u_hc3 = (sw * resid) / (1.0 - h)
    meat = Xw.T @ ((u_hc3 ** 2)[:, None] * Xw)
    cov = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    tvals = beta / se
    df_resid = int(max(n - k, 1))
    pvals = np.array([two_sided_p_from_t(t, df_resid) for t in tvals], dtype=float)
    crit = t_crit_975(df_resid)
    ci_low = beta - crit * se
    ci_high = beta + crit * se

    y_mean_w = float(np.average(y_arr, weights=w_arr))
    ss_res = float(np.sum(w_arr * (resid ** 2)))
    ss_tot = float(np.sum(w_arr * ((y_arr - y_mean_w) ** 2)))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return {
        "params": dict(zip(names, beta)),
        "bse": dict(zip(names, se)),
        "tvalues": dict(zip(names, tvals)),
        "pvalues": dict(zip(names, pvals)),
        "ci95_low": dict(zip(names, ci_low)),
        "ci95_high": dict(zip(names, ci_high)),
        "rsquared": r2,
    }


def build_design(d: pd.DataFrame, edge_col: str, include_site: bool = False) -> Tuple[pd.Series, pd.DataFrame, List[str], str]:
    y = pd.to_numeric(d["_outcome_"], errors="coerce")
    X = pd.DataFrame(index=d.index)
    X["intercept"] = 1.0
    X["tms_bin"] = d["_tms_bin_"].astype(float)
    X["edge_z"] = zscore(d[edge_col])
    X["tms_x_edge"] = X["tms_bin"] * X["edge_z"]
    covariates = ["tms_bin", "edge_z", "tms_bin × edge_z"]

    if "PCL_pre" in d.columns:
        X["baseline_pcl_z"] = zscore(d["PCL_pre"])
        covariates.append("baseline PCL")
    if "age" in d.columns:
        X["age_z"] = zscore(d["age"])
        covariates.append("age")
    if "pre_meanFD" in d.columns:
        X["meanFD_z"] = zscore(d["pre_meanFD"])
        covariates.append("mean FD")
    if "sex" in d.columns:
        sex_dum = make_dummies(d["sex"], "sex")
        X = pd.concat([X, sex_dum], axis=1)
        if sex_dum.shape[1] > 0:
            covariates.append("sex")
    if include_site and "_site_inferred_TMS_vs_nonTMS_" in d.columns:
        site_dum = make_dummies(d["_site_inferred_TMS_vs_nonTMS_"], "site")
        X = pd.concat([X, site_dum], axis=1)
        if site_dum.shape[1] > 0:
            covariates.append("site")

    X = X.apply(pd.to_numeric, errors="coerce")
    valid = y.notna() & X.notna().all(axis=1)
    y = y.loc[valid]
    X = X.loc[valid]
    X = remove_zero_variance_columns(X)
    warn = ""
    if "tms_x_edge" not in X.columns:
        warn = "交互项缺失或零方差，模型不可估计。"
    return y, X, covariates, warn


def fit_ols_model(
    d: pd.DataFrame,
    edge_id: str,
    value_column: str,
    edge_col: str,
    model_type: str,
    include_site: bool = False,
    weights: Optional[pd.Series] = None,
) -> ModelResult:
    y, X, covariates, build_warn = build_design(d, edge_col, include_site=include_site)
    tmp = d.loc[y.index]
    n_psy = int((tmp["_pathway_"] == "PSY").sum())
    n_tms = int((tmp["_pathway_"] == "TMS").sum())
    if len(y) < 10 or n_psy < 5 or n_tms < 5:
        return ModelResult(edge_id, value_column, edge_col, model_type, len(y), n_psy, n_tms,
                           np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                           " + ".join(covariates), False,
                           f"样本量不足或分组不足：n={len(y)}, PSY={n_psy}, TMS={n_tms}。{build_warn}")
    if build_warn:
        return ModelResult(edge_id, value_column, edge_col, model_type, len(y), n_psy, n_tms,
                           np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                           " + ".join(covariates), False, build_warn)
    ok, rank, ncol = design_rank_ok(X)
    if not ok:
        return ModelResult(edge_id, value_column, edge_col, model_type, len(y), n_psy, n_tms,
                           np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                           " + ".join(covariates), False,
                           f"设计矩阵不满秩：rank={rank}, columns={ncol}。常见原因是 treatment 与 site/trial 完全或高度重合。")
    try:
        if weights is not None:
            w = pd.to_numeric(weights.loc[y.index], errors="coerce")
            valid = w.notna() & np.isfinite(w) & (w > 0)
            y2, X2, w2 = y.loc[valid], X.loc[valid], w.loc[valid]
            if False and sm is not None:
                fit = sm.WLS(y2, X2, weights=w2).fit(cov_type="HC3")
                beta = float(fit.params.get("tms_x_edge", np.nan))
                se = float(fit.bse.get("tms_x_edge", np.nan))
                tval = float(fit.tvalues.get("tms_x_edge", np.nan))
                pval = float(fit.pvalues.get("tms_x_edge", np.nan))
                ci = fit.conf_int().loc["tms_x_edge"].tolist() if "tms_x_edge" in fit.params.index else [np.nan, np.nan]
                r2 = float(getattr(fit, "rsquared", np.nan))
            else:
                fit_np = fit_linear_hc3_numpy(y2, X2, weights=w2)
                beta = float(fit_np["params"].get("tms_x_edge", np.nan))
                se = float(fit_np["bse"].get("tms_x_edge", np.nan))
                tval = float(fit_np["tvalues"].get("tms_x_edge", np.nan))
                pval = float(fit_np["pvalues"].get("tms_x_edge", np.nan))
                ci = [fit_np["ci95_low"].get("tms_x_edge", np.nan), fit_np["ci95_high"].get("tms_x_edge", np.nan)]
                r2 = float(fit_np.get("rsquared", np.nan))
            y, X = y2, X2
        else:
            if False and sm is not None:
                fit = sm.OLS(y, X).fit(cov_type="HC3")
                beta = float(fit.params.get("tms_x_edge", np.nan))
                se = float(fit.bse.get("tms_x_edge", np.nan))
                tval = float(fit.tvalues.get("tms_x_edge", np.nan))
                pval = float(fit.pvalues.get("tms_x_edge", np.nan))
                ci = fit.conf_int().loc["tms_x_edge"].tolist() if "tms_x_edge" in fit.params.index else [np.nan, np.nan]
                r2 = float(getattr(fit, "rsquared", np.nan))
            else:
                fit_np = fit_linear_hc3_numpy(y, X, weights=None)
                beta = float(fit_np["params"].get("tms_x_edge", np.nan))
                se = float(fit_np["bse"].get("tms_x_edge", np.nan))
                tval = float(fit_np["tvalues"].get("tms_x_edge", np.nan))
                pval = float(fit_np["pvalues"].get("tms_x_edge", np.nan))
                ci = [fit_np["ci95_low"].get("tms_x_edge", np.nan), fit_np["ci95_high"].get("tms_x_edge", np.nan)]
                r2 = float(fit_np.get("rsquared", np.nan))
        return ModelResult(edge_id, value_column, edge_col, model_type, len(y), n_psy, n_tms,
                           beta, se, tval, pval, float(ci[0]), float(ci[1]), r2,
                           " + ".join(covariates), True,
                           "使用 numpy HC3 备用模型；结果用于敏感性审计，与 v5 旧版保持数值等价。")
    except Exception as e:
        return ModelResult(edge_id, value_column, edge_col, model_type, len(y), n_psy, n_tms,
                           np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                           " + ".join(covariates), False, f"模型拟合失败：{repr(e)}")


def smd_continuous(x: pd.Series, g: pd.Series) -> float:
    a = pd.to_numeric(x[g == "PSY"], errors="coerce").dropna()
    b = pd.to_numeric(x[g == "TMS"], errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    if pooled == 0 or not np.isfinite(pooled):
        return np.nan
    return float((b.mean() - a.mean()) / pooled)


def cramers_v(table: pd.DataFrame) -> float:
    if scipy_stats is None:
        return np.nan
    arr = table.to_numpy(dtype=float)
    if arr.sum() == 0 or min(arr.shape) < 2:
        return np.nan
    chi2 = scipy_stats.chi2_contingency(arr, correction=False)[0]
    n = arr.sum()
    r, k = arr.shape
    denom = n * (min(k - 1, r - 1))
    if denom <= 0:
        return np.nan
    return float(math.sqrt(chi2 / denom))


def bh_fdr(pvals: Sequence[float]) -> List[float]:
    p = np.asarray([np.nan if x is None else x for x in pvals], dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    valid = np.where(np.isfinite(p))[0]
    if len(valid) == 0:
        return out.tolist()
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out[valid[order]] = q
    return out.tolist()


def make_balance_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    g = d["_pathway_"]
    for col, label in [("PCL_pre", "baseline PCL"), ("age", "age"), ("pre_meanFD", "mean FD"), ("_outcome_", "PCL improvement / outcome")]:
        if col not in d.columns:
            continue
        x = pd.to_numeric(d[col], errors="coerce")
        rows.append({
            "variable": label,
            "type": "continuous",
            "n_PSY": int(x[g == "PSY"].notna().sum()),
            "mean_PSY": float(x[g == "PSY"].mean(skipna=True)),
            "sd_PSY": float(x[g == "PSY"].std(skipna=True, ddof=1)),
            "n_TMS": int(x[g == "TMS"].notna().sum()),
            "mean_TMS": float(x[g == "TMS"].mean(skipna=True)),
            "sd_TMS": float(x[g == "TMS"].std(skipna=True, ddof=1)),
            "SMD_TMS_minus_PSY": smd_continuous(x, g),
            "interpretation": "|SMD|<0.1 excellent; 0.1-0.2 small; >0.2 possible imbalance",
        })
    for col, label in [("sex", "sex"), ("_site_inferred_TMS_vs_nonTMS_", "site")]:
        if col not in d.columns:
            continue
        tab = pd.crosstab(d[col].astype("string"), g, dropna=False)
        rows.append({
            "variable": label,
            "type": "categorical",
            "n_PSY": int((g == "PSY").sum()),
            "mean_PSY": np.nan,
            "sd_PSY": np.nan,
            "n_TMS": int((g == "TMS").sum()),
            "mean_TMS": np.nan,
            "sd_TMS": np.nan,
            "SMD_TMS_minus_PSY": np.nan,
            "cramers_v": cramers_v(tab),
            "category_table": tab.to_json(force_ascii=False),
            "interpretation": "Cramer's V 越接近 1，类别变量与治疗路径越强相关。",
        })
    return pd.DataFrame(rows)


def make_site_audit(d: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    c = "_site_inferred_TMS_vs_nonTMS_"
    tab = pd.crosstab(d[c].astype("string"), d["_pathway_"], dropna=False)
    for col in ["PSY", "TMS"]:
        if col not in tab.columns:
            tab[col] = 0
    tab = tab[["PSY", "TMS"]]
    both = ((tab["PSY"] > 0) & (tab["TMS"] > 0)).sum()
    complete = both == 0 and tab.shape[0] > 1
    msg = "该因子每个水平只包含一种治疗路径：study/site effects cannot be fully separated from intervention-pathway effects（研究/站点效应无法与干预路径效应完全分离）。"
    audit = pd.DataFrame([{
        "factor": "site",
        "available": True,
        "n_levels": int(tab.shape[0]),
        "n_levels_with_both_PSY_and_TMS": int(both),
        "complete_confounding": bool(complete),
        "message": msg,
    }])
    trial = pd.DataFrame([{
        "factor": "trial",
        "available": False,
        "n_levels": 0,
        "n_levels_with_both_PSY_and_TMS": 0,
        "complete_confounding": np.nan,
        "message": "未识别到 trial 列。",
    }])
    return pd.concat([audit, trial], ignore_index=True), tab


def estimate_propensity_weights(d: pd.DataFrame, out_dir: Path) -> Tuple[Optional[pd.Series], str]:
    if LogisticRegression is None:
        return None, "sklearn 不可用，跳过 IPW（逆概率加权）模型。"
    X = pd.DataFrame(index=d.index)
    used = []
    if "PCL_pre" in d.columns:
        X["baseline_pcl_z"] = zscore(d["PCL_pre"]); used.append("baseline PCL")
    if "age" in d.columns:
        X["age_z"] = zscore(d["age"]); used.append("age")
    if "pre_meanFD" in d.columns:
        X["meanFD_z"] = zscore(d["pre_meanFD"]); used.append("mean FD")
    if "sex" in d.columns:
        X = pd.concat([X, make_dummies(d["sex"], "sex")], axis=1); used.append("sex")
    X = X.apply(pd.to_numeric, errors="coerce")
    valid = X.notna().all(axis=1) & d["_tms_bin_"].notna()
    X = X.loc[valid]
    y = d.loc[valid, "_tms_bin_"].astype(int)
    if X.shape[1] == 0 or y.nunique() < 2 or len(y) < 15:
        return None, "可用于 propensity score（倾向评分）的协变量不足，跳过 IPW（逆概率加权）。"
    try:
        clf = LogisticRegression(max_iter=2000, solver="lbfgs")
        clf.fit(X, y)
        e = pd.Series(clf.predict_proba(X)[:, 1], index=X.index)
        p_t = float(y.mean())
        e_clip = e.clip(0.05, 0.95)
        w = pd.Series(index=d.index, dtype=float)
        w.loc[X.index] = np.where(y == 1, p_t / e_clip, (1 - p_t) / (1 - e_clip))
        auc = float(roc_auc_score(y, e)) if roc_auc_score is not None else np.nan
        diag = pd.DataFrame({
            "subject_index": X.index,
            "tms_bin": y,
            "propensity_TMS": e,
            "propensity_TMS_clipped": e_clip,
            "stabilized_ipw": w.loc[X.index],
        })
        write_csv(diag, out_dir / "03b_IPW倾向评分诊断.csv")
        msg = f"IPW 使用协变量: {', '.join(used)}；in-sample AUC={auc:.3f}。注意：这是 observed-covariate adjustment（观测协变量校正），不能消除未测量的 trial/site 混杂。"
        return w, msg
    except Exception as e:
        return None, f"IPW 估计失败：{repr(e)}"


def scan_outputs_for_legacy_labels(out_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(out_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in [".csv", ".txt", ".json"]:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
        hits = []
        for label, pat in LEGACY_LABEL_REGEX.items():
            if re.search(pat, text, flags=re.IGNORECASE):
                hits.append(label)
        compact = re.sub(r"[^a-z0-9]+", "", text.lower())
        for term in COMPRESSED_LEGACY_TERMS:
            if term in compact:
                hits.append(term)
        rows.append({
            "file": path.name,
            "legacy_label_hit_count": len(sorted(set(hits))),
            "legacy_label_hits": " | ".join(sorted(set(hits))),
            "status": "PASS" if not hits else "FAIL",
        })
    audit = pd.DataFrame(rows)
    write_csv(audit, out_dir / "00_output_legacy_label_text_audit_v6.csv")
    return audit


def make_manuscript_suggestion(d: pd.DataFrame, results_df: pd.DataFrame, ipw_message: str) -> str:
    n_psy = int((d["_pathway_"] == "PSY").sum())
    n_tms = int((d["_pathway_"] == "TMS").sum())
    summary_lines = []
    if not results_df.empty:
        for m in results_df["model_type"].dropna().unique().tolist():
            sub = results_df[(results_df["model_type"] == m) & (results_df["estimable"] == True)]
            if sub.empty:
                summary_lines.append(f"- {m}: 没有可估计的候选边模型。")
                continue
            n_p05 = int((sub["p_interaction"] < 0.05).sum())
            n_q05 = int((sub["q_fdr_within_model"] < 0.05).sum()) if "q_fdr_within_model" in sub.columns else 0
            summary_lines.append(f"- {m}: 可估计 {len(sub)} 条边；interaction p<0.05 为 {n_p05} 条；FDR q<0.05 为 {n_q05} 条。")
    return f"""
{SCRIPT_NAME} 自动解释建议
============================================================

样本口径
------------------------------------------------------------
主动治疗样本：PSY n={n_psy}；TMS n={n_tms}。
PSY = psychotherapy pathway（心理干预路径）；TMS = transcranial magnetic stimulation pathway（经颅磁刺激路径）。

设计风险审计
------------------------------------------------------------
本数据中 intervention pathway（干预路径）与推断 site（站点）完全重合。因此，study/site effects could not be fully separated from intervention-pathway effects（研究/站点效应无法与干预路径效应完全分离）。

本脚本不能证明 causal treatment-assignment rule（因果性治疗分配规则），只能支持或削弱 exploratory pathway-dependent treatment-selection signal（探索性路径依赖治疗选择信号）的表述。

候选边身份规则
------------------------------------------------------------
本 v6 脚本只输出 EDGE_01–EDGE_10 与 value_column【取值列】身份；不输出旧 Brainnetome 解剖边名。解剖名与 BrainNet 图必须依赖后续单独审计通过的 ROI 映射。

Method 或 Statistical analysis 建议句子
------------------------------------------------------------
English:
Because intervention pathway was not randomized within a single head-to-head trial, treatment-selection signals were interpreted as exploratory pathway-dependent associations rather than validated causal treatment-assignment rules.

中文：
由于干预路径并非在同一个头对头试验中随机分配，治疗选择信号应解释为探索性的路径依赖关联，而不是已经验证的因果性治疗分配规则。

English:
Study/site effects could not be fully separated from intervention-pathway effects.

中文：
研究/站点效应无法与干预路径效应完全分离。

敏感性模型概览
------------------------------------------------------------
{chr(10).join(summary_lines) if summary_lines else '未生成模型结果。'}

IPW（inverse probability weighting，逆概率加权）说明
------------------------------------------------------------
{ipw_message}
""".strip() + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="46号 v6 value_column-only PSY/TMS 非头对头设计风险审计")
    p.add_argument("--input_file", type=str, default="", help="43_v4_2 value_column-only 被试层核心边表。留空则自动搜索。")
    p.add_argument("--output_dir", type=str, default="", help="输出目录。默认当前第四步目录下 v6 结果文件夹。")
    p.add_argument("--reverse_outcome", action="store_true", help="若 outcome 方向相反，乘以 -1。默认不反转。")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if sm is None:
        log(f"[依赖提示] statsmodels 不可用，将使用脚本内置 numpy HC3 备用模型。原始错误: {STATSMODELS_IMPORT_ERROR}")

    input_path = Path(args.input_file) if args.input_file else auto_find_safe_input_file()
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    out_dir = Path(args.output_dir) if args.output_dir else find_project_root() / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 100)
    log(f"{SCRIPT_NAME} 启动")
    log(f"input_file = {input_path}")
    log(f"output_dir = {out_dir}")
    log("=" * 100)

    df = safe_read_table(input_path)
    log(f"[读取] rows={df.shape[0]}, cols={df.shape[1]}")

    # 硬性安全检查
    required_cols = ["group", "pcl_improvement", "PCL_pre", "age", "sex", "pre_meanFD"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"安全输入表缺少必要列：{c}")
    for eid, vc in EDGE_VALUE_COLUMNS.items():
        for role in ["value_column", "pre", "post", "delta"]:
            c = f"{eid}__{role}"
            if c not in df.columns:
                raise ValueError(f"安全输入表缺少必要列：{c}")
        got = str(df[f"{eid}__value_column"].dropna().iloc[0])
        if got != vc:
            raise ValueError(f"{eid} value_column 不符合预期：got={got}, expected={vc}")

    d = df.copy()
    d["_pathway_"] = d["group"].apply(lambda x: standardize_treatment_value(x, ["PSY", "psychotherapy", "ACT", "MIN"], ["TMS", "rTMS"]))
    before = len(d)
    d = d[d["_pathway_"].isin(["PSY", "TMS"])].copy()
    d["_tms_bin_"] = (d["_pathway_"] == "TMS").astype(int)
    d["_outcome_"] = pd.to_numeric(d["pcl_improvement"], errors="coerce")
    if args.reverse_outcome:
        d["_outcome_"] = -d["_outcome_"]
    d["_site_inferred_TMS_vs_nonTMS_"] = np.where(d["_pathway_"].eq("TMS"), "site_TMS", "site_nonTMS_ACT_MIN_WL")
    log(f"[样本筛选] 主动治疗 PSY/TMS: {len(d)}/{before}; PSY={(d['_pathway_']=='PSY').sum()}, TMS={(d['_pathway_']=='TMS').sum()}")

    # 输出主动治疗样本表（安全身份）
    sample_cols = ["subject_key", "subject_id4", "group", "treatment_family", "pcl_improvement", "PCL_pre", "age", "sex", "pre_meanFD", "post_meanFD", "_pathway_", "_tms_bin_", "_outcome_", "_site_inferred_TMS_vs_nonTMS_"]
    for eid in EDGE_VALUE_COLUMNS:
        sample_cols += [f"{eid}__value_column", f"{eid}__pre", f"{eid}__post", f"{eid}__delta"]
    sample_cols = [c for c in sample_cols if c in d.columns]
    write_csv(d[sample_cols], out_dir / "00b_用于审计的主动治疗样本表.csv")

    site_msg = (
        "已按项目设计显式推断 site（站点）变量：TMS -> site_TMS；非TMS/PSY -> site_nonTMS_ACT_MIN_WL。"
        "该变量由 treatment pathway（治疗路径）派生，预期与 treatment 完全重合；"
        "M2 的 treatment + site 模型若不可估计，属于设计限制证据，而不是脚本失败。"
    )
    write_text(out_dir / "00c_site推断规则说明.txt", site_msg)

    run_info = {
        "script": SCRIPT_NAME,
        "input_file_name": input_path.name,
        "input_policy": "43_v4_2_value_column_only_required",
        "output_dir": str(out_dir),
        "columns_detected": {
            "subject": "subject_key",
            "treatment": "group",
            "outcome": "pcl_improvement",
            "baseline_pcl": "PCL_pre",
            "age": "age",
            "sex": "sex",
            "fd": "pre_meanFD",
            "site": "_site_inferred_TMS_vs_nonTMS_",
            "trial": None,
        },
        "reverse_outcome": bool(args.reverse_outcome),
        "include_post_delta": False,
        "edge_identity_policy": "EDGE_01_to_EDGE_10_plus_value_column_only_no_anatomical_label",
        "n_raw": int(df.shape[0]),
        "n_active_pathway": int(d.shape[0]),
        "n_PSY": int((d["_pathway_"] == "PSY").sum()),
        "n_TMS": int((d["_pathway_"] == "TMS").sum()),
    }
    write_text(out_dir / "00_运行参数和列识别.json", json.dumps(run_info, ensure_ascii=False, indent=2))

    audit_df, site_tab = make_site_audit(d)
    write_csv(audit_df, out_dir / "01_trial_site与治疗路径混杂审计.csv")
    site_tab.to_csv(out_dir / "01b_site_by_pathway交叉表.csv", encoding="utf-8-sig")
    balance_df = make_balance_table(d)
    write_csv(balance_df, out_dir / "02_PSY_TMS基线协变量平衡_SMD.csv")

    # 候选边匹配表：显示全部 triplets，但 only pre used for model
    edge_rows = []
    for eid, vc in EDGE_VALUE_COLUMNS.items():
        for role in ["pre", "post", "delta"]:
            col = f"{eid}__{role}"
            edge_rows.append({
                "requested_edge": f"{eid}__{role}",
                "edge": eid,
                "value_column": vc,
                "matched_column": col,
                "matched": col in d.columns,
                "used_for_model": role == "pre",
                "default_baseline_pre_only": True,
                "identity_policy": "value_column_only_no_anatomical_label",
            })
    edge_info = pd.DataFrame(edge_rows)
    write_csv(edge_info, out_dir / "03_候选边列名匹配表.csv")

    ipw_weights, ipw_message = estimate_propensity_weights(d, out_dir)
    log(f"[IPW] {ipw_message}")

    results: List[ModelResult] = []
    for i, (eid, vc) in enumerate(EDGE_VALUE_COLUMNS.items(), start=1):
        edge_col = f"{eid}__pre"
        log(f"[模型] {i}/10: {eid} ({vc})")
        results.append(fit_ols_model(d, eid, vc, edge_col, "M1_covariate_adjusted_no_site_trial"))
        results.append(fit_ols_model(d, eid, vc, edge_col, "M2_covariate_adjusted_plus_site_trial", include_site=True))
        if ipw_weights is not None:
            results.append(fit_ols_model(d, eid, vc, edge_col, "M3_IPW_observed_covariate_weighted_no_site_trial", weights=ipw_weights))

    res_df = pd.DataFrame([r.__dict__ for r in results])
    if not res_df.empty:
        res_df["q_fdr_within_model"] = np.nan
        for model_type, idx in res_df.groupby("model_type").groups.items():
            res_df.loc[idx, "q_fdr_within_model"] = bh_fdr(res_df.loc[idx, "p_interaction"].tolist())
        m1 = res_df[res_df["model_type"] == "M1_covariate_adjusted_no_site_trial"][["edge", "beta_interaction"]].rename(columns={"beta_interaction": "beta_M1"})
        res_df = res_df.merge(m1, on="edge", how="left")
        res_df["same_direction_as_M1"] = np.where(
            res_df["estimable"] & np.isfinite(res_df["beta_interaction"]) & np.isfinite(res_df["beta_M1"]),
            np.sign(res_df["beta_interaction"]) == np.sign(res_df["beta_M1"]),
            np.nan,
        )
    write_csv(res_df, out_dir / "04_候选边治疗路径交互敏感性模型结果.csv")

    compact_cols = [
        "edge", "value_column", "feature_col", "model_type", "n", "n_psy", "n_tms",
        "beta_interaction", "se_interaction", "p_interaction", "q_fdr_within_model",
        "ci95_low", "ci95_high", "r2", "estimable", "same_direction_as_M1", "warning",
    ]
    write_csv(res_df[[c for c in compact_cols if c in res_df.columns]].copy(), out_dir / "04b_模型结果紧凑摘要.csv")

    suggestion = make_manuscript_suggestion(d, res_df, ipw_message)
    write_text(out_dir / "05_论文口径和限制性表述建议.txt", suggestion)

    if not res_df.empty:
        pivot = res_df.pivot_table(index=["edge", "value_column"], columns="model_type", values=["beta_interaction", "p_interaction", "q_fdr_within_model"], aggfunc="first")
        pivot.columns = [f"{a}__{b}" for a, b in pivot.columns]
        pivot = pivot.reset_index()
        write_csv(pivot, out_dir / "06_每条候选边跨模型对照表.csv")

    # 身份审计
    identity_rows = []
    for eid, vc in EDGE_VALUE_COLUMNS.items():
        identity_rows.append({
            "edge": eid,
            "value_column": vc,
            "pre_col": f"{eid}__pre",
            "post_col": f"{eid}__post",
            "delta_col": f"{eid}__delta",
            "status": "PASS",
        })
    write_csv(pd.DataFrame(identity_rows), out_dir / "00_edge_value_column_identity_audit_v6.csv")

    legacy_audit = scan_outputs_for_legacy_labels(out_dir)
    n_fail = int((legacy_audit["status"] != "PASS").sum()) if not legacy_audit.empty else 0
    summary = {
        "script": SCRIPT_NAME,
        "safe_for_formal_use": bool(n_fail == 0),
        "n_legacy_fail_files": n_fail,
        "n_edges": len(EDGE_VALUE_COLUMNS),
        "n_model_rows": int(len(res_df)),
        "edge_identity_policy": "EDGE_01_to_EDGE_10_plus_value_column_only_no_anatomical_label",
    }
    write_text(out_dir / "00_v6_run_summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
    if n_fail > 0:
        raise RuntimeError("v6 输出中发现旧脑区标签残留，请检查 00_output_legacy_label_text_audit_v6.csv。")

    log("=" * 100)
    log("完成。v6 输出已通过旧标签审计，可进入数值等价审计。")
    log(f"结果目录：{out_dir}")
    log("=" * 100)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main()
