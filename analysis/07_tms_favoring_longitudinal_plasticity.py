# -*- coding: utf-8 -*-
"""
45号脚本：TMS偏向候选边_探索性纵向可塑性分析
==================================================
目的：
    在 44号脚本已经固定的候选边框架下，进一步给 TMS 组一个“有原则的、范围收缩的”
    treatment-process / longitudinal plasticity【治疗过程/纵向可塑性】探索性检查。

本脚本不是为了通过放宽阈值来“制造显著”，而是同时执行三个预先限定的方案：

    方案A：只分析 TMS-favoring edges【TMS偏向候选边】。
           默认边：EDGE_03、EDGE_05、EDGE_07、EDGE_09、EDGE_10；解剖标签不在本脚本输出。
           如果能从 44号模块1结果中识别 TMS-favoring，也会自动审计。

    方案B：只分析 TMS 结果最相关的核心结局。
           默认核心结局：PCL5, GAD-7, PHQ-9, PCL_D, PCL_E。
           形成较小检验族：约 5条TMS偏向边 × 5个核心结局 = 25个检验。

    方案C：不仅报告 p 值，还报告 effect size【效应量】、方向一致性、WL负控对照和证据等级。
           证据等级包括：FDR-supported, nominal-directional, trend, weak/no evidence。

输入：
    优先读取 44_v3 value_column-only【仅取值列身份】输出文件夹中的：
        00d_合并后数据预览前200行.csv
        01_自动识别_临床结局指标.csv
        02_自动识别_固定候选边.csv
        10_模块1_其他指标治疗调节效应.csv

输出：
    45_TMS偏向候选边_探索性纵向可塑性分析结果/
        00_输入审计.json
        01_TMS偏向边识别与来源.csv
        02_核心结局识别.csv
        10_方案A_TMS偏向边x全部结局_探索性deltaFC结果.csv
        20_方案B_TMS偏向边x核心结局_收缩检验族结果.csv
        30_方案C_效应量方向一致性与证据等级.csv
        31_可写入论文或补充材料的候选结果.csv
        99_结果解读提示.txt

运行示例：
    python 45_TMS偏向候选边_探索性纵向可塑性分析.py

更稳的手动指定：
    python 45_v2_value_column_only_TMS偏向候选边_探索性纵向可塑性分析_无旧标签正式版.py --input_dir "E:\\E_zhangzhihui\\从yv那边提取\\脚本\\第四步分析：探索\\44_v3_value_column_only_其他临床指标_固定候选边辅助验证与机制解释结果_无旧标签正式版"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception as e:
    stats = None
    print("[警告] 未能导入 scipy，将无法计算 Pearson/Spearman/p 值。错误：", repr(e))


SCRIPT_VERSION = "v2.0_2026-05-20_value_column_only_TMS_favoring_exploratory_plasticity"

DEFAULT_TMS_FAVORING_EDGES = [
    "EDGE_03",
    "EDGE_05",
    "EDGE_07",
    "EDGE_09",
    "EDGE_10",
]

SAFE_EDGE_VALUE_COLUMNS = {
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

LEGACY_LABEL_PATTERNS = [
    r"A45r", r"A11m", r"A5l", r"vId/vIg", r"vId_vIg", r"cpSTS", r"lsOccG", r"A7c",
    r"A9/46v", r"A9_46v", r"A7ip", r"A35/36c", r"A35_36c", r"A37elv", r"cLinG",
    r"A24rv", r"A32p", r"A9/46d", r"A9_46d", r"A24cd", r"A11l", r"aSTS",
    r"a45ra11m", r"a5lvidvig", r"cpstslsoccg", r"a3536cdia",
    r"a37elvcling", r"a946va7ip", r"a946da24cd", r"a11lasts",
]

CORE_OUTCOME_PATTERNS = {
    "PCL5_total": ["PCL5", "PCL_5总分"],
    "GAD7_anxiety": ["GAD_7总分", "GAD-7", "GAD7"],
    "PHQ9_depression": ["PHQ_9总分", "PHQ-9", "PHQ9"],
    "PCL_D_negative_cognition_mood": ["PCL_D认知情绪改变", "PCL_D"],
    "PCL_E_hyperarousal": ["PCL_E警觉反应", "PCL_E"],
}

PREFERRED_MODEL_TIER = "M2_age_sex_meanFD"


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def normalize_key(x: object) -> str:
    """Normalize labels so A9/46v–A7ip and A9_46v_A7ip can match."""
    s = str(x) if x is not None else ""
    s = s.replace("–", "_").replace("—", "_").replace("-", "_").replace("/", "_")
    s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower()


def safe_read_csv(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    # utf-8-sig 优先；失败后尝试 gbk，兼容 Excel 保存的中文 CSV。
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc, nrows=nrows)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, nrows=nrows)


def find_first_file(input_dir: Path, required_tokens: Iterable[str], suffix: str = ".csv") -> Optional[Path]:
    files = list(input_dir.glob(f"*{suffix}"))
    scored = []
    for f in files:
        name = f.name.lower()
        score = 0
        for tok in required_tokens:
            if tok.lower() in name:
                score += 1
        if score > 0:
            scored.append((score, len(name), f))
    if not scored:
        return None
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][2]


def auto_find_44_dir(start_dir: Path) -> Optional[Path]:
    """Only accept the locked 44_v3 value_column-only output directory.

    This prevents accidental reuse of the old 44_v2.3 folder that contains
    anatomical edge labels.
    """
    candidates = []
    for p in [start_dir] + [x for x in start_dir.iterdir() if x.is_dir()]:
        if not p.is_dir():
            continue
        audit = p / "00_output_legacy_label_text_audit_v3.csv"
        merged = p / "00d_合并后数据预览前200行.csv"
        edge = p / "02_自动识别_固定候选边.csv"
        if audit.exists() and merged.exists() and edge.exists() and "44_v3" in p.name and "value_column" in p.name:
            candidates.append((0, p))
    if candidates:
        return sorted(candidates, key=lambda x: len(str(x[1])))[0][1]
    return None


def coerce_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    # 兼容性别等分类变量
    out = pd.to_numeric(s, errors="coerce")
    if out.notna().sum() == 0 and s.notna().sum() > 0:
        vals = s.astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
        codes, uniques = pd.factorize(vals, sort=True)
        out = pd.Series(codes, index=s.index).replace(-1, np.nan).astype(float)
    return out


def zscore_series(s: pd.Series) -> pd.Series:
    x = coerce_numeric(s)
    sd = x.std(skipna=True, ddof=1)
    if pd.isna(sd) or sd == 0:
        return x * np.nan
    return (x - x.mean(skipna=True)) / sd


def fdr_bh(pvals: Iterable[object]) -> np.ndarray:
    p = pd.to_numeric(pd.Series(list(pvals)), errors="coerce").to_numpy(dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    idx = np.where(np.isfinite(p))[0]
    if len(idx) == 0:
        return q
    pv = p[idx]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    q[idx[order]] = adj
    return q


def pearson_spearman(df: pd.DataFrame, x_col: str, y_col: str) -> Dict[str, float]:
    out = {
        "n_pair": 0,
        "pearson_r": np.nan,
        "pearson_p": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
    }
    if stats is None:
        return out
    x = coerce_numeric(df[x_col])
    y = coerce_numeric(df[y_col])
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    out["n_pair"] = int(len(d))
    if len(d) < 4 or d["x"].nunique() < 2 or d["y"].nunique() < 2:
        return out
    try:
        r, p = stats.pearsonr(d["x"], d["y"])
        out["pearson_r"] = float(r)
        out["pearson_p"] = float(p)
    except Exception:
        pass
    try:
        rho, p2 = stats.spearmanr(d["x"], d["y"])
        out["spearman_rho"] = float(rho)
        out["spearman_p"] = float(p2)
    except Exception:
        pass
    return out


def fisher_z_compare(r1: float, n1: int, r2: float, n2: int) -> Tuple[float, float]:
    if stats is None:
        return np.nan, np.nan
    if not (np.isfinite(r1) and np.isfinite(r2)) or n1 < 4 or n2 < 4:
        return np.nan, np.nan
    r1 = float(np.clip(r1, -0.999999, 0.999999))
    r2 = float(np.clip(r2, -0.999999, 0.999999))
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    se = math.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z = (z1 - z2) / se
    p = 2 * stats.norm.sf(abs(z))
    return float(z), float(p)


def ols_delta_effect(df: pd.DataFrame, y_col: str, delta_col: str, covariates: List[str]) -> Dict[str, object]:
    """OLS: y ~ deltaFC + covariates. Return delta coefficient and p.
    Uses numpy only. Covariates are optional and dropped if not present or all missing.
    """
    result = {
        "ols_ok": False,
        "ols_error": "",
        "ols_n": 0,
        "ols_df_resid": np.nan,
        "ols_r2": np.nan,
        "ols_delta_beta": np.nan,
        "ols_delta_p": np.nan,
        "used_covariates": "",
    }
    if stats is None:
        result["ols_error"] = "scipy_not_available"
        return result
    use_cols = [y_col, delta_col]
    usable_covs = []
    for c in covariates:
        if c in df.columns and df[c].notna().sum() >= 4:
            usable_covs.append(c)
            use_cols.append(c)
    d = df[use_cols].copy()
    d[y_col] = coerce_numeric(d[y_col])
    d[delta_col] = coerce_numeric(d[delta_col])
    for c in usable_covs:
        d[c] = coerce_numeric(d[c])
    d = d.dropna()
    result["ols_n"] = int(len(d))
    result["used_covariates"] = " | ".join(usable_covs)
    if len(d) < max(8, 3 + len(usable_covs)):
        result["ols_error"] = f"样本量不足 n={len(d)}"
        return result
    if d[delta_col].nunique() < 2 or d[y_col].nunique() < 2:
        result["ols_error"] = "deltaFC或结局无变异"
        return result
    y = d[y_col].to_numpy(dtype=float)
    X_parts = [np.ones(len(d)), zscore_series(d[delta_col]).to_numpy(dtype=float)]
    for c in usable_covs:
        X_parts.append(zscore_series(d[c]).to_numpy(dtype=float))
    X = np.column_stack(X_parts)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        result["ols_error"] = "设计矩阵含非有限值"
        return result
    try:
        beta = np.linalg.pinv(X) @ y
        yhat = X @ beta
        resid = y - yhat
        n, k = X.shape
        df_resid = n - k
        if df_resid <= 0:
            result["ols_error"] = "自由度不足"
            return result
        sse = float(np.sum(resid ** 2))
        sst = float(np.sum((y - y.mean()) ** 2))
        mse = sse / df_resid
        cov_beta = mse * np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diag(cov_beta))
        if not np.isfinite(se[1]) or se[1] == 0:
            result["ols_error"] = "deltaFC标准误无效"
            return result
        tval = beta[1] / se[1]
        pval = 2 * stats.t.sf(abs(tval), df_resid)
        result.update({
            "ols_ok": True,
            "ols_df_resid": float(df_resid),
            "ols_r2": float(1 - sse / sst) if sst > 0 else np.nan,
            "ols_delta_beta": float(beta[1]),
            "ols_delta_p": float(pval),
            "ols_error": "",
        })
    except Exception as e:
        result["ols_error"] = repr(e)
    return result


def detect_group_col(df: pd.DataFrame) -> str:
    for c in ["group3", "treatment_family", "group", "最终分组", "clinical_group"]:
        if c in df.columns:
            return c
    raise ValueError("未找到分组列，例如 group3/treatment_family/group。")


def group_mask(df: pd.DataFrame, group_col: str, group_name: str) -> pd.Series:
    vals = df[group_col].astype(str).str.upper()
    if group_name.upper() == "TMS":
        return vals.str.contains("TMS", na=False)
    if group_name.upper() == "WL":
        return vals.str.contains("WL|WAIT", na=False)
    if group_name.upper() == "PSY":
        return vals.str.contains("PSY|ACT|MIN", na=False) & ~vals.str.contains("WL|TMS", na=False)
    return vals == group_name.upper()


def assert_no_legacy_text_in_value(x: object, context: str = "") -> None:
    s = str(x)
    for pat in LEGACY_LABEL_PATTERNS:
        if re.search(pat, s, flags=re.IGNORECASE):
            raise RuntimeError(f"发现旧脑区标签残留：{context} = {s}")

def output_has_legacy_text(text: str) -> bool:
    for pat in LEGACY_LABEL_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True
    return False

def scan_outputs_for_legacy_labels(out_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(out_dir.glob("*")):
        if not p.is_file() or p.suffix.lower() not in [".csv", ".txt", ".json"]:
            continue
        text = p.read_text(encoding="utf-8-sig", errors="ignore")
        hit = int(output_has_legacy_text(text))
        rows.append({"file": p.name, "legacy_label_hit_count": hit, "audit_status": "PASS" if hit == 0 else "FAIL"})
    audit = pd.DataFrame(rows)
    audit.to_csv(out_dir / "00_output_legacy_label_text_audit_v2.csv", index=False, encoding="utf-8-sig")
    if not audit.empty and int(audit["legacy_label_hit_count"].sum()) > 0:
        raise RuntimeError("45_v2 输出中仍发现旧脑区标签残留，请检查 00_output_legacy_label_text_audit_v2.csv。")
    return audit

def build_edge_meta(edge_df: Optional[pd.DataFrame], merged: pd.DataFrame) -> pd.DataFrame:
    """Build safe edge metadata from 44_v3 output.

    Required identity: EDGE_01--EDGE_10 plus value_column. Old anatomical labels
    are not accepted.
    """
    rows = []
    if edge_df is not None and len(edge_df) > 0:
        required = {"edge_label", "delta_fc_col"}
        if not required.issubset(edge_df.columns):
            raise RuntimeError("44_v3 的 02_自动识别_固定候选边.csv 缺少 edge_label/delta_fc_col。")
        for _, r in edge_df.iterrows():
            eid = str(r.get("canonical_edge_id", r.get("edge_label", ""))).strip()
            if not re.fullmatch(r"EDGE_\d{2}", eid):
                raise RuntimeError(f"45_v2 只接受 EDGE_01--EDGE_10 作为边身份；发现：{eid}")
            if eid not in SAFE_EDGE_VALUE_COLUMNS:
                continue
            value_column = str(r.get("value_column", SAFE_EDGE_VALUE_COLUMNS[eid])).strip() or SAFE_EDGE_VALUE_COLUMNS[eid]
            if value_column != SAFE_EDGE_VALUE_COLUMNS[eid]:
                raise RuntimeError(f"{eid} 的 value_column 与锁定映射不一致：{value_column}")
            base = str(r.get("baseline_fc_col", f"{eid}__pre"))
            post = str(r.get("post_fc_col", f"{eid}__post"))
            delta = str(r.get("delta_fc_col", f"{eid}__delta"))
            for val, ctx in [(eid, "edge_id"), (value_column, "value_column"), (base, "baseline_fc_col"), (post, "post_fc_col"), (delta, "delta_fc_col")]:
                assert_no_legacy_text_in_value(val, ctx)
            if delta not in merged.columns:
                raise RuntimeError(f"{eid} 的 delta 列不在合并数据中：{delta}")
            rows.append({
                "edge_label": eid,
                "canonical_edge_id": eid,
                "value_column": value_column,
                "edge_identity_policy": "value_column_only_no_anatomical_label",
                "edge_key": eid.lower(),
                "baseline_fc_col": base,
                "post_fc_col": post,
                "delta_fc_col": delta,
            })
    if not rows:
        # Last-resort safe inference from merged table.
        for i in range(1, 11):
            eid = f"EDGE_{i:02d}"
            base, post, delta = f"{eid}__pre", f"{eid}__post", f"{eid}__delta"
            if delta in merged.columns:
                rows.append({
                    "edge_label": eid,
                    "canonical_edge_id": eid,
                    "value_column": SAFE_EDGE_VALUE_COLUMNS[eid],
                    "edge_identity_policy": "value_column_only_no_anatomical_label",
                    "edge_key": eid.lower(),
                    "baseline_fc_col": base,
                    "post_fc_col": post,
                    "delta_fc_col": delta,
                })
    meta = pd.DataFrame(rows).drop_duplicates(subset=["edge_label"]).sort_values("edge_label")
    if len(meta) != 10:
        raise RuntimeError(f"未能从 44_v3 安全结果中识别完整 10 条 EDGE 候选边；当前 n={len(meta)}。")
    return meta


def detect_tms_favoring_edges(module10: Optional[pd.DataFrame], edge_meta: pd.DataFrame, default_edges: List[str]) -> pd.DataFrame:
    """Return the fixed TMS-favoring EDGE set.

    The set is locked to EDGE_03/05/07/09/10 to preserve the old script's
    statistical scope while removing anatomical labels from outputs.
    """
    rows = []
    meta_by_id = {str(r["edge_label"]): r for _, r in edge_meta.iterrows()}
    for eid in default_edges:
        if eid not in meta_by_id:
            raise RuntimeError(f"TMS 偏向固定边 {eid} 未在 44_v3 edge_meta 中找到。")
        r = meta_by_id[eid]
        rows.append({
            "edge_label": eid,
            "canonical_edge_id": eid,
            "value_column": str(r.get("value_column", SAFE_EDGE_VALUE_COLUMNS[eid])),
            "edge_identity_policy": "value_column_only_no_anatomical_label",
            "delta_fc_col": r.get("delta_fc_col", f"{eid}__delta"),
            "baseline_fc_col": r.get("baseline_fc_col", f"{eid}__pre"),
            "source": "locked_TMS_favoring_EDGE_set【锁定TMS偏向EDGE集合】",
            "n_tms_favoring_M2_core": np.nan,
            "n_psy_favoring_M2_core": np.nan,
        })
    out = pd.DataFrame(rows)
    return out


def build_outcome_meta(outcome_df: Optional[pd.DataFrame], merged: pd.DataFrame) -> pd.DataFrame:
    if outcome_df is not None and len(outcome_df) > 0 and "improvement_col" in outcome_df.columns:
        out = outcome_df.copy()
        out = out[out["improvement_col"].astype(str).isin(merged.columns)].copy()
        if len(out) > 0:
            return out
    rows = []
    for c in merged.columns:
        if str(c).startswith("OUTCOME_IMPROVEMENT__"):
            outcome = str(c).replace("OUTCOME_IMPROVEMENT__", "")
            rows.append({
                "outcome": outcome,
                "category": "unknown",
                "improvement_col": str(c),
                "direction_rule": "正值代表改善",
                "n_nonmissing": int(pd.to_numeric(merged[c], errors="coerce").notna().sum()),
            })
    return pd.DataFrame(rows)


def detect_core_outcomes(outcome_meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for core_name, patterns in CORE_OUTCOME_PATTERNS.items():
        best = None
        best_score = -1
        for _, r in outcome_meta.iterrows():
            text = f"{r.get('outcome','')} {r.get('improvement_col','')} {r.get('category','')}"
            # 排除 PCL5_change/post_minus_pre 等重复技术列，优先 PCL5 或 PCL_5总分。
            score = 0
            for pat in patterns:
                if pat.lower() in text.lower():
                    score += 10
            if "change_post_minus_pre" in text or "improvement_pre_minus_post" in text:
                score -= 3
            if core_name == "PCL5_total" and str(r.get("outcome", "")) == "PCL5":
                score += 5
            if score > best_score:
                best_score = score
                best = r
        if best is not None and best_score > 0:
            br = best.to_dict()
            br["core_outcome_name"] = core_name
            br["core_detect_score"] = best_score
            rows.append(br)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.drop_duplicates(subset=["core_outcome_name"])
    return out


def run_one_family(merged: pd.DataFrame, edges: pd.DataFrame, outcomes: pd.DataFrame, family_name: str, covariates: List[str]) -> pd.DataFrame:
    group_col = detect_group_col(merged)
    tms = merged[group_mask(merged, group_col, "TMS")].copy()
    wl = merged[group_mask(merged, group_col, "WL")].copy()
    rows = []
    total = max(1, len(edges) * len(outcomes))
    counter = 0
    for _, er in edges.iterrows():
        edge_label = str(er["edge_label"])
        delta_col = str(er["delta_fc_col"])
        if delta_col not in merged.columns:
            continue
        for _, orow in outcomes.iterrows():
            counter += 1
            if counter == 1 or counter % 10 == 0 or counter == total:
                log(f"{family_name} 进度：{counter}/{total}")
            outcome = str(orow.get("outcome", ""))
            imp_col = str(orow.get("improvement_col", ""))
            if imp_col not in merged.columns:
                continue
            # TMS association
            tms_corr = pearson_spearman(tms, delta_col, imp_col)
            wl_corr = pearson_spearman(wl, delta_col, imp_col)
            tms_ols = ols_delta_effect(tms, imp_col, delta_col, covariates)
            wl_ols = ols_delta_effect(wl, imp_col, delta_col, [])
            z_diff, p_diff = fisher_z_compare(
                tms_corr.get("pearson_r", np.nan), int(tms_corr.get("n_pair", 0) or 0),
                wl_corr.get("pearson_r", np.nan), int(wl_corr.get("n_pair", 0) or 0),
            )
            rows.append({
                "analysis_family": family_name,
                "edge_label": edge_label,
                "canonical_edge_id": edge_label,
                "value_column": str(er.get("value_column", "")),
                "edge_identity_policy": "value_column_only_no_anatomical_label",
                "delta_fc_col": delta_col,
                "outcome": outcome,
                "outcome_category": orow.get("category", ""),
                "improvement_col": imp_col,
                "edge_source": er.get("source", ""),
                "n_TMS": tms_corr.get("n_pair", 0),
                "TMS_pearson_r": tms_corr.get("pearson_r", np.nan),
                "TMS_pearson_p": tms_corr.get("pearson_p", np.nan),
                "TMS_spearman_rho": tms_corr.get("spearman_rho", np.nan),
                "TMS_spearman_p": tms_corr.get("spearman_p", np.nan),
                "TMS_ols_ok": tms_ols.get("ols_ok", False),
                "TMS_ols_error": tms_ols.get("ols_error", ""),
                "TMS_ols_n": tms_ols.get("ols_n", 0),
                "TMS_ols_delta_beta": tms_ols.get("ols_delta_beta", np.nan),
                "TMS_ols_delta_p": tms_ols.get("ols_delta_p", np.nan),
                "TMS_ols_r2": tms_ols.get("ols_r2", np.nan),
                "TMS_ols_covariates": tms_ols.get("used_covariates", ""),
                "n_WL": wl_corr.get("n_pair", 0),
                "WL_pearson_r": wl_corr.get("pearson_r", np.nan),
                "WL_pearson_p": wl_corr.get("pearson_p", np.nan),
                "WL_spearman_rho": wl_corr.get("spearman_rho", np.nan),
                "WL_spearman_p": wl_corr.get("spearman_p", np.nan),
                "WL_ols_n": wl_ols.get("ols_n", 0),
                "WL_ols_delta_beta": wl_ols.get("ols_delta_beta", np.nan),
                "WL_ols_delta_p": wl_ols.get("ols_delta_p", np.nan),
                "TMS_vs_WL_fisher_z": z_diff,
                "TMS_vs_WL_fisher_p": p_diff,
            })
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    # FDR within the family
    for pcol, qcol in [
        ("TMS_pearson_p", "TMS_pearson_q"),
        ("TMS_spearman_p", "TMS_spearman_q"),
        ("TMS_ols_delta_p", "TMS_ols_delta_q"),
        ("WL_pearson_p", "WL_pearson_q"),
        ("TMS_vs_WL_fisher_p", "TMS_vs_WL_fisher_q"),
    ]:
        out[qcol] = fdr_bh(out[pcol]) if pcol in out.columns else np.nan
    return out


def add_evidence_rating(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()

    def sgn(x):
        if pd.isna(x) or not np.isfinite(x):
            return 0
        return 1 if x > 0 else (-1 if x < 0 else 0)

    ratings = []
    for _, r in out.iterrows():
        signs = [sgn(r.get("TMS_pearson_r", np.nan)), sgn(r.get("TMS_spearman_rho", np.nan)), sgn(r.get("TMS_ols_delta_beta", np.nan))]
        nonzero = [x for x in signs if x != 0]
        direction_consistency = "insufficient【不足】"
        if len(nonzero) >= 2:
            direction_consistency = "consistent_positive【一致正向】" if all(x > 0 for x in nonzero) else ("consistent_negative【一致负向】" if all(x < 0 for x in nonzero) else "mixed【方向混合】")
        abs_r = abs(r.get("TMS_pearson_r", np.nan)) if np.isfinite(r.get("TMS_pearson_r", np.nan)) else np.nan
        wl_abs_r = abs(r.get("WL_pearson_r", np.nan)) if np.isfinite(r.get("WL_pearson_r", np.nan)) else np.nan
        stronger_than_wl = bool(np.isfinite(abs_r) and (not np.isfinite(wl_abs_r) or abs_r > wl_abs_r))
        wl_not_nominal = not (np.isfinite(r.get("WL_pearson_p", np.nan)) and r.get("WL_pearson_p", 1) < 0.05)
        fdr_any = (
            (np.isfinite(r.get("TMS_pearson_q", np.nan)) and r.get("TMS_pearson_q") < 0.05) or
            (np.isfinite(r.get("TMS_spearman_q", np.nan)) and r.get("TMS_spearman_q") < 0.05) or
            (np.isfinite(r.get("TMS_ols_delta_q", np.nan)) and r.get("TMS_ols_delta_q") < 0.05)
        )
        nominal_any = (
            (np.isfinite(r.get("TMS_pearson_p", np.nan)) and r.get("TMS_pearson_p") < 0.05) or
            (np.isfinite(r.get("TMS_spearman_p", np.nan)) and r.get("TMS_spearman_p") < 0.05) or
            (np.isfinite(r.get("TMS_ols_delta_p", np.nan)) and r.get("TMS_ols_delta_p") < 0.05)
        )
        trend_any = (
            (np.isfinite(r.get("TMS_pearson_p", np.nan)) and r.get("TMS_pearson_p") < 0.10) or
            (np.isfinite(r.get("TMS_spearman_p", np.nan)) and r.get("TMS_spearman_p") < 0.10) or
            (np.isfinite(r.get("TMS_ols_delta_p", np.nan)) and r.get("TMS_ols_delta_p") < 0.10)
        )

        if fdr_any and direction_consistency.startswith("consistent") and stronger_than_wl:
            grade = "FDR-supported TMS process evidence【FDR支持的TMS过程证据】"
        elif nominal_any and np.isfinite(abs_r) and abs_r >= 0.30 and direction_consistency.startswith("consistent") and stronger_than_wl and wl_not_nominal:
            grade = "nominal directional TMS support【名义方向性TMS支持】"
        elif trend_any and np.isfinite(abs_r) and abs_r >= 0.25:
            grade = "trend-level exploratory support【趋势级探索性支持】"
        else:
            grade = "weak_or_no_TMS_process_evidence【弱或无TMS过程证据】"
        ratings.append({
            "direction_consistency_TMS": direction_consistency,
            "abs_TMS_pearson_r": abs_r,
            "stronger_than_WL_by_abs_r": stronger_than_wl,
            "WL_not_nominal": wl_not_nominal,
            "evidence_grade": grade,
        })
    ratings_df = pd.DataFrame(ratings)
    return pd.concat([out.reset_index(drop=True), ratings_df], axis=1)


def write_txt(path: Path, text: str) -> None:
    try:
        path.write_text(text, encoding="utf-8")
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path.with_name(path.stem + f"_{ts}" + path.suffix).write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="", help="44号 v2.3 输出文件夹。默认自动搜索当前目录下的44号结果文件夹。")
    ap.add_argument("--out_dir", default="", help="输出文件夹。默认在当前目录创建45_TMS偏向候选边_探索性纵向可塑性分析结果。")
    ap.add_argument("--min_n", type=int, default=8, help="最小有效样本量，默认8。")
    args = ap.parse_args()

    cwd = Path.cwd()
    log("=" * 100)
    log("45号 v2：value_column-only TMS偏向候选边_探索性纵向可塑性分析 启动")
    log(f"脚本版本：{SCRIPT_VERSION}")
    log(f"当前目录：{cwd}")

    input_dir = Path(args.input_dir) if args.input_dir else (auto_find_44_dir(cwd) or cwd)
    if not input_dir.exists():
        raise FileNotFoundError(f"输入文件夹不存在：{input_dir}")
    if not ("44_v3" in input_dir.name and "value_column" in input_dir.name):
        raise RuntimeError("45_v2 只允许读取 44_v3 value_column-only 结果目录，避免旧44结果继续传播错误标签。请先运行44_v3。")
    audit_v3 = input_dir / "00_output_legacy_label_text_audit_v3.csv"
    if not audit_v3.exists():
        raise RuntimeError("未找到 44_v3 旧标签审计文件 00_output_legacy_label_text_audit_v3.csv。请先完成44_v3并确认审计通过。")


    out_dir = Path(args.out_dir) if args.out_dir else cwd / "45_v2_value_column_only_TMS偏向候选边_探索性纵向可塑性分析结果_无旧标签正式版"
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"44_v3 value_column-only结果目录：{input_dir}")
    log(f"输出目录：{out_dir}")
    log("=" * 100)

    merged_file = find_first_file(input_dir, ["00d"])
    outcome_file = find_first_file(input_dir, ["01_"]) or find_first_file(input_dir, ["01"])
    edge_file = find_first_file(input_dir, ["02_"]) or find_first_file(input_dir, ["02"])
    module10_file = find_first_file(input_dir, ["10_"]) or find_first_file(input_dir, ["10"])
    if merged_file is None:
        # 更宽松搜索
        cands = list(input_dir.glob("*00d*.csv"))
        if cands:
            merged_file = cands[0]
    if merged_file is None:
        raise FileNotFoundError("未找到 00d_合并后数据预览 CSV。请先运行 44_v3 value_column-only，或用 --input_dir 指向 44_v3 结果文件夹。")

    log(f"读取合并数据：{merged_file}")
    merged = safe_read_csv(merged_file)
    log(f"合并数据维度：rows={len(merged)}, cols={len(merged.columns)}")

    outcome_df = safe_read_csv(outcome_file) if outcome_file and outcome_file.exists() else None
    edge_df = safe_read_csv(edge_file) if edge_file and edge_file.exists() else None
    module10 = safe_read_csv(module10_file) if module10_file and module10_file.exists() else None

    edge_meta = build_edge_meta(edge_df, merged)
    outcome_meta = build_outcome_meta(outcome_df, merged)
    tms_edges = detect_tms_favoring_edges(module10, edge_meta, DEFAULT_TMS_FAVORING_EDGES)
    core_outcomes = detect_core_outcomes(outcome_meta)

    if tms_edges.empty:
        raise RuntimeError("没有识别到TMS偏向EDGE边。请确认输入目录是 44_v3 value_column-only 结果。")
    if core_outcomes.empty:
        raise RuntimeError("没有识别到核心结局。请检查01_自动识别_临床结局指标.csv。")

    group_col = detect_group_col(merged)
    group_counts = {
        "TMS": int(group_mask(merged, group_col, "TMS").sum()),
        "PSY": int(group_mask(merged, group_col, "PSY").sum()),
        "WL": int(group_mask(merged, group_col, "WL").sum()),
    }
    log(f"分组列：{group_col}; 分组计数：{group_counts}")
    log(f"TMS偏向边数量：{len(tms_edges)}")
    log(f"全部结局数量：{len(outcome_meta)}；核心结局数量：{len(core_outcomes)}")

    # 保存识别文件
    tms_edges.to_csv(out_dir / "01_TMS偏向边识别与来源.csv", index=False, encoding="utf-8-sig")
    core_outcomes.to_csv(out_dir / "02_核心结局识别.csv", index=False, encoding="utf-8-sig")

    covariates = [c for c in ["age", "sex", "pre_meanFD"] if c in merged.columns]

    # 方案A：TMS偏向边 × 全部结局
    log("方案A启动：TMS偏向边 × 全部结局。")
    res_a = run_one_family(merged, tms_edges, outcome_meta, "A_TMS_favoring_edges_all_outcomes【方案A：TMS偏向边x全部结局】", covariates)
    res_a = add_evidence_rating(res_a)
    res_a.to_csv(out_dir / "10_方案A_TMS偏向边x全部结局_探索性deltaFC结果.csv", index=False, encoding="utf-8-sig")

    # 方案B：TMS偏向边 × 核心结局
    log("方案B启动：TMS偏向边 × 5个核心结局。")
    res_b = run_one_family(merged, tms_edges, core_outcomes, "B_TMS_favoring_edges_core_outcomes【方案B：TMS偏向边x核心结局】", covariates)
    res_b = add_evidence_rating(res_b)
    res_b.to_csv(out_dir / "20_方案B_TMS偏向边x核心结局_收缩检验族结果.csv", index=False, encoding="utf-8-sig")

    # 方案C：效应量、方向一致性、证据等级，基于方案B
    log("方案C启动：汇总效应量、方向一致性、WL负控和证据等级。")
    if not res_b.empty:
        rank = res_b.copy()
        # 排序：证据等级、q/p、效应量
        grade_order = {
            "FDR-supported TMS process evidence【FDR支持的TMS过程证据】": 0,
            "nominal directional TMS support【名义方向性TMS支持】": 1,
            "trend-level exploratory support【趋势级探索性支持】": 2,
            "weak_or_no_TMS_process_evidence【弱或无TMS过程证据】": 3,
        }
        rank["evidence_rank"] = rank["evidence_grade"].map(grade_order).fillna(9)
        rank = rank.sort_values([
            "evidence_rank", "TMS_ols_delta_q", "TMS_pearson_q", "TMS_pearson_p", "abs_TMS_pearson_r"
        ], ascending=[True, True, True, True, False])
    else:
        rank = res_b
    rank.to_csv(out_dir / "30_方案C_效应量方向一致性与证据等级.csv", index=False, encoding="utf-8-sig")

    writeable = rank[rank.get("evidence_grade", pd.Series(dtype=str)).astype(str).str.contains("FDR-supported|nominal directional|trend-level", na=False)].copy() if not rank.empty else rank
    writeable.to_csv(out_dir / "31_可写入论文或补充材料的候选结果.csv", index=False, encoding="utf-8-sig")

    audit = {
        "script_version": SCRIPT_VERSION,
        "input_dir": str(input_dir),
        "out_dir": str(out_dir),
        "merged_file": str(merged_file),
        "outcome_file": str(outcome_file) if outcome_file else None,
        "edge_file": str(edge_file) if edge_file else None,
        "module10_file": str(module10_file) if module10_file else None,
        "n_rows_merged": int(len(merged)),
        "group_col": group_col,
        "group_counts": group_counts,
        "n_tms_favoring_edges": int(len(tms_edges)),
        "n_all_outcomes": int(len(outcome_meta)),
        "n_core_outcomes": int(len(core_outcomes)),
        "covariates_used_if_available": covariates,
        "scheme_A_tests": int(len(res_a)),
        "scheme_B_tests": int(len(res_b)),
        "scheme_B_FDR_supported": int(rank["evidence_grade"].astype(str).str.contains("FDR-supported", na=False).sum()) if not rank.empty else 0,
        "scheme_B_nominal_directional": int(rank["evidence_grade"].astype(str).str.contains("nominal directional", na=False).sum()) if not rank.empty else 0,
        "scheme_B_trend_level": int(rank["evidence_grade"].astype(str).str.contains("trend-level", na=False).sum()) if not rank.empty else 0,
    }
    (out_dir / "00_输入审计.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    def count_grade(df: pd.DataFrame, keyword: str) -> int:
        if df is None or df.empty or "evidence_grade" not in df.columns:
            return 0
        return int(df["evidence_grade"].astype(str).str.contains(keyword, na=False).sum())

    top_lines = []
    if not rank.empty:
        for _, r in rank.head(8).iterrows():
            top_lines.append(
                f"- {r['edge_label']} × {r['outcome']}: "
                f"TMS Pearson r={r.get('TMS_pearson_r', np.nan):.3f}, p={r.get('TMS_pearson_p', np.nan):.4g}, "
                f"q={r.get('TMS_pearson_q', np.nan):.4g}; "
                f"OLS beta={r.get('TMS_ols_delta_beta', np.nan):.3f}, p={r.get('TMS_ols_delta_p', np.nan):.4g}, "
                f"grade={r.get('evidence_grade','')}"
            )

    readme = f"""45号 v2 value_column-only TMS偏向候选边探索性纵向可塑性分析：结果解读提示
============================================================

脚本版本：{SCRIPT_VERSION}
输入目录：{input_dir}
合并数据：{merged_file.name}
样本计数：{group_counts}

本脚本执行的三个方案：
1. 方案A：只分析 TMS-favoring edges【TMS偏向边】，但覆盖所有可识别临床结局。
2. 方案B：只分析 TMS-favoring edges【TMS偏向边】 × 5个核心结局，形成较小检验族。
3. 方案C：基于方案B报告 effect size【效应量】、方向一致性、WL负控对照和证据等级。

识别到的 TMS 偏向边数量：{len(tms_edges)}
识别到的全部结局数量：{len(outcome_meta)}
识别到的核心结局数量：{len(core_outcomes)}
方案A检验数：{len(res_a)}
方案B检验数：{len(res_b)}

方案B证据等级计数：
- FDR-supported TMS process evidence【FDR支持的TMS过程证据】：{count_grade(rank, 'FDR-supported')}
- nominal directional TMS support【名义方向性TMS支持】：{count_grade(rank, 'nominal directional')}
- trend-level exploratory support【趋势级探索性支持】：{count_grade(rank, 'trend-level')}
- weak/no evidence【弱或无证据】：{count_grade(rank, 'weak_or_no')}

优先查看文件：
- 20_方案B_TMS偏向边x核心结局_收缩检验族结果.csv
- 30_方案C_效应量方向一致性与证据等级.csv
- 31_可写入论文或补充材料的候选结果.csv

排序靠前的结果：
{os.linesep.join(top_lines) if top_lines else '- 无可排序结果。'}

写作建议：
- 如果出现 FDR-supported 结果，可以写为探索性但较强的 TMS process evidence【TMS治疗过程证据】。
- 如果只有 nominal directional / trend-level 结果，只能写为 exploratory directional support【探索性方向支持】，不能写成稳健机制证据。
- 如果大多数为 weak/no evidence，则说明 TMS 的主要证据仍应定位为 baseline treatment-selection signal【基线治疗选择信号】，而不是 deltaFC plasticity marker【纵向可塑性标志物】。
- 本脚本没有通过放宽 FDR 阈值来制造显著，而是通过预先限定 TMS偏向边和核心结局来减少检验族。
"""
    write_txt(out_dir / "99_结果解读提示.txt", readme)

    scan_outputs_for_legacy_labels(out_dir)

    log("=" * 100)
    log("45号脚本运行完成")
    log(f"请优先查看：{out_dir / '99_结果解读提示.txt'}")
    log("=" * 100)


if __name__ == "__main__":
    main()
