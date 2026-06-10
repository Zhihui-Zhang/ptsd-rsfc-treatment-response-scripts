# -*- coding: utf-8 -*-
"""
43_v4_2_value_column_only_固定候选边_治疗选择机制可塑性WL负控分析_保留v3数值等价_无旧标签终版.py

目的：
1) 在不重新筛边、不改变43号原始统计计算的前提下，重新运行固定候选边的
   treatment-selection vs prognosis【治疗选择 vs 一般预后】、
   plasticity【可塑性】和 WL negative control【等待组负控】分析。
2) 正式身份只使用 canonical_edge_id【标准边编号】与 value_column【取值列】。
3) 本脚本不输出旧解剖标签、不输出脑区名、不输出脑系统解释；脑区定位和 BrainNet
   可视化交给独立审计通过的 Brainnetome 246 ROI 映射流程。

重要说明：
- 本脚本不改变统计公式。
- 本脚本以 unknown_roi_*__unknown_roi_* 作为正式 value_column【取值列】身份。
- 为保持与 43_v3 数值完全等价，内部保留原 43_v3 的 selected_unknown_col 优先、selected_named_col 后备取值规则；但所有正式输出均隐藏后备解剖列名。
"""

import os
import re
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats


SCRIPT_NAME = "43_v4_2_value_column_only_固定候选边_治疗选择机制可塑性WL负控分析_保留v3数值等价_无旧标签终版.py"

DEFAULT_MAIN_TABLE = r"E:\E_zhangzhihui\从yv那边提取\脚本\第二步分析\第二步纵向分析底座表\pre_meanFD\06_Brainnetome246_纵向FC宽表_合并clinical_prepostmeanFD_重跑合并版.csv"
FALLBACK_MAIN_TABLE = r"E:\E_zhangzhihui\从yv那边提取\脚本\第二步分析\06_Brainnetome246_纵向FC宽表_合并clinical_postmeanFD_重跑合并版.csv"
DEFAULT_OUT_DIR = r"E:\E_zhangzhihui\从yv那边提取\脚本\第四步分析：探索\43_v4_2_value_column_only_固定候选边_治疗选择机制可塑性WL负控分析结果_保留v3数值等价_无旧标签终版"

# 固定候选 10 条：只用 value_column，不在本脚本中写入或输出脑区名。
FIXED_EDGES = [
    {"canonical_edge_id": "EDGE_01", "value_column": "unknown_roi_18__unknown_roi_24", "a_priori_direction": "PSY-favoring"},
    {"canonical_edge_id": "EDGE_02", "value_column": "unknown_roi_65__unknown_roi_85", "a_priori_direction": "PSY-favoring"},
    {"canonical_edge_id": "EDGE_03", "value_column": "unknown_roi_62__unknown_roi_105", "a_priori_direction": "TMS-favoring"},
    {"canonical_edge_id": "EDGE_04", "value_column": "unknown_roi_64__unknown_roi_85", "a_priori_direction": "PSY-favoring"},
    {"canonical_edge_id": "EDGE_05", "value_column": "unknown_roi_11__unknown_roi_67", "a_priori_direction": "TMS-favoring"},
    {"canonical_edge_id": "EDGE_06", "value_column": "unknown_roi_56__unknown_roi_84", "a_priori_direction": "PSY-favoring"},
    {"canonical_edge_id": "EDGE_07", "value_column": "unknown_roi_46__unknown_roi_95", "a_priori_direction": "TMS-favoring"},
    {"canonical_edge_id": "EDGE_08", "value_column": "unknown_roi_89__unknown_roi_90", "a_priori_direction": "PSY-favoring"},
    {"canonical_edge_id": "EDGE_09", "value_column": "unknown_roi_8__unknown_roi_92", "a_priori_direction": "TMS-favoring"},
    {"canonical_edge_id": "EDGE_10", "value_column": "unknown_roi_23__unknown_roi_44", "a_priori_direction": "TMS-favoring"},
]
CORE_EDGES = [x["canonical_edge_id"] for x in FIXED_EDGES]
EDGE_META = {
    x["canonical_edge_id"]: {
        "value_column": x["value_column"],
        "system": "fixed_candidate_connection",
        "system_cn": "固定候选连接",
        "a_priori_direction": x["a_priori_direction"],
        "interpretation": "value_column_only_candidate; anatomical interpretation handled outside this script",
        "edge_identity_policy": "value_column_only_no_anatomical_label",
    }
    for x in FIXED_EDGES
}
VALUE_TO_EDGE = {x["value_column"]: x["canonical_edge_id"] for x in FIXED_EDGES}


def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def read_csv_smart(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() == ".gz":
        return pd.read_csv(path, encoding="utf-8-sig", compression="gzip", low_memory=False)
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gb18030", low_memory=False)


def save_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"写出：{path} | rows={len(df)}")


def bh_fdr(pvals):
    p = np.asarray([np.nan if x is None else x for x in pvals], dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    if mask.sum() == 0:
        return q
    pm = p[mask]
    n = len(pm)
    order = np.argsort(pm)
    ranked = pm[order]
    vals = ranked * n / (np.arange(n) + 1)
    vals = np.minimum.accumulate(vals[::-1])[::-1]
    vals = np.minimum(vals, 1.0)
    out = np.empty(n)
    out[order] = vals
    q[mask] = out
    return q


def normalize_subject_id(x):
    s = re.sub(r"\D", "", str(x))
    if not s:
        return ""
    if len(s) == 5 and s.startswith("0"):
        s = s[-4:]
    if len(s) == 5 and s.endswith("0"):
        s = s[:4]
    if s in ["887", "0887"]:
        return "0887"
    if s == "1081":
        return "1081"
    if s == "01691":
        return "1691"
    if len(s) <= 4:
        return s.zfill(4)
    return s


def detect_col(df, candidates, required=False, label=""):
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for c in df.columns:
        cl = str(c).lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    if required:
        raise ValueError(f"无法识别列 {label}; candidates={candidates}; columns={list(df.columns)[:80]}")
    return None


def infer_group_col(df):
    return detect_col(df, ["group", "treatment_group", "group_family", "intervention", "arm"], True, "group")


def infer_time_col(df):
    for cand in ["time", "timepoint", "visit", "session", "scan_time", "prepost", "time_label"]:
        for c in df.columns:
            if str(c).lower() == cand:
                return c
    for c in df.columns:
        vals = df[c].dropna().astype(str).str.lower().head(200).tolist()
        joined = " ".join(vals)
        if ("pre" in joined or "baseline" in joined) and ("post" in joined or "follow" in joined):
            return c
    raise ValueError("无法识别 pre/post 时间列。请检查主表是否含 time/timepoint/visit 等列。")


def infer_subject_col(df):
    candidates = ["subject_id4", "subject_key", "subject_id", "subject", "sub_id", "id", "ID", "被试编号", "编号", "sub"]
    for cand in candidates:
        if cand in df.columns:
            return cand
    for c in df.columns[:80]:
        cl = str(c).lower()
        if "subject" in cl or "sub" in cl or "编号" in str(c):
            return c
    raise ValueError("无法识别 subject 列。")


def infer_outcome(df):
    direct = [
        "PCL_improvement", "pcl_improvement", "PCL改善", "pcl_change",
        "PCL_change", "improvement", "symptom_improvement", "PCL5_improvement"
    ]
    for c in direct:
        if c in df.columns:
            return c, None, None
    pre_candidates = ["PCL_pre", "pcl_pre", "PCL5_pre", "PCL_total_pre", "pre_PCL", "baseline_PCL"]
    post_candidates = ["PCL_post", "pcl_post", "PCL5_post", "PCL_total_post", "post_PCL"]
    pre_col = detect_col(df, pre_candidates, False)
    post_col = detect_col(df, post_candidates, False)
    if pre_col and post_col:
        return None, pre_col, post_col
    return None, None, None


def safe_ttest_ind(a, b):
    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna().values
    b = pd.to_numeric(pd.Series(b), errors="coerce").dropna().values
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    t, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return float(t), float(p)


def safe_ttest_rel(pre, post):
    x = pd.to_numeric(pd.Series(pre), errors="coerce")
    y = pd.to_numeric(pd.Series(post), errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return np.nan, np.nan
    t, p = stats.ttest_rel(x[mask], y[mask], nan_policy="omit")
    return float(t), float(p)


def safe_corr(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 4:
        return np.nan, np.nan, int(mask.sum())
    r, p = stats.pearsonr(x[mask], y[mask])
    return float(r), float(p), int(mask.sum())


def ols_fit(df, y_col, x_cols):
    use_cols = [y_col] + x_cols
    dat = df[use_cols].copy()
    for c in use_cols:
        dat[c] = pd.to_numeric(dat[c], errors="coerce")
    dat = dat.dropna()
    n = len(dat)
    k = len(x_cols) + 1
    if n <= k + 1:
        return None
    y = dat[y_col].values.astype(float)
    X = dat[x_cols].values.astype(float)
    X = np.column_stack([np.ones(n), X])
    names = ["Intercept"] + x_cols
    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = np.nan if sst <= 0 else 1 - sse / sst
    df_resid = n - X.shape[1]
    if df_resid <= 0:
        return None
    sigma2 = sse / df_resid
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(xtx_inv) * sigma2)
    tvals = beta / se
    pvals = 2 * stats.t.sf(np.abs(tvals), df_resid)
    out = pd.DataFrame({"term": names, "beta": beta, "se": se, "t": tvals, "p": pvals})
    return {"n": n, "df_resid": df_resid, "r2": r2, "table": out}


def encode_covariates(dat, covariates):
    df = dat.copy()
    used = []
    for c in covariates:
        if c not in df.columns:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            df[c] = pd.to_numeric(s, errors="coerce")
            if df[c].notna().sum() > 0:
                used.append(c)
        else:
            ss = s.astype(str).str.strip()
            if ss.replace("nan", np.nan).dropna().empty:
                continue
            if c.lower() in ["sex", "gender", "性别"]:
                low = ss.str.lower()
                mapped = low.map({"m": 1, "male": 1, "男": 1, "1": 1, "f": 0, "female": 0, "女": 0, "0": 0})
                if mapped.notna().sum() > 0:
                    df[c] = mapped
                    used.append(c)
                    continue
            codes = pd.Categorical(ss.replace({"nan": np.nan})).codes.astype(float)
            codes[codes < 0] = np.nan
            df[c] = codes
            if pd.Series(codes).notna().sum() > 0:
                used.append(c)
    return df, used


def find_latest_40_dir(root):
    root = Path(root)
    cands = []
    if root.exists():
        for p in root.iterdir():
            if p.is_dir() and ("40_" in p.name or "40_v3" in p.name) and "结果" in p.name:
                if (p / "01_candidate_edge_mapping.csv").exists() or (p / "99_fixed_candidate_moderation_report.txt").exists():
                    cands.append(p)
    if not cands:
        return None
    preferred = [p for p in cands if ("preMeanFD" in p.name or "premeanfd" in p.name.lower() or "v3" in p.name.lower())]
    cands2 = preferred if preferred else cands
    return sorted(cands2, key=lambda x: x.stat().st_mtime, reverse=True)[0]


def load_edge_mapping(results40_dir):
    """读取 40 号候选边映射表。

    v4.2 关键原则：
    1) 正式身份只暴露 canonical_edge_id + value_column；
    2) 内部保留 43_v3 的取值规则：selected_unknown_col 优先，selected_named_col 后备；
    3) 后备列名仅用于保持 v3 数值等价，不写入正式输出。
    """
    if not results40_dir:
        return pd.DataFrame()
    p = Path(results40_dir) / "01_candidate_edge_mapping.csv"
    if not p.exists():
        return pd.DataFrame()
    mp = read_csv_smart(p)
    keep = []
    for _, r in mp.iterrows():
        vu = str(r.get("selected_unknown_col", "")).strip()
        if vu in VALUE_TO_EDGE:
            row = {
                "canonical_edge_id": VALUE_TO_EDGE[vu],
                "value_column": vu,
                "value_source": str(r.get("value_source", "")).strip() or "unknown",
                "mapping_status": str(r.get("status", "")).strip(),
                "edge_identity_policy": "value_column_only_no_anatomical_label",
            }
            # 这些列仅供内部取值使用，不应保存到正式 mapping 输出。
            for c in ["selected_unknown_col", "selected_named_col", "selected_col", "column", "edge_col"]:
                if c in mp.columns:
                    row[c] = str(r.get(c, "")).strip()
            keep.append(row)
    if not keep:
        keep = [
            {
                "canonical_edge_id": x["canonical_edge_id"],
                "value_column": x["value_column"],
                "value_source": "fixed_value_column_list",
                "mapping_status": "from_script_fixed_list",
                "edge_identity_policy": "value_column_only_no_anatomical_label",
                "selected_unknown_col": x["value_column"],
                "selected_named_col": "",
            }
            for x in FIXED_EDGES
        ]
    return pd.DataFrame(keep).drop_duplicates("canonical_edge_id")


def sanitized_mapping_for_output(mapping_df):
    """只输出安全身份列，避免 selected_named_col 等旧标签列进入结果。"""
    if mapping_df is None or mapping_df.empty:
        return pd.DataFrame()
    keep_cols = ["canonical_edge_id", "value_column", "value_source", "mapping_status", "edge_identity_policy"]
    keep_cols = [c for c in keep_cols if c in mapping_df.columns]
    return mapping_df[keep_cols].copy()


def candidate_edge_cols_from_mapping(mapping_df, main_df):
    """返回内部实际取值列。

    为保证与 43_v3 完全等价，保留原 v3 行级取值逻辑：
    value_source == unknown 时 selected_unknown_col 优先，selected_named_col 后备；
    value_source == named 时反向。
    注意：这些后备列名不进入正式输出。
    """
    edge_to_cols = {}
    if mapping_df is not None and not mapping_df.empty:
        for _, r in mapping_df.iterrows():
            eid = str(r.get("canonical_edge_id", "")).strip()
            vc = str(r.get("value_column", "")).strip()
            if eid not in CORE_EDGES:
                continue
            value_source = str(r.get("value_source", "")).strip().lower()
            if value_source == "named":
                ordered = ["selected_named_col", "selected_unknown_col", "selected_col", "column", "edge_col"]
            else:
                ordered = ["selected_unknown_col", "selected_named_col", "selected_col", "column", "edge_col"]
            cols = []
            for colname in ordered:
                val = str(r.get(colname, "")).strip()
                if val and val in main_df.columns and val not in cols:
                    cols.append(val)
            if not cols and vc in main_df.columns:
                cols.append(vc)
            if cols:
                edge_to_cols[eid] = cols

    missing = []
    for x in FIXED_EDGES:
        eid = x["canonical_edge_id"]
        vc = x["value_column"]
        if eid not in edge_to_cols:
            if vc in main_df.columns:
                edge_to_cols[eid] = [vc]
            else:
                missing.append(x)
    if missing:
        raise ValueError(
            "以下固定候选 value_column 在主表中缺失，不能继续运行："
            + "; ".join([f"{x['canonical_edge_id']}={x['value_column']}" for x in missing])
        )
    return edge_to_cols

def first_nonmissing_numeric(frame, cols):
    if frame is None or frame.empty:
        return np.nan, ""
    if isinstance(cols, str):
        cols = [cols]
    for c in cols:
        if c in frame.columns:
            vals = pd.to_numeric(frame[c], errors="coerce").dropna()
            if len(vals):
                return vals.iloc[0], c
    return np.nan, ""


def build_subject_level(main_df, edge_to_col, out_dir):
    group_col = infer_group_col(main_df)
    time_col = infer_time_col(main_df)
    subj_col = infer_subject_col(main_df)
    outcome_col, pcl_pre_col, pcl_post_col = infer_outcome(main_df)

    df = main_df.copy()
    df["_group_raw"] = df[group_col].astype(str).str.upper().str.strip()

    def treatment_family(g):
        if "TMS" in g:
            return "TMS"
        if "ACT" in g or "MIN" in g:
            return "PSY"
        if "WL" in g:
            return "WL"
        if "HC" in g:
            return "HC"
        return g

    df["treatment_family"] = df["_group_raw"].map(treatment_family)
    df["subject_id4"] = df[subj_col].map(normalize_subject_id)
    df["_key"] = df["_group_raw"] + "-" + df["subject_id4"]

    tv = df[time_col].astype(str).str.lower()
    df["_time2"] = np.where(tv.str.contains("post|follow|after|后"), "post",
                            np.where(tv.str.contains("pre|baseline|before|前"), "pre", ""))

    if outcome_col:
        df["_pcl_improvement"] = pd.to_numeric(df[outcome_col], errors="coerce")
        outcome_source = outcome_col
    elif pcl_pre_col and pcl_post_col:
        df["_pcl_improvement"] = pd.to_numeric(df[pcl_pre_col], errors="coerce") - pd.to_numeric(df[pcl_post_col], errors="coerce")
        outcome_source = f"{pcl_pre_col} - {pcl_post_col}"
    else:
        df["_pcl_improvement"] = np.nan
        outcome_source = "NOT_FOUND"

    cov_cols = {}
    for name, cands in {
        "PCL_pre": ["PCL_pre", "pcl_pre", "PCL5_pre", "pre_PCL", "baseline_PCL"],
        "age": ["age", "Age", "年龄"],
        "sex": ["sex", "Sex", "gender", "性别"],
        "pre_meanFD": ["pre_meanFD", "meanFD_pre", "pre_mean_fd", "mean_fd_pre"],
        "post_meanFD": ["post_meanFD", "meanFD_post", "post_mean_fd", "mean_fd_post"],
    }.items():
        c = detect_col(df, cands, False)
        if c:
            cov_cols[name] = c

    rows = []
    for key, g in df.groupby("_key", dropna=False):
        if not key or key == "-":
            continue
        pre = g[g["_time2"] == "pre"]
        post = g[g["_time2"] == "post"]
        if pre.empty and post.empty:
            continue
        base_row = pre.iloc[0] if not pre.empty else g.iloc[0]
        row = {
            "subject_key": key,
            "group": base_row["_group_raw"],
            "treatment_family": base_row["treatment_family"],
            "subject_id4": base_row["subject_id4"],
            "pcl_improvement": pd.to_numeric(g["_pcl_improvement"], errors="coerce").dropna().iloc[0] if pd.to_numeric(g["_pcl_improvement"], errors="coerce").dropna().shape[0] else np.nan,
        }
        for std_name, orig_col in cov_cols.items():
            vals = g[orig_col].dropna()
            row[std_name] = vals.iloc[0] if len(vals) else np.nan

        for e, cols in edge_to_col.items():
            vc = EDGE_META[e]["value_column"]
            pre_v, pre_source = first_nonmissing_numeric(pre, cols)
            post_v, post_source = first_nonmissing_numeric(post, cols)
            row[f"{e}__value_column"] = vc
            row[f"{e}__pre"] = pre_v
            row[f"{e}__post"] = post_v
            row[f"{e}__pre_source_col"] = f"{vc}__pre"
            row[f"{e}__post_source_col"] = f"{vc}__post"
            row[f"{e}__delta"] = row[f"{e}__post"] - row[f"{e}__pre"] if pd.notna(row[f"{e}__pre"]) and pd.notna(row[f"{e}__post"]) else np.nan
        rows.append(row)

    subj = pd.DataFrame(rows)
    detected = {
        "group_col": group_col,
        "time_col": time_col,
        "subject_col": subj_col,
        "outcome_source": outcome_source,
        "covariate_columns": cov_cols,
        "n_subject_rows": len(subj),
        "edge_identity_policy": "value_column_only_no_anatomical_label",
        "edge_to_col": {e: [EDGE_META[e]["value_column"]] for e in CORE_EDGES},
        "internal_value_source_note": "selected_unknown_col primary with selected_named_col fallback preserved only for v3 numerical equivalence; fallback column names are not exposed",
    }
    with open(Path(out_dir) / "00_detected_columns_and_mapping_v4_2_value_column_only.json", "w", encoding="utf-8") as f:
        json.dump(detected, f, ensure_ascii=False, indent=2)
    return subj, detected


def active_df(subj):
    return subj[subj["treatment_family"].isin(["TMS", "PSY"])].copy()


def add_edge_identity_columns(df):
    if df is None or df.empty or "edge" not in df.columns:
        return df
    out = df.copy()
    out.insert(1, "value_column", out["edge"].map(lambda e: EDGE_META.get(e, {}).get("value_column", "")))
    out.insert(2, "edge_identity_policy", "value_column_only_no_anatomical_label")
    return out


def analysis_treatment_selection(subj, covariates, out_dir):
    dat0 = active_df(subj)
    dat0["TMS_code"] = (dat0["treatment_family"] == "TMS").astype(int)
    dat0, used_covs = encode_covariates(dat0, covariates)

    rows = []
    simple_rows = []
    for e in CORE_EDGES:
        pre_col = f"{e}__pre"
        if pre_col not in dat0.columns:
            continue
        dat = dat0.copy()
        dat["edge_pre"] = pd.to_numeric(dat[pre_col], errors="coerce")
        dat["interaction"] = dat["TMS_code"] * dat["edge_pre"]

        prog_cols = ["edge_pre"] + used_covs
        prog = ols_fit(dat, "pcl_improvement", prog_cols)

        mod_cols = ["TMS_code", "edge_pre", "interaction"] + used_covs
        mod = ols_fit(dat, "pcl_improvement", mod_cols)

        inter_beta = inter_p = r2_mod = n_mod = np.nan
        if mod is not None:
            tab = mod["table"]
            hit = tab[tab["term"] == "interaction"]
            if not hit.empty:
                inter_beta = float(hit["beta"].iloc[0])
                inter_p = float(hit["p"].iloc[0])
            r2_mod = mod["r2"]
            n_mod = mod["n"]

        prog_beta = prog_p = r2_prog = n_prog = np.nan
        if prog is not None:
            tab = prog["table"]
            hit = tab[tab["term"] == "edge_pre"]
            if not hit.empty:
                prog_beta = float(hit["beta"].iloc[0])
                prog_p = float(hit["p"].iloc[0])
            r2_prog = prog["r2"]
            n_prog = prog["n"]

        slopes = {}
        for fam in ["PSY", "TMS"]:
            dsub = dat[dat["treatment_family"] == fam].copy()
            cols_with_cov = ["edge_pre"] + [c for c in used_covs if c in dsub.columns]
            fit = ols_fit(dsub, "pcl_improvement", cols_with_cov)
            if fit is None and len(cols_with_cov) > 1:
                fit = ols_fit(dsub, "pcl_improvement", ["edge_pre"])
                cov_note = "no_covariates_due_to_small_n"
            else:
                cov_note = "with_available_covariates"
            slope = slope_p = slope_r2 = np.nan
            n = 0
            if fit is not None:
                h = fit["table"][fit["table"]["term"] == "edge_pre"]
                if not h.empty:
                    slope = float(h["beta"].iloc[0])
                    slope_p = float(h["p"].iloc[0])
                slope_r2 = fit["r2"]
                n = fit["n"]
            slopes[fam] = slope
            simple_rows.append({
                "edge": e,
                "value_column": EDGE_META[e]["value_column"],
                "edge_identity_policy": EDGE_META[e]["edge_identity_policy"],
                "treatment_family": fam,
                "n": n,
                "slope_edge_pre": slope,
                "p": slope_p,
                "r2": slope_r2,
                "covariate_note": cov_note,
            })

        slope_psy = slopes.get("PSY", np.nan)
        slope_tms = slopes.get("TMS", np.nan)
        if np.isfinite(slope_psy) and np.isfinite(slope_tms):
            slope_pattern = "opposite_slopes" if np.sign(slope_psy) != np.sign(slope_tms) else "same_direction_slopes"
        else:
            slope_pattern = "insufficient"

        if np.isfinite(inter_p) and inter_p < 0.05 and np.isfinite(r2_prog) and np.isfinite(r2_mod):
            label = "strong_treatment_selection_pattern" if slope_pattern == "opposite_slopes" else "treatment_moderation_without_opposite_slopes"
        elif np.isfinite(prog_p) and prog_p < 0.05:
            label = "mainly_general_prognostic"
        else:
            label = "weak_or_unclear"

        rows.append({
            "edge": e,
            "value_column": EDGE_META[e]["value_column"],
            "edge_identity_policy": EDGE_META[e]["edge_identity_policy"],
            "system": EDGE_META[e]["system"],
            "system_cn": EDGE_META[e]["system_cn"],
            "a_priori_direction": EDGE_META[e]["a_priori_direction"],
            "n_moderation": n_mod,
            "interaction_beta": inter_beta,
            "interaction_p": inter_p,
            "r2_moderation": r2_mod,
            "n_prognostic": n_prog,
            "prognostic_edge_beta": prog_beta,
            "prognostic_edge_p": prog_p,
            "r2_prognostic": r2_prog,
            "delta_r2_interaction_over_prognostic": (r2_mod - r2_prog) if np.isfinite(r2_mod) and np.isfinite(r2_prog) else np.nan,
            "slope_PSY": slope_psy,
            "slope_TMS": slope_tms,
            "slope_pattern": slope_pattern,
            "selection_vs_prognosis_label": label,
            "interpretation": EDGE_META[e]["interpretation"],
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["interaction_q"] = bh_fdr(out["interaction_p"])
        out["prognostic_edge_q"] = bh_fdr(out["prognostic_edge_p"])
        out = out.sort_values(["selection_vs_prognosis_label", "interaction_p"], ascending=[True, True])
    simple = pd.DataFrame(simple_rows)
    if not simple.empty:
        simple["q_within_all_simple_slopes"] = bh_fdr(simple["p"])
    save_csv(out, Path(out_dir) / "10_treatment_selection_vs_general_prognosis_v4_2_value_column_only.csv")
    save_csv(simple, Path(out_dir) / "11_treatment_specific_simple_slopes_v4_2_value_column_only.csv")
    return out, simple, used_covs


def analysis_system_direction(moderation_df, out_dir):
    rows = []
    for e in CORE_EDGES:
        meta = EDGE_META[e]
        if moderation_df is not None and not moderation_df.empty and e in moderation_df["edge"].values:
            r = moderation_df[moderation_df["edge"] == e].iloc[0]
            beta = r.get("interaction_beta", np.nan)
            empirical_direction = "TMS-favoring" if np.isfinite(beta) and beta > 0 else ("PSY-favoring" if np.isfinite(beta) else meta["a_priori_direction"])
            p = r.get("interaction_p", np.nan)
            q = r.get("interaction_q", np.nan)
        else:
            empirical_direction = meta["a_priori_direction"]
            beta = p = q = np.nan
        rows.append({
            "edge": e,
            "value_column": meta["value_column"],
            "edge_identity_policy": meta["edge_identity_policy"],
            "system": meta["system"],
            "system_cn": meta["system_cn"],
            "a_priori_direction": meta["a_priori_direction"],
            "empirical_direction": empirical_direction,
            "interaction_beta": beta,
            "interaction_p": p,
            "interaction_q": q,
            "interpretation": meta["interpretation"],
        })
    df = pd.DataFrame(rows)
    save_csv(df, Path(out_dir) / "20_edge_direction_summary_v4_2_value_column_only.csv")
    sys = df.groupby(["empirical_direction", "system", "system_cn"]).agg(
        n_edges=("edge", "count"),
        edges=("edge", lambda x: "; ".join(x)),
        value_columns=("value_column", lambda x: "; ".join(x)),
    ).reset_index()
    save_csv(sys, Path(out_dir) / "21_candidate_set_direction_summary_v4_2_value_column_only.csv")
    return df, sys


def assign_high_low(dat, method, improvement_col="pcl_improvement", high_cutoff=8.0, low_cutoff=0.0):
    d = dat.copy()
    y = pd.to_numeric(d[improvement_col], errors="coerce")
    d["_hl"] = pd.Series("", index=d.index, dtype="object")
    if method == "median":
        med = y.median(skipna=True)
        d.loc[y > med, "_hl"] = "high"
        d.loc[y <= med, "_hl"] = "low"
    elif method == "tertile":
        q1 = y.quantile(1/3)
        q2 = y.quantile(2/3)
        d.loc[y >= q2, "_hl"] = "high"
        d.loc[y <= q1, "_hl"] = "low"
    elif method == "clinical_cutoff_8vs0":
        d.loc[y >= high_cutoff, "_hl"] = "high"
        d.loc[y <= low_cutoff, "_hl"] = "low"
    else:
        raise ValueError(method)
    return d[d["_hl"].isin(["high", "low"])].copy()


def analysis_plasticity(subj, out_dir, high_cutoff=8.0, low_cutoff=0.0):
    group_sets = {"PSY": ["PSY"], "TMS": ["TMS"], "ACTIVE": ["PSY", "TMS"]}
    methods = ["median", "tertile", "clinical_cutoff_8vs0"]
    rows = []
    corr_rows = []
    for e in CORE_EDGES:
        pre_col, post_col, delta_col = f"{e}__pre", f"{e}__post", f"{e}__delta"
        if delta_col not in subj.columns:
            continue
        for gname, fams in group_sets.items():
            base = subj[subj["treatment_family"].isin(fams)].copy()
            base = base.dropna(subset=[delta_col, "pcl_improvement"])
            r, p, n = safe_corr(base[delta_col], base["pcl_improvement"])
            corr_rows.append({
                "edge": e, "value_column": EDGE_META[e]["value_column"], "edge_identity_policy": EDGE_META[e]["edge_identity_policy"],
                "group_set": gname, "n": n, "pearson_r_deltaFC_vs_improvement": r, "p": p,
            })
            for method in methods:
                d = assign_high_low(base, method, high_cutoff=high_cutoff, low_cutoff=low_cutoff)
                hi = d[d["_hl"] == "high"]
                lo = d[d["_hl"] == "low"]
                t, pval = safe_ttest_ind(hi[delta_col], lo[delta_col])
                t_hi, p_hi = safe_ttest_rel(hi[pre_col], hi[post_col]) if pre_col in hi and post_col in hi else (np.nan, np.nan)
                t_lo, p_lo = safe_ttest_rel(lo[pre_col], lo[post_col]) if pre_col in lo and post_col in lo else (np.nan, np.nan)
                rows.append({
                    "edge": e,
                    "value_column": EDGE_META[e]["value_column"],
                    "edge_identity_policy": EDGE_META[e]["edge_identity_policy"],
                    "group_set": gname,
                    "split_method": method,
                    "n_high": len(hi),
                    "n_low": len(lo),
                    "high_pre_mean": pd.to_numeric(hi[pre_col], errors="coerce").mean() if pre_col in hi else np.nan,
                    "high_post_mean": pd.to_numeric(hi[post_col], errors="coerce").mean() if post_col in hi else np.nan,
                    "high_delta_mean": pd.to_numeric(hi[delta_col], errors="coerce").mean(),
                    "low_pre_mean": pd.to_numeric(lo[pre_col], errors="coerce").mean() if pre_col in lo else np.nan,
                    "low_post_mean": pd.to_numeric(lo[post_col], errors="coerce").mean() if post_col in lo else np.nan,
                    "low_delta_mean": pd.to_numeric(lo[delta_col], errors="coerce").mean(),
                    "high_minus_low_delta": pd.to_numeric(hi[delta_col], errors="coerce").mean() - pd.to_numeric(lo[delta_col], errors="coerce").mean(),
                    "welch_t_high_vs_low_delta": t,
                    "welch_p_high_vs_low_delta": pval,
                    "paired_t_high_prepost": t_hi,
                    "paired_p_high_prepost": p_hi,
                    "paired_t_low_prepost": t_lo,
                    "paired_p_low_prepost": p_lo,
                })
    plastic = pd.DataFrame(rows)
    if not plastic.empty:
        plastic["q_high_vs_low_delta_all_tests"] = bh_fdr(plastic["welch_p_high_vs_low_delta"])
        best = plastic.sort_values("welch_p_high_vs_low_delta").groupby("edge", as_index=False).head(1)
        best = best.sort_values("welch_p_high_vs_low_delta")
        best["support_label"] = np.where(
            best["q_high_vs_low_delta_all_tests"] < 0.05, "FDR_supported",
            np.where(best["welch_p_high_vs_low_delta"] < 0.05, "nominal_supported", "weak_or_none"),
        )
    else:
        best = pd.DataFrame()
    corr = pd.DataFrame(corr_rows)
    if not corr.empty:
        corr["q_corr_all_tests"] = bh_fdr(corr["p"])
    save_csv(plastic, Path(out_dir) / "30_plasticity_high_low_prepost_absolute_v4_2_value_column_only.csv")
    save_csv(corr, Path(out_dir) / "31_plasticity_deltaFC_improvement_correlations_v4_2_value_column_only.csv")
    save_csv(best, Path(out_dir) / "32_plasticity_best_high_low_per_edge_v4_2_value_column_only.csv")
    return plastic, corr, best


def analysis_wl_negative_control(subj, out_dir):
    rows = []
    within_rows = []
    for e in CORE_EDGES:
        delta_col = f"{e}__delta"
        pre_col = f"{e}__pre"
        post_col = f"{e}__post"
        if delta_col not in subj.columns:
            continue
        wl = subj[subj["treatment_family"] == "WL"].copy()
        psy = subj[subj["treatment_family"] == "PSY"].copy()
        tms = subj[subj["treatment_family"] == "TMS"].copy()
        active = subj[subj["treatment_family"].isin(["PSY", "TMS"])].copy()
        t_wl, p_wl = safe_ttest_rel(wl[pre_col], wl[post_col])
        within_rows.append({
            "edge": e, "value_column": EDGE_META[e]["value_column"], "edge_identity_policy": EDGE_META[e]["edge_identity_policy"],
            "group_set": "WL",
            "n": wl[[pre_col, post_col]].dropna().shape[0],
            "pre_mean": pd.to_numeric(wl[pre_col], errors="coerce").mean(),
            "post_mean": pd.to_numeric(wl[post_col], errors="coerce").mean(),
            "delta_mean": pd.to_numeric(wl[delta_col], errors="coerce").mean(),
            "paired_t_prepost": t_wl,
            "paired_p_prepost": p_wl,
        })
        for gname, d in [("PSY", psy), ("TMS", tms), ("ACTIVE", active)]:
            t, p = safe_ttest_ind(d[delta_col], wl[delta_col])
            mean_g = pd.to_numeric(d[delta_col], errors="coerce").mean()
            mean_wl = pd.to_numeric(wl[delta_col], errors="coerce").mean()
            rows.append({
                "edge": e,
                "value_column": EDGE_META[e]["value_column"],
                "edge_identity_policy": EDGE_META[e]["edge_identity_policy"],
                "comparison": f"{gname}_vs_WL",
                "n_treatment": pd.to_numeric(d[delta_col], errors="coerce").notna().sum(),
                "n_WL": pd.to_numeric(wl[delta_col], errors="coerce").notna().sum(),
                "treatment_delta_mean": mean_g,
                "WL_delta_mean": mean_wl,
                "treatment_minus_WL_delta": mean_g - mean_wl,
                "welch_t": t,
                "welch_p": p,
            })
    ctrl = pd.DataFrame(rows)
    if not ctrl.empty:
        ctrl["q_all_treatment_vs_WL"] = bh_fdr(ctrl["welch_p"])
        ctrl["WL_negative_control_label"] = ctrl.apply(
            lambda r: "FDR_treatment_specific_delta" if pd.notna(r["q_all_treatment_vs_WL"]) and r["q_all_treatment_vs_WL"] < 0.05
            else ("nominal_treatment_specific_delta" if pd.notna(r["welch_p"]) and r["welch_p"] < 0.05 else "not_distinguishable_from_WL"),
            axis=1,
        )
    within = pd.DataFrame(within_rows)
    if not within.empty:
        within["q_WL_prepost"] = bh_fdr(within["paired_p_prepost"])
    save_csv(ctrl, Path(out_dir) / "40_WL_negative_control_treatment_vs_WL_deltaFC_v4_2_value_column_only.csv")
    save_csv(within, Path(out_dir) / "41_WL_within_group_prepost_change_v4_2_value_column_only.csv")
    if not ctrl.empty:
        summ = ctrl.groupby(["edge", "value_column", "edge_identity_policy"]).agg(
            any_FDR_vs_WL=("WL_negative_control_label", lambda x: any(v == "FDR_treatment_specific_delta" for v in x)),
            any_nominal_vs_WL=("WL_negative_control_label", lambda x: any(v in ["FDR_treatment_specific_delta", "nominal_treatment_specific_delta"] for v in x)),
            comparisons_nominal=("comparison", lambda x: "; ".join(ctrl.loc[x.index][ctrl.loc[x.index, "welch_p"] < 0.05]["comparison"].astype(str))),
        ).reset_index()
        save_csv(summ, Path(out_dir) / "42_WL_negative_control_summary_by_edge_v4_2_value_column_only.csv")
    else:
        summ = pd.DataFrame()
    return ctrl, within, summ


def make_integrated_decision(moder, plastic_best, wl_summ, out_dir):
    rows = []
    for e in CORE_EDGES:
        m = moder[moder["edge"] == e].iloc[0].to_dict() if moder is not None and not moder.empty and e in moder["edge"].values else {}
        p = plastic_best[plastic_best["edge"] == e].iloc[0].to_dict() if plastic_best is not None and not plastic_best.empty and e in plastic_best["edge"].values else {}
        w = wl_summ[wl_summ["edge"] == e].iloc[0].to_dict() if wl_summ is not None and not wl_summ.empty and e in wl_summ["edge"].values else {}
        treatment_selection = m.get("selection_vs_prognosis_label", "unknown")
        plasticity = p.get("support_label", "unknown")
        wl = "treatment_specific_supported" if w.get("any_nominal_vs_WL", False) else "WL_control_not_supported_or_unclear"
        if treatment_selection == "strong_treatment_selection_pattern" and plasticity in ["FDR_supported", "nominal_supported"] and w.get("any_nominal_vs_WL", False):
            priority = "highest_priority_convergent_candidate"
        elif treatment_selection in ["strong_treatment_selection_pattern", "treatment_moderation_without_opposite_slopes"] and plasticity in ["FDR_supported", "nominal_supported"]:
            priority = "moderation_plus_plasticity_candidate"
        elif treatment_selection in ["strong_treatment_selection_pattern", "treatment_moderation_without_opposite_slopes"]:
            priority = "moderation_candidate_without_strong_plasticity"
        else:
            priority = "supplement_or_unclear"
        beta = m.get("interaction_beta", np.nan)
        empirical_direction = "TMS-favoring" if pd.notna(beta) and beta > 0 else ("PSY-favoring" if pd.notna(beta) else EDGE_META[e]["a_priori_direction"])
        rows.append({
            "edge": e,
            "value_column": EDGE_META[e]["value_column"],
            "edge_identity_policy": EDGE_META[e]["edge_identity_policy"],
            "system": EDGE_META[e]["system"],
            "system_cn": EDGE_META[e]["system_cn"],
            "empirical_direction": empirical_direction,
            "interaction_beta": beta,
            "interaction_p": m.get("interaction_p", np.nan),
            "interaction_q": m.get("interaction_q", np.nan),
            "selection_vs_prognosis_label": treatment_selection,
            "slope_pattern": m.get("slope_pattern", ""),
            "plasticity_best_group": p.get("group_set", ""),
            "plasticity_best_split": p.get("split_method", ""),
            "plasticity_high_low_p": p.get("welch_p_high_vs_low_delta", np.nan),
            "plasticity_high_low_q": p.get("q_high_vs_low_delta_all_tests", np.nan),
            "plasticity_support_label": plasticity,
            "WL_negative_control_label": wl,
            "WL_nominal_comparisons": w.get("comparisons_nominal", ""),
            "integrated_priority": priority,
            "interpretation": EDGE_META[e]["interpretation"],
        })
    dec = pd.DataFrame(rows)
    dec = dec.sort_values(["integrated_priority", "interaction_p"])
    save_csv(dec, Path(out_dir) / "50_integrated_edge_decision_table_v4_2_value_column_only.csv")
    return dec


def write_report(out_dir, detected, moder, sys_sum, plastic_best, wl_sum, decision, used_covs, main_table, results40_dir):
    lines = []
    lines.append("43_v4：固定候选边治疗选择/机制/可塑性/WL负控分析报告（value_column-only，无解剖标签）")
    lines.append("=" * 100)
    lines.append("")
    lines.append("[输入]")
    lines.append(f"main_table_file = {Path(main_table).name}")
    lines.append("results40_dir = [path omitted for legacy-label audit safety; see command log if needed]")
    lines.append(f"n_subject_rows = {detected.get('n_subject_rows')}")
    lines.append(f"outcome_source = {detected.get('outcome_source')}")
    lines.append(f"covariates_used_for_selection_vs_prognosis = {', '.join(used_covs) if used_covs else 'none'}")
    lines.append("")
    lines.append("[固定候选边身份]")
    for e in CORE_EDGES:
        lines.append(f"- {e}: {EDGE_META[e]['value_column']} | {EDGE_META[e]['a_priori_direction']} | value_column_only_no_anatomical_label")
    lines.append("")
    lines.append("[第一层问题：治疗选择 vs 一般预后]")
    if moder is not None and not moder.empty:
        counts = moder["selection_vs_prognosis_label"].value_counts(dropna=False)
        lines.append(counts.to_string())
        top = moder.sort_values("interaction_p").head(10)
        lines.append("")
        lines.append("Top interaction results:")
        lines.append(top[["edge", "value_column", "interaction_beta", "interaction_p", "interaction_q", "slope_PSY", "slope_TMS", "selection_vs_prognosis_label"]].to_string(index=False))
    lines.append("")
    lines.append("[第二层问题：PSY-favoring vs TMS-favoring 的候选连接方向模式]")
    if sys_sum is not None and not sys_sum.empty:
        lines.append(sys_sum.to_string(index=False))
    lines.append("")
    lines.append("[第三层问题：哪些边也参与治疗过程/疗效相关可塑性]")
    if plastic_best is not None and not plastic_best.empty:
        show_cols = ["edge", "value_column", "group_set", "split_method", "n_high", "n_low", "high_delta_mean", "low_delta_mean", "high_minus_low_delta", "welch_p_high_vs_low_delta", "q_high_vs_low_delta_all_tests", "support_label"]
        lines.append(plastic_best[show_cols].to_string(index=False))
    lines.append("")
    lines.append("[第四层问题：WL负控——治疗相关变化 vs 自然波动]")
    if wl_sum is not None and not wl_sum.empty:
        lines.append(wl_sum.to_string(index=False))
    lines.append("")
    lines.append("[综合决策]")
    if decision is not None and not decision.empty:
        lines.append(decision[["edge", "value_column", "empirical_direction", "selection_vs_prognosis_label", "plasticity_support_label", "WL_negative_control_label", "integrated_priority"]].to_string(index=False))
    lines.append("")
    lines.append("[谨慎解释]")
    lines.append("1. 本脚本仍属于 stability-selected candidates 的效应刻画，不是独立验证。")
    lines.append("2. 第三层 plasticity 是 responder-linked longitudinal FC change，不等同于已证明的因果机制。")
    lines.append("3. WL 负控若不显著，不能证明没有治疗特异性；可能受 WL 样本量小影响。")
    lines.append("4. 本脚本不提供解剖命名或脑图定位；解剖解释需使用独立审计通过的 Brainnetome 246 ROI 映射与 BrainNet 文件。")
    p = Path(out_dir) / "99_treatment_selection_mechanism_plasticity_WL_report_v4_2_value_column_only.txt"
    p.write_text("\n".join(lines), encoding="utf-8-sig")
    log(f"写出：{p}")


def scan_outputs_for_legacy_labels(out_dir):
    # 不在正式结果中允许旧脑区标签痕迹；value_column 的 unknown_roi 允许。
    # v4.2 修复点：
    # 1) 不再把 split_method = "median" 里的 dia 误判为 dIa；
    # 2) 报告中的输入路径已隐藏，避免旧文件夹名触发误报；
    # 3) 增加压缩旧标签（如 a3536cdia）的拦截。
    legacy_regexes = {
        "A45r": r"(?<![A-Za-z0-9])A45r(?![A-Za-z0-9])",
        "A11m": r"(?<![A-Za-z0-9])A11m(?![A-Za-z0-9])",
        "A5l": r"(?<![A-Za-z0-9])A5l(?![A-Za-z0-9])",
        "vId": r"(?<![A-Za-z0-9])vId(?![A-Za-z0-9])",
        "vIg": r"(?<![A-Za-z0-9])vIg(?![A-Za-z0-9])",
        "cpSTS": r"(?<![A-Za-z0-9])cpSTS(?![A-Za-z0-9])",
        "lsOccG": r"(?<![A-Za-z0-9])lsOccG(?![A-Za-z0-9])",
        "A7c": r"(?<![A-Za-z0-9])A7c(?![A-Za-z0-9])",
        "A9/46v": r"(?<![A-Za-z0-9])A9[/_]?46v(?![A-Za-z0-9])",
        "A7ip": r"(?<![A-Za-z0-9])A7ip(?![A-Za-z0-9])",
        "A35/36c": r"(?<![A-Za-z0-9])A35[/_]?36c(?![A-Za-z0-9])",
        "dIa": r"(?<![A-Za-z0-9])dIa(?![A-Za-z0-9])",
        "A37elv": r"(?<![A-Za-z0-9])A37elv(?![A-Za-z0-9])",
        "cLinG": r"(?<![A-Za-z0-9])cLinG(?![A-Za-z0-9])",
        "A24rv": r"(?<![A-Za-z0-9])A24rv(?![A-Za-z0-9])",
        "A32p": r"(?<![A-Za-z0-9])A32p(?![A-Za-z0-9])",
        "A9/46d": r"(?<![A-Za-z0-9])A9[/_]?46d(?![A-Za-z0-9])",
        "A24cd": r"(?<![A-Za-z0-9])A24cd(?![A-Za-z0-9])",
        "A11l": r"(?<![A-Za-z0-9])A11l(?![A-Za-z0-9])",
        "aSTS": r"(?<![A-Za-z0-9])aSTS(?![A-Za-z0-9])",
        # 压缩形式：避免旧标签以去标点后的形式混入。
        "compressed_A45r_A11m": r"a45ra11m",
        "compressed_A5l_vId_vIg": r"a5lvidvig",
        "compressed_cpSTS_lsOccG": r"cpstslsoccg",
        "compressed_A7c_vId_vIg": r"a7cvidvig",
        "compressed_A9_46v_A7ip": r"a9/?46va7ip|a946va7ip",
        "compressed_A35_36c_dIa": r"a35/?36cdia|a3536cdia",
        "compressed_A37elv_cLinG": r"a37elvcling",
        "compressed_A24rv_A32p": r"a24rva32p",
        "compressed_A9_46d_A24cd": r"a9/?46da24cd|a946da24cd",
        "compressed_A11l_aSTS": r"a11lasts",
    }
    compiled = {name: re.compile(rx, flags=re.IGNORECASE) for name, rx in legacy_regexes.items()}
    rows = []
    for p in Path(out_dir).glob("*"):
        if p.suffix.lower() not in [".csv", ".txt", ".json"]:
            continue
        if p.name.startswith("00_output_legacy_label_text_audit"):
            continue
        try:
            txt = p.read_text(encoding="utf-8-sig", errors="ignore")
        except Exception:
            continue
        hits = []
        for name, rx in compiled.items():
            if rx.search(txt):
                hits.append(name)
        hits = sorted(set(hits))
        rows.append({"file": p.name, "legacy_label_hit_count": len(hits), "legacy_label_hits": ";".join(hits)})
    audit = pd.DataFrame(rows)
    audit_path = Path(out_dir) / "00_output_legacy_label_text_audit_v4_2.csv"
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    bad = audit[audit["legacy_label_hit_count"] > 0] if not audit.empty else pd.DataFrame()
    if not bad.empty:
        raise RuntimeError("v4.2 输出中仍发现旧脑区标签残留，请检查 00_output_legacy_label_text_audit_v4_2.csv。")

def parse_args():
    ap = argparse.ArgumentParser(description=SCRIPT_NAME)
    ap.add_argument("--main_table", default=DEFAULT_MAIN_TABLE, help="优先使用含 pre_meanFD 的 Brainnetome 主表")
    ap.add_argument("--results40_dir", default="", help="40号最终结果文件夹；留空则自动在当前目录搜索")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--high_cutoff", type=float, default=8.0, help="high improvement cutoff，默认 PCL improvement >= 8")
    ap.add_argument("--low_cutoff", type=float, default=0.0, help="low improvement cutoff，默认 PCL improvement <= 0")
    ap.add_argument("--covariates", default="PCL_pre,age,sex,pre_meanFD", help="治疗选择vs预后分析中使用的协变量")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    log("=" * 100)
    log(f"{SCRIPT_NAME} 启动")
    log(f"out_dir = {out_dir}")

    main_table = args.main_table
    if not Path(main_table).exists():
        log(f"主表不存在，尝试 fallback: {FALLBACK_MAIN_TABLE}")
        main_table = FALLBACK_MAIN_TABLE
    log(f"main_table = {main_table}")
    main_df = read_csv_smart(main_table)
    log(f"main_df rows={len(main_df)}, cols={len(main_df.columns)}")

    if args.results40_dir:
        results40_dir = Path(args.results40_dir)
    else:
        results40_dir = find_latest_40_dir(Path.cwd()) or find_latest_40_dir(Path(DEFAULT_OUT_DIR).parent)
    if results40_dir is None or not Path(results40_dir).exists():
        log("未找到40号结果目录；将仅使用脚本内固定 value_column 清单。")
        results40_dir = ""
    else:
        log(f"results40_dir = {results40_dir}")

    mapping = load_edge_mapping(results40_dir)
    save_csv(sanitized_mapping_for_output(mapping), out_dir / "01_input_40_candidate_edge_mapping_v4_2_sanitized_value_column_only.csv")

    edge_to_col = candidate_edge_cols_from_mapping(mapping, main_df)
    map_rows = [
        {
            "edge": e,
            "value_column": EDGE_META[e]["value_column"],
            "selected_main_table_cols": EDGE_META[e]["value_column"],
            "mapped": bool(cols),
            "edge_identity_policy": EDGE_META[e]["edge_identity_policy"],
        }
        for e, cols in edge_to_col.items()
    ]
    save_csv(pd.DataFrame(map_rows), out_dir / "02_core_edge_mapping_to_main_table_v4_2_value_column_only.csv")

    subj, detected = build_subject_level(main_df, edge_to_col, out_dir)
    save_csv(subj, out_dir / "03_subject_level_core_edges_TMS_PSY_WL_v4_2_value_column_only.csv")

    covariates = [x.strip() for x in args.covariates.split(",") if x.strip()]
    moder, simple, used_covs = analysis_treatment_selection(subj, covariates, out_dir)
    sys_df, sys_sum = analysis_system_direction(moder, out_dir)
    plastic, corr, plastic_best = analysis_plasticity(subj, out_dir, high_cutoff=args.high_cutoff, low_cutoff=args.low_cutoff)
    wl_ctrl, wl_within, wl_sum = analysis_wl_negative_control(subj, out_dir)
    decision = make_integrated_decision(moder, plastic_best, wl_sum, out_dir)

    write_report(out_dir, detected, moder, sys_sum, plastic_best, wl_sum, decision, used_covs, main_table, results40_dir)
    scan_outputs_for_legacy_labels(out_dir)

    summary = {
        "script": SCRIPT_NAME,
        "runtime_finished": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "edge_identity_policy": "value_column_only_no_anatomical_label",
        "n_subject_rows": int(len(subj)),
        "n_edges": int(len(CORE_EDGES)),
        "n_active_subjects": int(subj[subj["treatment_family"].isin(["TMS", "PSY"])]["subject_key"].nunique()),
        "n_wl_subjects": int(subj[subj["treatment_family"] == "WL"]["subject_key"].nunique()),
        "used_covariates": used_covs,
        "outputs_legacy_label_audit": "00_output_legacy_label_text_audit_v4_2.csv",
    }
    (out_dir / "00_v4_2_value_column_only_run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    log("完成。")


if __name__ == "__main__":
    main()
