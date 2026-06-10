# -*- coding: utf-8 -*-
"""
36_v3_全脑基线FC治疗调节效应扫描_值列身份标准化_无解剖标签版.py

目的：
    从全脑/大范围 baseline FC 宽表或长表中，对每一条连接运行同一个治疗调节模型：
        PCL_improvement = b0 + b1*baseline_FC_z + b2*TMS + b3*(baseline_FC_z*TMS) + covariates + error
    其中 b3 就是 treatment × baseline FC 的交互效应。

推荐运行：
    python -u 36_全脑基线FC治疗调节效应扫描_v2_被试去重和标签增强版.py

如自动找表不准确，可手动指定：
    python -u 36_全脑基线FC治疗调节效应扫描_v2_被试去重和标签增强版.py --input_file "你的FC临床合并宽表.csv"

可选加入协变量敏感性分析：
    python -u 36_全脑基线FC治疗调节效应扫描_v2_被试去重和标签增强版.py --input_file "你的FC临床合并宽表.csv" --covariates age,sex,meanFD

说明：
    1. 默认仅比较 TMS vs PSY，其中 PSY = ACT + MIN + psychotherapy 等；WL/HC 默认排除。
    2. outcome 默认为 PCL improvement；若找不到 improvement 列，会尝试用 PCL_pre - PCL_post 计算。
    3. 输出包括所有边结果、未校正 p<0.05 边、FDR q<0.10 边、摘要报告。
    4. 这是探索性全脑扫描；论文解释必须以 FDR 或预先固定候选集为准。
    5. v2 默认按被试去重，优先保留 pre/baseline 前测行，避免同一被试 pre/post 重复进入模型。
    6. v3 为解决 Brainnetome 标签/半脑映射风险，默认不做任何解剖命名映射。
       全脑扫描阶段只使用原始 value_column【取值列】作为连接身份。
       --roi_label_file 与 --edge_label_map 参数会被保留但忽略，避免错误标签重新进入结果。
       如需解剖标签，请在后续独立审计通过的 246 ROI 映射流程中附加，不在本脚本内完成。
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception as e:
    stats = None


# -----------------------------
# 基础工具
# -----------------------------

def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now_str()}] {msg}", flush=True)


def safe_mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if suffix in [".csv", ".txt", ".tsv"]:
        sep = "\t" if suffix == ".tsv" else ","
        for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
            try:
                return pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
            except Exception:
                pass
        return pd.read_csv(path, sep=sep, low_memory=False)
    raise ValueError(f"不支持的文件类型: {path}")


def to_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    ss = s.astype(str).str.strip()
    ss = ss.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NA": np.nan, "N/A": np.nan})
    return pd.to_numeric(ss, errors="coerce")


def zscore_arr(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return np.full_like(x, np.nan, dtype=float)
    return (x - mu) / sd


def bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR。"""
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    m = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    q_ranked = ranked * m / (np.arange(1, m + 1))
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)
    temp = np.empty_like(q_ranked)
    temp[order] = q_ranked
    q[ok] = temp
    return q


def pearson_r_p(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan, np.nan
    if np.nanstd(x[ok], ddof=1) == 0 or np.nanstd(y[ok], ddof=1) == 0:
        return np.nan, np.nan
    if stats is None:
        return float(np.corrcoef(x[ok], y[ok])[0, 1]), np.nan
    r, p = stats.pearsonr(x[ok], y[ok])
    return float(r), float(p)


# -----------------------------
# 自动识别列
# -----------------------------

SUBJECT_PATTERNS = ["subject", "subj", "participant", "bids", "sub_id", "subject_id", "participant_id", "id", "被试", "编号"]
GROUP_PATTERNS = ["group", "treatment", "treat", "arm", "intervention", "condition", "组别", "分组", "治疗"]
OUTCOME_PATTERNS = ["improvement", "improve", "pcl_improvement", "pcl5_improvement", "pcl_change", "pcl_delta", "改善"]

CLINICAL_KEYWORDS = [
    "subject", "participant", "group", "treatment", "arm", "intervention", "condition",
    "diagnosis", "site", "age", "sex", "gender", "fd", "meanfd", "mean_fd",
    "pcl", "caps", "ham", "bdi", "bai", "score", "improvement", "improve", "change", "delta",
    "responder", "response", "time", "visit", "prepost", "postmeanfd", "motion",
    "被试", "编号", "组别", "分组", "治疗", "年龄", "性别", "量表", "改善", "疗效", "站点",
]


def norm_name(c: str) -> str:
    return re.sub(r"\s+", "", str(c)).lower()


def find_first_col(cols: Sequence[str], patterns: Sequence[str]) -> Optional[str]:
    nmap = {c: norm_name(c) for c in cols}
    # 先找完全匹配或强匹配
    for pat in patterns:
        patn = norm_name(pat)
        for c, nc in nmap.items():
            if nc == patn:
                return c
    # 再找包含
    for pat in patterns:
        patn = norm_name(pat)
        for c, nc in nmap.items():
            if patn in nc:
                return c
    return None


def detect_group_col(df: pd.DataFrame, explicit: Optional[str] = None) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"指定 group_col 不在表中: {explicit}")
        return explicit
    candidates = []
    for c in df.columns:
        nc = norm_name(c)
        score = 0
        if any(p in nc for p in GROUP_PATTERNS):
            score += 4
        vals = df[c].dropna().astype(str).str.upper().str.strip().unique()[:30]
        vals_set = set(vals)
        if any(v in vals_set for v in ["TMS", "ACT", "MIN", "WL", "HC", "PSY", "PSYCHOTHERAPY"]):
            score += 5
        if 0 < df[c].nunique(dropna=True) <= 10:
            score += 1
        if score > 0:
            candidates.append((score, c))
    if not candidates:
        raise ValueError("未能自动识别治疗分组列。请用 --group_col 指定。")
    candidates.sort(reverse=True)
    return candidates[0][1]


def detect_subject_col(df: pd.DataFrame, explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"指定 subject_col 不在表中: {explicit}")
        return explicit
    c = find_first_col(df.columns, SUBJECT_PATTERNS)
    return c


def detect_outcome_col(df: pd.DataFrame, explicit: Optional[str], out_dir: Path) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"指定 outcome_col 不在表中: {explicit}")
        return explicit

    # 1) 优先找 PCL + improvement/change/delta/改善
    candidates = []
    for c in df.columns:
        nc = norm_name(c)
        score = 0
        if "pcl" in nc and any(k in nc for k in ["improvement", "improve", "改善"]):
            score += 10
        if "pcl" in nc and any(k in nc for k in ["change", "delta", "diff", "变化"]):
            score += 6
        if any(k in nc for k in ["pcl_improvement", "pcl5_improvement"]):
            score += 4
        if score > 0 and pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce")):
            valid = to_numeric_series(df[c]).notna().sum()
            score += min(valid / max(len(df), 1), 1)
            candidates.append((score, c))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]

    # 2) 找 pre/post PCL，用 pre - post 计算改善，症状下降为正改善
    pre_cols = []
    post_cols = []
    for c in df.columns:
        nc = norm_name(c)
        if "pcl" not in nc:
            continue
        if any(k in nc for k in ["pre", "baseline", "before", "前测", "治疗前"]):
            pre_cols.append(c)
        if any(k in nc for k in ["post", "after", "后测", "治疗后"]):
            post_cols.append(c)
    if pre_cols and post_cols:
        # 选择缺失最少的一对
        best = None
        for pc in pre_cols:
            for qc in post_cols:
                valid = (to_numeric_series(df[pc]).notna() & to_numeric_series(df[qc]).notna()).sum()
                score = valid
                if best is None or score > best[0]:
                    best = (score, pc, qc)
        if best is not None and best[0] > 0:
            _, pc, qc = best
            new_col = "PCL_improvement_auto_pre_minus_post"
            df[new_col] = to_numeric_series(df[pc]) - to_numeric_series(df[qc])
            with open(out_dir / "00_outcome_auto_created.txt", "w", encoding="utf-8") as f:
                f.write(f"自动生成结局列: {new_col}\n")
                f.write(f"计算方式: {pc} - {qc}\n")
                f.write("解释: PCL下降越多，improvement 越大。\n")
            return new_col

    raise ValueError(
        "未能自动识别 PCL improvement 结局列，也未找到可配对的 PCL pre/post 列。"
        "请用 --outcome_col 指定，或确认表中有 PCL_pre 与 PCL_post。"
    )


def normalize_group_value(x: object) -> str:
    s = str(x).strip().upper()
    s = s.replace(" ", "")
    return s


def build_treatment_binary(
    group_s: pd.Series,
    tms_groups: Sequence[str],
    psy_groups: Sequence[str],
    exclude_groups: Sequence[str],
) -> pd.Series:
    tms_set = {normalize_group_value(x) for x in tms_groups}
    psy_set = {normalize_group_value(x) for x in psy_groups}
    excl_set = {normalize_group_value(x) for x in exclude_groups}

    out = pd.Series(np.nan, index=group_s.index, dtype=float)
    for idx, val in group_s.items():
        g = normalize_group_value(val)
        if g in tms_set:
            out.loc[idx] = 1.0
        elif g in psy_set:
            out.loc[idx] = 0.0
        elif g in excl_set:
            out.loc[idx] = np.nan
        else:
            # 允许包含式匹配，例如 Psychotherapy_ACT
            if any(t in g for t in tms_set):
                out.loc[idx] = 1.0
            elif any(p in g for p in psy_set):
                out.loc[idx] = 0.0
            else:
                out.loc[idx] = np.nan
    return out


def is_edge_like_col(c: str) -> bool:
    raw = str(c)
    nc = norm_name(raw)

    # 排除明显临床/元数据/纵向变化列
    if any(k in nc for k in CLINICAL_KEYWORDS):
        # 注意：pre/baseline 边列可能也含 pre；这里不把 pre 单独作为临床关键词
        # 如果列名同时有明显 ROI pair，下面会放行
        has_pair_marker = ("__" in raw) or ("--" in raw) or ("_to_" in nc) or ("-" in raw)
        has_roi = bool(re.search(r"(bn\d{1,3}|a\d{1,2}|opc|a32p|a11m|a37dl|roi)", nc))
        if not (has_pair_marker and has_roi):
            return False

    # 跳过 post/delta/change 边，优先只做 baseline/pre
    if any(k in nc for k in ["post", "delta", "change", "变化", "diff"]):
        if not any(k in nc for k in ["baseline", "pre", "前测", "治疗前"]):
            return False

    # 常见边命名：BN001__BN185, A11m__A32p, A37dl-OPC, ROI1_to_ROI2
    patterns = [
        r"bn\d{1,3}.*(__|--|-|_to_|to).*bn\d{1,3}",
        r"a\d{1,2}[a-z/]*.*(__|--|-|_to_|to).*a\d{1,2}[a-z/]*",
        r"a\d{1,2}[a-z/]*.*(__|--|-|_to_|to).*opc",
        r"opc.*(__|--|-|_to_|to).*a\d{1,2}[a-z/]*",
        r"roi\d+.*(__|--|-|_to_|to).*roi\d+",
    ]
    if any(re.search(p, nc) for p in patterns):
        return True

    # 如果含双下划线，且不是明显临床列，也可能是边名
    if "__" in raw and len(raw) < 200:
        return True

    return False


def detect_edge_columns_wide(df: pd.DataFrame, exclude_cols: Sequence[str], explicit_regex: Optional[str]) -> List[str]:
    exclude = set([c for c in exclude_cols if c])
    cols = []
    regex = re.compile(explicit_regex) if explicit_regex else None
    for c in df.columns:
        if c in exclude:
            continue
        # 必须可转成一定数量的数值
        num = to_numeric_series(df[c])
        valid_n = num.notna().sum()
        if valid_n < max(10, int(0.25 * len(df))):
            continue
        if regex is not None:
            if regex.search(str(c)):
                cols.append(c)
            continue
        if is_edge_like_col(c):
            cols.append(c)

    # 如果严格模式找到太少，使用更宽松的备用策略：大量数值列、排除临床列、列名不太短
    if len(cols) < 50:
        backup = []
        for c in df.columns:
            if c in exclude:
                continue
            nc = norm_name(c)
            if any(k in nc for k in CLINICAL_KEYWORDS):
                continue
            if any(k in nc for k in ["post", "delta", "change", "变化", "diff"]):
                continue
            num = to_numeric_series(df[c])
            valid_n = num.notna().sum()
            if valid_n >= max(10, int(0.5 * len(df))):
                # 避免把少量人口学连续变量当成边：要求整体候选数足够多时才使用
                backup.append(c)
        if len(backup) >= 200:
            cols = backup

    return list(dict.fromkeys(cols))


def detect_long_format(df: pd.DataFrame) -> Optional[Dict[str, str]]:
    cols = list(df.columns)
    edge_col = None
    for c in cols:
        nc = norm_name(c)
        if nc in ["edge", "edge_name", "connection", "connection_name", "pair", "roi_pair", "边", "连接"]:
            edge_col = c
            break
    if edge_col is None:
        return None

    fc_candidates = []
    for c in cols:
        nc = norm_name(c)
        if c == edge_col:
            continue
        score = 0
        if "baseline" in nc or "pre" in nc or "前测" in nc or "治疗前" in nc:
            score += 3
        if "fc" in nc or "z" == nc or "fisher" in nc or "corr" in nc or "connect" in nc or "连接" in nc:
            score += 2
        if score > 0 and to_numeric_series(df[c]).notna().sum() > 10:
            fc_candidates.append((score, c))
    if not fc_candidates:
        return None
    fc_candidates.sort(reverse=True)
    return {"edge_col": edge_col, "fc_col": fc_candidates[0][1]}


# -----------------------------
# 自动找输入文件
# -----------------------------

def likely_file_score(path: Path) -> int:
    name = path.name.lower()
    score = 0
    for k in ["brainnetome", "bna", "bn246", "246", "fc", "宽表", "wide", "合并", "clinical", "pcl", "baseline", "纵向"]:
        if k in name:
            score += 2
    for bad in ["result", "结果", "summary", "摘要", "top", "fdr", "permutation", "置换", "report", "报告"]:
        if bad in name:
            score -= 1
    if path.suffix.lower() in [".csv", ".xlsx", ".xls", ".tsv"]:
        score += 1
    return score


def iter_candidate_files(search_roots: Sequence[Path], max_depth: int = 4) -> Iterable[Path]:
    seen = set()
    allowed = {".csv", ".xlsx", ".xls", ".tsv"}
    for root in search_roots:
        if not root.exists():
            continue
        root = root.resolve()
        for dirpath, dirnames, filenames in os.walk(root):
            d = Path(dirpath)
            try:
                rel_depth = len(d.relative_to(root).parts)
            except Exception:
                rel_depth = 0
            if rel_depth > max_depth:
                dirnames[:] = []
                continue
            # 跳过明显输出/临时目录
            lowd = str(d).lower()
            if any(bad in lowd for bad in ["__pycache__", "backup", "备份", "zip", "diagnostic"]):
                pass
            for fn in filenames:
                p = d / fn
                if p.suffix.lower() not in allowed:
                    continue
                if p.name.startswith("~$"):
                    continue
                key = str(p).lower()
                if key in seen:
                    continue
                seen.add(key)
                yield p


def auto_find_input_file() -> Optional[Path]:
    cwd = Path.cwd()
    roots = [cwd]
    likely_roots = [
        Path(r"E:\E_zhangzhihui\从yv那边提取\脚本"),
        Path(r"E:\E_zhangzhihui\提取roi"),
        Path(r"D:\自科＋脑中心论文选题\PAI选题\工作站传输\第二步纵向分析脚本"),
        Path(r"D:\自科＋脑中心论文选题\PAI选题\工作站传输"),
        Path(r"D:\自科＋脑中心论文选题\PAI选题\causal_forest\outputs"),
    ]
    for r in likely_roots:
        if r.exists() and r not in roots:
            roots.append(r)

    scored = []
    for p in iter_candidate_files(roots, max_depth=4):
        fs = likely_file_score(p)
        if fs <= 0:
            continue
        # 简单读取前几行判断列数，避免读超大文件太慢
        try:
            if p.suffix.lower() in [".csv", ".tsv"]:
                sep = "\t" if p.suffix.lower() == ".tsv" else ","
                sample = None
                for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
                    try:
                        sample = pd.read_csv(p, sep=sep, encoding=enc, nrows=5, low_memory=False)
                        break
                    except Exception:
                        continue
                if sample is None:
                    continue
            else:
                sample = pd.read_excel(p, nrows=5)
            ncol = sample.shape[1]
            coltxt = " ".join(map(str, sample.columns)).lower()
            score = fs + min(ncol / 100, 15)
            if any(k in coltxt for k in ["pcl", "improvement", "改善"]):
                score += 5
            if any(k in coltxt for k in ["tms", "act", "min", "group", "组别", "治疗"]):
                score += 4
            if ncol >= 500:
                score += 8
            scored.append((score, p))
        except Exception:
            continue

    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    log("自动候选输入文件 Top 5：")
    for s, p in scored[:5]:
        log(f"  score={s:.1f} | {p}")
    return scored[0][1]



# -----------------------------
# v2：时间点识别、被试去重、标签映射
# -----------------------------

TIMEPOINT_PATTERNS = [
    "timepoint", "time_point", "visit", "session", "ses", "scan", "phase", "wave",
    "prepost", "pre_post", "assessment", "测量", "时间", "时间点", "访视", "阶段", "前后测",
]
BASELINE_TOKENS = ["pre", "baseline", "base", "before", "t0", "v1", "visit1", "session1", "ses1", "前测", "治疗前", "基线"]
POST_TOKENS = ["post", "after", "t1", "v2", "visit2", "session2", "ses2", "后测", "治疗后"]


def truthy_text(x: object) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on", "是"}


def falsy_text(x: object) -> bool:
    return str(x).strip().lower() in {"0", "false", "no", "n", "off", "否"}


def detect_timepoint_col(df: pd.DataFrame, explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"指定 timepoint_col 不在表中: {explicit}")
        return explicit
    best = None
    for c in df.columns:
        nc = norm_name(c)
        score = 0.0
        if any(p in nc for p in TIMEPOINT_PATTERNS):
            score += 4.0
        vals = df[c].dropna().astype(str).str.lower().str.strip()
        if vals.empty:
            continue
        sample = vals.head(min(len(vals), 200))
        joined = " ".join(sample.tolist())
        if any(t in joined for t in BASELINE_TOKENS):
            score += 4.0
        if any(t in joined for t in POST_TOKENS):
            score += 4.0
        nunique = df[c].nunique(dropna=True)
        if 1 < nunique <= 8:
            score += 1.0
        if score > 0:
            if best is None or score > best[0]:
                best = (score, c)
    return best[1] if best else None


def timepoint_priority_value(x: object) -> int:
    s = str(x).strip().lower()
    compact = re.sub(r"[\s_\-]+", "", s)
    if any(t in s for t in BASELINE_TOKENS) or any(re.sub(r"[\s_\-]+", "", t) in compact for t in BASELINE_TOKENS):
        return 0
    if any(t in s for t in POST_TOKENS) or any(re.sub(r"[\s_\-]+", "", t) in compact for t in POST_TOKENS):
        return 10
    return 5


def normalize_subject_id_for_dedup(x: object) -> str:
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "na"}:
        return ""
    # 优先保留数字主体；例如 sub-preACT1466 与 1466 会映射为 1466
    nums = re.findall(r"\d+", s)
    if nums:
        # 多段数字时取最长段；常见 subject_digits 已经是纯数字
        longest = max(nums, key=len)
        try:
            return str(int(longest))
        except Exception:
            return longest.lstrip("0") or longest
    return re.sub(r"\s+", "", s).upper()


def make_fc_completeness_score(df: pd.DataFrame, edge_cols: Optional[Sequence[str]] = None) -> pd.Series:
    if edge_cols:
        cols = [c for c in edge_cols if c in df.columns]
        if cols:
            # 只抽样部分列以节省时间；用于排序足够了
            sample_cols = cols[: min(len(cols), 1000)]
            return df[sample_cols].apply(lambda row: pd.to_numeric(row, errors="coerce").notna().sum(), axis=1)
    return pd.Series(0, index=df.index, dtype=float)


def deduplicate_wide_subject_rows(
    df: pd.DataFrame,
    treatment_binary: pd.Series,
    subject_col: Optional[str],
    group_col: Optional[str],
    outcome_col: str,
    edge_cols: Optional[Sequence[str]],
    out_dir: Path,
    timepoint_col: Optional[str] = None,
    enable: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, object]]:
    """宽表：先保留 TMS/PSY 行，再每名被试只保留一行。"""
    meta: Dict[str, object] = {}
    tmp = df.copy()
    tmp["__treat_binary__"] = treatment_binary
    tmp["__orig_row__"] = np.arange(len(tmp))

    meta["n_rows_before_group_filter"] = int(len(tmp))
    tmp = tmp[tmp["__treat_binary__"].notna()].copy()
    meta["n_rows_after_group_filter_before_dedup"] = int(len(tmp))
    meta["n_TMS_rows_before_dedup"] = int((tmp["__treat_binary__"] == 1).sum())
    meta["n_PSY_rows_before_dedup"] = int((tmp["__treat_binary__"] == 0).sum())

    if not enable:
        meta["deduplication_enabled"] = "false"
        out = tmp.drop(columns=["__orig_row__"], errors="ignore")
        return out, out["__treat_binary__"].astype(float), meta

    if not subject_col or subject_col not in tmp.columns:
        log("未识别 subject_col，无法按被试去重；将仅使用组别过滤后的行。")
        meta["deduplication_enabled"] = "requested_but_no_subject_col"
        out = tmp.drop(columns=["__orig_row__"], errors="ignore")
        return out, out["__treat_binary__"].astype(float), meta

    tp_col = detect_timepoint_col(tmp, timepoint_col)
    meta["timepoint_col_used_for_dedup"] = tp_col if tp_col else "none"
    tmp["__subject_key__"] = tmp[subject_col].map(normalize_subject_id_for_dedup)
    tmp = tmp[tmp["__subject_key__"].astype(str).str.len() > 0].copy()
    meta["n_unique_subjects_before_dedup"] = int(tmp["__subject_key__"].nunique(dropna=True))

    if tp_col:
        tmp["__timepoint_priority__"] = tmp[tp_col].map(timepoint_priority_value).astype(int)
    else:
        tmp["__timepoint_priority__"] = 5

    tmp["__outcome_nonmissing__"] = to_numeric_series(tmp[outcome_col]).notna().astype(int)
    tmp["__fc_nonmissing_score__"] = make_fc_completeness_score(tmp, edge_cols)

    # 若同一被试同一组有 pre/post 两行，优先：baseline/pre；有 outcome；FC 完整；原始顺序靠前
    tmp = tmp.sort_values(
        by=["__subject_key__", "__timepoint_priority__", "__outcome_nonmissing__", "__fc_nonmissing_score__", "__orig_row__"],
        ascending=[True, True, False, False, True],
    )
    chosen = tmp.drop_duplicates(subset=["__subject_key__"], keep="first").copy()

    meta["deduplication_enabled"] = "true"
    meta["n_rows_after_subject_dedup"] = int(len(chosen))
    meta["n_TMS_subjects_after_dedup"] = int((chosen["__treat_binary__"] == 1).sum())
    meta["n_PSY_subjects_after_dedup"] = int((chosen["__treat_binary__"] == 0).sum())
    meta["n_duplicate_rows_removed_by_subject_dedup"] = int(len(tmp) - len(chosen))

    audit_cols = [c for c in [subject_col, "__subject_key__", group_col, tp_col, outcome_col, "__treat_binary__", "__timepoint_priority__", "__outcome_nonmissing__", "__fc_nonmissing_score__", "__orig_row__"] if c and c in chosen.columns]
    try:
        chosen[audit_cols].to_csv(out_dir / "00_subject_dedup_chosen_rows.csv", index=False, encoding="utf-8-sig")
        tmp[tmp.duplicated(subset=["__subject_key__"], keep=False)][audit_cols].to_csv(out_dir / "00_subject_dedup_duplicate_candidates.csv", index=False, encoding="utf-8-sig")
    except Exception as e:
        log(f"去重审计表写入失败，不影响主分析: {e}")

    drop_cols = ["__orig_row__", "__subject_key__", "__timepoint_priority__", "__outcome_nonmissing__", "__fc_nonmissing_score__"]
    out = chosen.drop(columns=drop_cols, errors="ignore")
    return out, out["__treat_binary__"].astype(float), meta


def filter_long_to_baseline_rows(
    df: pd.DataFrame,
    treatment_binary: pd.Series,
    out_dir: Path,
    timepoint_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, object]]:
    """长表：先排除 WL/HC，再优先保留 baseline/pre 时间点；随后 run_long_scan 会按 subject+edge 去重。"""
    meta: Dict[str, object] = {}
    tmp = df.copy()
    tmp["__treat_binary__"] = treatment_binary
    tmp = tmp[tmp["__treat_binary__"].notna()].copy()
    meta["n_rows_after_group_filter_before_baseline_filter"] = int(len(tmp))
    tp_col = detect_timepoint_col(tmp, timepoint_col)
    meta["timepoint_col_used_for_baseline_filter"] = tp_col if tp_col else "none"
    if tp_col:
        tmp["__timepoint_priority__"] = tmp[tp_col].map(timepoint_priority_value)
        base = tmp[tmp["__timepoint_priority__"] == 0].copy()
        if not base.empty:
            meta["n_rows_after_baseline_filter"] = int(len(base))
            tmp = base
        else:
            meta["n_rows_after_baseline_filter"] = "no_baseline_rows_detected_keep_all"
    out = tmp.drop(columns=["__timepoint_priority__"], errors="ignore")
    return out, out["__treat_binary__"].astype(float), meta


def load_edge_label_map(edge_label_map: Optional[str] = None, roi_label_file: Optional[str] = None) -> Dict[str, str]:
    """加载边名或 ROI 编号映射。返回 key->label。支持两类：完整边名映射，或 ROI index/name 映射。"""
    mapping: Dict[str, str] = {}
    # 完整边名映射：至少要有 edge/raw_edge 和 label/readable_edge 等列
    if edge_label_map:
        p = Path(edge_label_map)
        if p.exists():
            try:
                tab = read_table(p)
                cols = list(tab.columns)
                raw_col = None
                lab_col = None
                for c in cols:
                    nc = norm_name(c)
                    if nc in {"edge", "raw_edge", "edge_name", "connection", "原始边", "连接"}:
                        raw_col = c
                    if nc in {"label", "readable_edge", "edge_label", "connection_label", "roi_pair_label", "标签", "可读边名"}:
                        lab_col = c
                if raw_col and lab_col:
                    for _, r in tab[[raw_col, lab_col]].dropna().iterrows():
                        mapping[str(r[raw_col])] = str(r[lab_col])
                    log(f"已加载完整边名映射: {p}, n={len(mapping)}")
            except Exception as e:
                log(f"完整边名映射读取失败: {p}; {e}")
    # ROI 标签表：支持 Brainnetome 常见列，如 index/roi_id/label/name/abbr
    if roi_label_file:
        p = Path(roi_label_file)
        if p.exists():
            try:
                tab = read_table(p)
                id_col = None
                label_col = None
                for c in tab.columns:
                    nc = norm_name(c)
                    if id_col is None and any(k in nc for k in ["index", "roi", "id", "number", "label_id", "编号", "序号"]):
                        if to_numeric_series(tab[c]).notna().sum() > 0:
                            id_col = c
                    if label_col is None and any(k in nc for k in ["abbr", "name", "label", "region", "roi_name", "subregion", "名称", "标签", "脑区"]):
                        # 不要把纯数字列当 label
                        if to_numeric_series(tab[c]).notna().sum() < len(tab) * 0.8:
                            label_col = c
                if id_col and label_col:
                    for _, r in tab[[id_col, label_col]].dropna().iterrows():
                        try:
                            k = str(int(float(r[id_col])))
                        except Exception:
                            k = str(r[id_col]).strip()
                        label = str(r[label_col]).strip()
                        if k and label:
                            mapping[f"ROI::{k}"] = label
                            mapping[f"ROI::{k.zfill(3)}"] = label
                    log(f"已加载 ROI 标签映射: {p}, id_col={id_col}, label_col={label_col}, n={sum(1 for k in mapping if k.startswith('ROI::'))}")
                else:
                    log(f"ROI 标签表未能自动识别 id/name 列: {p}")
            except Exception as e:
                log(f"ROI 标签表读取失败: {p}; {e}")
    return mapping


def split_edge_tokens(edge: str) -> Optional[Tuple[str, str]]:
    s = str(edge)
    for sep in ["__", "--", "_to_", "-", " to "]:
        if sep in s:
            parts = s.split(sep)
            if len(parts) >= 2:
                return parts[0].strip(), parts[1].strip()
    return None


def roi_token_to_label(tok: str, mapping: Dict[str, str]) -> str:
    raw = str(tok).strip()
    low = raw.lower()
    # 完整 ROI 文本本身已经可读，如 A11m、A32p、A37dl、OPC
    if re.search(r"a\d{1,2}[a-z/]*", low) or "opc" in low or "orbital" in low or "cing" in low:
        return raw
    # unknown_roi_16, ROI16, BN016 都尝试映射到 ROI index
    nums = re.findall(r"\d+", raw)
    if nums:
        n = str(int(nums[-1]))
        for key in [f"ROI::{n}", f"ROI::{n.zfill(3)}"]:
            if key in mapping:
                return mapping[key]
    return raw


def add_readable_edge_labels(res: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    if res.empty or "edge" not in res.columns:
        return res
    res = res.copy()
    labels = []
    for e in res["edge"].astype(str):
        if e in mapping:
            labels.append(mapping[e])
            continue
        toks = split_edge_tokens(e)
        if toks:
            a, b = toks
            labels.append(f"{roi_token_to_label(a, mapping)}__{roi_token_to_label(b, mapping)}")
        else:
            labels.append(e)
    res.insert(1, "edge_readable", labels)
    return res

# -----------------------------
# 回归模型
# -----------------------------

def prepare_covariates(df: pd.DataFrame, covariate_names: Sequence[str]) -> Tuple[pd.DataFrame, List[str]]:
    cov_df = pd.DataFrame(index=df.index)
    used = []
    for name in covariate_names:
        name = name.strip()
        if not name:
            continue
        if name not in df.columns:
            log(f"协变量未找到，跳过: {name}")
            continue
        s = df[name]
        if pd.api.types.is_numeric_dtype(s) or to_numeric_series(s).notna().sum() >= max(10, int(0.3 * len(df))):
            vals = to_numeric_series(s).astype(float)
            z = zscore_arr(vals.to_numpy())
            cov_df[name] = z
            used.append(name)
        else:
            # 类别协变量转 dummy，drop first
            dummies = pd.get_dummies(s.astype(str).fillna("NA"), prefix=name, drop_first=True, dtype=float)
            for dc in dummies.columns:
                vals = dummies[dc].astype(float).to_numpy()
                if np.nanstd(vals, ddof=1) > 0:
                    cov_df[dc] = vals
                    used.append(dc)
    return cov_df, used


def ols_interaction(y: np.ndarray, fc: np.ndarray, tr: np.ndarray, cov: Optional[np.ndarray] = None) -> Optional[Dict[str, float]]:
    """
    y: raw PCL improvement
    fc: raw baseline FC, will be z-scored within complete sample
    tr: 1=TMS, 0=PSY
    cov: already prepared covariates, rows aligned; may contain nan
    """
    y = np.asarray(y, dtype=float)
    fc = np.asarray(fc, dtype=float)
    tr = np.asarray(tr, dtype=float)

    ok = np.isfinite(y) & np.isfinite(fc) & np.isfinite(tr)
    if cov is not None and cov.size > 0:
        ok = ok & np.all(np.isfinite(cov), axis=1)
    if ok.sum() < 20:
        return None

    y2 = y[ok]
    tr2 = tr[ok]
    fc2 = fc[ok]
    cov2 = cov[ok, :] if cov is not None and cov.size > 0 else None

    n_psy = int(np.sum(tr2 == 0))
    n_tms = int(np.sum(tr2 == 1))
    if n_psy < 8 or n_tms < 8:
        return None
    if np.nanstd(fc2, ddof=1) == 0 or np.nanstd(y2, ddof=1) == 0:
        return None

    fcz = zscore_arr(fc2)
    if not np.all(np.isfinite(fcz)):
        return None
    inter = fcz * tr2

    X_parts = [np.ones(len(y2)), fcz, tr2, inter]
    if cov2 is not None and cov2.size > 0:
        for j in range(cov2.shape[1]):
            X_parts.append(cov2[:, j])
    X = np.column_stack(X_parts)

    # 去掉完全共线/零方差列，保留前4列关键列
    keep = []
    for j in range(X.shape[1]):
        if j == 0:
            keep.append(True)
        else:
            keep.append(np.nanstd(X[:, j], ddof=1) > 0)
    keep = np.array(keep, dtype=bool)
    if not keep[:4].all():
        return None
    Xk = X[:, keep]

    try:
        beta, residuals, rank, svals = np.linalg.lstsq(Xk, y2, rcond=None)
        if rank < Xk.shape[1]:
            return None
        yhat = Xk @ beta
        resid = y2 - yhat
        n = len(y2)
        p = Xk.shape[1]
        df = n - p
        if df <= 1:
            return None
        rss = float(np.sum(resid ** 2))
        sigma2 = rss / df
        xtx_inv = np.linalg.inv(Xk.T @ Xk)
        se = np.sqrt(np.diag(xtx_inv) * sigma2)
        tvals = beta / se
        if stats is not None:
            pvals = 2 * stats.t.sf(np.abs(tvals), df)
        else:
            pvals = np.full_like(tvals, np.nan, dtype=float)

        # 因为关键前4列保留，index 不变
        b0, b_fc, b_tr, b_int = beta[0], beta[1], beta[2], beta[3]
        se_int = se[3]
        t_int = tvals[3]
        p_int = pvals[3]

        # 标准化 outcome 的 beta，便于不同表比较
        yz = zscore_arr(y2)
        beta_std = np.linalg.lstsq(Xk, yz, rcond=None)[0]
        b_int_std = beta_std[3]

        r2 = 1 - rss / np.sum((y2 - np.mean(y2)) ** 2)
        r_psy, p_psy = pearson_r_p(fcz[tr2 == 0], y2[tr2 == 0])
        r_tms, p_tms = pearson_r_p(fcz[tr2 == 1], y2[tr2 == 1])

        return {
            "n_total": int(n),
            "n_psy": n_psy,
            "n_tms": n_tms,
            "beta_fc_psy_rawY_per1SDfc": float(b_fc),
            "beta_tms_minus_psy_intercept_rawY": float(b_tr),
            "beta_interaction_rawY_per1SDfc": float(b_int),
            "se_interaction": float(se_int),
            "t_interaction": float(t_int),
            "p_interaction": float(p_int),
            "beta_interaction_stdY_per1SDfc": float(b_int_std),
            "slope_psy_rawY_per1SDfc": float(b_fc),
            "slope_tms_rawY_per1SDfc": float(b_fc + b_int),
            "r_psy": float(r_psy),
            "p_r_psy": float(p_psy),
            "r_tms": float(r_tms),
            "p_r_tms": float(p_tms),
            "model_r2": float(r2),
            "df_resid": int(df),
        }
    except Exception:
        return None


def run_wide_scan(
    df: pd.DataFrame,
    edge_cols: Sequence[str],
    outcome_col: str,
    treatment_binary: pd.Series,
    cov_df: pd.DataFrame,
    min_n: int,
) -> pd.DataFrame:
    y_all = to_numeric_series(df[outcome_col]).to_numpy(dtype=float)
    tr_all = treatment_binary.to_numpy(dtype=float)
    cov_all = cov_df.to_numpy(dtype=float) if cov_df is not None and cov_df.shape[1] > 0 else None

    rows = []
    t0 = time.time()
    total = len(edge_cols)
    for i, edge in enumerate(edge_cols, 1):
        fc_all = to_numeric_series(df[edge]).to_numpy(dtype=float)
        # 先看有效样本数，减少无效回归
        ok = np.isfinite(y_all) & np.isfinite(tr_all) & np.isfinite(fc_all)
        if cov_all is not None:
            ok = ok & np.all(np.isfinite(cov_all), axis=1)
        if ok.sum() >= min_n:
            res = ols_interaction(y_all, fc_all, tr_all, cov_all)
            if res is not None and res["n_total"] >= min_n:
                res["edge"] = edge
                rows.append(res)

        if i == 1 or i % 500 == 0 or i == total:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else np.nan
            remain = (total - i) / rate if rate and rate > 0 else np.nan
            log(f"全脑交互扫描进度: {i}/{total}; 已得到有效边 {len(rows)}; elapsed={elapsed/60:.1f}min; remain≈{remain/60:.1f}min")

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # 调整列顺序
    first = ["edge", "n_total", "n_psy", "n_tms", "beta_interaction_rawY_per1SDfc", "se_interaction", "t_interaction", "p_interaction", "beta_interaction_stdY_per1SDfc", "slope_psy_rawY_per1SDfc", "slope_tms_rawY_per1SDfc", "r_psy", "p_r_psy", "r_tms", "p_r_tms", "model_r2", "df_resid"]
    rest = [c for c in out.columns if c not in first]
    return out[first + rest]


def run_long_scan(
    df: pd.DataFrame,
    long_info: Dict[str, str],
    outcome_col: str,
    treatment_binary: pd.Series,
    cov_df: pd.DataFrame,
    min_n: int,
    subject_col: Optional[str],
) -> pd.DataFrame:
    edge_col = long_info["edge_col"]
    fc_col = long_info["fc_col"]
    tmp = df.copy()
    tmp["__treat_binary__"] = treatment_binary
    if cov_df is not None and cov_df.shape[1] > 0:
        for c in cov_df.columns:
            tmp[f"__cov__{c}"] = cov_df[c]
    cov_cols = [c for c in tmp.columns if c.startswith("__cov__")]

    rows = []
    edges = tmp[edge_col].dropna().astype(str).unique().tolist()
    t0 = time.time()
    for i, edge in enumerate(edges, 1):
        sub = tmp[tmp[edge_col].astype(str) == edge].copy()
        if subject_col and subject_col in sub.columns:
            # 如果同一被试同一边重复，保留第一条有效记录
            sub = sub.drop_duplicates(subset=[subject_col, edge_col], keep="first")
        y = to_numeric_series(sub[outcome_col]).to_numpy(dtype=float)
        fc = to_numeric_series(sub[fc_col]).to_numpy(dtype=float)
        tr = to_numeric_series(sub["__treat_binary__"]).to_numpy(dtype=float)
        cov = sub[cov_cols].to_numpy(dtype=float) if cov_cols else None
        ok = np.isfinite(y) & np.isfinite(fc) & np.isfinite(tr)
        if cov is not None:
            ok = ok & np.all(np.isfinite(cov), axis=1)
        if ok.sum() >= min_n:
            res = ols_interaction(y, fc, tr, cov)
            if res is not None and res["n_total"] >= min_n:
                res["edge"] = edge
                rows.append(res)
        if i == 1 or i % 500 == 0 or i == len(edges):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else np.nan
            remain = (len(edges) - i) / rate if rate and rate > 0 else np.nan
            log(f"长表交互扫描进度: {i}/{len(edges)}; 已得到有效边 {len(rows)}; elapsed={elapsed/60:.1f}min; remain≈{remain/60:.1f}min")
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    first = ["edge", "n_total", "n_psy", "n_tms", "beta_interaction_rawY_per1SDfc", "se_interaction", "t_interaction", "p_interaction", "beta_interaction_stdY_per1SDfc", "slope_psy_rawY_per1SDfc", "slope_tms_rawY_per1SDfc", "r_psy", "p_r_psy", "r_tms", "p_r_tms", "model_r2", "df_resid"]
    rest = [c for c in out.columns if c not in first]
    return out[first + rest]


def add_direction_columns(res: pd.DataFrame) -> pd.DataFrame:
    if res.empty:
        return res
    res = res.copy()
    res["q_interaction_fdr_bh"] = bh_fdr(res["p_interaction"].to_numpy(dtype=float))
    res["abs_t_interaction"] = res["t_interaction"].abs()
    res["abs_beta_interaction_stdY"] = res["beta_interaction_stdY_per1SDfc"].abs()

    def direction(row):
        b = row["beta_interaction_rawY_per1SDfc"]
        if not np.isfinite(b):
            return "NA"
        if b > 0:
            return "higher baseline FC favors stronger TMS slope vs PSY"
        if b < 0:
            return "higher baseline FC favors stronger PSY slope vs TMS"
        return "no interaction direction"

    res["direction_interpretation"] = res.apply(direction, axis=1)
    return res.sort_values(["p_interaction", "q_interaction_fdr_bh"], ascending=[True, True]).reset_index(drop=True)


def find_key_edges(res: pd.DataFrame) -> pd.DataFrame:
    """v3 strict mode: do not search or label named anatomical key edges.

    This script intentionally keeps edge identity as original value_column only.
    Anatomical naming must be added only by a separately audited 246-ROI label
    mapping workflow. Returning an empty table prevents legacy labels such as
    A11m-A32p or A37dl-OPC from re-entering this upstream full-brain scan.
    """
    return pd.DataFrame()

def save_outputs(res: pd.DataFrame, out_dir: Path, meta: Dict[str, object]) -> None:
    safe_mkdir(out_dir)
    all_csv = out_dir / "01_fullbrain_treatment_by_baselineFC_interaction_all_edges.csv"
    res.to_csv(all_csv, index=False, encoding="utf-8-sig")

    if not res.empty:
        p005 = res[res["p_interaction"] < 0.05].copy()
        p001 = res[res["p_interaction"] < 0.01].copy()
        q10 = res[res["q_interaction_fdr_bh"] < 0.10].copy()
        q05 = res[res["q_interaction_fdr_bh"] < 0.05].copy()
        p005.to_csv(out_dir / "02_uncorrected_p_lt_0p05_edges.csv", index=False, encoding="utf-8-sig")
        p001.to_csv(out_dir / "03_uncorrected_p_lt_0p01_edges.csv", index=False, encoding="utf-8-sig")
        q10.to_csv(out_dir / "04_fdr_q_lt_0p10_edges.csv", index=False, encoding="utf-8-sig")
        q05.to_csv(out_dir / "05_fdr_q_lt_0p05_edges.csv", index=False, encoding="utf-8-sig")
        top50 = res.head(50).copy()
        top50.to_csv(out_dir / "06_top50_edges_by_interaction_p.csv", index=False, encoding="utf-8-sig")
        key = find_key_edges(res)
        if not key.empty:
            key.to_csv(out_dir / "07_key_edges_A11m_A32p_A37dl_OPC_if_found.csv", index=False, encoding="utf-8-sig")

        # Excel 便于快速看，但失败不影响主输出
        try:
            with pd.ExcelWriter(out_dir / "08_interaction_scan_summary_tables.xlsx") as writer:
                res.head(200).to_excel(writer, sheet_name="top200_all", index=False)
                p005.to_excel(writer, sheet_name="p_lt_0p05", index=False)
                q10.to_excel(writer, sheet_name="q_lt_0p10", index=False)
                if not key.empty:
                    key.to_excel(writer, sheet_name="key_edges_if_found", index=False)
        except Exception as e:
            log(f"Excel 汇总写入失败，不影响 CSV: {e}")

    with open(out_dir / "99_summary_report.txt", "w", encoding="utf-8") as f:
        f.write("全脑 baseline FC × treatment 交互扫描摘要\n")
        f.write("=" * 80 + "\n")
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")
        if res.empty:
            f.write("没有得到有效结果。请检查输入表、分组列、结局列、边列识别。\n")
            return
        f.write(f"有效边数: {len(res)}\n")
        f.write(f"uncorrected p<0.05 边数: {int((res['p_interaction'] < 0.05).sum())}\n")
        f.write(f"uncorrected p<0.01 边数: {int((res['p_interaction'] < 0.01).sum())}\n")
        f.write(f"FDR q<0.10 边数: {int((res['q_interaction_fdr_bh'] < 0.10).sum())}\n")
        f.write(f"FDR q<0.05 边数: {int((res['q_interaction_fdr_bh'] < 0.05).sum())}\n")
        f.write("\nTop 20 by interaction p-value:\n")
        cols = [c for c in ["edge", "edge_readable", "n_total", "n_psy", "n_tms", "beta_interaction_rawY_per1SDfc", "p_interaction", "q_interaction_fdr_bh", "slope_psy_rawY_per1SDfc", "slope_tms_rawY_per1SDfc", "direction_interpretation"] if c in res.columns]
        f.write(res[cols].head(20).to_string(index=False))
        f.write("\n\n解释规则：\n")
        f.write("1. beta_interaction_rawY_per1SDfc > 0 表示 baseline FC 每升高 1 SD 时，TMS 组的疗效斜率比 PSY 组更正。\n")
        f.write("2. beta_interaction_rawY_per1SDfc < 0 表示 baseline FC 每升高 1 SD 时，PSY 组的疗效斜率相对更有利。\n")
        f.write("3. 全脑扫描必须优先看 FDR q 值；未校正 p 值只能作为探索线索。\n")
        f.write("4. v3 严格模式不附加任何解剖标签；edge 字段为原始 value_column。\n")
        f.write("5. 如需 Brainnetome 解剖名，必须使用独立审计通过的 246 ROI 映射流程，不能在本脚本内直接命名。\n")

    log(f"结果已保存到: {out_dir}")


# -----------------------------
# 主程序
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="全脑 baseline FC × treatment 交互效应扫描")
    parser.add_argument("--input_file", default="", help="输入 FC+clinical 合并表；留空则自动搜索。")
    parser.add_argument("--out_dir", default="", help="输出目录；默认在当前目录生成结果文件夹。")
    parser.add_argument("--subject_col", default="", help="被试编号列名，可选。")
    parser.add_argument("--group_col", default="", help="治疗分组列名，例如 group/treatment。")
    parser.add_argument("--outcome_col", default="", help="结局列名，例如 PCL_improvement。")
    parser.add_argument("--edge_regex", default="", help="手动指定边列正则表达式，例如 'BN\\d+__BN\\d+'。")
    parser.add_argument("--edge_mode", default="auto", choices=["auto", "wide", "long"], help="输入表格式：auto/wide/long。")
    parser.add_argument("--covariates", default="", help="逗号分隔协变量列名，例如 age,sex,meanFD,site。默认不加协变量。")
    parser.add_argument("--min_n", type=int, default=40, help="每条边进入模型的最小完整样本量。默认40。")
    parser.add_argument("--max_edges", type=int, default=0, help="只测试前 N 条边，用于试跑；0 表示全部。")
    parser.add_argument("--deduplicate_subjects", default="true", help="是否按被试去重，默认 true。")
    parser.add_argument("--timepoint_col", default="", help="时间点列名，可选；脚本会优先保留 pre/baseline 行。")
    parser.add_argument("--roi_label_file", default="", help="保留兼容参数；v3严格模式会忽略，不在全脑扫描阶段附加解剖标签。")
    parser.add_argument("--edge_label_map", default="", help="保留兼容参数；v3严格模式会忽略，不在全脑扫描阶段附加解剖标签。")
    parser.add_argument("--tms_groups", default="TMS", help="TMS 组标签，逗号分隔。")
    parser.add_argument("--psy_groups", default="ACT,MIN,PSY,PSYCHOTHERAPY,PSYCHOTHERAPY_ACT,PSYCHOTHERAPY_MIN,心理治疗", help="心理治疗组标签，逗号分隔。")
    parser.add_argument("--exclude_groups", default="WL,HC,CONTROL,HEALTHYCONTROL,健康对照,等待名单", help="排除组标签，逗号分隔。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    log("=" * 100)
    log("36号v3：全脑 baseline FC × treatment 交互效应扫描 启动（值列身份标准化；不附加解剖标签）")
    log("=" * 100)

    if args.out_dir:
        out_dir = safe_mkdir(Path(args.out_dir))
    else:
        out_dir = safe_mkdir(Path.cwd() / "36_v3_全脑基线FC治疗调节效应扫描结果_值列身份标准化_无解剖标签版")
    log(f"out_dir = {out_dir}")

    if args.input_file:
        input_file = Path(args.input_file)
        if not input_file.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_file}")
    else:
        input_file = auto_find_input_file()
        if input_file is None:
            raise FileNotFoundError(
                "未能自动找到合适输入文件。请把脚本放在包含 FC+clinical 宽表的目录运行，"
                "或使用 --input_file 手动指定。"
            )
    log(f"input_file = {input_file}")

    df = read_table(input_file)
    log(f"读取完成: rows={df.shape[0]}, cols={df.shape[1]}")

    subject_col = detect_subject_col(df, args.subject_col or None)
    group_col = detect_group_col(df, args.group_col or None)
    outcome_col = detect_outcome_col(df, args.outcome_col or None, out_dir)
    log(f"subject_col = {subject_col}")
    log(f"group_col = {group_col}")
    log(f"outcome_col = {outcome_col}")

    tms_groups = [x.strip() for x in args.tms_groups.split(",") if x.strip()]
    psy_groups = [x.strip() for x in args.psy_groups.split(",") if x.strip()]
    exclude_groups = [x.strip() for x in args.exclude_groups.split(",") if x.strip()]
    tr_bin_raw = build_treatment_binary(df[group_col], tms_groups, psy_groups, exclude_groups)
    log("原始分组计数：")
    log(str(df[group_col].value_counts(dropna=False)))
    log(f"去重前纳入行数：PSY=0 行 {int((tr_bin_raw == 0).sum())}; TMS=1 行 {int((tr_bin_raw == 1).sum())}; 排除/未识别行 {int(tr_bin_raw.isna().sum())}")

    if int((tr_bin_raw == 0).sum()) < 8 or int((tr_bin_raw == 1).sum()) < 8:
        raise ValueError("TMS 或 PSY 可用样本过少。请检查 --psy_groups / --tms_groups / --group_col。")

    long_info = detect_long_format(df) if args.edge_mode in ["auto", "long"] else None
    if args.edge_mode == "long" and long_info is None:
        raise ValueError("指定 edge_mode=long，但未能识别 edge 列和 baseline FC 列。")

    dedup_enabled = not falsy_text(args.deduplicate_subjects)
    dedup_meta: Dict[str, object] = {}

    if long_info is not None and args.edge_mode in ["auto", "long"]:
        log(f"识别为长表: edge_col={long_info['edge_col']}, fc_col={long_info['fc_col']}")
        df_scan, tr_bin, dedup_meta = filter_long_to_baseline_rows(df, tr_bin_raw, out_dir, args.timepoint_col or None)
        covariate_names = [x.strip() for x in args.covariates.split(",") if x.strip()]
        cov_df, used_covs = prepare_covariates(df_scan, covariate_names)
        log(f"used_covariates = {used_covs if used_covs else '无'}")
        log(f"实际纳入唯一被试数估计：PSY={int((tr_bin == 0).sum())}; TMS={int((tr_bin == 1).sum())}（长表会按 subject+edge 再去重）")
        res = run_long_scan(df_scan, long_info, outcome_col, tr_bin, cov_df, args.min_n, subject_col)
        n_edges_detected = int(df_scan[long_info["edge_col"]].nunique(dropna=True))
        edge_mode_used = "long"
    else:
        exclude_cols_initial = [subject_col, group_col, outcome_col]
        edge_cols = detect_edge_columns_wide(df, exclude_cols_initial, args.edge_regex or None)
        if args.max_edges and args.max_edges > 0:
            edge_cols = edge_cols[: args.max_edges]
        if not edge_cols:
            raise ValueError(
                "未能自动识别 baseline FC 边列。请检查列名，或使用 --edge_regex 指定边列模式。"
            )
        log(f"识别为宽表: baseline FC edge columns = {len(edge_cols)}")
        log("前10条边列示例: " + "; ".join(map(str, edge_cols[:10])))
        try:
            pd.DataFrame({
                "edge_index": range(1, len(edge_cols) + 1),
                "value_column": edge_cols,
                "identity_policy": "value_column_only_no_anatomical_label",
            }).to_csv(out_dir / "00_detected_edge_value_columns_audit.csv", index=False, encoding="utf-8-sig")
        except Exception as e:
            log(f"边列身份审计表写入失败，不影响主分析: {e}")

        df_scan, tr_bin, dedup_meta = deduplicate_wide_subject_rows(
            df=df,
            treatment_binary=tr_bin_raw,
            subject_col=subject_col,
            group_col=group_col,
            outcome_col=outcome_col,
            edge_cols=edge_cols,
            out_dir=out_dir,
            timepoint_col=args.timepoint_col or None,
            enable=dedup_enabled,
        )
        log(f"去重后实际纳入：PSY=0 被试/行 {int((tr_bin == 0).sum())}; TMS=1 被试/行 {int((tr_bin == 1).sum())}")
        if int((tr_bin == 0).sum()) < 8 or int((tr_bin == 1).sum()) < 8:
            raise ValueError("被试去重后 TMS 或 PSY 可用样本过少。请检查时间点识别和分组列。")

        covariate_names = [x.strip() for x in args.covariates.split(",") if x.strip()]
        cov_df, used_covs = prepare_covariates(df_scan, covariate_names)
        log(f"used_covariates = {used_covs if used_covs else '无'}")
        res = run_wide_scan(df_scan, edge_cols, outcome_col, tr_bin, cov_df, args.min_n)
        n_edges_detected = len(edge_cols)
        edge_mode_used = "wide"

    # v3 strict policy: do not attach anatomical labels in the full-brain scan.
    # Edge identity is the original value_column only. This prevents any 123-row
    # or left-hemisphere-only label table from contaminating the upstream results.
    if args.edge_label_map or args.roi_label_file:
        log("v3安全策略：--edge_label_map / --roi_label_file 已被忽略；本脚本不附加解剖标签。")
    label_map = {}
    res = add_direction_columns(res)

    meta = {
        "input_file": str(input_file),
        "script_version": "v3_value_column_identity_only_no_anatomical_labels",
        "edge_identity_policy": "value_column_only",
        "anatomical_label_mapping_enabled": "false",
        "roi_label_file_ignored": args.roi_label_file if args.roi_label_file else "none",
        "edge_label_map_ignored": args.edge_label_map if args.edge_label_map else "none",
        "edge_mode_used": edge_mode_used,
        "n_rows_input": df.shape[0],
        "n_cols_input": df.shape[1],
        "n_edges_detected_or_unique": n_edges_detected,
        "n_edges_valid_model": len(res),
        "subject_col": subject_col,
        "group_col": group_col,
        "outcome_col": outcome_col,
        "tms_groups": ",".join(tms_groups),
        "psy_groups": ",".join(psy_groups),
        "exclude_groups": ",".join(exclude_groups),
        "n_TMS_included_after_dedup_or_filter": int((tr_bin == 1).sum()),
        "n_PSY_included_after_dedup_or_filter": int((tr_bin == 0).sum()),
        "covariates_used": ",".join(used_covs) if used_covs else "none",
        "min_n": args.min_n,
        "roi_label_file": "ignored_in_v3",
        "edge_label_map": "ignored_in_v3",
        "runtime_minutes": round((time.time() - t0) / 60, 3),
    }
    meta.update(dedup_meta)
    save_outputs(res, out_dir, meta)

    log("=" * 100)
    if res.empty:
        log("完成，但没有有效边结果。")
    else:
        log(f"完成。有效边={len(res)}; p<0.05={int((res['p_interaction'] < 0.05).sum())}; q<0.10={int((res['q_interaction_fdr_bh'] < 0.10).sum())}; q<0.05={int((res['q_interaction_fdr_bh'] < 0.05).sum())}")
        log("Top 10:")
        show_cols = [c for c in ["edge", "edge_readable", "n_total", "n_psy", "n_tms", "beta_interaction_rawY_per1SDfc", "p_interaction", "q_interaction_fdr_bh", "direction_interpretation"] if c in res.columns]
        print(res[show_cols].head(10).to_string(index=False), flush=True)
    log("=" * 100)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("运行失败：" + repr(e))
        log("建议：1）用 --input_file 手动指定 FC+clinical 合并表；2）用 --group_col / --outcome_col 指定列名；3）用 --edge_regex 指定边列。")
        raise
