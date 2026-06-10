# -*- coding: utf-8 -*-
"""
44_其他临床指标_固定候选边辅助验证与机制解释_v3_value_column_only.py

核心修正：
1. 不再默认从 Brainnetome246 大宽表里自动抓候选边。
2. 优先读取 43 号脚本生成的被试层候选边表，例如：
   43_固定候选边_治疗选择机制可塑性WL负控分析结果\03_subject_level_core_edges_TMS_PSY_WL.csv
3. 另行读取临床量表表，例如：
   04_临床量表清洗后_修正版.csv
   05_第二步纵向分析_被试层主表_重跑合并版.csv
4. 用标准化被试编号 subject_key 合并候选边表与临床表。
5. 模型分三层：
   A. 主模型：不强制协变量，避免 pre_meanFD 缺失导致 n=0。
   B. 基础协变量模型：age + sex，如果可用且样本量足够。
   C. 头动敏感性模型：age + sex + meanFD，仅在 meanFD 不删光样本时运行。
6. 如果仍然识别不到其他临床指标，会输出疑似临床列审计，告诉你哪些列没有形成 pre/post 配对。

运行方式：
    python 44_其他临床指标_固定候选边辅助验证与机制解释_v3_value_column_only.py

也可以手动指定：
    python 44_其他临床指标_固定候选边辅助验证与机制解释_v3_value_column_only.py --candidate_csv "43号候选边表.csv" --clinical_csv "临床量表表.csv"

输出目录默认：
    44_其他临床指标_固定候选边辅助验证与机制解释结果_v2_43候选边合并临床表

说明：
    本脚本只围绕固定候选边做 secondary outcome / mechanism analysis【次要结局/机制分析】，不做全脑重新筛选。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from scipy import stats  # type: ignore
except Exception as e:  # pragma: no cover
    stats = None
    print("[提示] 未能导入 scipy，将使用正态近似计算 p 值；建议正式结果安装 scipy。错误：", repr(e))

SCRIPT_VERSION = "v3.0_2026-05-20_value_column_only_no_anatomical_label"
DEFAULT_OUT_DIR_NAME = "44_v3_value_column_only_其他临床指标_固定候选边辅助验证与机制解释结果_无旧标签正式版"

# 固定候选边关键词：用于筛选重要边；如果 43 号表中还有更多候选边，脚本也会自动纳入。
IMPORTANT_EDGE_TERMS = [
    # v3 正式安全版只接受 43_v4.2 输出的 EDGE_01-EDGE_10 + value_column 身份。
    # 旧脑区名不再作为候选边身份，也不用于正式输出。
    "EDGE_01", "EDGE_02", "EDGE_03", "EDGE_04", "EDGE_05",
    "EDGE_06", "EDGE_07", "EDGE_08", "EDGE_09", "EDGE_10",
    "unknown_roi_18__unknown_roi_24", "unknown_roi_65__unknown_roi_85",
    "unknown_roi_62__unknown_roi_105", "unknown_roi_64__unknown_roi_85",
    "unknown_roi_11__unknown_roi_67", "unknown_roi_56__unknown_roi_84",
    "unknown_roi_46__unknown_roi_95", "unknown_roi_89__unknown_roi_90",
    "unknown_roi_8__unknown_roi_92", "unknown_roi_23__unknown_roi_44",
]

CLINICAL_KEYWORDS = [
    # PTSD / PCL / CAPS
    "pcl", "pcl5", "pcl_5", "ptsd", "caps",
    "intrusion", "reexperiencing", "re_experiencing", "re-experiencing", "闯入", "再体验",
    "avoidance", "avoid", "回避",
    "hyperarousal", "arousal", "警觉", "高警觉", "唤醒",
    "negative", "nacm", "cognition", "mood", "负性", "认知", "情绪",
    # depression / anxiety
    "hamd", "hama", "bdi", "bai", "phq", "gad", "sas", "sds",
    "depress", "depression", "抑郁", "anxiety", "anxious", "焦虑",
    # sleep / function / general
    "psqi", "sleep", "insomnia", "睡眠", "失眠",
    "function", "functional", "disability", "whodas", "quality", "qol", "生活质量", "功能", "社会功能",
    "symptom", "症状", "总分", "分量表", "score", "量表",
]

# 默认多数临床症状量表高分更差，改善 = pre - post。
# 如果某个指标高分更好，可在这里加入关键词，改善 = post - pre。
HIGHER_IS_BETTER_KEYWORDS = ["qol", "qualityoflife", "生活质量", "functioning_good", "cdrisc", "cd_risc", "resilience", "韧性", "ffmq", "mindfulness", "正念", "ptgi", "posttraumaticgrowth", "创伤后成长"]

GROUP_KEYWORDS = {
    "WL": ["wl", "wait", "waiting", "waitlist", "等待"],
    "TMS": ["tms", "rtms", "刺激"],
    "PSY": ["psy", "psych", "psychotherapy", "心理", "act", "min"],
}

EXCLUDE_EDGE_COL_KEYWORDS = [
    "subject", "sub", "participant", "id", "编号", "被试", "group", "组别", "treatment", "治疗", "arm", "condition",
    "age", "年龄", "sex", "gender", "性别", "site", "站点", "center", "中心",
    "meanfd", "mean_fd", "fd", "头动", "motion", "framewise",
    "p", "q", "fdr", "beta", "coef", "se", "tvalue", "t_value", "pvalue", "p_value", "r2", "aic",
    "n_", "count", "rank", "score", "summary", "备注", "说明",
]

# =============================================================================
# 基础工具
# =============================================================================

def log(msg: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def norm_text(x: Any) -> str:
    s = "" if x is None else str(x)
    s = s.lower().strip()
    s = s.replace("＋", "+")
    s = s.replace("/", "_")
    s = re.sub(r"[\s\-–—_:/\\|,;，；()（）\[\]{}<>\.]+", "", s)
    return s


def simple_name(x: Any) -> str:
    s = "" if x is None else str(x).strip()
    s = s.replace("＋", "+")
    s = re.sub(r"[\s\-–—:/\\|,;，；()（）\[\]{}<>]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:120] if len(s) > 120 else s


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamped_path(path: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    df_out = sanitize_dataframe_for_output(df)
    try:
        df_out.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        alt = timestamped_path(path)
        log(f"[权限提示] {path.name} 被占用，改写到 {alt.name}")
        df_out.to_csv(alt, index=False, encoding="utf-8-sig")
        return alt


def write_json(obj: Any, path: Path) -> Path:
    obj_out = sanitize_obj_for_output(obj)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj_out, f, ensure_ascii=False, indent=2)
        return path
    except PermissionError:
        alt = timestamped_path(path)
        log(f"[权限提示] {path.name} 被占用，改写到 {alt.name}")
        with open(alt, "w", encoding="utf-8") as f:
            json.dump(obj_out, f, ensure_ascii=False, indent=2)
        return alt


def write_text(text: str, path: Path) -> Path:
    text_out = sanitize_legacy_text(text)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text_out)
        return path
    except PermissionError:
        alt = timestamped_path(path)
        log(f"[权限提示] {path.name} 被占用，改写到 {alt.name}")
        with open(alt, "w", encoding="utf-8") as f:
            f.write(text_out)
        return alt



# =============================================================================
# v3 安全策略：所有正式输出仅允许 EDGE_XX + value_column，不允许旧脑区名残留
# =============================================================================
LEGACY_LABEL_TOKENS = [
    "A45r", "A11m", "A5l", "vId", "vIg", "cpSTS", "lsOccG", "A7c",
    "A9/46v", "A7ip", "A35/36c", "dIa", "A37elv", "cLinG", "A24rv", "A32p",
    "A9/46d", "A24cd", "A11l", "aSTS", "A40c", "A7m", "A32sg", "msOccG",
    "A23c", "dCa", "cHipp", "mAmyg", "lAmyg", "rHipp", "A40rv",
]
LEGACY_COMPRESSED_TOKENS = [
    "a45ra11m", "a5lvidvig", "cpstslsoccg", "a7cvidvig", "a946va7ip",
    "a3536cdia", "a37elvcling", "a24rva32p", "a946da24cd", "a11lasts",
    "a40cla7ml", "a32sgrmsoccgr", "a23cldcar", "a32sglmsoccgr", "a7iprclingr",
    "a32plmsoccgl", "vidviglchippr", "mamyrlamygl", "a7pclrhippl", "a40rvrdial",
]
LEGACY_REPLACEMENTS = {
    "A45r–A11m": "EDGE_01", "A45r-A11m": "EDGE_01", "A45r_A11m": "EDGE_01", "a45ra11m": "EDGE_01",
    "A5l–vId/vIg": "EDGE_02", "A5l-vId/vIg": "EDGE_02", "A5l_vId_vIg": "EDGE_02", "a5lvidvig": "EDGE_02",
    "cpSTS–lsOccG": "EDGE_03", "cpSTS-lsOccG": "EDGE_03", "cpSTS_lsOccG": "EDGE_03", "cpstslsoccg": "EDGE_03",
    "A7c–vId/vIg": "EDGE_04", "A7c-vId/vIg": "EDGE_04", "A7c_vId_vIg": "EDGE_04", "a7cvidvig": "EDGE_04",
    "A9/46v–A7ip": "EDGE_05", "A9/46v-A7ip": "EDGE_05", "A9_46v_A7ip": "EDGE_05", "a946va7ip": "EDGE_05",
    "A35/36c–dIa": "EDGE_06", "A35/36c-dIa": "EDGE_06", "A35_36c_dIa": "EDGE_06", "a3536cdia": "EDGE_06",
    "A37elv–cLinG": "EDGE_07", "A37elv-cLinG": "EDGE_07", "A37elv_cLinG": "EDGE_07", "a37elvcling": "EDGE_07",
    "A24rv–A32p": "EDGE_08", "A24rv-A32p": "EDGE_08", "A24rv_A32p": "EDGE_08", "a24rva32p": "EDGE_08",
    "A9/46d–A24cd": "EDGE_09", "A9/46d-A24cd": "EDGE_09", "A9_46d_A24cd": "EDGE_09", "a946da24cd": "EDGE_09",
    "A11l–aSTS": "EDGE_10", "A11l-aSTS": "EDGE_10", "A11l_aSTS": "EDGE_10", "a11lasts": "EDGE_10",
}


def sanitize_legacy_text(x: Any) -> Any:
    """仅清理输出中的旧脑区标签文本；不改变数值。"""
    if not isinstance(x, str):
        return x
    s = x
    for old, new in LEGACY_REPLACEMENTS.items():
        s = s.replace(old, new)
    # 删除/替换单个旧 ROI token，防止路径、说明、列名残留旧标签。
    # 注意：使用边界，避免 median 中的 dia 误报。
    for tok in LEGACY_LABEL_TOKENS:
        s = re.sub(rf"(?<![A-Za-z0-9/]){re.escape(tok)}(?![A-Za-z0-9/])", "LEGACYROI", s)
    return s


def sanitize_dataframe_for_output(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(sanitize_legacy_text)
    out.columns = [sanitize_legacy_text(str(c)) for c in out.columns]
    return out


def sanitize_obj_for_output(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {sanitize_legacy_text(str(k)): sanitize_obj_for_output(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_obj_for_output(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_obj_for_output(v) for v in obj]
    return sanitize_legacy_text(obj)


def output_has_legacy_text(text: str) -> bool:
    # 单 token 检查使用边界，避免 median/dia 误报；压缩旧标签单独检查。
    for tok in LEGACY_LABEL_TOKENS:
        if re.search(rf"(?<![A-Za-z0-9/]){re.escape(tok)}(?![A-Za-z0-9/])", text):
            return True
    low = re.sub(r"[^a-z0-9]+", "", text.lower())
    return any(t in low for t in LEGACY_COMPRESSED_TOKENS)


def scan_outputs_for_legacy_labels(out_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(out_dir.glob("*")):
        if not p.is_file() or p.suffix.lower() not in [".csv", ".txt", ".json"]:
            continue
        try:
            txt = p.read_text(encoding="utf-8-sig", errors="ignore")
        except Exception:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        hits = int(output_has_legacy_text(txt))
        rows.append({"file": p.name, "legacy_label_hit_count": hits, "audit_status": "PASS" if hits == 0 else "FAIL"})
    audit = pd.DataFrame(rows)
    # 直接写原始审计表，避免写入过程触发递归。
    audit.to_csv(out_dir / "00_output_legacy_label_text_audit_v3.csv", index=False, encoding="utf-8-sig")
    if not audit.empty and int(audit["legacy_label_hit_count"].sum()) > 0:
        raise RuntimeError("44_v3 输出中仍发现旧脑区标签残留，请检查 00_output_legacy_label_text_audit_v3.csv。")
    return audit

def read_csv_smart(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, nrows=nrows, low_memory=False)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"读取失败：{path}\n最后错误：{last_err}")


def safe_float_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "NA": np.nan, "N/A": np.nan, "无": np.nan}),
        errors="coerce",
    )


def get_first_series(df: pd.DataFrame, col: str, name: Optional[str] = None) -> pd.Series:
    """安全取得一列。若合并后出现重复列名，df[col] 会返回 DataFrame；这里取第一列，避免 pandas concat 报错。"""
    x = df[col]
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    if name is not None:
        x = x.rename(name)
    return x


def subject_key_overlap(a: pd.Series, b: pd.Series) -> Dict[str, Any]:
    ak = set(a.dropna().astype(str))
    bk = set(b.dropna().astype(str))
    ov = sorted(ak & bk, key=lambda x: (len(x), x))
    return {
        "n_a": len(ak),
        "n_b": len(bk),
        "n_overlap": len(ov),
        "overlap_examples": " | ".join(ov[:20]),
        "candidate_only_examples": " | ".join(sorted(ak - bk, key=lambda x: (len(x), x))[:20]),
        "clinical_only_examples": " | ".join(sorted(bk - ak, key=lambda x: (len(x), x))[:20]),
    }


def is_numeric_like(df: pd.DataFrame, col: str, min_nonmissing: int = 6) -> bool:
    if col not in df.columns:
        return False
    x = safe_float_series(df[col])
    return int(x.notna().sum()) >= min_nonmissing


def has_keyword(text: Any, keywords: Sequence[str]) -> bool:
    nt = norm_text(text)
    return any(norm_text(k) in nt for k in keywords if str(k).strip())


def contains_all_terms(text: Any, term: str) -> bool:
    nt = norm_text(text)
    tokens = [t for t in re.split(r"[\s_\-:/\\|,;，；()（）\[\]{}<>]+", str(term)) if t.strip()]
    return all(norm_text(t) in nt for t in tokens)


def find_first_col(cols: Iterable[str], keywords: Sequence[str]) -> Optional[str]:
    cols = list(cols)
    for kw in keywords:
        nkw = norm_text(kw)
        for c in cols:
            if norm_text(c) == nkw:
                return c
    for kw in keywords:
        nkw = norm_text(kw)
        for c in cols:
            if nkw and nkw in norm_text(c):
                return c
    return None


def bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray([np.nan if x is None else x for x in pvals], dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    if mask.sum() == 0:
        return q
    pv = p[mask]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    q_ranked = ranked * m / (np.arange(m) + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)
    q_sub = np.empty_like(q_ranked)
    q_sub[order] = q_ranked
    q[mask] = q_sub
    return q


def zscore(s: pd.Series) -> pd.Series:
    x = safe_float_series(s)
    sd = x.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (x - x.mean()) / sd


def extract_subject_key(x: Any) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "":
        return None
    # 优先取数字，兼容 sub-preACT0887 / 887。
    digits = re.findall(r"\d+", s)
    if digits:
        joined = "".join(digits)
        # 去前导零，但保留全零情况。
        stripped = joined.lstrip("0")
        return stripped if stripped else "0"
    return norm_text(s)


def canonical_group_value(x: Any) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if not s:
        return None
    if any(k.lower() in s for k in GROUP_KEYWORDS["WL"]):
        return "WL"
    if any(k.lower() in s for k in GROUP_KEYWORDS["TMS"]):
        return "TMS"
    if any(k.lower() in s for k in GROUP_KEYWORDS["PSY"]):
        return "PSY"
    return None


def detect_subject_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "subject_key", "subject_digits", "subject_id", "sub_id", "participant_id", "participant", "subject", "sub", "id",
        "被试编号", "标准编号", "编号", "被试", "ID",
    ]
    c = find_first_col(df.columns, candidates)
    if c:
        return c
    # 如果列名不明显，寻找唯一值较多且含数字的列。
    best = None
    best_score = -1
    for col in df.columns[:80]:
        vals = df[col].dropna().astype(str).head(200).tolist()
        if not vals:
            continue
        digit_ratio = np.mean([bool(re.search(r"\d", v)) for v in vals])
        unique_n = df[col].nunique(dropna=True)
        score = digit_ratio * 10 + min(unique_n, 100) / 50
        if score > best_score and unique_n >= max(5, len(df) * 0.2):
            best_score = score
            best = col
    return best


def detect_group_col(df: pd.DataFrame) -> Optional[str]:
    best_col = None
    best_score = -1
    for col in df.columns:
        if df[col].nunique(dropna=True) > 40:
            continue
        vals = df[col].astype(str).str.lower().fillna("").tolist()
        joined = " ".join(vals[:500])
        score = 0
        for canon, kws in GROUP_KEYWORDS.items():
            if any(k.lower() in joined for k in kws):
                score += 5
        if has_keyword(col, ["group", "treatment", "arm", "condition", "干预", "治疗", "组别", "分组"]):
            score += 5
        if score > best_score:
            best_score = score
            best_col = col
    return best_col if best_score > 0 else None


def detect_covariate_cols(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    out = {
        "age": find_first_col(df.columns, ["age", "年龄"]),
        "sex": find_first_col(df.columns, ["sex", "gender", "性别"]),
        "site": find_first_col(df.columns, ["site", "center", "站点", "中心"]),
        "meanFD": find_first_col(df.columns, ["pre_meanFD", "meanFD_pre", "mean_fd_pre", "meanFD", "mean_fd", "fd", "头动"]),
    }
    return out

# =============================================================================
# 搜索候选边表和临床表
# =============================================================================

def iter_csv_files(roots: Sequence[Path], max_files: int = 6000) -> List[Path]:
    seen = set()
    out: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file() and root.suffix.lower() == ".csv":
            key = str(root.resolve())
            if key not in seen:
                out.append(root)
                seen.add(key)
            continue
        for p in root.rglob("*.csv"):
            if len(out) >= max_files:
                return out
            key = str(p.resolve())
            if key not in seen:
                out.append(p)
                seen.add(key)
    return out


def candidate_table_score(path: Path) -> Dict[str, Any]:
    score = 0
    reasons = []
    name = norm_text(path.name)
    full = norm_text(str(path))
    raw_path = str(path)
    if "v42" in full or "v4_2" in raw_path or "valuecolumnonly" in full or "无旧标签" in raw_path:
        score += 500; reasons.append("优先：43_v4.2 value_column-only 安全候选边表")
    if "03subjectlevelcoreedges" in name or "subjectlevelcoreedges" in name:
        score += 180; reasons.append("文件名像43号subject-level core edges")
    if "43" in full and ("候选边" in str(path) or "coreedges" in name or "固定候选" in str(path)):
        score += 90; reasons.append("路径像43号固定候选边结果")
    if "43_固定候选边_治疗选择机制可塑性wl负控分析结果" in full and "valuecolumnonly" not in full:
        score -= 400; reasons.append("旧43结果包，含旧脑区标签，强扣分")
    if any(k in name for k in ["candidate", "coreedge", "edge", "候选边", "固定候选"]):
        score += 40; reasons.append("文件名含候选边/edge")
    if any(k in name for k in ["summary", "module", "模块", "diagn", "诊断", "readme", "audit", "审计", "result"]):
        # 不是绝对排除，因为 03 表可能在结果文件夹里，但摘要表要扣分。
        score -= 20; reasons.append("文件名像摘要/诊断，扣分")
    try:
        df0 = read_csv_smart(path, nrows=80)
        cols = list(df0.columns)
        if detect_subject_col(df0):
            score += 20; reasons.append("有被试列")
        if detect_group_col(df0):
            score += 20; reasons.append("有分组列")
        edge_like = [c for c in cols if is_edge_like_column_name(c)]
        if len(edge_like) >= 3:
            score += min(60, len(edge_like) * 6); reasons.append(f"疑似边列{len(edge_like)}个")
        important_hits = sum(1 for t in IMPORTANT_EDGE_TERMS if any(contains_all_terms(c, t) for c in cols))
        if important_hits:
            score += 30 + min(important_hits, 20); reasons.append(f"匹配重要候选边关键词{important_hits}个")
    except Exception as e:
        reasons.append(f"预览失败:{repr(e)[:80]}")
        score -= 30
    return {"path": str(path), "score": score, "reasons": "; ".join(reasons)}


def clinical_table_score(path: Path) -> Dict[str, Any]:
    score = 0
    reasons = []
    name_raw = path.name
    name = norm_text(path.name)
    full_raw = str(path)
    full = norm_text(full_raw)
    if "04临床量表清洗后修正版" in full or "临床量表清洗后" in full_raw:
        score += 160; reasons.append("文件名像04临床量表清洗后")
    # v2.2：同名旧表和修正版常常并存；优先选择修正版，避免 subject_key 对不上。
    if "修正版" in full_raw or "修正" in full_raw or "fixed" in full or "corrected" in full:
        score += 90; reasons.append("文件名含修正版/修正，优先")
    if "重跑合并版" in full_raw or "final" in full or "正式" in full_raw:
        score += 20; reasons.append("文件名像较新正式版本")
    if "05第二步纵向分析被试层主表" in full or "第二步纵向分析" in full_raw and "主表" in full_raw:
        score += 100; reasons.append("文件名像05第二步被试层主表")
    if any(k in full_raw for k in ["临床", "量表", "clinical"]):
        score += 40; reasons.append("路径/文件名含临床量表")
    if any(k in name for k in ["brainnetome", "aal", "fc", "edge", "matrix", "纵向fc宽表"]):
        score -= 50; reasons.append("文件名像FC宽表，扣分")
    if any(k in name for k in ["summary", "module", "模块", "diagn", "诊断", "readme", "audit", "审计"]):
        score -= 40; reasons.append("文件名像摘要/诊断，扣分")
    try:
        df0 = read_csv_smart(path, nrows=80)
        cols = list(df0.columns)
        if detect_subject_col(df0):
            score += 20; reasons.append("有被试列")
        if detect_group_col(df0):
            score += 10; reasons.append("有分组列")
        clin_cols = [c for c in cols if has_keyword(c, CLINICAL_KEYWORDS)]
        if len(clin_cols) >= 2:
            score += min(80, len(clin_cols) * 5); reasons.append(f"疑似临床列{len(clin_cols)}个")
        prepost_hits = [c for c in clin_cols if parse_timepoint(c)[0] in ["pre", "post", "delta", "improvement"]]
        if len(prepost_hits) >= 2:
            score += min(60, len(prepost_hits) * 5); reasons.append(f"疑似pre/post/change临床列{len(prepost_hits)}个")
    except Exception as e:
        reasons.append(f"预览失败:{repr(e)[:80]}")
        score -= 30
    return {"path": str(path), "score": score, "reasons": "; ".join(reasons)}


def choose_table(manual: Optional[str], roots: Sequence[Path], out_dir: Path, kind: str) -> Tuple[Path, pd.DataFrame]:
    if manual:
        p = Path(manual)
        if not p.exists():
            raise FileNotFoundError(f"指定的 {kind} CSV 不存在：{p}")
        log(f"使用手动指定{kind}表：{p}")
        return p, read_csv_smart(p)
    log(f"未手动指定{kind}表，开始自动搜索 CSV……")
    files = iter_csv_files(roots)
    log(f"共发现 CSV {len(files)} 个，开始给{kind}表评分。")
    scorer = candidate_table_score if kind == "候选边" else clinical_table_score
    scored = pd.DataFrame([scorer(p) for p in files]).sort_values("score", ascending=False)
    write_csv(scored, out_dir / ("00a_候选边表自动搜索评分.csv" if kind == "候选边" else "00b_临床表自动搜索评分.csv"))
    if scored.empty or float(scored.iloc[0]["score"]) < 30:
        raise RuntimeError(f"没有找到合适的{kind}表。请用 --candidate_csv 或 --clinical_csv 手动指定。")
    p = Path(scored.iloc[0]["path"])
    log(f"自动选择{kind}表：{p}")
    log(f"选择理由：{scored.iloc[0].get('reasons', '')}")
    return p, read_csv_smart(p)


def choose_clinical_table_with_overlap(manual: Optional[str], roots: Sequence[Path], out_dir: Path, candidate_subject_keys: Sequence[Any]) -> Tuple[Path, pd.DataFrame]:
    """v2.2：临床表不只按文件名评分，还按与43号候选边表的 subject_key 重叠数选择。

    这样可以避免自动选到旧版 04_临床量表清洗后.csv，导致合并 rows=0。
    """
    if manual:
        return choose_table(manual, roots, out_dir, kind="临床")
    cand_keys = set(pd.Series(candidate_subject_keys).dropna().astype(str))
    log("未手动指定临床表，开始按 文件名评分 + subject_key重叠数 自动搜索 CSV……")
    files = iter_csv_files(roots)
    log(f"共发现 CSV {len(files)} 个，开始给临床表评分并计算编号重叠。")
    rows = []
    for pth in files:
        rec = clinical_table_score(pth)
        rec["overlap_n"] = 0
        rec["overlap_examples"] = ""
        rec["subject_col_preview"] = ""
        if rec.get("score", 0) >= 30:
            try:
                prev = read_csv_smart(pth, nrows=1000)
                subj_col = detect_subject_col(prev)
                rec["subject_col_preview"] = subj_col or ""
                if subj_col:
                    keys = prev[subj_col].map(extract_subject_key).dropna().astype(str)
                    ov = sorted(cand_keys & set(keys), key=lambda x: (len(x), x))
                    rec["overlap_n"] = len(ov)
                    rec["overlap_examples"] = " | ".join(ov[:15])
                    if len(ov) > 0:
                        rec["score"] = rec.get("score", 0) + min(220, len(ov) * 4)
                        rec["reasons"] = str(rec.get("reasons", "")) + f"; 与候选边表subject_key重叠{len(ov)}个"
                    else:
                        rec["score"] = rec.get("score", 0) - 80
                        rec["reasons"] = str(rec.get("reasons", "")) + "; 与候选边表subject_key重叠0个，扣分"
            except Exception as e:
                rec["reasons"] = str(rec.get("reasons", "")) + f"; 重叠预览失败:{repr(e)[:60]}"
        rows.append(rec)
    scored = pd.DataFrame(rows).sort_values(["overlap_n", "score"], ascending=[False, False])
    write_csv(scored, out_dir / "00b_临床表自动搜索评分.csv")
    if scored.empty or float(scored.iloc[0].get("score", 0)) < 30 or int(scored.iloc[0].get("overlap_n", 0)) == 0:
        raise RuntimeError("没有找到与候选边表 subject_key 有重叠的临床表。请用 --clinical_csv 手动指定 04_临床量表清洗后_修正版.csv。")
    p = Path(scored.iloc[0]["path"])
    log(f"自动选择临床表：{p}")
    log(f"选择理由：{scored.iloc[0].get('reasons', '')}")
    return p, read_csv_smart(p)

# =============================================================================
# 候选边表标准化
# =============================================================================

def is_edge_like_column_name(col: str) -> bool:
    n = norm_text(col)
    raw = str(col)
    low = raw.lower()
    # v3 正式安全表：EDGE_01__pre/post/delta。
    if re.search(r"^EDGE_\d{2}__(pre|post|delta)$", raw, flags=re.I):
        return True
    # v2.1 修正：43号被试层候选边表的标准列名是 “ROI1–ROI2__pre/post/delta”。
    # v3 不再把旧 ROI 名作为正式身份；旧列如果进入，会在候选表安全校验处硬停。
    if re.search(r"__(pre|post|delta)$", low):
        return True
    if has_keyword(raw, EXCLUDE_EDGE_COL_KEYWORDS) and not any(contains_all_terms(raw, t) for t in IMPORTANT_EDGE_TERMS):
        # 注意：如果列名本身含重要边关键词，不排除。
        return False
    if any(contains_all_terms(raw, t) for t in IMPORTANT_EDGE_TERMS):
        return True
    # Brainnetome / ROI 连接常见形式
    if re.search(r"BN\d+.*BN\d+", raw, flags=re.I):
        return True
    if "__" in raw and any(k in raw for k in ["A", "BN", "OPC", "fc", "FC"]):
        return True
    if any(k in n for k in ["baselinefc", "pre_fc", "postfc", "deltafc", "changefc", "edgefc", "fc"]):
        # 只有含 fc 还不够，排除 meanFD 已在上面处理。
        return True
    return False


def infer_edge_timepoint(col: str) -> Optional[str]:
    raw = str(col)
    low = raw.lower()
    n = norm_text(raw)
    if any(k in n for k in ["deltafc", "fcchange", "changefc", "deltaz", "prepost", "postpre", "变化"]):
        return "delta"
    if re.search(r"(^|[_\-\.])(delta|change|diff|d)([_\-\.]|$)", low):
        return "delta"
    if re.search(r"(^|[_\-\.])(post|followup|follow_up|fu)([_\-\.]|$)", low) or any(k in raw for k in ["后测", "治疗后", "随访"]):
        return "post"
    if re.search(r"(^|[_\-\.])(pre|baseline|base)([_\-\.]|$)", low) or any(k in raw for k in ["前测", "治疗前", "基线"]):
        return "pre"
    return None


def edge_stem_from_col(col: str) -> str:
    s = str(col)
    m = re.match(r"^(EDGE_\d{2})__(pre|post|delta)$", s, flags=re.I)
    if m:
        return m.group(1).upper()
    # 去掉常见时间点和FC词，但保留 ROI/BN 信息。
    s2 = re.sub(r"(^|[_\-\.])(baseline|base|pre|post|followup|follow_up|fu|delta|change|diff|fc|z|r)([_\-\.]|$)", "_", s, flags=re.I)
    s2 = re.sub(r"(baseline|pre|post|delta|change|diff|fc|FC|功能连接|基线|前测|后测|治疗前|治疗后|变化)", "_", s2)
    s2 = re.sub(r"_+", "_", s2).strip("_")
    return simple_name(s2) if s2 else simple_name(s)


def detect_long_edge_table(df: pd.DataFrame) -> Tuple[bool, Optional[str], Optional[str], Optional[str], Optional[str]]:
    """判断候选边表是否为 long format【长表】。

    注意：wide format【宽表】中常有列名 EDGE__A11m__A32p__baseline_fc，
    不能因为列名含 edge 就误判为 edge label 列。因此这里要求 edge label 列
    通常是非数值列，且唯一值数量明显小于行数。
    """
    exact_candidates = [
        c for c in df.columns
        if norm_text(c) in [norm_text(x) for x in ["edge_label", "edge", "connection", "pair", "roi_pair", "候选边", "连接", "边"]]
    ]
    if not exact_candidates:
        # 只允许较明确的名称，不做宽松 substring 匹配，避免把 EDGE__xxx__baseline_fc 当成长表标签列。
        exact_candidates = [c for c in df.columns if has_keyword(c, ["edge_label", "roi_pair", "connection_label", "候选边名称"])]
    edge_col = None
    for c in exact_candidates:
        if is_numeric_like(df, c, min_nonmissing=5):
            continue
        nunique = df[c].nunique(dropna=True)
        if 1 < nunique <= max(100, len(df) * 0.8):
            edge_col = c
            break
    if not edge_col:
        return False, None, None, None, None

    # 长表的数值列也要求更明确，避免把某条边本身当作 baseline_col。
    baseline_col = find_first_col(df.columns, ["baseline_fc", "pre_fc", "fc_pre", "基线FC", "前测FC"])
    post_col = find_first_col(df.columns, ["post_fc", "fc_post", "后测FC", "治疗后FC"])
    delta_col = find_first_col(df.columns, ["delta_fc", "fc_delta", "change_fc", "fc_change", "FC变化", "deltaFC"])
    if edge_col and (baseline_col or post_col or delta_col):
        return True, edge_col, baseline_col, post_col, delta_col
    return False, edge_col, baseline_col, post_col, delta_col



def validate_value_column_only_candidate_table(df: pd.DataFrame) -> Dict[str, Any]:
    """44_v3 正式版只接受 43_v4.2 的安全被试层候选边表。"""
    edge_ids = sorted({m.group(1).upper() for c in df.columns for m in [re.match(r"^(EDGE_\d{2})__(pre|post|delta|value_column)$", str(c), flags=re.I)] if m})
    required = []
    missing = []
    for eid in [f"EDGE_{i:02d}" for i in range(1, 11)]:
        for suf in ["pre", "post", "delta", "value_column"]:
            col = f"{eid}__{suf}"
            required.append(col)
            if col not in df.columns:
                missing.append(col)
    if missing:
        raise RuntimeError(
            "44_v3 正式安全版需要读取 43_v4.2 的 value_column-only 候选边被试表；"
            f"当前候选边表缺少这些列：{missing[:12]}。请先运行 43_v4_2，或用 --candidate_csv 手动指定 "
            "03_subject_level_core_edges_TMS_PSY_WL_v4_2_value_column_only.csv。"
        )
    # 如果出现旧脑区名列，硬停，防止半脑标签链路回流。
    bad_cols = [c for c in df.columns if output_has_legacy_text(str(c))]
    if bad_cols:
        raise RuntimeError(
            "候选边表中发现旧脑区标签列名，44_v3 不允许使用旧 43 结果作为正式输入。"
            f"示例：{bad_cols[:10]}"
        )
    return {
        "candidate_value_column_only": True,
        "edge_ids": edge_ids,
        "n_required_safe_columns": len(required),
    }

def standardize_candidate_table(raw: pd.DataFrame, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    df = raw.copy()
    value_column_audit = validate_value_column_only_candidate_table(df)
    subj_col = detect_subject_col(df)
    group_col = detect_group_col(df)
    if not subj_col:
        raise RuntimeError("候选边表未能识别被试列。请检查 43 号候选边表是否含 subject/id 列。")
    df["subject_key"] = df[subj_col].map(extract_subject_key)
    if group_col:
        df["group_from_candidate"] = df[group_col].map(canonical_group_value)
    else:
        df["group_from_candidate"] = None

    is_long, edge_col, long_base, long_post, long_delta = detect_long_edge_table(df)
    audit = {
        **value_column_audit,
        "candidate_subject_col": subj_col,
        "candidate_group_col": group_col,
        "candidate_is_long_format": bool(is_long),
        "candidate_long_edge_col": edge_col,
        "candidate_long_baseline_col": long_base,
        "candidate_long_post_col": long_post,
        "candidate_long_delta_col": long_delta,
    }

    if is_long and edge_col:
        log("候选边表看起来是 long format【长表】，正在转换为 subject-level wide format【被试层宽表】。")
        keep = ["subject_key", "group_from_candidate", edge_col]
        value_cols = [c for c in [long_base, long_post, long_delta] if c and c in df.columns]
        long_df = df[keep + value_cols].copy()
        wide_parts = []
        for value_col, suffix in [(long_base, "baseline_fc"), (long_post, "post_fc"), (long_delta, "delta_fc")]:
            if not value_col or value_col not in long_df.columns:
                continue
            tmp = long_df.pivot_table(index="subject_key", columns=edge_col, values=value_col, aggfunc="first")
            tmp.columns = [f"EDGE__{simple_name(c)}__{suffix}" for c in tmp.columns]
            wide_parts.append(tmp)
        if not wide_parts:
            raise RuntimeError("候选边长表转换失败：没有可用的 baseline/post/delta FC 列。")
        wide = pd.concat(wide_parts, axis=1).reset_index()
        group_map = long_df.dropna(subset=["subject_key"]).groupby("subject_key")["group_from_candidate"].agg(lambda x: next((v for v in x if pd.notna(v)), np.nan)).reset_index()
        wide = wide.merge(group_map, on="subject_key", how="left")
        df_std = wide
    else:
        df_std = df.copy()

    # 如果一个被试多行，压缩为一行：数值列取第一个非缺失，分组取第一个非缺失。
    if df_std["subject_key"].duplicated().any():
        log("候选边表存在重复 subject_key，正在压缩为一行一个被试。")
        def first_nonnull(x: pd.Series) -> Any:
            vals = x.dropna()
            return vals.iloc[0] if len(vals) else np.nan
        df_std = df_std.groupby("subject_key", as_index=False).agg(first_nonnull)

    edge_meta = detect_candidate_edges_from_wide(df_std, out_dir)
    return df_std, edge_meta, audit


def detect_candidate_edges_from_wide(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    # v3 优先识别 43_v4.2 安全列：EDGE_01__pre/post/delta + EDGE_01__value_column。
    safe_rows = []
    for i in range(1, 11):
        eid = f"EDGE_{i:02d}"
        base = f"{eid}__pre"
        post = f"{eid}__post"
        delta = f"{eid}__delta"
        vc_col = f"{eid}__value_column"
        if base in df.columns and post in df.columns and delta in df.columns and vc_col in df.columns:
            value_column = str(df[vc_col].dropna().iloc[0]) if df[vc_col].dropna().shape[0] else ""
            safe_rows.append({
                "edge_key": eid.lower(),
                "edge_label": eid,
                "canonical_edge_id": eid,
                "value_column": value_column,
                "baseline_fc_col": base,
                "post_fc_col": post,
                "delta_fc_col": delta,
                "all_matched_cols": f"{base} | {post} | {delta} | {vc_col}",
                "n_baseline_nonmissing": int(safe_float_series(df[base]).notna().sum()),
                "n_delta_nonmissing": int(safe_float_series(df[delta]).notna().sum()),
                "important_edge_keyword_hit": True,
                "edge_identity_policy": "value_column_only_no_anatomical_label",
            })
    if len(safe_rows) == 10:
        edge_meta = pd.DataFrame(safe_rows)
        write_csv(edge_meta, out_dir / "02_自动识别_固定候选边.csv")
        return edge_meta

    numeric_cols = [c for c in df.columns if is_numeric_like(df, c, min_nonmissing=5)]
    edge_like_cols = []
    for c in numeric_cols:
        if is_edge_like_column_name(c):
            edge_like_cols.append(c)
    # 如果 43 号表的列名很干净但没有 fc/BN/A 等标记，尝试纳入非临床、非协变量的数值列。
    if len(edge_like_cols) < 3:
        for c in numeric_cols:
            if c in edge_like_cols:
                continue
            if has_keyword(c, CLINICAL_KEYWORDS + EXCLUDE_EDGE_COL_KEYWORDS):
                continue
            # 排除很明显的统计摘要列
            if df[c].notna().sum() >= 5:
                edge_like_cols.append(c)

    by_stem: Dict[str, Dict[str, Any]] = {}
    for col in edge_like_cols:
        tp = infer_edge_timepoint(col)
        stem = edge_stem_from_col(col)
        key = norm_text(stem)
        if not key:
            key = norm_text(col)
        by_stem.setdefault(key, {"edge_label": stem, "matched_cols": []})["matched_cols"].append(col)
        if tp == "pre":
            by_stem[key].setdefault("baseline_fc_col", col)
        elif tp == "post":
            by_stem[key].setdefault("post_fc_col", col)
        elif tp == "delta":
            by_stem[key].setdefault("delta_fc_col", col)
        else:
            # 无时间点列通常就是 baseline/raw candidate FC。
            by_stem[key].setdefault("raw_fc_col", col)

    rows = []
    for key, d in by_stem.items():
        baseline = d.get("baseline_fc_col") or d.get("raw_fc_col")
        post = d.get("post_fc_col")
        delta = d.get("delta_fc_col")
        if baseline and post and not delta:
            new_delta = f"EDGE__{simple_name(d.get('edge_label', key))}__delta_fc_AUTO_post_minus_pre"
            if new_delta not in df.columns:
                df[new_delta] = safe_float_series(df[post]) - safe_float_series(df[baseline])
            delta = new_delta
        if not baseline:
            continue
        rows.append({
            "edge_key": key,
            "edge_label": d.get("edge_label", key),
            "baseline_fc_col": baseline,
            "post_fc_col": post,
            "delta_fc_col": delta,
            "all_matched_cols": " | ".join(map(str, d.get("matched_cols", []))),
            "n_baseline_nonmissing": int(safe_float_series(df[baseline]).notna().sum()) if baseline else 0,
            "n_delta_nonmissing": int(safe_float_series(df[delta]).notna().sum()) if delta else 0,
            "important_edge_keyword_hit": any(contains_all_terms(baseline, t) or contains_all_terms(d.get("edge_label", ""), t) for t in IMPORTANT_EDGE_TERMS),
        })
    edge_meta = pd.DataFrame(rows)
    if not edge_meta.empty:
        edge_meta = edge_meta.drop_duplicates(subset=["baseline_fc_col"], keep="first")
        # 优先重要候选边，再按非缺失数排序。最多保留 60 条，避免错误纳入过多列。
        edge_meta = edge_meta.sort_values(["important_edge_keyword_hit", "n_baseline_nonmissing"], ascending=[False, False]).head(60)
    write_csv(edge_meta, out_dir / "02_自动识别_固定候选边.csv")
    return edge_meta

# =============================================================================
# 临床表标准化和结局识别
# =============================================================================

def parse_timepoint(col: str) -> Tuple[Optional[str], str]:
    raw = str(col)
    low = raw.lower()
    # 先识别 pre_minus_post / post_minus_pre 这类已经计算好的列。
    if re.search(r"pre.*minus.*post|pre.*post|前.*后", low) or "前减后" in raw:
        stem = re.sub(r"pre.*minus.*post|pre.*post", "", raw, flags=re.I)
        stem = raw.replace("前减后", "").replace("前后差", "")
        return "improvement", simple_name(stem) or simple_name(raw)
    if re.search(r"post.*minus.*pre|post.*pre|后.*前", low) or "后减前" in raw:
        stem = re.sub(r"post.*minus.*pre|post.*pre", "", raw, flags=re.I)
        stem = raw.replace("后减前", "").replace("后前差", "")
        return "worsening_delta", simple_name(stem) or simple_name(raw)
    if re.search(r"(^|[_\-\.])(improvement|improve|reduction|reduce|改善)([_\-\.]|$)", low):
        stem = re.sub(r"(^|[_\-\.])(improvement|improve|reduction|reduce)([_\-\.]|$)", "_", raw, flags=re.I)
        stem = stem.replace("改善", "")
        return "improvement", simple_name(stem)
    if re.search(r"(^|[_\-\.])(delta|change|diff|变化)([_\-\.]|$)", low):
        stem = re.sub(r"(^|[_\-\.])(delta|change|diff)([_\-\.]|$)", "_", raw, flags=re.I)
        stem = stem.replace("变化", "")
        return "delta_unknown", simple_name(stem)
    if re.search(r"(^|[_\-\.])(pre|baseline|base)([_\-\.]|$)", low) or any(k in raw for k in ["前测", "治疗前", "基线"]):
        stem = re.sub(r"(^|[_\-\.])(pre|baseline|base)([_\-\.]|$)", "_", raw, flags=re.I)
        stem = stem.replace("前测", "").replace("治疗前", "").replace("基线", "")
        return "pre", simple_name(stem)
    if re.search(r"(^|[_\-\.])(post|followup|follow_up|fu)([_\-\.]|$)", low) or any(k in raw for k in ["后测", "治疗后", "随访"]):
        stem = re.sub(r"(^|[_\-\.])(post|followup|follow_up|fu)([_\-\.]|$)", "_", raw, flags=re.I)
        stem = stem.replace("后测", "").replace("治疗后", "").replace("随访", "")
        return "post", simple_name(stem)
    return None, simple_name(raw)


def outcome_category(name: str) -> str:
    n = norm_text(name)
    # v2.1 修正：PCL分维度要先于 PCL 总分识别，否则 PCL_B/C/D/E 会被误归为 total。
    if any(k in n for k in ["intrusion", "reexperiencing", "闯入", "再体验"]):
        return "PTSD_intrusion_reexperiencing"
    if any(k in n for k in ["avoid", "回避"]):
        return "PTSD_avoidance"
    if any(k in n for k in ["arousal", "hyperarousal", "警觉", "唤醒"]):
        return "PTSD_arousal"
    if any(k in n for k in ["negative", "nacm", "cognition", "mood", "负性", "认知", "情绪"]):
        return "PTSD_negative_cognition_mood"
    if any(k in n for k in ["pcl", "caps", "ptsd"]):
        return "PTSD_total_or_global"
    if any(k in n for k in ["hamd", "bdi", "phq", "depress", "抑郁"]):
        return "depression"
    if any(k in n for k in ["hama", "bai", "gad", "anxiety", "焦虑"]):
        return "anxiety"
    if any(k in n for k in ["psqi", "sleep", "insomnia", "睡眠", "失眠"]):
        return "sleep"
    if any(k in n for k in ["function", "disability", "whodas", "qol", "quality", "功能", "生活质量"]):
        return "function_or_quality_of_life"
    return "other_clinical"


def improvement_direction(stem: str) -> int:
    n = norm_text(stem)
    if any(norm_text(k) in n for k in HIGHER_IS_BETTER_KEYWORDS):
        return -1
    return 1


def standardize_clinical_table(raw: pd.DataFrame, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    df = raw.copy()
    subj_col = detect_subject_col(df)
    group_col = detect_group_col(df)
    if not subj_col:
        raise RuntimeError("临床表未能识别被试列。请检查 clinical_csv 是否含 subject/id 列。")
    df["subject_key"] = df[subj_col].map(extract_subject_key)
    if group_col:
        df["group_from_clinical"] = df[group_col].map(canonical_group_value)
    else:
        df["group_from_clinical"] = None

    if df["subject_key"].duplicated().any():
        log("临床表存在重复 subject_key，正在压缩为一行一个被试。")
        def first_nonnull(x: pd.Series) -> Any:
            vals = x.dropna()
            return vals.iloc[0] if len(vals) else np.nan
        df = df.groupby("subject_key", as_index=False).agg(first_nonnull)

    clinical_cols = [c for c in df.columns if has_keyword(c, CLINICAL_KEYWORDS) and is_numeric_like(df, c, min_nonmissing=5)]
    by_stem: Dict[str, Dict[str, Any]] = {}
    unmatched_rows = []
    for col in clinical_cols:
        tp, stem = parse_timepoint(col)
        key = norm_text(stem)
        if not key:
            key = norm_text(col)
        if tp is None:
            unmatched_rows.append({"column": col, "reason": "含临床关键词，但未识别pre/post/change/improvement时间点", "n_nonmissing": int(safe_float_series(df[col]).notna().sum())})
            continue
        by_stem.setdefault(key, {"outcome": stem, "cols": []})["cols"].append(col)
        by_stem[key][tp] = col

    outcome_rows = []
    for key, d in by_stem.items():
        outcome = simple_name(d.get("outcome", key))
        pre = d.get("pre")
        post = d.get("post")
        imp = d.get("improvement")
        worsening = d.get("worsening_delta")
        delta_unknown = d.get("delta_unknown")
        direction = improvement_direction(outcome)
        new_col = None
        source = None
        rule = None
        if pre and post:
            new_col = f"OUTCOME_IMPROVEMENT__{outcome}"
            df[new_col] = direction * (safe_float_series(df[pre]) - safe_float_series(df[post]))
            source = "computed_from_pre_post"
            rule = "pre - post；高分更差，正值代表改善" if direction == 1 else "post - pre；高分更好，正值代表改善"
        elif imp:
            new_col = f"OUTCOME_IMPROVEMENT__{outcome}"
            df[new_col] = safe_float_series(df[imp])
            source = "existing_improvement_or_pre_minus_post"
            rule = "直接使用已有改善列，默认正值代表改善"
        elif worsening:
            new_col = f"OUTCOME_IMPROVEMENT__{outcome}"
            df[new_col] = -safe_float_series(df[worsening])
            source = "existing_post_minus_pre_reversed"
            rule = "由 post-minus-pre 取负，正值代表改善"
        elif delta_unknown:
            new_col = f"OUTCOME_DELTA_UNKNOWN__{outcome}"
            df[new_col] = safe_float_series(df[delta_unknown])
            source = "existing_delta_direction_unclear"
            rule = "直接使用delta/change列；请人工确认正值含义"
        else:
            unmatched_rows.append({"column": " | ".join(d.get("cols", [])), "reason": "只有pre或只有post，无法形成改善量", "n_nonmissing": ""})
            continue
        n_nonmiss = int(df[new_col].notna().sum())
        outcome_rows.append({
            "outcome": outcome,
            "category": outcome_category(outcome),
            "improvement_col": new_col,
            "source_type": source,
            "pre_col": pre,
            "post_col": post,
            "existing_col": imp or worsening or delta_unknown,
            "direction_rule": rule,
            "n_nonmissing": n_nonmiss,
            "mean": float(df[new_col].mean()) if n_nonmiss else np.nan,
            "sd": float(df[new_col].std(ddof=1)) if n_nonmiss > 1 else np.nan,
            "all_source_cols": " | ".join(d.get("cols", [])),
        })

    outcome_df = pd.DataFrame(outcome_rows)
    if not outcome_df.empty:
        def pri(row: pd.Series) -> int:
            cat = row.get("category", "")
            out = norm_text(row.get("outcome", ""))
            if "pcl" in out and cat == "PTSD_total_or_global": return 0
            if str(cat).startswith("PTSD"): return 1
            if cat in ["depression", "anxiety", "sleep", "function_or_quality_of_life"]: return 2
            return 3
        outcome_df["priority"] = outcome_df.apply(pri, axis=1)
        outcome_df = outcome_df.sort_values(["priority", "outcome"]).drop(columns=["priority"])
    unmatched_df = pd.DataFrame(unmatched_rows)
    write_csv(outcome_df, out_dir / "01_自动识别_临床结局指标.csv")
    write_csv(unmatched_df, out_dir / "01b_疑似临床列但未形成结局.csv")
    audit = {
        "clinical_subject_col": subj_col,
        "clinical_group_col": group_col,
        "n_raw_clinical_cols_keyword_hit": len(clinical_cols),
        "n_outcomes_created": int(len(outcome_df)),
        "n_unmatched_clinical_like_cols": int(len(unmatched_df)),
    }
    return df, outcome_df, unmatched_df, audit

# =============================================================================
# 合并和模型
# =============================================================================

def merge_candidate_and_clinical(candidate_df: pd.DataFrame, clinical_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    # 避免重复列冲突：保留 subject_key、group，其他同名列加来源后缀。
    cdf = candidate_df.copy()
    ldf = clinical_df.copy()
    cdf["subject_key"] = cdf["subject_key"].astype(str)
    ldf["subject_key"] = ldf["subject_key"].astype(str)

    ov = subject_key_overlap(cdf["subject_key"], ldf["subject_key"])
    write_csv(pd.DataFrame([ov]), out_dir / "00c0_subject_key重叠诊断.csv")

    merged = cdf.merge(ldf, on="subject_key", how="inner", suffixes=("__candidate", "__clinical"))
    # 统一 group。
    if "group_from_candidate" in merged.columns and merged["group_from_candidate"].notna().any():
        merged["group3"] = merged["group_from_candidate"]
    elif "group_from_clinical" in merged.columns:
        merged["group3"] = merged["group_from_clinical"]
    else:
        merged["group3"] = None
    # 如果 candidate 组别缺失，clinical 补。
    if "group_from_clinical" in merged.columns:
        merged["group3"] = merged["group3"].where(merged["group3"].notna(), merged["group_from_clinical"])
    audit_rows = []
    if len(merged) == 0:
        audit_rows.append({"group3": "NO_OVERLAP", "n_merged_rows": 0, "n_unique_subject": 0})
    else:
        for g, sub in merged.groupby("group3", dropna=False):
            audit_rows.append({"group3": g, "n_merged_rows": len(sub), "n_unique_subject": sub["subject_key"].nunique()})
    write_csv(pd.DataFrame(audit_rows), out_dir / "00c_合并后样本审计.csv")
    return merged


def encode_sex(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return safe_float_series(s)
    m = {
        "m": 1, "male": 1, "man": 1, "男": 1, "1": 1,
        "f": 0, "female": 0, "woman": 0, "女": 0, "0": 0,
    }
    return s.astype(str).str.strip().str.lower().map(lambda x: m.get(x, np.nan))


def build_covariate_matrix(d: pd.DataFrame, cov_cols: Sequence[str]) -> Tuple[pd.DataFrame, List[str]]:
    parts = []
    names = []
    for cov in cov_cols:
        if not cov or cov not in d.columns:
            continue
        base = simple_name(cov)
        if has_keyword(cov, ["sex", "gender", "性别"]):
            x = encode_sex(get_first_series(d, cov))
            parts.append(x.rename(base))
            names.append(base)
        elif has_keyword(cov, ["site", "center", "站点", "中心"]):
            cov_s = get_first_series(d, cov)
            cats = cov_s.astype(str).where(cov_s.notna(), np.nan)
            dum = pd.get_dummies(cats, prefix=base, dummy_na=False, drop_first=True)
            # 避免过多站点哑变量
            for c in dum.columns[:5]:
                parts.append(dum[c].astype(float).rename(c))
                names.append(c)
        else:
            parts.append(safe_float_series(get_first_series(d, cov)).rename(base))
            names.append(base)
    if parts:
        Xcov = pd.concat(parts, axis=1)
    else:
        Xcov = pd.DataFrame(index=d.index)
    return Xcov, names


def ols_fit(y: pd.Series, X: pd.DataFrame, term_names: List[str]) -> Dict[str, Any]:
    data = pd.concat([y.rename("y"), X], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(data)
    p = X.shape[1]
    if n <= p + 1 or n < 5:
        return {"ok": False, "error": f"样本量不足 n={n}, p={p}", "n": n}
    yv = data["y"].astype(float).to_numpy()
    Xv = data[X.columns].astype(float).to_numpy()
    # 去掉零方差列，保留 intercept。
    keep = []
    keep_names = []
    for j, name in enumerate(X.columns):
        col = Xv[:, j]
        if name == "Intercept" or np.nanstd(col) > 1e-12:
            keep.append(j); keep_names.append(name)
    Xv = Xv[:, keep]
    p2 = Xv.shape[1]
    if n <= p2 + 1:
        return {"ok": False, "error": f"去零方差后样本量不足 n={n}, p={p2}", "n": n}
    try:
        beta = np.linalg.lstsq(Xv, yv, rcond=None)[0]
        resid = yv - Xv @ beta
        sse = float(np.sum(resid ** 2))
        tss = float(np.sum((yv - yv.mean()) ** 2))
        df_resid = n - p2
        sigma2 = sse / df_resid if df_resid > 0 else np.nan
        xtx_inv = np.linalg.pinv(Xv.T @ Xv)
        se = np.sqrt(np.diag(xtx_inv) * sigma2)
        tvals = beta / se
        if stats is not None:
            pvals = 2 * stats.t.sf(np.abs(tvals), df=df_resid)
        else:
            pvals = 2 * (1 - 0.5 * (1 + np.vectorize(math.erf)(np.abs(tvals) / math.sqrt(2))))
        r2 = 1 - sse / tss if tss > 0 else np.nan
        adj_r2 = 1 - (1 - r2) * (n - 1) / df_resid if df_resid > 0 and np.isfinite(r2) else np.nan
        aic = n * np.log(max(sse / n, 1e-12)) + 2 * p2
        out = {
            "ok": True, "error": "", "n": n, "p_model": p2, "df_resid": df_resid,
            "r2": r2, "adj_r2": adj_r2, "aic": aic,
            "terms": keep_names,
            "beta": dict(zip(keep_names, beta)),
            "se": dict(zip(keep_names, se)),
            "t": dict(zip(keep_names, tvals)),
            "p": dict(zip(keep_names, pvals)),
        }
        return out
    except Exception as e:
        return {"ok": False, "error": repr(e), "n": n}


def run_treatment_selection_model(
    df: pd.DataFrame,
    y_col: str,
    edge_col: str,
    group_col: str,
    cov_cols: Sequence[str],
    min_n: int,
) -> Dict[str, Any]:
    if df.empty:
        return {"model_ok": False, "model_error": "merged表为空，无法建模", "n_key_complete": 0, "n_cov_complete": 0, "n_PSY_key": 0, "n_TMS_key": 0, "n_PSY_cov": 0, "n_TMS_cov": 0}
    needed = [y_col, edge_col, group_col] + [c for c in cov_cols if c in df.columns]
    d = df.loc[:, needed].copy()
    group_s = get_first_series(d, group_col)
    d = d[group_s.isin(["PSY", "TMS"])].copy()
    if d.empty:
        return {"model_ok": False, "model_error": "PSY/TMS筛选后为空", "n_key_complete": 0, "n_cov_complete": 0, "n_PSY_key": 0, "n_TMS_key": 0, "n_PSY_cov": 0, "n_TMS_cov": 0}
    d["y_z"] = zscore(get_first_series(d, y_col))
    d["edge_z"] = zscore(get_first_series(d, edge_col))
    d["tms_vs_psy"] = get_first_series(d, group_col).map({"PSY": 0.0, "TMS": 1.0})
    d["interaction"] = d["edge_z"] * d["tms_vs_psy"]
    Xcov, cov_names = build_covariate_matrix(d, cov_cols)
    # full: treatment + edge + treatment*edge + covariates
    X_full = pd.concat([
        pd.Series(1.0, index=d.index, name="Intercept"),
        d["tms_vs_psy"].rename("TMS_vs_PSY"),
        d["edge_z"].rename("baselineFC_z"),
        d["interaction"].rename("TMSxBaselineFC"),
        Xcov,
    ], axis=1)
    full = ols_fit(d["y_z"], X_full, list(X_full.columns))
    # prognostic: treatment + edge + covariates, no interaction
    X_prog = pd.concat([
        pd.Series(1.0, index=d.index, name="Intercept"),
        d["tms_vs_psy"].rename("TMS_vs_PSY"),
        d["edge_z"].rename("baselineFC_z"),
        Xcov,
    ], axis=1)
    prog = ols_fit(d["y_z"], X_prog, list(X_prog.columns))

    # complete-case counts by group for key vars only and with covariates
    key_cc = pd.concat([get_first_series(d, y_col, y_col), get_first_series(d, edge_col, edge_col), get_first_series(d, group_col, group_col)], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    all_cols = [y_col, edge_col, group_col] + [c for c in cov_cols if c in d.columns]
    all_cc_parts = [get_first_series(d, c, c) for c in all_cols]
    all_cc = pd.concat(all_cc_parts, axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    out: Dict[str, Any] = {
        "n_key_complete": int(len(key_cc)),
        "n_cov_complete": int(len(all_cc)),
        "n_PSY_key": int((key_cc[group_col] == "PSY").sum()) if len(key_cc) else 0,
        "n_TMS_key": int((key_cc[group_col] == "TMS").sum()) if len(key_cc) else 0,
        "n_PSY_cov": int((all_cc[group_col] == "PSY").sum()) if len(all_cc) else 0,
        "n_TMS_cov": int((all_cc[group_col] == "TMS").sum()) if len(all_cc) else 0,
    }
    if out["n_cov_complete"] < min_n or out["n_PSY_cov"] < 3 or out["n_TMS_cov"] < 3:
        out.update({
            "model_ok": False,
            "model_error": f"协变量层完整样本不足：n={out['n_cov_complete']}, PSY={out['n_PSY_cov']}, TMS={out['n_TMS_cov']}",
        })
        return out
    out.update({
        "model_ok": bool(full.get("ok")),
        "model_error": full.get("error", ""),
        "n_model": full.get("n", np.nan),
        "full_adj_r2": full.get("adj_r2", np.nan),
        "full_r2": full.get("r2", np.nan),
        "full_aic": full.get("aic", np.nan),
        "prog_adj_r2": prog.get("adj_r2", np.nan),
        "prog_r2": prog.get("r2", np.nan),
        "prog_aic": prog.get("aic", np.nan),
        "delta_r2_full_minus_prog": (full.get("r2", np.nan) - prog.get("r2", np.nan)) if full.get("ok") and prog.get("ok") else np.nan,
        "delta_aic_full_minus_prog": (full.get("aic", np.nan) - prog.get("aic", np.nan)) if full.get("ok") and prog.get("ok") else np.nan,
        "interaction_beta": full.get("beta", {}).get("TMSxBaselineFC", np.nan) if full.get("ok") else np.nan,
        "interaction_p": full.get("p", {}).get("TMSxBaselineFC", np.nan) if full.get("ok") else np.nan,
        "baselineFC_main_beta_in_full": full.get("beta", {}).get("baselineFC_z", np.nan) if full.get("ok") else np.nan,
        "baselineFC_main_p_in_full": full.get("p", {}).get("baselineFC_z", np.nan) if full.get("ok") else np.nan,
        "baselineFC_prognostic_beta": prog.get("beta", {}).get("baselineFC_z", np.nan) if prog.get("ok") else np.nan,
        "baselineFC_prognostic_p": prog.get("p", {}).get("baselineFC_z", np.nan) if prog.get("ok") else np.nan,
    })
    beta = out.get("interaction_beta", np.nan)
    if np.isfinite(beta):
        out["higher_baselineFC_favors"] = "TMS" if beta > 0 else "PSY"
    else:
        out["higher_baselineFC_favors"] = "unknown"
    if np.isfinite(out.get("interaction_p", np.nan)) and out.get("interaction_p") < 0.05:
        out["signal_interpretation"] = "treatment-selection leaning【更像治疗选择信号】"
    elif np.isfinite(out.get("baselineFC_prognostic_p", np.nan)) and out.get("baselineFC_prognostic_p") < 0.05:
        out["signal_interpretation"] = "general prognostic leaning【更像一般预后信号】"
    elif np.isfinite(out.get("interaction_p", np.nan)) and out.get("interaction_p") < 0.10:
        out["signal_interpretation"] = "trend-level treatment-selection【趋势性治疗选择信号】"
    else:
        out["signal_interpretation"] = "weak_or_null【弱或无证据】"
    return out


def run_delta_process_model(
    df: pd.DataFrame,
    y_col: str,
    delta_col: str,
    group_col: str,
    cov_cols: Sequence[str],
    active_only: bool,
    min_n: int,
) -> Dict[str, Any]:
    if df.empty:
        return {"model_ok": False, "model_error": "merged表为空，无法建模", "n_key_complete": 0, "n_cov_complete": 0}
    needed = [y_col, delta_col, group_col] + [c for c in cov_cols if c in df.columns]
    d = df.loc[:, needed].copy()
    group_s = get_first_series(d, group_col)
    if active_only:
        d = d[group_s.isin(["PSY", "TMS"])].copy()
    else:
        d = d[group_s.isin(["WL"])].copy()
    if d.empty:
        return {"model_ok": False, "model_error": "分组筛选后为空", "n_key_complete": 0, "n_cov_complete": 0}
    d["y_z"] = zscore(get_first_series(d, y_col))
    d["delta_z"] = zscore(get_first_series(d, delta_col))
    if active_only:
        d["tms_vs_psy"] = get_first_series(d, group_col).map({"PSY": 0.0, "TMS": 1.0})
    Xcov, cov_names = build_covariate_matrix(d, cov_cols)
    if active_only:
        X = pd.concat([
            pd.Series(1.0, index=d.index, name="Intercept"),
            d["tms_vs_psy"].rename("TMS_vs_PSY"),
            d["delta_z"].rename("deltaFC_z"),
            Xcov,
        ], axis=1)
    else:
        X = pd.concat([
            pd.Series(1.0, index=d.index, name="Intercept"),
            d["delta_z"].rename("deltaFC_z"),
            Xcov,
        ], axis=1)
    key_cc = pd.concat([get_first_series(d, y_col, y_col), get_first_series(d, delta_col, delta_col), get_first_series(d, group_col, group_col)], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    all_cols = [y_col, delta_col, group_col] + [c for c in cov_cols if c in d.columns]
    all_cc = pd.concat([get_first_series(d, c, c) for c in all_cols], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    out = {
        "n_key_complete": int(len(key_cc)),
        "n_cov_complete": int(len(all_cc)),
        "n_PSY": int((all_cc[group_col] == "PSY").sum()) if active_only and len(all_cc) else 0,
        "n_TMS": int((all_cc[group_col] == "TMS").sum()) if active_only and len(all_cc) else 0,
        "n_WL": int((all_cc[group_col] == "WL").sum()) if (not active_only and len(all_cc)) else 0,
    }
    if out["n_cov_complete"] < min_n:
        out.update({"model_ok": False, "model_error": f"完整样本不足 n={out['n_cov_complete']}"})
        return out
    if active_only and (out["n_PSY"] < 3 or out["n_TMS"] < 3):
        out.update({"model_ok": False, "model_error": f"治疗组分组样本不足 PSY={out['n_PSY']}, TMS={out['n_TMS']}"})
        return out
    fit = ols_fit(d["y_z"], X, list(X.columns))
    out.update({
        "model_ok": bool(fit.get("ok")),
        "model_error": fit.get("error", ""),
        "n_model": fit.get("n", np.nan),
        "adj_r2": fit.get("adj_r2", np.nan),
        "r2": fit.get("r2", np.nan),
        "deltaFC_beta": fit.get("beta", {}).get("deltaFC_z", np.nan) if fit.get("ok") else np.nan,
        "deltaFC_p": fit.get("p", {}).get("deltaFC_z", np.nan) if fit.get("ok") else np.nan,
    })
    return out

# =============================================================================
# 主分析模块
# =============================================================================

def covariate_tiers(merged: pd.DataFrame) -> List[Tuple[str, List[str]]]:
    covs = detect_covariate_cols(merged)
    tiers: List[Tuple[str, List[str]]] = [("M0_no_covariate【无协变量主模型】", [])]
    age_sex = [c for c in [covs.get("age"), covs.get("sex")] if c and c in merged.columns]
    if age_sex:
        tiers.append(("M1_age_sex【年龄性别模型】", age_sex))
    motion = [c for c in [covs.get("age"), covs.get("sex"), covs.get("meanFD")] if c and c in merged.columns]
    if motion and len(motion) > len(age_sex):
        tiers.append(("M2_age_sex_meanFD【头动敏感性模型】", motion))
    return tiers


def run_modules(merged: pd.DataFrame, edge_meta: pd.DataFrame, outcome_df: pd.DataFrame, out_dir: Path, min_n: int) -> Dict[str, pd.DataFrame]:
    results: Dict[str, pd.DataFrame] = {}
    tiers = covariate_tiers(merged)
    group_col = "group3"

    # 创建数值列，避免列名后缀问题。
    for _, e in edge_meta.iterrows():
        for colkey in ["baseline_fc_col", "delta_fc_col"]:
            col = e.get(colkey)
            if isinstance(col, str) and col in merged.columns:
                merged[col] = safe_float_series(merged[col])
    for _, o in outcome_df.iterrows():
        col = o.get("improvement_col")
        if isinstance(col, str) and col in merged.columns:
            merged[col] = safe_float_series(merged[col])

    log("模块1/2启动：固定候选边 × 其他结局的治疗调节效应，以及治疗选择 vs 一般预后比较。")
    rows = []
    total = max(1, len(edge_meta) * len(outcome_df) * len(tiers))
    done = 0
    for _, e in edge_meta.iterrows():
        edge_col = str(e["baseline_fc_col"])
        for _, o in outcome_df.iterrows():
            y_col = str(o["improvement_col"])
            for tier_name, covs in tiers:
                done += 1
                if done == 1 or done == total or done % 50 == 0:
                    log(f"模块1/2进度：{done}/{total}")
                res = run_treatment_selection_model(merged, y_col, edge_col, group_col, covs, min_n=min_n)
                res.update({
                    "edge_label": e.get("edge_label"),
                    "baseline_fc_col": edge_col,
                    "outcome": o.get("outcome"),
                    "outcome_category": o.get("category"),
                    "improvement_col": y_col,
                    "model_tier": tier_name,
                    "covariates": " | ".join(covs),
                })
                rows.append(res)
    mod12 = pd.DataFrame(rows)
    if not mod12.empty:
        for col in ["interaction_p", "baselineFC_prognostic_p"]:
            mod12[col.replace("_p", "_q")] = bh_fdr(mod12[col].tolist()) if col in mod12.columns else np.nan
    write_csv(mod12, out_dir / "10_模块1_其他指标治疗调节效应.csv")
    write_csv(mod12, out_dir / "20_模块2_治疗选择vs一般预后模型比较.csv")
    results["mod12"] = mod12

    log("模块3启动：PSY偏向边与TMS偏向边的症状维度画像。")
    if not mod12.empty:
        m0 = mod12[mod12["model_tier"].astype(str).str.startswith("M0")].copy()
        profile_rows = []
        group_cols = ["edge_label", "outcome_category", "higher_baselineFC_favors"]
        for keys, sub in m0.groupby(group_cols, dropna=False):
            edge_label, cat, favors = keys
            pvals = pd.to_numeric(sub.get("interaction_p"), errors="coerce")
            betas = pd.to_numeric(sub.get("interaction_beta"), errors="coerce")
            profile_rows.append({
                "edge_label": edge_label,
                "outcome_category": cat,
                "higher_baselineFC_favors": favors,
                "n_tests": len(sub),
                "n_p_lt_05": int((pvals < 0.05).sum()),
                "n_p_lt_10": int((pvals < 0.10).sum()),
                "min_interaction_p": float(pvals.min()) if pvals.notna().any() else np.nan,
                "mean_interaction_beta": float(betas.mean()) if betas.notna().any() else np.nan,
                "outcomes": " | ".join(sub["outcome"].astype(str).tolist()),
            })
        profile = pd.DataFrame(profile_rows).sort_values(["edge_label", "min_interaction_p"])
    else:
        profile = pd.DataFrame()
    write_csv(profile, out_dir / "31_模块3_按候选边和症状维度汇总.csv")
    results["profile"] = profile

    log("模块4启动：deltaFC 与症状改善的治疗过程证据，以及 WL 负控。")
    delta_rows = []
    edge_delta = edge_meta[edge_meta["delta_fc_col"].notna()].copy() if not edge_meta.empty else pd.DataFrame()
    total4 = max(1, len(edge_delta) * len(outcome_df) * len(tiers))
    done4 = 0
    for _, e in edge_delta.iterrows():
        delta_col = str(e["delta_fc_col"])
        if delta_col not in merged.columns:
            continue
        merged[delta_col] = safe_float_series(merged[delta_col])
        for _, o in outcome_df.iterrows():
            y_col = str(o["improvement_col"])
            for tier_name, covs in tiers:
                done4 += 1
                if done4 == 1 or done4 == total4 or done4 % 50 == 0:
                    log(f"模块4进度：{done4}/{total4}")
                active_res = run_delta_process_model(merged, y_col, delta_col, group_col, covs, active_only=True, min_n=min_n)
                wl_res = run_delta_process_model(merged, y_col, delta_col, group_col, covs, active_only=False, min_n=max(5, min_n // 2))
                row = {
                    "edge_label": e.get("edge_label"),
                    "delta_fc_col": delta_col,
                    "outcome": o.get("outcome"),
                    "outcome_category": o.get("category"),
                    "improvement_col": y_col,
                    "model_tier": tier_name,
                    "covariates": " | ".join(covs),
                    "active_model_ok": active_res.get("model_ok"),
                    "active_error": active_res.get("model_error"),
                    "active_n": active_res.get("n_model"),
                    "active_n_PSY": active_res.get("n_PSY"),
                    "active_n_TMS": active_res.get("n_TMS"),
                    "active_deltaFC_beta": active_res.get("deltaFC_beta"),
                    "active_deltaFC_p": active_res.get("deltaFC_p"),
                    "WL_model_ok": wl_res.get("model_ok"),
                    "WL_error": wl_res.get("model_error"),
                    "WL_n": wl_res.get("n_model"),
                    "WL_deltaFC_beta": wl_res.get("deltaFC_beta"),
                    "WL_deltaFC_p": wl_res.get("deltaFC_p"),
                }
                delta_rows.append(row)
    mod4 = pd.DataFrame(delta_rows)
    if not mod4.empty:
        mod4["active_deltaFC_q"] = bh_fdr(mod4["active_deltaFC_p"].tolist()) if "active_deltaFC_p" in mod4.columns else np.nan
        mod4["WL_deltaFC_q"] = bh_fdr(mod4["WL_deltaFC_p"].tolist()) if "WL_deltaFC_p" in mod4.columns else np.nan
        def interp(row: pd.Series) -> str:
            ap = row.get("active_deltaFC_p", np.nan)
            wp = row.get("WL_deltaFC_p", np.nan)
            if np.isfinite(ap) and ap < 0.05 and (not np.isfinite(wp) or wp >= 0.10):
                return "active-specific process support【治疗组特异过程支持】"
            if np.isfinite(ap) and ap < 0.10:
                return "trend-level process support【趋势性过程支持】"
            if np.isfinite(wp) and wp < 0.05:
                return "also seen in WL; possible natural fluctuation【等待组也存在，需警惕自然波动】"
            return "weak_or_null【弱或无证据】"
        mod4["process_interpretation"] = mod4.apply(interp, axis=1)
    write_csv(mod4, out_dir / "40_模块4_deltaFC与其他指标改善相关.csv")
    # 单独输出 WL 负控视角
    write_csv(mod4[[c for c in mod4.columns if c.startswith("WL") or c in ["edge_label", "outcome", "outcome_category", "model_tier", "process_interpretation"]]] if not mod4.empty else mod4, out_dir / "41_模块4_治疗组vsWL自然波动负控.csv")
    results["mod4"] = mod4

    log("综合证据评分启动。")
    summary_rows = []
    if not mod12.empty:
        m0 = mod12[mod12["model_tier"].astype(str).str.startswith("M0")].copy()
        for _, row in m0.iterrows():
            score = 0
            ip = row.get("interaction_p", np.nan)
            iq = row.get("interaction_q", np.nan)
            pp = row.get("baselineFC_prognostic_p", np.nan)
            if np.isfinite(ip) and ip < 0.05: score += 3
            elif np.isfinite(ip) and ip < 0.10: score += 1
            if np.isfinite(iq) and iq < 0.10: score += 2
            if str(row.get("signal_interpretation", "")).startswith("treatment-selection"):
                score += 2
            if np.isfinite(pp) and pp < 0.05:
                score -= 1  # 如果只是一般预后，降低“治疗选择”权重
            # 加入模块4过程证据
            process_hit = False
            if not mod4.empty:
                sub4 = mod4[(mod4["edge_label"] == row.get("edge_label")) & (mod4["outcome"] == row.get("outcome")) & (mod4["model_tier"].astype(str).str.startswith("M0"))]
                if len(sub4):
                    ap = pd.to_numeric(sub4["active_deltaFC_p"], errors="coerce")
                    if (ap < 0.05).any():
                        score += 2
                        process_hit = True
            summary_rows.append({
                "edge_label": row.get("edge_label"),
                "outcome": row.get("outcome"),
                "outcome_category": row.get("outcome_category"),
                "higher_baselineFC_favors": row.get("higher_baselineFC_favors"),
                "interaction_beta": row.get("interaction_beta"),
                "interaction_p": row.get("interaction_p"),
                "interaction_q": row.get("interaction_q"),
                "prognostic_p": row.get("baselineFC_prognostic_p"),
                "signal_interpretation": row.get("signal_interpretation"),
                "has_deltaFC_process_p_lt_05": process_hit,
                "evidence_score": score,
            })
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(["evidence_score", "interaction_p"], ascending=[False, True])
    write_csv(summary, out_dir / "50_综合证据评分_候选边x指标.csv")
    results["summary"] = summary
    return results

# =============================================================================
# README
# =============================================================================

def make_readme(
    out_dir: Path,
    candidate_path: Path,
    clinical_path: Path,
    merged: pd.DataFrame,
    edge_meta: pd.DataFrame,
    outcome_df: pd.DataFrame,
    unmatched_df: pd.DataFrame,
    results: Dict[str, pd.DataFrame],
    audit: Dict[str, Any],
) -> str:
    lines = []
    lines.append("44号 v3 value_column-only 结果解读提示")
    lines.append("=" * 90)
    lines.append(f"脚本版本：{SCRIPT_VERSION}")
    lines.append(f"候选边表：{candidate_path}")
    lines.append(f"临床表：{clinical_path}")
    lines.append(f"合并后总行数：{len(merged)}；唯一被试数：{merged['subject_key'].nunique() if 'subject_key' in merged.columns else 'NA'}")
    if len(merged) == 0:
        lines.append("[严重提示] 合并后 rows=0：请查看 00c0_subject_key重叠诊断.csv。通常是临床表选错或被试编号列识别错。")
    if "group3" in merged.columns:
        lines.append(f"合并后分组计数：{merged['group3'].value_counts(dropna=False).to_dict()}")
    lines.append("")
    lines.append("一、输入是否修好了？")
    lines.append(f"- 固定候选边数量：{len(edge_meta)}")
    if len(edge_meta):
        lines.append(f"- 有 deltaFC【功能连接变化】的候选边数量：{int(edge_meta['delta_fc_col'].notna().sum())}")
        lines.append("- 前几条候选边：" + "；".join(edge_meta['edge_label'].astype(str).head(10).tolist()))
    lines.append(f"- 识别到临床结局数量：{len(outcome_df)}")
    if len(outcome_df):
        cats = outcome_df['category'].value_counts(dropna=False).to_dict()
        lines.append(f"- 临床结局类别：{cats}")
        lines.append("- 前几个结局：" + "；".join(outcome_df['outcome'].astype(str).head(12).tolist()))
    if len(unmatched_df):
        lines.append(f"- 仍有 {len(unmatched_df)} 个疑似临床列未形成可分析结局，请看 01b_疑似临床列但未形成结局.csv")
    lines.append("")

    mod12 = results.get("mod12", pd.DataFrame())
    mod4 = results.get("mod4", pd.DataFrame())
    summary = results.get("summary", pd.DataFrame())

    lines.append("二、模块1/2：其他指标是否支持治疗选择效应？")
    if mod12.empty:
        lines.append("- 模块1/2为空：通常说明没有候选边或没有临床结局。")
    else:
        m0 = mod12[mod12['model_tier'].astype(str).str.startswith('M0')].copy()
        ok = int(m0.get('model_ok', pd.Series(dtype=bool)).fillna(False).sum()) if len(m0) else 0
        lines.append(f"- M0无协变量主模型成功数量：{ok}/{len(m0)}")
        if 'interaction_p' in m0.columns:
            p = pd.to_numeric(m0['interaction_p'], errors='coerce')
            q = pd.to_numeric(m0.get('interaction_q', pd.Series(np.nan, index=m0.index)), errors='coerce')
            lines.append(f"- 交互项 p<0.05 数量：{int((p < 0.05).sum())}；p<0.10 数量：{int((p < 0.10).sum())}；q<0.10 数量：{int((q < 0.10).sum())}")
            top = m0[p.notna()].sort_values('interaction_p').head(8)
            if len(top):
                lines.append("- 最靠前的治疗调节结果：")
                for _, r in top.iterrows():
                    lines.append(f"  * {r.get('edge_label')} × {r.get('outcome')}：beta={r.get('interaction_beta'):.4g}, p={r.get('interaction_p'):.4g}, favor={r.get('higher_baselineFC_favors')}")
        if 'baselineFC_prognostic_p' in m0.columns:
            pp = pd.to_numeric(m0['baselineFC_prognostic_p'], errors='coerce')
            lines.append(f"- 一般预后主效应 p<0.05 数量：{int((pp < 0.05).sum())}")
    lines.append("")

    lines.append("三、模块4：deltaFC 是否支持治疗过程，且区别于 WL 自然波动？")
    if mod4.empty:
        lines.append("- 模块4为空：说明候选边表仍没有 post/delta FC【后测/变化功能连接】。")
    else:
        m40 = mod4[mod4['model_tier'].astype(str).str.startswith('M0')].copy()
        ap = pd.to_numeric(m40.get('active_deltaFC_p', pd.Series(dtype=float)), errors='coerce')
        wp = pd.to_numeric(m40.get('WL_deltaFC_p', pd.Series(dtype=float)), errors='coerce')
        lines.append(f"- 治疗组 active deltaFC p<0.05 数量：{int((ap < 0.05).sum())}；p<0.10 数量：{int((ap < 0.10).sum())}")
        lines.append(f"- WL负控 deltaFC p<0.05 数量：{int((wp < 0.05).sum())}")
        top4 = m40[ap.notna()].sort_values('active_deltaFC_p').head(8)
        if len(top4):
            lines.append("- 最靠前的治疗过程结果：")
            for _, r in top4.iterrows():
                lines.append(f"  * {r.get('edge_label')} ΔFC × {r.get('outcome')}：beta={r.get('active_deltaFC_beta'):.4g}, p={r.get('active_deltaFC_p'):.4g}, WL_p={r.get('WL_deltaFC_p')}")
    lines.append("")

    lines.append("四、综合证据评分")
    if summary.empty:
        lines.append("- 综合证据表为空。")
    else:
        lines.append("- 分数最高的候选边×结局：")
        for _, r in summary.head(10).iterrows():
            lines.append(f"  * score={r.get('evidence_score')} | {r.get('edge_label')} × {r.get('outcome')} | p={r.get('interaction_p')} | {r.get('signal_interpretation')}")
    lines.append("")

    lines.append("五、如何写进论文")
    lines.append("- 如果 M0 中 PCL 仍最强，而其他 PTSD 分维度方向一致：可写为主结果具有 PTSD 相关 convergent validity【聚合效度】。")
    lines.append("- 如果抑郁/焦虑/睡眠也有同方向趋势：可写为可能反映 broader affective recovery【更广泛的情绪恢复】。")
    lines.append("- 如果一般预后模型强于交互模型：需保守，说明该边更像 general prognostic marker【一般预后标志物】。")
    lines.append("- 如果交互模型强于一般预后模型：更支持 treatment-selection signal【治疗选择信号】。")
    lines.append("- 如果 deltaFC 在治疗组相关而 WL 不相关：可作为 treatment-related plasticity【治疗相关可塑性】的探索性支持。")
    lines.append("- 如果本报告显示仍只有 PCL 结局，请优先检查 01b 文件，并把其他量表 pre/post 列补进 clinical_csv。")
    return "\n".join(lines) + "\n"

# =============================================================================
# 主程序
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="44号 v3：value_column-only 安全版；读取43_v4.2候选边表 + 临床量表表，移除旧脑区标签输出。")
    parser.add_argument("--candidate_csv", type=str, default=None, help="43号被试层候选边表 CSV，例如 03_subject_level_core_edges_TMS_PSY_WL.csv")
    parser.add_argument("--clinical_csv", type=str, default=None, help="临床量表表 CSV，例如 04_临床量表清洗后_修正版.csv")
    parser.add_argument("--out_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--search_root", type=str, default=None, help="自动搜索根目录；默认当前目录和上两级目录")
    parser.add_argument("--min_n", type=int, default=12, help="模型最小完整样本量，默认12")
    args = parser.parse_args()

    cwd = Path.cwd()
    out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else cwd / DEFAULT_OUT_DIR_NAME)
    log("=" * 100)
    log("44号 v3：其他临床指标_固定候选边辅助验证与机制解释 value_column-only 安全版启动")
    log(f"脚本版本：{SCRIPT_VERSION}")
    log(f"当前目录：{cwd}")
    log(f"输出目录：{out_dir}")
    log("=" * 100)

    if args.search_root:
        roots = [Path(args.search_root)]
    else:
        # 当前第四步目录 + 上级“脚本”目录，能找到 43 号结果和第二步临床表。
        roots = [cwd, cwd.parent, cwd.parent.parent]
        # 去重
        uniq = []
        seen = set()
        for r in roots:
            key = str(r.resolve()) if r.exists() else str(r)
            if key not in seen:
                uniq.append(r); seen.add(key)
        roots = uniq

    candidate_path, candidate_raw = choose_table(args.candidate_csv, roots, out_dir, kind="候选边")

    log("标准化候选边表……")
    candidate_df, edge_meta, candidate_audit = standardize_candidate_table(candidate_raw, out_dir)
    log(f"候选边表标准化完成：subjects={candidate_df['subject_key'].nunique()}, edges={len(edge_meta)}")

    clinical_path, clinical_raw = choose_clinical_table_with_overlap(args.clinical_csv, roots, out_dir, candidate_df["subject_key"].dropna().astype(str).tolist())

    log("标准化临床表并生成改善指标……")
    clinical_df, outcome_df, unmatched_df, clinical_audit = standardize_clinical_table(clinical_raw, out_dir)
    log(f"临床表标准化完成：subjects={clinical_df['subject_key'].nunique()}, outcomes={len(outcome_df)}")

    log("合并候选边表和临床表……")
    merged = merge_candidate_and_clinical(candidate_df, clinical_df, out_dir)
    write_csv(merged.head(200), out_dir / "00d_合并后数据预览前200行.csv")
    log(f"合并完成：rows={len(merged)}, unique_subjects={merged['subject_key'].nunique()}")

    # 样本覆盖审计：每条边×每个结局的最小样本。
    sample_rows = []
    for _, e in edge_meta.iterrows():
        edge_col = e.get("baseline_fc_col")
        if not isinstance(edge_col, str) or edge_col not in merged.columns:
            continue
        for _, o in outcome_df.iterrows():
            y_col = o.get("improvement_col")
            if not isinstance(y_col, str) or y_col not in merged.columns:
                continue
            d = merged[["group3", edge_col, y_col]].copy()
            d[edge_col] = safe_float_series(d[edge_col])
            d[y_col] = safe_float_series(d[y_col])
            cc = d[d["group3"].isin(["PSY", "TMS"])].dropna()
            sample_rows.append({
                "edge_label": e.get("edge_label"),
                "outcome": o.get("outcome"),
                "n_active_complete_key_vars": len(cc),
                "n_PSY": int((cc["group3"] == "PSY").sum()),
                "n_TMS": int((cc["group3"] == "TMS").sum()),
                "n_WL_key_vars": int(merged[merged["group3"].eq("WL")][[edge_col, y_col]].dropna().shape[0]),
            })
    write_csv(pd.DataFrame(sample_rows), out_dir / "00e_候选边x结局_关键变量样本量审计.csv")

    audit = {
        "script_version": SCRIPT_VERSION,
        "candidate_csv": str(candidate_path),
        "clinical_csv": str(clinical_path),
        "out_dir": str(out_dir),
        "candidate_audit": candidate_audit,
        "clinical_audit": clinical_audit,
        "n_candidate_raw_rows": int(len(candidate_raw)),
        "n_clinical_raw_rows": int(len(clinical_raw)),
        "n_candidate_subjects": int(candidate_df["subject_key"].nunique()),
        "n_clinical_subjects": int(clinical_df["subject_key"].nunique()),
        "n_merged_rows": int(len(merged)),
        "n_merged_subjects": int(merged["subject_key"].nunique()),
        "group_counts_merged": merged["group3"].value_counts(dropna=False).to_dict() if "group3" in merged.columns else {},
        "n_edges_detected": int(len(edge_meta)),
        "n_edges_with_deltaFC": int(edge_meta["delta_fc_col"].notna().sum()) if len(edge_meta) else 0,
        "n_outcomes_detected": int(len(outcome_df)),
        "n_unmatched_clinical_like_cols": int(len(unmatched_df)),
        "covariate_cols_detected": detect_covariate_cols(merged),
        "min_n": int(args.min_n),
    }
    write_json(audit, out_dir / "00_输入审计.json")

    if edge_meta.empty:
        log("[警告] 未识别到候选边，后续模块将为空。")
    if outcome_df.empty:
        log("[警告] 未识别到临床结局，后续模块将为空。")
    if merged.empty:
        log("[错误] 候选边表与临床表合并后 rows=0。已输出 00c0_subject_key重叠诊断.csv；本次不继续建模，避免产生误导性报错。")
        empty = pd.DataFrame()
        for fname in [
            "10_模块1_其他指标治疗调节效应.csv",
            "20_模块2_治疗选择vs一般预后模型比较.csv",
            "31_模块3_按候选边和症状维度汇总.csv",
            "40_模块4_deltaFC与其他指标改善相关.csv",
            "41_模块4_治疗组vsWL自然波动负控.csv",
            "50_综合证据评分_候选边x指标.csv",
        ]:
            write_csv(empty, out_dir / fname)
        results = {"mod12": empty, "profile": empty, "mod4": empty, "summary": empty}
        readme = make_readme(out_dir, candidate_path, clinical_path, merged, edge_meta, outcome_df, unmatched_df, results, audit)
        readme_path = write_text(readme, out_dir / "99_结果解读提示.txt")
        scan_outputs_for_legacy_labels(out_dir)
        log(f"请优先查看：{readme_path}")
        return

    results = run_modules(merged, edge_meta, outcome_df, out_dir, min_n=args.min_n)
    readme = make_readme(out_dir, candidate_path, clinical_path, merged, edge_meta, outcome_df, unmatched_df, results, audit)
    readme_path = write_text(readme, out_dir / "99_结果解读提示.txt")

    log("开始执行旧标签残留审计……")
    scan_outputs_for_legacy_labels(out_dir)

    log("=" * 100)
    log("44号 v3 value_column-only 运行完成")
    log(f"请优先查看：{readme_path}")
    log("如果临床结局仍然只有 PCL，请查看：01b_疑似临床列但未形成结局.csv")
    log("=" * 100)


if __name__ == "__main__":
    main()
