# -*- coding: utf-8 -*-
r"""
49_Table1_固定非重复前测临床指标_主文和补充材料一键生成_v5_4_自动清理历史.py

用途【Purpose】
----------------
完整重写 Table 1【表1】生成脚本：
1) 不读取旧版 Table1【表1】结果作为数值来源；旧 Word 只能作为格式参考，不能作为数据源。
2) 主文列固定为 HC / WL / PSY / TMS / Total active treatment。
3) PSY = ACT + MIN；Total active treatment = PSY + TMS，不包括 WL。
4) 使用固定白名单选择非重复 baseline/pre-treatment【基线/治疗前】临床指标，避免同一构念的 HCPTSD 备份列与正式前测列重复进入主文表。
5) 输出主文 Table 1【表1】、补充材料版、候选临床指标识别审计、缺失审计、mean FD 回填审计。
6) P 值列仅为 PSY vs TMS【心理干预 vs TMS】描述性比较；HC/WL 不参与 P 值计算。
7) 自动清理/归档旧版 Table1【表1】脚本和结果；不触碰 40/42/44/46 等正式分析源文件。

运行【Run】
-----------
cd /d "D:\自科＋脑中心论文选题\PAI选题\工作站传输\第四步分析-codex"
python -u 49_Table1_固定非重复前测临床指标_主文和补充材料一键生成_v5_4_自动清理历史.py

重要说明【Important notes】
--------------------------
- 默认安全归档旧版 Table1 文件；如确需永久删除，将 PERMANENT_DELETE_OLD_TABLE1 改为 True。
- 临床指标识别是“自动候选 + 审计输出”：如果自动识别到过多或过少，请先查看
  03_审计与追溯/03_clinical_indicator_detection_audit.csv。
- 主文默认纳入所有自动识别为 selected_for_main 的前测临床指标；如果想只纳入部分指标，
  可在 MAIN_CLINICAL_INDICATOR_INCLUDE_KEYWORDS / EXCLUDE_KEYWORDS 中调整。
"""

from __future__ import annotations

import os
import re
import json
import shutil
import math
import random
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:
    stats = None

# =========================
# 用户可调参数【User parameters】
# =========================
OUTPUT_DIR_NAME = "Table1_正式重制结果_v5_4_固定非重复前测临床指标_自动清理历史"
PERMANENT_DELETE_OLD_TABLE1 = False
INCLUDE_PSY_TMS_P_COLUMN = True
RANDOM_SEED = 20260521
MONTE_CARLO_N = 20000

# 主文临床指标自动纳入控制【Clinical indicator inclusion control】
# 默认：自动识别到的所有 baseline/pre-treatment 临床总分/分量表都进入主文表。
# 如果之后觉得主文太长，可把 AUTO_INCLUDE_ALL_DETECTED_CLINICAL_IN_MAIN 改为 False，
# 再通过 MAIN_CLINICAL_INDICATOR_INCLUDE_KEYWORDS 控制纳入。
AUTO_INCLUDE_ALL_DETECTED_CLINICAL_IN_MAIN = True
MAX_MAIN_CLINICAL_INDICATORS = 40  # 防止误识别过多 item-level 列；超过时会只纳入优先级最高的前 40 个并在报告提醒。
MAIN_CLINICAL_INDICATOR_INCLUDE_KEYWORDS = [
    "pcl", "ptsd", "gad", "phq", "ptgi", "cd-risc", "cdrisc", "risc", "韧性", "复原", "成长",
    "焦虑", "抑郁", "侵入", "回避", "认知", "情绪", "警觉", "睡眠", "isi", "psqi", "bdi", "bai", "caps", "dass", "ies", "des"
]
MAIN_CLINICAL_INDICATOR_EXCLUDE_KEYWORDS = [
    "后测", "post", "posttest", "follow", "随访", "改善", "变化", "差值", "delta", "change", "improvement", "减分",
    "有效", "responder", "分组", "标签", "备注", "说明", "时间", "date", "日期"
]

# 固定的非重复临床指标白名单【Fixed non-duplicated clinical-indicator whitelist】
# 说明：每个 target 只允许一个正式来源进入主文表；HCPTSD 表列仅作为备选来源，不与正式“（前测）”列重复进入。
# 若你希望正文更短，可把 selected_for_main 改为 False，并把该指标放到补充材料。
CANONICAL_CLINICAL_INDICATORS = [
    {
        "target": "pcl5_pre",
        "label": "Baseline PCL-5【基线 PCL-5】",
        "priority": 1,
        "source_candidates": ["PCL_5总分（前测）", "PCL_5总分(前测)", "PCL-5总分（前测）", "PCL-5总分(前测)", "HCPTSD表_PCL_5总分"],
        "selected_for_main": True,
    },
    {
        "target": "pcl_b_intrusion_pre",
        "label": "Baseline PCL-5 intrusion symptoms【基线 PCL-5 侵入症状】",
        "priority": 2,
        "source_candidates": ["PCL_B侵入（前测）", "PCL_B侵入(前测)", "HCPTSD表_PCL_B"],
        "selected_for_main": True,
    },
    {
        "target": "pcl_c_avoidance_pre",
        "label": "Baseline PCL-5 avoidance symptoms【基线 PCL-5 回避症状】",
        "priority": 3,
        "source_candidates": ["PCL_C回避（前测）", "PCL_C回避(前测)", "HCPTSD表_PCL_C"],
        "selected_for_main": True,
    },
    {
        "target": "pcl_d_negative_mood_cognition_pre",
        "label": "Baseline PCL-5 negative cognition/mood【基线 PCL-5 负性认知/情绪】",
        "priority": 4,
        "source_candidates": ["PCL_D认知情绪改变（前测）", "PCL_D认知情绪改变(前测)", "HCPTSD表_PCL_D"],
        "selected_for_main": True,
    },
    {
        "target": "pcl_e_arousal_pre",
        "label": "Baseline PCL-5 arousal symptoms【基线 PCL-5 警觉反应】",
        "priority": 5,
        "source_candidates": ["PCL_E警觉反应（前测）", "PCL_E警觉反应(前测)", "HCPTSD表_PCL_E"],
        "selected_for_main": True,
    },
    {
        "target": "gad7_pre",
        "label": "Baseline GAD-7【基线 GAD-7】",
        "priority": 10,
        "source_candidates": ["GAD_7总分（前测）", "GAD_7总分(前测)", "GAD-7总分（前测）", "GAD-7总分(前测)", "HCPTSD表_GAD_7总分"],
        "selected_for_main": True,
    },
    {
        "target": "phq9_pre",
        "label": "Baseline PHQ-9【基线 PHQ-9】",
        "priority": 11,
        "source_candidates": ["PHQ_9总分（前测）", "PHQ_9总分(前测)", "PHQ-9总分（前测）", "PHQ-9总分(前测)", "HCPTSD表_PHQ_9总分"],
        "selected_for_main": True,
    },
    {
        "target": "ptgi_sf_pre",
        "label": "Baseline PTGI-SF【基线 PTGI-SF】",
        "priority": 20,
        "source_candidates": ["PTGI_SF总分（前测）", "PTGI_SF总分(前测)", "PTGI-SF总分（前测）", "PTGI-SF总分(前测)"],
        "selected_for_main": True,
    },
    {
        "target": "aaq_ii_pre",
        "label": "Baseline AAQ-II【基线 AAQ-II】",
        "priority": 30,
        "source_candidates": ["AAQ_II总分（前测）", "AAQ_II总分(前测)", "AAQ-II总分（前测）", "AAQ-II总分(前测)"],
        "selected_for_main": True,
    },
    {
        "target": "cd_risc_10_pre",
        "label": "Baseline CD-RISC-10【基线 CD-RISC-10】",
        "priority": 31,
        "source_candidates": ["CD_RISC_10总分（前测）", "CD_RISC_10总分(前测)", "CD-RISC-10总分（前测）", "CD-RISC-10总分(前测)"],
        "selected_for_main": True,
    },
    {
        "target": "ffmq_pre",
        "label": "Baseline FFMQ【基线 FFMQ】",
        "priority": 32,
        "source_candidates": ["FFMQ总分（前测）", "FFMQ总分(前测)"],
        "selected_for_main": True,
    },
    {
        "target": "experiential_avoidance_pre",
        "label": "Baseline experiential avoidance【基线 经验性回避】",
        "priority": 40,
        "source_candidates": ["经验性回避（前测）", "经验性回避(前测)"],
        "selected_for_main": True,
    },
    {
        "target": "cognitive_fusion_pre",
        "label": "Baseline cognitive fusion【基线 认知融合】",
        "priority": 41,
        "source_candidates": ["认知融合（前测）", "认知融合(前测)"],
        "selected_for_main": True,
    },
    {
        "target": "cognitive_defusion_pre",
        "label": "Baseline cognitive defusion【基线 认知解离】",
        "priority": 42,
        "source_candidates": ["认知解离（前测）", "认知解离(前测)"],
        "selected_for_main": True,
    },
]


GROUP_COLS = ["HC", "WL", "PSY", "TMS", "Total active treatment"]
GROUP_LABELS = {
    "HC": "HC\n(n = {n})",
    "WL": "WL\n(n = {n})",
    "PSY": "PSY\n(n = {n})",
    "TMS": "TMS\n(n = {n})",
    "Total active treatment": "Total active treatment\n(n = {n})",
}

# =========================
# 基础工具函数【Utilities】
# =========================

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log(msg: str) -> None:
    print(f"[{now_str()}] {msg}", flush=True)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_read_csv(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"无法读取 CSV: {path}; last error={last_err}")


def safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def safe_to_excel(sheets: Dict[str, pd.DataFrame], path: Path) -> None:
    ensure_dir(path.parent)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            sheet_name = re.sub(r"[\\/*?:\[\]]", "_", str(name))[:31] or "Sheet"
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]
            for col_idx, col in enumerate(df.columns, start=1):
                values = [str(col)] + ["" if pd.isna(x) else str(x) for x in df[col].head(200).tolist()]
                width = min(max(len(v) for v in values) + 2, 46)
                ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = max(width, 10)
            ws.freeze_panes = "A2"


def normalize_colname(x: Any) -> str:
    s = str(x).strip().lower()
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"\s+", "", s)
    s = s.replace("－", "-").replace("–", "-").replace("—", "-")
    return s


def normalize_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = s.replace("（", "(").replace("）", ")")
    return re.sub(r"\s+", "", s)


def extract_subject_num(x: Any) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    s0 = s.upper().replace("SUB-", "").replace("SUB", "").replace("SCAN-", "").replace("SCAN", "")
    nums = re.findall(r"[0-9]+", s0)
    if nums:
        n = nums[-1]
        try:
            return str(int(n))
        except Exception:
            return n.lstrip("0") or "0"
    return s0


def standardize_raw_group(x: Any) -> Optional[str]:
    if pd.isna(x):
        return None
    raw = str(x)
    s = raw.strip().upper()
    s = s.replace("心理干预", "PSY").replace("等待名单", "WL").replace("健康对照", "HC")
    s = s.replace("经颅磁刺激", "TMS")
    if "TMS" in s or "SEAT" in s:
        return "TMS"
    if "HC" in s or "CONTROL" in s or "健康" in raw:
        return "HC"
    if "WL" in s or "WAIT" in s:
        return "WL"
    if "ACT" in s:
        return "ACT"
    if "MIN" in s:
        return "MIN"
    if "PSY" in s or "PSYCH" in s:
        return "PSY"
    return None


def to_table_group(raw_group: Any, subject_num: Any = None) -> Optional[str]:
    g = standardize_raw_group(raw_group)
    sn = extract_subject_num(subject_num)
    # 项目内编号纠错规则【project-specific ID corrections】
    if sn in {"887", "1081"}:
        g = "MIN"
    if g in {"ACT", "MIN", "PSY"}:
        return "PSY"
    if g in {"HC", "WL", "TMS"}:
        return g
    return None


def subject_key_from_group_num(group: Any, num: Any) -> Optional[str]:
    n = extract_subject_num(num)
    if n is None:
        return None
    g = standardize_raw_group(group)
    if g == "HC":
        return f"HC__{n}"
    if g in {"WL", "ACT", "MIN", "PSY", "TMS"}:
        return f"{g}__{n}"
    return f"UNK__{n}"


def choose_col(df: pd.DataFrame, include_any: List[str], include_all: Optional[List[str]] = None,
               exclude_any: Optional[List[str]] = None, prefer_any: Optional[List[str]] = None,
               label: str = "") -> Tuple[Optional[str], pd.DataFrame]:
    include_all = include_all or []
    exclude_any = exclude_any or []
    prefer_any = prefer_any or []
    rows = []
    for c in df.columns:
        nc = normalize_colname(c)
        score = 0
        if include_any and any(k.lower() in nc for k in include_any):
            score += 10
        elif include_any:
            score -= 100
        if all(k.lower() in nc for k in include_all):
            score += 10 * len(include_all)
        else:
            score -= 100
        if any(k.lower() in nc for k in exclude_any):
            score -= 100
        for k in prefer_any:
            if k.lower() in nc:
                score += 3
        nonmissing = int(df[c].notna().sum())
        if nonmissing > 0:
            score += 1
        rows.append({"label": label, "column": c, "score": score, "nonmissing": nonmissing})
    aud = pd.DataFrame(rows).sort_values("score", ascending=False) if rows else pd.DataFrame()
    if aud.empty or aud.iloc[0]["score"] < 0:
        return None, aud
    return str(aud.iloc[0]["column"]), aud


def fmt_p(p: Any) -> str:
    try:
        if p is None or pd.isna(p):
            return "—"
        p = float(p)
    except Exception:
        return "—"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def fmt_mean_sd(x: pd.Series, digits: int = 1) -> str:
    s = pd.to_numeric(x, errors="coerce").dropna()
    if len(s) == 0:
        return "NA"
    return f"{s.mean():.{digits}f} ± {s.std(ddof=1):.{digits}f}" if len(s) > 1 else f"{s.mean():.{digits}f} ± NA"


def fmt_mean_sd3(x: pd.Series) -> str:
    s = pd.to_numeric(x, errors="coerce").dropna()
    if len(s) == 0:
        return "NA"
    return f"{s.mean():.3f} ± {s.std(ddof=1):.3f}" if len(s) > 1 else f"{s.mean():.3f} ± NA"


def fmt_n_pct(success: pd.Series, valid: pd.Series) -> str:
    valid_bool = valid.fillna(False).astype(bool)
    denom = int(valid_bool.sum())
    if denom == 0:
        return "NA"
    succ = int((success.fillna(False).astype(bool) & valid_bool).sum())
    return f"{succ}/{denom} ({100 * succ / denom:.1f}%)"


def mannwhitney_p(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    y = pd.to_numeric(y, errors="coerce").dropna()
    if len(x) < 2 or len(y) < 2 or stats is None:
        return np.nan
    try:
        return float(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def fisher_p_two_by_two(a: int, b: int, c: int, d: int) -> float:
    if stats is None:
        return np.nan
    try:
        return float(stats.fisher_exact([[a, b], [c, d]], alternative="two-sided")[1])
    except Exception:
        return np.nan


def chi2_stat(table: np.ndarray) -> float:
    table = np.asarray(table, dtype=float)
    if table.size == 0 or table.sum() == 0:
        return np.nan
    row = table.sum(axis=1, keepdims=True)
    col = table.sum(axis=0, keepdims=True)
    exp = row @ col / table.sum()
    mask = exp > 0
    return float(((table - exp) ** 2 / np.where(mask, exp, np.nan))[mask].sum())


def monte_carlo_permutation_p(groups: pd.Series, cats: pd.Series, n_perm: int = 20000, seed: int = 20260521) -> float:
    df = pd.DataFrame({"g": groups, "c": cats}).dropna()
    if df["g"].nunique() != 2 or df["c"].nunique() < 2:
        return np.nan
    obs = pd.crosstab(df["g"], df["c"]).values
    obs_stat = chi2_stat(obs)
    if pd.isna(obs_stat):
        return np.nan
    rng = np.random.default_rng(seed)
    g = df["g"].to_numpy().copy()
    c = df["c"].to_numpy().copy()
    count = 1
    for _ in range(n_perm):
        gp = rng.permutation(g)
        stat = chi2_stat(pd.crosstab(gp, c).values)
        if stat >= obs_stat - 1e-12:
            count += 1
    return count / (n_perm + 1)

# =========================
# 清理旧版本【Cleanup old Table1 versions】
# =========================

def cleanup_old_table1(root: Path, current_script_name: str, out_dir: Path) -> pd.DataFrame:
    log("清理/归档旧版 Table1 脚本和旧版结果【cleanup/archive old Table1 versions】...")
    archive_dir = ensure_dir(root / f"00_Table1旧版本自动归档_{stamp()}")
    rows = []
    candidates: List[Path] = []
    for p in root.glob("49_Table1*.py"):
        if p.name != current_script_name:
            candidates.append(p)
    supp = root / "论文写作" / "补充材料"
    if supp.exists():
        for p in supp.glob("Table1_正式重制结果*"):
            # 不清理当前输出目录本身
            if p.resolve() != out_dir.resolve():
                candidates.append(p)
    for p in root.glob("Table1_正式重制结果*.zip"):
        candidates.append(p)
    for p in root.glob("Table1_*旧*.zip"):
        candidates.append(p)
    for p in sorted(set(candidates), key=lambda x: str(x)):
        try:
            dest = archive_dir / p.name
            if PERMANENT_DELETE_OLD_TABLE1:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
                action = "deleted"
                dest_s = ""
            else:
                if dest.exists():
                    dest = archive_dir / f"{p.stem}_{stamp()}{p.suffix}"
                shutil.move(str(p), str(dest))
                action = "archived"
                dest_s = str(dest)
            rows.append({"path": str(p), "action": action, "archive_path": dest_s, "status": "OK"})
        except Exception as e:
            rows.append({"path": str(p), "action": "failed", "archive_path": "", "status": str(e)})
    return pd.DataFrame(rows)

# =========================
# 输入定位【Input resolution】
# =========================

def find_first_file(root: Path, exact_relatives: Optional[List[str]] = None, patterns: Optional[List[str]] = None) -> Optional[Path]:
    exact_relatives = exact_relatives or []
    patterns = patterns or []
    for rel in exact_relatives:
        p = root / rel
        if p.exists():
            return p
    hits: List[Path] = []
    for pat in patterns:
        hits.extend(root.glob(pat))
    hits = [p for p in hits if p.is_file()]
    if not hits:
        return None
    # 优先非归档、非旧版目录
    def score(p: Path) -> Tuple[int, int, str]:
        s = str(p)
        bad = int(("旧版本自动归档" in s) or ("archive" in s.lower()) or ("旧" in s))
        return (bad, len(s), s)
    return sorted(hits, key=score)[0]


def resolve_inputs(root: Path) -> Dict[str, Optional[Path]]:
    log("发现正式输入文件【resolving formal input files】...")
    inputs: Dict[str, Optional[Path]] = {}
    inputs["clinical_master"] = find_first_file(root, exact_relatives=[
        "HC＋TMS＋ACT＋WL＋MIN(终版).xlsx",
        "论文分析过程文件备份/HC＋TMS＋ACT＋WL＋MIN(终版).xlsx",
    ], patterns=["**/HC＋TMS＋ACT＋WL＋MIN(终版).xlsx"])
    inputs["core40_subject"] = find_first_file(root, patterns=["**/40_v3*结果*/02_subject_level_core_edges.csv"])
    inputs["core40_covariate_model"] = find_first_file(root, patterns=["**/40_v3*结果*/11_fixed_candidate_moderation_covariates.csv"])
    inputs["longitudinal_fc_clinical_postmeanfd"] = find_first_file(root, exact_relatives=[
        "论文分析过程文件备份/06_Brainnetome246_纵向FC宽表_合并clinical_postmeanFD_重跑合并版.csv"
    ], patterns=["**/06_Brainnetome246_纵向FC宽表_合并clinical_postmeanFD_重跑合并版.csv"])
    inputs["hc_subject_level"] = find_first_file(root, patterns=["**/05b_HC_subject_level_core_edges_from_matrices_v8_6_2.csv"])
    inputs["wl_negative_control"] = find_first_file(root, patterns=["**/03_WL_negative_control_deltaFC_v8_6_2.csv"])
    inputs["baseline_meanfd_included_sample"] = find_first_file(root, patterns=["**/00_age_sex_meanFD模型纳入样本.csv"])
    inputs["baseline_meanfd_matching_detail"] = find_first_file(root, patterns=["**/01_age_sex_meanFD_样本匹配明细.csv"])
    inputs["baseline_meanfd_180"] = find_first_file(root, patterns=["**/baseline_meanFD_180.csv"])
    inputs["site_by_pathway"] = find_first_file(root, patterns=["**/46_v6*结果*/01b_site_by_pathway交叉表.csv"])
    inputs["site_trial_audit"] = find_first_file(root, patterns=["**/46_v6*结果*/01_trial_site与治疗路径混杂审计.csv"])
    return inputs


def read_inputs(inputs: Dict[str, Optional[Path]]) -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    if inputs.get("clinical_master") is None:
        raise FileNotFoundError("找不到临床/人口学主表 HC＋TMS＋ACT＋WL＋MIN(终版).xlsx")
    log("读取人口学/临床主表【demographic/clinical master table】...")
    dfs["clinical_master"] = pd.read_excel(inputs["clinical_master"], sheet_name=0)
    for key in ["core40_subject", "core40_covariate_model", "longitudinal_fc_clinical_postmeanfd", "hc_subject_level", "wl_negative_control", "baseline_meanfd_included_sample", "baseline_meanfd_matching_detail", "baseline_meanfd_180", "site_by_pathway", "site_trial_audit"]:
        p = inputs.get(key)
        if p is None:
            dfs[key] = pd.DataFrame()
        else:
            log(f"读取 {key}: {p.name}")
            dfs[key] = safe_read_csv(p)
    return dfs

# =========================
# 临床指标自动识别【Clinical indicator auto-detection】
# =========================

def is_numeric_like(series: pd.Series) -> Tuple[int, float, float, float]:
    vals = pd.to_numeric(series, errors="coerce")
    n = int(vals.notna().sum())
    return n, float(vals.mean()) if n else np.nan, float(vals.min()) if n else np.nan, float(vals.max()) if n else np.nan


def clinical_col_priority(col: str) -> int:
    nc = normalize_colname(col)
    # 越小越靠前
    if any(k in nc for k in ["pcl_5总分", "pcl-5总分", "pcl5总分", "pcl_pre", "hcptsd表_pcl_5总分"]):
        return 1
    if "pcl" in nc and any(k in nc for k in ["b", "侵入"]):
        return 2
    if "pcl" in nc and any(k in nc for k in ["c", "回避"]):
        return 3
    if "pcl" in nc and any(k in nc for k in ["d", "认知", "情绪"]):
        return 4
    if "pcl" in nc and any(k in nc for k in ["e", "警觉"]):
        return 5
    if "gad" in nc:
        return 10
    if "phq" in nc:
        return 11
    if "ptgi" in nc or "成长" in nc:
        return 20
    if "cd-risc" in nc or "cdrisc" in nc or "韧性" in nc or "复原" in nc:
        return 21
    if any(k in nc for k in ["isi", "psqi", "睡眠"]):
        return 30
    return 50


def make_indicator_label(col: str) -> str:
    s = str(col).strip()
    s = s.replace("（", "(").replace("）", ")")
    # 常用精确映射
    nc = normalize_colname(s)
    mapping = [
        (["pcl_5总分", "pcl-5总分", "pcl5总分", "hcptsd表_pcl_5总分", "pcl_pre"], "Baseline PCL-5【基线 PCL-5】"),
        (["gad_7总分", "gad-7总分", "gad7总分", "hcptsd表_gad_7总分"], "Baseline GAD-7【基线 GAD-7】"),
        (["phq_9总分", "phq-9总分", "phq9总分", "hcptsd表_phq_9总分"], "Baseline PHQ-9【基线 PHQ-9】"),
    ]
    for keys, lab in mapping:
        if any(k in nc for k in keys):
            return lab
    # 其他列保留原始可读名，但去掉前测标识
    cleaned = s
    cleaned = re.sub(r"HCPTSD表[_\-]*", "", cleaned, flags=re.I)
    cleaned = cleaned.replace("(前测)", "").replace("（前测）", "")
    cleaned = cleaned.replace("前测", "").replace("baseline", "").replace("Baseline", "")
    cleaned = re.sub(r"[_\-]+$", "", cleaned).strip(" _-")
    return f"Baseline {cleaned}【基线 {cleaned}】"


def canonical_source_match(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """按照候选顺序寻找正式来源列；列名采用标准化后精确匹配，避免误把备份列重复纳入。"""
    norm_to_col = {normalize_colname(c): c for c in df.columns}
    for cand in candidates:
        nc = normalize_colname(cand)
        if nc in norm_to_col:
            col = norm_to_col[nc]
            n_numeric, _, _, _ = is_numeric_like(df[col])
            if n_numeric >= 5:
                return col
    # 兜底：允许轻微包含匹配，但只在没有精确匹配时使用。
    for cand in candidates:
        nc = normalize_colname(cand)
        for col in df.columns:
            ncol = normalize_colname(col)
            if nc and (nc in ncol or ncol in nc):
                n_numeric, _, _, _ = is_numeric_like(df[col])
                if n_numeric >= 5:
                    return col
    return None


def detect_clinical_indicator_columns(df: pd.DataFrame, audit_dir: Path) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    """按固定白名单选择非重复前测/基线临床指标列。

    与 v5.3 的区别：
    - 不再“所有自动识别列都进主文表”；
    - 同一构念只保留一个正式来源列，例如 PCL_5总分（前测）优先于 HCPTSD表_PCL_5总分；
    - HCPTSD 表列如果只是备份来源，会进入审计但不进入主文表。
    """
    selected: List[Dict[str, str]] = []
    selected_cols = set()
    selected_targets = set()

    # 先按白名单选出主文指标
    for item in CANONICAL_CLINICAL_INDICATORS:
        source_col = canonical_source_match(df, item["source_candidates"])
        if source_col is None:
            selected.append({
                "target": item["target"],
                "source_col": "",
                "label": item["label"],
                "priority": item["priority"],
                "selected_for_main": False,
                "selection_status": "not_found",
            })
            continue
        selected.append({
            "target": item["target"],
            "source_col": str(source_col),
            "label": item["label"],
            "priority": item["priority"],
            "selected_for_main": bool(item.get("selected_for_main", True)),
            "selection_status": "selected_canonical_source",
        })
        selected_cols.add(str(source_col))
        selected_targets.add(item["target"])

    # 再生成全列审计，说明哪些临床列被纳入、哪些是备份/重复/后测/排除
    rows = []
    canonical_candidate_cols = set()
    candidate_to_target = {}
    candidate_to_label = {}
    for item in CANONICAL_CLINICAL_INDICATORS:
        for cand in item["source_candidates"]:
            nc = normalize_colname(cand)
            canonical_candidate_cols.add(nc)
            candidate_to_target[nc] = item["target"]
            candidate_to_label[nc] = item["label"]

    for col in df.columns:
        nc = normalize_colname(col)
        n_numeric, mean_v, min_v, max_v = is_numeric_like(df[col])
        baseline_like = any(k in nc for k in ["前测", "pre", "baseline", "基线", "hcptsd表"])
        clinical_like = any(k.lower() in nc for k in MAIN_CLINICAL_INDICATOR_INCLUDE_KEYWORDS) or any(k in nc for k in ["总分", "分量表", "量表"])
        is_canonical_candidate = False
        matched_target = ""
        matched_label = ""
        for cand_norm in canonical_candidate_cols:
            if nc == cand_norm or (cand_norm and cand_norm in nc):
                is_canonical_candidate = True
                matched_target = candidate_to_target.get(cand_norm, "")
                matched_label = candidate_to_label.get(cand_norm, "")
                break

        if str(col) in selected_cols:
            include = True
            selected_for_main = True
            reason = "selected_canonical_source"
            target = matched_target or next((x["target"] for x in selected if x["source_col"] == str(col)), "")
            label = matched_label or next((x["label"] for x in selected if x["source_col"] == str(col)), "")
            priority = next((x["priority"] for x in selected if x["source_col"] == str(col)), 999)
        elif is_canonical_candidate:
            include = True
            selected_for_main = False
            reason = "canonical_backup_or_duplicate_not_selected"
            target = matched_target
            label = matched_label
            priority = next((x["priority"] for x in selected if x["target"] == matched_target), 999)
        else:
            include = False
            selected_for_main = False
            target = ""
            label = ""
            priority = 999
            if any(k.lower() in nc for k in MAIN_CLINICAL_INDICATOR_EXCLUDE_KEYWORDS):
                reason = "post_or_change_or_followup"
            elif n_numeric < 5:
                reason = "too_few_numeric_values"
            elif any(k in nc for k in ["id", "编号", "subject", "姓名", "name", "组别", "分组", "group", "年龄", "age", "性别", "sex", "教育", "edu", "婚姻", "mar", "work", "工作", "收入", "money", "职业", "carrer", "region", "地区"]):
                reason = "metadata_or_demographic"
            elif baseline_like and clinical_like:
                reason = "clinical_like_but_not_in_fixed_whitelist"
            else:
                reason = "not_selected"

        rows.append({
            "column": str(col),
            "normalized_column": nc,
            "n_numeric": n_numeric,
            "mean": mean_v,
            "min": min_v,
            "max": max_v,
            "baseline_like": baseline_like,
            "clinical_like": clinical_like,
            "included_as_candidate": include,
            "selected_for_main": selected_for_main,
            "target": target,
            "label": label,
            "priority": priority,
            "exclude_or_include_reason": reason,
        })

    audit = pd.DataFrame(rows).sort_values(
        ["included_as_candidate", "selected_for_main", "priority", "column"],
        ascending=[False, False, True, True],
    )
    selected_df = pd.DataFrame(selected)
    safe_to_csv(audit, audit_dir / "03_clinical_indicator_detection_audit.csv")
    safe_to_csv(selected_df, audit_dir / "03b_selected_clinical_indicators_for_table.csv")
    return audit, selected


# =========================
# 输入标准化【Standardization】
# =========================

def standardize_clinical_master(df: pd.DataFrame, audit_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    col_audits = []

    def nonempty_count(col: str) -> int:
        if col is None or col not in df.columns:
            return 0
        return int(df[col].map(lambda v: str(v).strip() if not pd.isna(v) else "").ne("").sum())

    def pick_existing(candidates: List[str], label: str, required: bool = False) -> Optional[str]:
        rows = []
        picked = None
        for c in candidates:
            exists = c in df.columns
            n_nonempty = nonempty_count(c) if exists else 0
            rows.append({"target": label, "column": c, "exists": exists, "nonempty_count": n_nonempty, "selected": False})
            if picked is None and exists and n_nonempty > 0:
                picked = c
        for r in rows:
            r["selected"] = (r["column"] == picked)
        if required and picked is None:
            raise RuntimeError(f"临床主表缺少必要列: {label}; candidates={candidates}")
        col_audits.append(pd.DataFrame(rows))
        return picked

    id_col = pick_existing(["受试者ID", "受试者原始编号", "受试者标准化编号_用于分组", "subject_id", "subject", "编号", "ID"], "subject_id", required=True)
    group_col = pick_existing(["最终分组", "有HC表组别标签", "有HC表组别编码", "分组", "组别", "group", "Group", "治疗分组"], "group", required=False)
    age_col = pick_existing(["年龄", "age", "Age"], "age")
    sex_col = pick_existing(["性别编码", "sex", "Sex", "性别"], "sex")
    edu_col = pick_existing(["教育编码", "edu", "education", "教育", "学历"], "education")
    mar_col = pick_existing(["婚姻编码", "mar", "marriage", "婚姻"], "marriage")
    work_col = pick_existing(["工作状态编码", "work", "工作", "employment"], "work")
    money_col = pick_existing(["收入编码", "money", "收入", "经济状况"], "money")

    fixed_audit = pd.concat(col_audits, ignore_index=True) if col_audits else pd.DataFrame()
    safe_to_csv(fixed_audit, audit_dir / "01_column_detection_audit_demographic_fixed.csv")

    clinical_audit, clinical_indicators = detect_clinical_indicator_columns(df, audit_dir)
    selected_df = pd.DataFrame(clinical_indicators)

    out = pd.DataFrame(index=df.index)
    out["source_row_index"] = np.arange(len(df))
    out["subject_raw"] = df[id_col] if id_col else np.nan
    out["subject_num"] = out["subject_raw"].map(extract_subject_num)

    if group_col:
        out["raw_group"] = df[group_col].map(standardize_raw_group)
    else:
        out["raw_group"] = pd.NA
    missing_g = out["raw_group"].isna()
    out.loc[missing_g, "raw_group"] = out.loc[missing_g, "subject_raw"].map(standardize_raw_group)
    out["table_group"] = [to_table_group(g, n) for g, n in zip(out["raw_group"], out["subject_num"])]
    out["subject_key"] = [subject_key_from_group_num(g, n) for g, n in zip(out["raw_group"], out["subject_num"])]

    out["age"] = pd.to_numeric(df[age_col], errors="coerce") if age_col else np.nan
    out["sex_code"] = pd.to_numeric(df[sex_col], errors="coerce") if sex_col else np.nan
    out["female"] = out["sex_code"].eq(2)
    out["male"] = out["sex_code"].eq(1)
    out["sex_nonmissing"] = out["sex_code"].isin([1, 2])

    out["edu_code"] = pd.to_numeric(df[edu_col], errors="coerce") if edu_col else np.nan
    out["mar_code"] = pd.to_numeric(df[mar_col], errors="coerce") if mar_col else np.nan
    out["work_code"] = pd.to_numeric(df[work_col], errors="coerce") if work_col else np.nan
    out["money_code"] = pd.to_numeric(df[money_col], errors="coerce") if money_col else np.nan

    out["edu_nonmissing"] = out["edu_code"].isin([1, 2, 3, 4])
    out["mar_nonmissing"] = out["mar_code"].isin([1, 2, 3, 4])
    out["work_nonmissing"] = out["work_code"].isin([1, 2, 3, 4])
    out["money_nonmissing"] = out["money_code"].isin([1, 2, 3, 4])

    out["edu_high_school"] = out["edu_code"].eq(1)
    out["edu_college"] = out["edu_code"].eq(2)
    out["edu_bachelor"] = out["edu_code"].eq(3)
    out["edu_master_plus"] = out["edu_code"].eq(4)
    out["college_or_above"] = out["edu_code"].isin([2, 3, 4])
    out["married_cohabiting"] = out["mar_code"].eq(1)
    out["employed_or_studying"] = out["work_code"].isin([1, 2, 3])
    out["income_insufficiency"] = out["money_code"].isin([3, 4])

    for item in clinical_indicators:
        target = item["target"]
        col = item["source_col"]
        out[target] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
        out[f"{target}_source_col"] = col

    out = out[out["subject_num"].notna()].copy()
    value_cols = ["age", "sex_code", "edu_code", "mar_code", "work_code", "money_code"] + [x["target"] for x in clinical_indicators]
    existing_value_cols = [c for c in value_cols if c in out.columns]
    out["nonmissing_score"] = out[existing_value_cols].notna().sum(axis=1)
    out = out.sort_values(["subject_num", "nonmissing_score"], ascending=[True, False])
    dup_audit = out[out.duplicated("subject_num", keep=False)].copy()
    safe_to_csv(dup_audit, audit_dir / "02_clinical_master_duplicate_subject_candidates.csv")
    out = out.drop_duplicates("subject_num", keep="first").drop(columns=["nonmissing_score"])
    return out, fixed_audit, selected_df


def standardize_subject_table(df: pd.DataFrame, default_group: Optional[str] = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["subject_num", "raw_group", "table_group", "subject_key"])
    d = df.copy()
    id_col, _ = choose_col(d, ["subject", "sub", "id", "编号", "subject_num", "subject_key"], exclude_any=["edge"], prefer_any=["subject_key", "subject_num", "subject", "编号"], label="subject")
    group_col, _ = choose_col(d, ["group", "组别", "treatment", "pathway", "arm"], prefer_any=["group", "pathway", "组别"], label="group")
    if id_col is None:
        d["subject_num"] = [extract_subject_num(x) for x in d.index]
    else:
        d["subject_num"] = d[id_col].map(extract_subject_num)
    if group_col:
        d["raw_group"] = d[group_col].map(standardize_raw_group)
    else:
        d["raw_group"] = default_group
    if default_group is not None:
        d["raw_group"] = d["raw_group"].fillna(default_group)
    else:
        d["raw_group"] = d["raw_group"].where(d["raw_group"].notna(), np.nan)
    d["table_group"] = [to_table_group(g, n) for g, n in zip(d["raw_group"], d["subject_num"])]
    d["subject_key"] = [subject_key_from_group_num(g, n) for g, n in zip(d["raw_group"], d["subject_num"])]
    return d


def standardize_core40(df: pd.DataFrame) -> pd.DataFrame:
    d = standardize_subject_table(df)
    if d.empty:
        return d
    cols = {}
    specs = {
        "core_pcl_pre": (["pcl_pre", "pcl5_pre", "pcl", "pcl-5"], ["pre"], ["post", "delta", "improvement", "change", "分量", "pcl_b", "pcl_c", "pcl_d", "pcl_e"]),
        "core_age": (["age", "年龄"], [], []),
        "core_sex_code": (["sex", "性别"], [], []),
        "core_pre_mean_fd": (["pre_meanfd", "pre_mean_fd", "meanfd_pre", "pre_fd", "mean_fd"], [], ["post"]),
    }
    for target, spec in specs.items():
        col, _ = choose_col(df, include_any=spec[0], include_all=spec[1], exclude_any=spec[2], prefer_any=spec[0], label=target)
        cols[target] = col
    for target, col in cols.items():
        d[target] = pd.to_numeric(df[col], errors="coerce") if col else np.nan
    d["included_in_core40_active_moderation_sample"] = d["table_group"].isin(["PSY", "TMS"])
    keep = ["subject_num", "raw_group", "table_group", "subject_key", "included_in_core40_active_moderation_sample", "core_pcl_pre", "core_age", "core_sex_code", "core_pre_mean_fd"]
    return d[keep].drop_duplicates("subject_key")


def standardize_longitudinal(df: pd.DataFrame) -> pd.DataFrame:
    d = standardize_subject_table(df)
    if d.empty:
        return d
    pre_col, _ = choose_col(df, ["pre_meanfd", "pre_mean_fd", "meanfd_pre", "pre_fd", "mean_fd_pre", "meanfd"], exclude_any=["post"], prefer_any=["pre", "meanfd"], label="long_pre_mean_fd")
    post_col, _ = choose_col(df, ["post_meanfd", "post_mean_fd", "meanfd_post", "post_fd", "mean_fd_post"], prefer_any=["post", "meanfd"], label="long_post_mean_fd")
    d["long_pre_mean_fd"] = pd.to_numeric(df[pre_col], errors="coerce") if pre_col else np.nan
    d["post_mean_fd"] = pd.to_numeric(df[post_col], errors="coerce") if post_col else np.nan
    d["available_in_longitudinal_fc_clinical_postmeanfd"] = True
    d["score"] = d[["long_pre_mean_fd", "post_mean_fd"]].notna().sum(axis=1)
    d = d.sort_values(["subject_key", "score"], ascending=[True, False]).drop_duplicates("subject_key", keep="first")
    return d[["subject_num", "raw_group", "table_group", "subject_key", "available_in_longitudinal_fc_clinical_postmeanfd", "long_pre_mean_fd", "post_mean_fd"]]


def standardize_baseline_meanfd_sources(dfs: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    audits = []
    for key in ["baseline_meanfd_included_sample", "baseline_meanfd_matching_detail", "baseline_meanfd_180"]:
        df = dfs.get(key, pd.DataFrame())
        if df.empty:
            continue
        d = standardize_subject_table(df)
        fd_col, aud = choose_col(df, ["meanfd", "mean_fd", "fd", "framewise"], exclude_any=["post"], prefer_any=["meanfd", "mean_fd"], label=f"{key}_pre_mean_fd")
        aud["source_file_key"] = key
        audits.append(aud)
        if fd_col is None:
            continue
        tmp = d[["subject_num", "raw_group", "table_group", "subject_key"]].copy()
        tmp["pre_mean_fd_from_baseline_source"] = pd.to_numeric(df[fd_col], errors="coerce")
        tmp["pre_mean_fd_source_file_key"] = key
        tmp["pre_mean_fd_source_col"] = fd_col
        rows.append(tmp)
    if not rows:
        return pd.DataFrame(), pd.concat(audits, ignore_index=True) if audits else pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out[out["pre_mean_fd_from_baseline_source"].notna()].copy()
    priority = {"baseline_meanfd_180": 1, "baseline_meanfd_included_sample": 2, "baseline_meanfd_matching_detail": 3}
    out["priority"] = out["pre_mean_fd_source_file_key"].map(priority).fillna(99)
    out = out.sort_values(["subject_key", "priority"]).drop_duplicates("subject_key", keep="first")
    return out.drop(columns=["priority"]), pd.concat(audits, ignore_index=True) if audits else pd.DataFrame()

# =========================
# 构建分析样本【Build analytic sample】
# =========================

def build_base_sample(clin: pd.DataFrame, core: pd.DataFrame, longi: pd.DataFrame, hc: pd.DataFrame, wl: pd.DataFrame, baseline_fd: pd.DataFrame, clinical_meta: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    log("构建 Table 1 分析样本【building Table 1 analytic sample】...")
    audits: Dict[str, pd.DataFrame] = {}
    hc_std = standardize_subject_table(hc, default_group="HC")
    wl_std = standardize_subject_table(wl, default_group="WL")
    if wl_std.empty or wl_std["subject_key"].dropna().nunique() < 5:
        wl_std = longi[longi["table_group"].eq("WL")].copy()

    target_rows = []
    if not hc_std.empty:
        target_rows.append(hc_std[["subject_num", "raw_group", "table_group", "subject_key"]].assign(sample_source="42_HC_reference"))
    if not wl_std.empty:
        target_rows.append(wl_std[["subject_num", "raw_group", "table_group", "subject_key"]].assign(sample_source="42_WL_negative_control_or_longitudinal"))
    if not core.empty:
        active = core[core["table_group"].isin(["PSY", "TMS"])][["subject_num", "raw_group", "table_group", "subject_key"]].assign(sample_source="40_core_active")
        target_rows.append(active)
    if not target_rows:
        raise RuntimeError("无法构建目标样本：HC/WL/core40 均为空。")
    target = pd.concat(target_rows, ignore_index=True)
    target = target[target["subject_key"].notna()].drop_duplicates(["table_group", "subject_num"], keep="first")
    audits["target_sample_keys"] = target.copy()

    clin_merge = clin.copy()
    clin_merge = clin_merge.drop(columns=[c for c in ["subject_key", "raw_group", "table_group"] if c in clin_merge.columns], errors="ignore")
    base = target.merge(clin_merge, on="subject_num", how="left", suffixes=("", "_clin"))

    core_keep = core[[c for c in core.columns if c in {"subject_key", "core_pcl_pre", "core_age", "core_sex_code", "core_pre_mean_fd", "included_in_core40_active_moderation_sample"}]].copy()
    base = base.merge(core_keep, on="subject_key", how="left")
    active_mask = base["table_group"].isin(["PSY", "TMS"])
    if "pcl5_pre" in base.columns:
        base.loc[active_mask & base["core_pcl_pre"].notna(), "pcl5_pre"] = base.loc[active_mask & base["core_pcl_pre"].notna(), "core_pcl_pre"]
    base.loc[active_mask & base["core_age"].notna(), "age"] = base.loc[active_mask & base["core_age"].notna(), "core_age"]
    base.loc[active_mask & base["core_sex_code"].notna(), "sex_code"] = base.loc[active_mask & base["core_sex_code"].notna(), "core_sex_code"]
    base["female"] = pd.to_numeric(base["sex_code"], errors="coerce").eq(2)
    base["male"] = pd.to_numeric(base["sex_code"], errors="coerce").eq(1)
    base["sex_nonmissing"] = pd.to_numeric(base["sex_code"], errors="coerce").isin([1, 2])

    long_keep = longi[[c for c in longi.columns if c in {"subject_key", "available_in_longitudinal_fc_clinical_postmeanfd", "long_pre_mean_fd", "post_mean_fd"}]].copy()
    base = base.merge(long_keep, on="subject_key", how="left")

    base["pre_mean_fd"] = np.nan
    base["pre_mean_fd_source"] = pd.Series([pd.NA] * len(base), dtype="object")
    idx = active_mask & base["core_pre_mean_fd"].notna()
    base.loc[idx, "pre_mean_fd"] = base.loc[idx, "core_pre_mean_fd"]
    base.loc[idx, "pre_mean_fd_source"] = "40_core_active_pre_meanFD"
    idx = base["pre_mean_fd"].isna() & base["long_pre_mean_fd"].notna()
    base.loc[idx, "pre_mean_fd"] = base.loc[idx, "long_pre_mean_fd"]
    base.loc[idx, "pre_mean_fd_source"] = "longitudinal_table_pre_meanFD"
    if not baseline_fd.empty:
        bfd = baseline_fd[["subject_key", "pre_mean_fd_from_baseline_source", "pre_mean_fd_source_file_key", "pre_mean_fd_source_col"]].copy()
        base = base.merge(bfd, on="subject_key", how="left")
        idx = base["pre_mean_fd"].isna() & base["pre_mean_fd_from_baseline_source"].notna()
        base.loc[idx, "pre_mean_fd"] = base.loc[idx, "pre_mean_fd_from_baseline_source"]
        base.loc[idx, "pre_mean_fd_source"] = base.loc[idx, "pre_mean_fd_source_file_key"].astype("object")
    else:
        base["pre_mean_fd_from_baseline_source"] = np.nan
        base["pre_mean_fd_source_file_key"] = pd.NA
        base["pre_mean_fd_source_col"] = pd.NA

    base["paired_prepost_fmri"] = base["table_group"].isin(["WL", "PSY", "TMS"])
    base.loc[base["table_group"].eq("HC"), "paired_prepost_fmri"] = pd.NA

    audits["clean_subject_level_table1_sample"] = base.copy()
    pre_fd_cols = ["subject_key", "subject_num", "table_group", "pre_mean_fd", "pre_mean_fd_source", "pre_mean_fd_from_baseline_source", "pre_mean_fd_source_file_key", "post_mean_fd"]
    audits["pre_post_meanFD_fill_audit"] = base[[c for c in pre_fd_cols if c in base.columns]].copy()

    # 各临床指标样本可用性审计
    clin_cols = [c for c in clinical_meta.get("target", pd.Series(dtype=str)).tolist() if c in base.columns]
    avail_rows = []
    for c in clin_cols:
        label = clinical_meta.loc[clinical_meta["target"].eq(c), "label"].iloc[0] if c in clinical_meta["target"].values else c
        for g in GROUP_COLS:
            gd = group_df(base, g)
            avail_rows.append({"target": c, "label": label, "group": g, "n": int(gd["subject_key"].nunique()), "n_nonmissing": int(pd.to_numeric(gd[c], errors="coerce").notna().sum())})
    audits["clinical_indicator_availability_by_group"] = pd.DataFrame(avail_rows)
    return base, audits

# =========================
# 生成 Table 1【Build table】
# =========================

def group_df(base: pd.DataFrame, group: str) -> pd.DataFrame:
    if group == "Total active treatment":
        return base[base["table_group"].isin(["PSY", "TMS"])].copy()
    return base[base["table_group"].eq(group)].copy()


def group_ns(base: pd.DataFrame) -> Dict[str, int]:
    return {g: int(group_df(base, g)["subject_key"].nunique()) for g in GROUP_COLS}


def p_psy_tms_cont(base: pd.DataFrame, col: str) -> str:
    if col not in base.columns:
        return "—"
    p = mannwhitney_p(group_df(base, "PSY")[col], group_df(base, "TMS")[col])
    return fmt_p(p)


def p_psy_tms_binary(base: pd.DataFrame, bool_col: str, denom_col: str) -> str:
    p1 = group_df(base, "PSY")
    p2 = group_df(base, "TMS")
    valid1 = p1[denom_col].fillna(False)
    valid2 = p2[denom_col].fillna(False)
    s1 = int((p1[bool_col].fillna(False) & valid1).sum())
    f1 = int(valid1.sum() - s1)
    s2 = int((p2[bool_col].fillna(False) & valid2).sum())
    f2 = int(valid2.sum() - s2)
    return fmt_p(fisher_p_two_by_two(s1, f1, s2, f2))


def p_psy_tms_education(base: pd.DataFrame) -> str:
    df = base[base["table_group"].isin(["PSY", "TMS"]) & base["edu_nonmissing"].fillna(False)].copy()
    if df.empty:
        return "—"
    p = monte_carlo_permutation_p(df["table_group"], df["edu_code"], n_perm=MONTE_CARLO_N, seed=RANDOM_SEED)
    return fmt_p(p)


def selected_clinical_for_main(clinical_meta: pd.DataFrame) -> pd.DataFrame:
    if clinical_meta is None or clinical_meta.empty:
        return pd.DataFrame(columns=["target", "label", "source_col", "priority", "selected_for_main"])
    df = clinical_meta.copy()
    df = df[df["selected_for_main"].astype(bool)].copy()
    df["priority"] = pd.to_numeric(df["priority"], errors="coerce").fillna(999)
    df = df.sort_values(["priority", "label", "target"])
    return df


def build_main_table(base: pd.DataFrame, clinical_meta: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    log("生成主文 Table 1【main manuscript Table 1】...")
    ns = group_ns(base)
    col_names = ["Variable"] + [GROUP_LABELS[g].format(n=ns[g]) for g in GROUP_COLS]
    if INCLUDE_PSY_TMS_P_COLUMN:
        col_names.append("PSY vs TMS P")
    rows = []
    tests = []

    def add_row(var: str, values: Dict[str, str], p: str = "—", test: str = ""):
        row = {"Variable": var}
        for g in GROUP_COLS:
            row[GROUP_LABELS[g].format(n=ns[g])] = values.get(g, "")
        if INCLUDE_PSY_TMS_P_COLUMN:
            row["PSY vs TMS P"] = p
        rows.append(row)
        tests.append({"Variable": var, "PSY_vs_TMS_P": p, "test": test})

    add_row("N【样本量】", {g: str(ns[g]) for g in GROUP_COLS}, p="—", test="not tested")
    add_row("Age, years【年龄】", {g: fmt_mean_sd(group_df(base, g)["age"]) for g in GROUP_COLS}, p=p_psy_tms_cont(base, "age"), test="Mann-Whitney U test")
    add_row("Female, n/N (%)【女性】", {g: fmt_n_pct(group_df(base, g)["female"], group_df(base, g)["sex_nonmissing"]) for g in GROUP_COLS}, p=p_psy_tms_binary(base, "female", "sex_nonmissing"), test="Fisher exact test")

    edu_p = p_psy_tms_education(base)
    add_row("Education, n/N (%)【教育】", {g: "" for g in GROUP_COLS}, p=edu_p, test=f"Monte Carlo permutation chi-square test, n_perm={MONTE_CARLO_N}")
    for label, col in [
        ("  High school/technical secondary【高中/中专】", "edu_high_school"),
        ("  College diploma【大专】", "edu_college"),
        ("  Bachelor's degree【本科】", "edu_bachelor"),
        ("  Master's or above【硕士及以上】", "edu_master_plus"),
    ]:
        add_row(label, {g: fmt_n_pct(group_df(base, g)[col], group_df(base, g)["edu_nonmissing"]) for g in GROUP_COLS}, p="", test="education category count")

    # 自动临床指标分区
    clin_main = selected_clinical_for_main(clinical_meta)
    if not clin_main.empty:
        add_row("Baseline/pre-treatment clinical indicators【基线/治疗前临床指标】", {g: "" for g in GROUP_COLS}, p="", test="section header")
        for _, item in clin_main.iterrows():
            col = str(item["target"])
            label = str(item["label"])
            if col not in base.columns:
                continue
            add_row(label, {g: fmt_mean_sd(group_df(base, g)[col]) for g in GROUP_COLS}, p=p_psy_tms_cont(base, col), test="Mann-Whitney U test")
    else:
        add_row("Baseline/pre-treatment clinical indicators【基线/治疗前临床指标】", {g: "No eligible clinical indicators detected" for g in GROUP_COLS}, p="—", test="none detected")

    paired_values = {}
    for g in GROUP_COLS:
        gd = group_df(base, g)
        if g == "HC":
            paired_values[g] = "N/A"
        else:
            denom = pd.Series(True, index=gd.index)
            paired_values[g] = fmt_n_pct(gd["paired_prepost_fmri"].fillna(False), denom)
    add_row("Paired pre-post fMRI, n/N (%)【配对 fMRI】", paired_values, p="—", test="not tested; shown descriptively")

    add_row("Pre-treatment mean FD【治疗前平均 FD】", {g: fmt_mean_sd3(group_df(base, g)["pre_mean_fd"]) for g in GROUP_COLS}, p=p_psy_tms_cont(base, "pre_mean_fd"), test="Mann-Whitney U test")
    post_values = {g: ("N/A" if g == "HC" else fmt_mean_sd3(group_df(base, g)["post_mean_fd"])) for g in GROUP_COLS}
    add_row("Post-treatment mean FD【治疗后平均 FD】", post_values, p=p_psy_tms_cont(base, "post_mean_fd"), test="Mann-Whitney U test")

    table = pd.DataFrame(rows, columns=col_names)
    tests_df = pd.DataFrame(tests)
    return table, tests_df


def missingness_table(base: pd.DataFrame, clinical_meta: pd.DataFrame) -> pd.DataFrame:
    clinical_cols = [c for c in clinical_meta.get("target", pd.Series(dtype=str)).tolist() if c in base.columns]
    vars_to_check = ["age", "sex_code", "edu_code"] + clinical_cols + ["pre_mean_fd", "post_mean_fd"]
    rows = []
    for var in vars_to_check:
        for g in GROUP_COLS:
            gd = group_df(base, g)
            n_total = int(gd["subject_key"].nunique())
            n_non = int(pd.to_numeric(gd[var], errors="coerce").notna().sum()) if var in gd.columns else 0
            rows.append({"variable": var, "group": g, "n_total": n_total, "n_nonmissing": n_non, "n_missing": n_total - n_non})
    return pd.DataFrame(rows)


def clinical_indicator_audit(base: pd.DataFrame, clinical_meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, item in clinical_meta.iterrows():
        target = str(item["target"])
        if target not in base.columns:
            continue
        for g in GROUP_COLS:
            gd = group_df(base, g)
            vals = pd.to_numeric(gd[target], errors="coerce")
            rows.append({
                "target": target,
                "label": item.get("label", target),
                "source_col": item.get("source_col", ""),
                "selected_for_main": bool(item.get("selected_for_main", False)),
                "group": g,
                "n_total": int(gd["subject_key"].nunique()),
                "n_nonmissing": int(vals.notna().sum()),
                "mean": vals.mean(),
                "sd": vals.std(ddof=1),
                "min": vals.min(),
                "max": vals.max(),
            })
    return pd.DataFrame(rows)


def variable_dictionary(clinical_meta: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ["sex", "性别", "1 = male; 2 = female", "用户确认编码"],
        ["education", "教育", "1 = high school/technical secondary; 2 = college diploma; 3 = bachelor's degree; 4 = master's or above", "用户确认编码"],
        ["marriage", "婚姻", "1 = married/cohabiting; 2 = stable relationship; 3 = single; 4 = separated", "用户确认编码"],
        ["work", "工作状态", "1 = full-time; 2 = part-time; 3 = student; 4 = unstable/no stable work", "用户确认编码"],
        ["money", "收入/经济", "1 = fully sufficient; 2 = basically sufficient; 3 = basically insufficient; 4 = completely insufficient", "用户确认编码"],
        ["PSY", "心理干预", "ACT + MIN", "主文合并口径"],
        ["Total active treatment", "主动治疗总样本", "PSY + TMS; excludes WL", "主文合并口径"],
        ["Pre-treatment mean FD", "治疗前平均 FD", "baseline/pre-treatment framewise displacement", "影像 QC 协变量"],
        ["Post-treatment mean FD", "治疗后平均 FD", "post-treatment framewise displacement; N/A for HC", "主文扩展行/QC"],
    ]
    if clinical_meta is not None and not clinical_meta.empty:
        for _, r in clinical_meta.iterrows():
            rows.append([r.get("target", ""), str(r.get("label", "")), f"source column = {r.get('source_col', '')}", "自动识别的基线/治疗前临床指标"])
    return pd.DataFrame(rows, columns=["variable", "中文", "coding_or_definition", "source_or_note"])

# =========================
# Word 输出【DOCX】
# =========================

def write_docx_table(table: pd.DataFrame, out_path: Path) -> bool:
    try:
        from docx import Document
        from docx.shared import Pt, Inches
        from docx.enum.section import WD_ORIENTATION
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
    except Exception as e:
        log(f"python-docx 不可用，跳过 Word 输出: {e}")
        return False

    def set_cell_border(cell, top=None, bottom=None, left=None, right=None):
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        for edge, val in [("top", top), ("bottom", bottom), ("left", left), ("right", right)]:
            tag = "w:{}".format(edge)
            element = tcPr.find(qn(tag))
            if element is None:
                element = OxmlElement(tag)
                tcPr.append(element)
            if val is None:
                element.set(qn("w:val"), "nil")
            else:
                element.set(qn("w:val"), val.get("val", "single"))
                element.set(qn("w:sz"), str(val.get("sz", 4)))
                element.set(qn("w:space"), "0")
                element.set(qn("w:color"), val.get("color", "000000"))

    doc = Document()
    sec = doc.sections[0]
    sec.orientation = WD_ORIENTATION.LANDSCAPE
    sec.page_width, sec.page_height = sec.page_height, sec.page_width
    sec.top_margin = Inches(0.45)
    sec.bottom_margin = Inches(0.45)
    sec.left_margin = Inches(0.35)
    sec.right_margin = Inches(0.35)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = p.add_run("Table 1. Demographic, clinical, and neuroimaging quality-control characteristics of the analytic sample")
    r.bold = True
    r.font.size = Pt(9.5)

    tbl = doc.add_table(rows=1, cols=len(table.columns))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    hdr = tbl.rows[0].cells
    for j, col in enumerate(table.columns):
        hdr[j].text = str(col)
        hdr[j].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for para in hdr[j].paragraphs:
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
            for run in para.runs:
                run.bold = True
                run.font.size = Pt(7.2)
        set_cell_border(hdr[j], top={"sz": 12}, bottom={"sz": 6}, left=None, right=None)

    section_headers = ["Education", "Baseline/pre-treatment clinical indicators"]
    for _, row in table.iterrows():
        cells = tbl.add_row().cells
        var = str(row["Variable"])
        for j, col in enumerate(table.columns):
            cells[j].text = "" if pd.isna(row[col]) else str(row[col])
            cells[j].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for para in cells[j].paragraphs:
                para.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
                for run in para.runs:
                    run.font.size = Pt(6.8)
        if var.startswith("  "):
            for para in cells[0].paragraphs:
                para.paragraph_format.left_indent = Inches(0.10)
        if any(var.startswith(h) for h in section_headers):
            for c in cells:
                for para in c.paragraphs:
                    for run in para.runs:
                        run.bold = True
                set_cell_border(c, top={"sz": 4}, bottom=None, left=None, right=None)
        else:
            for c in cells:
                set_cell_border(c, top=None, bottom=None, left=None, right=None)
    for c in tbl.rows[-1].cells:
        set_cell_border(c, bottom={"sz": 8}, left=None, right=None)

    # 列宽，适配 7 列
    widths = [2.75, 1.00, 1.00, 1.00, 1.00, 1.45, 0.95] if len(table.columns) == 7 else [3.0] + [1.1] * (len(table.columns)-1)
    for row in tbl.rows:
        for j, cell in enumerate(row.cells):
            if j < len(widths):
                cell.width = Inches(widths[j])

    note = doc.add_paragraph()
    note.paragraph_format.space_before = Pt(4)
    note.paragraph_format.space_after = Pt(0)
    note_text = (
        "Note. Values are mean ± SD or n/N (%), with percentages calculated using non-missing denominators. "
        "Clinical indicators were automatically detected from baseline/pre-treatment clinical columns and are fully audited in the supplementary output. "
        "PSY was defined as ACT + MIN, and total active treatment was defined as PSY + TMS. "
        "The P column is descriptive and compares PSY with TMS only; HC and WL were not included in P-value calculations. "
        "Continuous variables were compared using Mann–Whitney U tests, binary variables using Fisher exact tests, and education using a Monte Carlo permutation chi-square test. "
        "Treatment pathway and source/design stratum were highly aligned; therefore, PSY–TMS differences should not be interpreted as effects from a fully head-to-head randomized design. "
        "FD = framewise displacement; QC = quality control; SD = standard deviation."
    )
    run = note.add_run(note_text)
    run.font.size = Pt(6.7)
    ensure_dir(out_path.parent)
    doc.save(out_path)
    return True

# =========================
# 报告【Report】
# =========================

def write_report(out_dir: Path, inputs: Dict[str, Optional[Path]], base: pd.DataFrame, table: pd.DataFrame, cleanup_df: pd.DataFrame, clinical_meta: pd.DataFrame) -> None:
    ns = group_ns(base)
    lines: List[str] = []
    lines.append("49 Table1 v5.4 固定非重复前测临床指标运行报告")
    lines.append("=" * 90)
    lines.append(f"运行时间: {now_str()}")
    lines.append("")
    lines.append("一、主样本量")
    for g in GROUP_COLS:
        lines.append(f"- {g}: n = {ns[g]}")
    lines.append("")
    lines.append("二、输入文件")
    for k, p in inputs.items():
        lines.append(f"- {k}: {'FOUND' if p else 'MISSING'} | {p if p else ''}")
    lines.append("")
    lines.append("三、固定纳入的非重复前测/基线临床指标")
    if clinical_meta is None or clinical_meta.empty:
        lines.append("- 未识别到任何候选临床指标。请查看 03_clinical_indicator_detection_audit.csv。")
    else:
        lines.append(f"- 候选临床指标数: {len(clinical_meta)}")
        lines.append(f"- 纳入主文表的临床指标数: {int(clinical_meta['selected_for_main'].astype(bool).sum())}")
        for _, r in clinical_meta.sort_values(["selected_for_main", "priority", "label"], ascending=[False, True, True]).iterrows():
            flag = "MAIN" if bool(r.get("selected_for_main", False)) else "SUPP_ONLY"
            lines.append(f"  [{flag}] {r.get('label', '')} <= {r.get('source_col', '')}")
    lines.append("")
    lines.append("四、关键缺失情况")
    key_vars = ["pre_mean_fd", "post_mean_fd"] + [c for c in clinical_meta.get("target", pd.Series(dtype=str)).tolist() if c in base.columns]
    for var in key_vars:
        lines.append(f"[{var}]")
        for g in GROUP_COLS:
            gd = group_df(base, g)
            if var == "post_mean_fd" and g == "HC":
                lines.append(f"  {g}: N/A for HC")
            else:
                n_non = int(pd.to_numeric(gd[var], errors="coerce").notna().sum()) if var in gd.columns else 0
                lines.append(f"  {g}: {n_non}/{gd['subject_key'].nunique()} non-missing")
    lines.append("")
    lines.append("五、清理旧版本")
    if cleanup_df.empty:
        lines.append("- 没有发现需要清理/归档的旧版 Table1 文件。")
    else:
        lines.append(f"- 已处理 {len(cleanup_df)} 个旧版 Table1 文件/目录。详见 03_审计与追溯/00_old_Table1_cleanup_audit.csv")
    lines.append("")
    lines.append("六、输出")
    lines.append(f"- 主文 CSV: {out_dir / '01_正式正文Table1' / 'Table1_main_auto_clinical_with_total_PSYTMSP.csv'}")
    lines.append(f"- 主文 Excel: {out_dir / '01_正式正文Table1' / 'Table1_main_auto_clinical_with_total_PSYTMSP.xlsx'}")
    lines.append(f"- 主文 Word: {out_dir / '01_正式正文Table1' / 'Table1_main_auto_clinical_with_total_PSYTMSP.docx'}")
    lines.append("")
    lines.append("七、使用口径")
    lines.append("- PSY = ACT + MIN。")
    lines.append("- Total active treatment = PSY + TMS，不包括 WL。")
    lines.append("- P 值列仅为 PSY vs TMS 描述性比较；HC 和 WL 不参与 P 值计算。")
    lines.append("- 临床指标从临床主表自动识别并审计；不从旧 Word 抄写数字。")
    lines.append("- post-treatment mean FD 对 HC 为 N/A。")
    (out_dir / "99_总报告.txt").write_text("\n".join(lines), encoding="utf-8")

# =========================
# 主流程【Main】
# =========================

def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    root = Path.cwd()
    out_dir = root / "论文写作" / "补充材料" / OUTPUT_DIR_NAME
    out_main = ensure_dir(out_dir / "01_正式正文Table1")
    out_supp = ensure_dir(out_dir / "02_正式补充材料")
    out_audit = ensure_dir(out_dir / "03_审计与追溯")
    out_intermediate = ensure_dir(out_dir / "04_中间清洗表")

    log("=" * 100)
    log(f"{Path(__file__).name} 启动")
    log(f"root = {root}")
    log("本脚本为 v5.3 完整重写版本：自动识别全部前测/基线临床指标，不读取旧版 Table1 数字。")
    log("=" * 100)

    cleanup_df = cleanup_old_table1(root, Path(__file__).name, out_dir)
    safe_to_csv(cleanup_df, out_audit / "00_old_Table1_cleanup_audit.csv")

    inputs = resolve_inputs(root)
    input_audit = pd.DataFrame([{"input_key": k, "status": "FOUND" if p else "MISSING", "path": str(p) if p else ""} for k, p in inputs.items()])
    safe_to_csv(input_audit, out_audit / "00_resolved_input_paths.csv")
    dfs = read_inputs(inputs)

    clin, fixed_col_audit, clinical_meta = standardize_clinical_master(dfs["clinical_master"], out_audit)
    safe_to_csv(clin, out_intermediate / "01_clinical_master_subject_level_cleaned.csv")
    safe_to_csv(clinical_meta, out_audit / "03b_selected_clinical_indicators_for_table.csv")

    core = standardize_core40(dfs["core40_subject"])
    safe_to_csv(core, out_intermediate / "02_core40_active_subject_level_standardized.csv")
    longi = standardize_longitudinal(dfs["longitudinal_fc_clinical_postmeanfd"])
    safe_to_csv(longi, out_intermediate / "03_longitudinal_postmeanFD_subject_level_standardized.csv")

    baseline_fd, baseline_fd_audit = standardize_baseline_meanfd_sources(dfs)
    safe_to_csv(baseline_fd, out_intermediate / "04_baseline_meanFD_sources_standardized.csv")
    safe_to_csv(baseline_fd_audit, out_audit / "05_baseline_meanFD_column_detection_audit.csv")

    hc_std = standardize_subject_table(dfs["hc_subject_level"], default_group="HC")
    wl_std = standardize_subject_table(dfs["wl_negative_control"], default_group="WL")
    safe_to_csv(hc_std, out_intermediate / "05_HC_reference_subject_keys_standardized.csv")
    safe_to_csv(wl_std, out_intermediate / "06_WL_negative_control_subject_keys_standardized.csv")

    base, audits = build_base_sample(clin, core, longi, dfs["hc_subject_level"], dfs["wl_negative_control"], baseline_fd, clinical_meta)
    for name, df in audits.items():
        safe_to_csv(df, out_audit / f"{name}.csv")
    safe_to_csv(base, out_intermediate / "10_Table1_final_subject_level_dataset.csv")

    table, tests_df = build_main_table(base, clinical_meta)
    miss_df = missingness_table(base, clinical_meta)
    clin_audit = clinical_indicator_audit(base, clinical_meta)
    var_dict = variable_dictionary(clinical_meta)

    safe_to_csv(table, out_main / "Table1_main_auto_clinical_with_total_PSYTMSP.csv")
    safe_to_csv(tests_df, out_main / "Table1_main_auto_clinical_with_total_PSYTMSP_tests.csv")
    safe_to_csv(miss_df, out_audit / "04_missingness_by_variable_and_group.csv")
    safe_to_csv(clin_audit, out_audit / "06_clinical_indicator_value_range_audit.csv")
    safe_to_csv(var_dict, out_supp / "Supplementary_Table_variable_dictionary_and_coding.csv")

    safe_to_excel({
        "Table1_main": table,
        "PSY_vs_TMS_tests": tests_df,
        "clinical_indicator_detection": pd.read_csv(out_audit / "03_clinical_indicator_detection_audit.csv", encoding="utf-8-sig"),
        "clinical_indicator_audit": clin_audit,
        "missingness": miss_df,
        "variable_dictionary": var_dict,
    }, out_main / "Table1_main_auto_clinical_with_total_PSYTMSP.xlsx")

    safe_to_excel({
        "clean_subject_level": base,
        "clinical_indicator_meta": clinical_meta,
        "clinical_indicator_audit": clin_audit,
        "missingness": miss_df,
        "pre_post_meanFD_audit": audits.get("pre_post_meanFD_fill_audit", pd.DataFrame()),
        "input_paths": input_audit,
        "cleanup": cleanup_df,
        "variable_dictionary": var_dict,
    }, out_dir / "Table1_full_outputs_and_audits.xlsx")

    write_docx_table(table, out_main / "Table1_main_auto_clinical_with_total_PSYTMSP.docx")
    write_report(out_dir, inputs, base, table, cleanup_df, clinical_meta)

    log("=" * 100)
    log("完成。")
    log(f"输出目录: {out_dir}")
    log("请优先查看: 99_总报告.txt")
    log("=" * 100)


if __name__ == "__main__":
    main()
