# -*- coding: utf-8 -*-
r"""
39_v4_codex_固定候选边_value_column谱系审计_无解剖标签正式版.py

用途
----
这是面向当前 Codex 文件夹结构的 39 号安全版脚本。

它不重新跑 38/39/40 的统计模型，只做 value_column【取值列】谱系审计：
1) 从当前 Codex 文件夹中自动定位：
   - 36_v3 全脑扫描结果
   - 39_v3 已生成的候选边 value_column 谱系审计结果
   - 40_v3 preMeanFD 修复版正式固定候选边结果
   - 论文分析过程文件备份中的 Brainnetome246 纵向 FC 宽表
2) 以原始 value_column 作为唯一正式边身份；
3) 禁止把旧 edge_label_mapped / edge_label_short / Axx 脑区名写入正式输出；
4) 审计 10 条固定候选边是否在 36_v3、39_v3、40_v3 和正式宽表中一致存在；
5) 生成供后续 40/42/43/论文表格使用的“无解剖标签正式候选边表”。

重要原则
--------
- 不改变基本计算方法；
- 不重新筛边；
- 不直接命名脑区；
- 不使用旧错误标签；
- 解剖命名和 BrainNet 作图必须由独立审计通过的 246 ROI 映射流程完成，
  不能在本脚本中完成。

推荐运行
--------
cd /d "D:\自科＋脑中心论文选题\PAI选题\工作站传输\第四步分析-codex"
python -u 39_v4_codex_固定候选边_value_column谱系审计_无解剖标签正式版.py

可选显式指定 root：
python -u 39_v4_codex_固定候选边_value_column谱系审计_无解剖标签正式版.py --root "D:\自科＋脑中心论文选题\PAI选题\工作站传输\第四步分析-codex"

输出目录
--------
39_v4_codex_固定候选边_value_column谱系审计_无解剖标签正式版结果
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ENCODINGS = ["utf-8-sig", "utf-8", "gb18030", "gbk"]

DEFAULT_OUT_NAME = "39_v4_codex_固定候选边_value_column谱系审计_无解剖标签正式版结果"

# 明确禁止进入正式输出的旧标签列/解剖命名列
FORBIDDEN_LABEL_COLUMNS = {
    "edge_label_mapped", "edge_label_short", "edge_readable", "readable_edge",
    "label", "labels", "roi_label", "roi_labels", "bna_label", "brainnetome_label",
    "corrected_label", "corrected_bna_label", "corrected_edge_label",
    "node1_label", "node2_label", "roi1_label", "roi2_label",
    "region1", "region2", "anatomical_name", "anatomical_label",
    "edge_name_mapped", "old_edge_name", "legacy_edge_name",
}

# 对正式 value_column 本身的保守检查。unknown_roi / BN 数字列名是安全的；
# A45r/A11m/OPC 这类解剖名不应作为正式 value_column。
ANATOMICAL_TOKEN_RE = re.compile(
    r"(^|[_\-\s])A\d{1,2}[A-Za-z/]*($|[_\-\s])|"
    r"(OPC|LinG|Cing|Amyg|Hipp|STS|Occ|Orb|Frontal|Temporal|Parietal|Limbic)",
    re.IGNORECASE,
)

SAFE_VALUE_COLUMN_RE = re.compile(
    r"(unknown[_\-]?roi[_\-]?\d+.*(__|--|-|_to_|to).*unknown[_\-]?roi[_\-]?\d+)|"
    r"(BN\d{1,3}.*(__|--|-|_to_|to).*BN\d{1,3})|"
    r"(roi\d+.*(__|--|-|_to_|to).*roi\d+)",
    re.IGNORECASE,
)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def ensure_dir(p: Path | str) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_csv_any(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    last = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, nrows=nrows)
        except Exception as e:
            last = e
    raise RuntimeError(f"CSV 读取失败: {path}; last_error={repr(last)}")


def read_table_any(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in [".csv", ".txt", ".tsv"]:
        if suffix == ".tsv":
            last = None
            for enc in ENCODINGS:
                try:
                    return pd.read_csv(path, sep="\t", encoding=enc, low_memory=False, nrows=nrows)
                except Exception as e:
                    last = e
            raise RuntimeError(f"TSV 读取失败: {path}; last_error={repr(last)}")
        return read_csv_any(path, nrows=nrows)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path, nrows=nrows)
    raise ValueError(f"不支持的表格类型: {path}")


def ensure_unique_columns(df: pd.DataFrame, table_name: str = "table") -> pd.DataFrame:
    """Return a copy with unique column names.

    Some upstream CSV files can contain duplicated headers such as two
    ``value_column`` columns after compatibility exports. Pandas then treats
    ``df["value_column"]`` as a DataFrame, and merge() fails with
    "The column label 'value_column' is not unique". For lineage auditing,
    the first occurrence is kept unchanged and later duplicates receive a
    ``__dup`` suffix.
    """
    cols = list(map(str, df.columns))
    seen: Dict[str, int] = {}
    new_cols: List[str] = []
    changed = False
    for c in cols:
        if c not in seen:
            seen[c] = 1
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}__dup{seen[c]}")
            changed = True
    if changed:
        df = df.copy()
        df.columns = new_cols
        log(f"{table_name}: 检测到重复列名，已保留第一列并给后续重复列加 __dup 后缀。")
    return df


def write_csv(df: pd.DataFrame, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def norm_col(c: object) -> str:
    return re.sub(r"\s+", "", str(c)).lower()


def clean_value(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def first_existing(root: Path, rels: Sequence[str]) -> Optional[Path]:
    for rel in rels:
        p = root / rel
        if p.exists():
            return p
    return None


def find_one(root: Path, patterns: Sequence[str], prefer_contains: Sequence[str] = ()) -> Optional[Path]:
    """在 root 下根据 glob pattern 找一个文件/目录。"""
    found: List[Path] = []
    for pat in patterns:
        found.extend(root.glob(pat))
    found = [p for p in found if p.exists()]
    if not found:
        return None

    def score(p: Path) -> Tuple[int, int, str]:
        s = str(p)
        sc = 0
        for t in prefer_contains:
            if t in s:
                sc += 10
        # 越短越可能是目标路径，避免扫到深层备份
        return (sc, -len(p.parts), s)

    found = sorted(set(found), key=score, reverse=True)
    return found[0]


def load_json_if_exists(path: Optional[Path]) -> Dict:
    if not path or not path.exists():
        return {}
    for enc in ENCODINGS:
        try:
            return json.loads(path.read_text(encoding=enc))
        except Exception:
            continue
    return {}


def resolve_paths(root: Path) -> Dict[str, Optional[str]]:
    """根据当前 Codex 文件夹结构自动定位关键输入。"""
    paths: Dict[str, Optional[Path]] = {}

    paths["wide_table"] = first_existing(root, [
        r"论文分析过程文件备份\06_Brainnetome246_纵向FC宽表_合并clinical_postmeanFD_重跑合并版.csv",
        r"06_Brainnetome246_纵向FC宽表_合并clinical_postmeanFD_重跑合并版.csv",
    ]) or find_one(root, [
        "**/06_Brainnetome246_纵向FC宽表_合并clinical_postmeanFD_重跑合并版.csv",
        "**/*Brainnetome246*纵向FC宽表*clinical*postmeanFD*.csv",
    ], prefer_contains=["论文分析过程文件备份"])

    paths["out36_dir"] = first_existing(root, [
        r"36_v3_全脑基线FC治疗调节效应扫描结果_值列身份标准化_无解剖标签版",
    ]) or find_one(root, [
        "36_v3*结果*值列身份标准化*无解剖标签版",
        "**/36_v3*结果*值列身份标准化*无解剖标签版",
    ])

    out36 = paths.get("out36_dir")
    paths["out36_edge_audit"] = (out36 / "00_detected_edge_value_columns_audit.csv") if out36 else None
    paths["out36_all_edges"] = (out36 / "01_fullbrain_treatment_by_baselineFC_interaction_all_edges.csv") if out36 else None

    paths["out39_dir"] = first_existing(root, [
        r"39_v3_稳定性选择候选边_value_column谱系审计_修正BNA标签结果",
    ]) or find_one(root, [
        "39_v3*value_column*结果",
        "**/39_v3*value_column*结果",
    ])

    out39 = paths.get("out39_dir")
    paths["out39_canonical"] = (out39 / "03_canonical_fixed_10_candidate_edges_v3.csv") if out39 else None
    paths["out39_lineage"] = (out39 / "02_fixed_10_value_column_lineage_audit.csv") if out39 else None
    paths["out39_long"] = (out39 / "01_upstream_39_candidate_value_columns_long.csv") if out39 else None

    paths["out40_dir"] = first_existing(root, [
        r"40_v3_稳定性选择候选边_正式固定候选治疗调节和可塑性分析结果_修正BNA标签_preMeanFD修复版",
    ]) or find_one(root, [
        "40_v3*正式固定候选治疗调节*preMeanFD修复版",
        "**/40_v3*正式固定候选治疗调节*preMeanFD修复版",
    ])

    out40 = paths.get("out40_dir")
    paths["out40_mapping"] = (out40 / "01_candidate_edge_mapping.csv") if out40 else None
    paths["out40_subject"] = (out40 / "02_subject_level_core_edges.csv") if out40 else None
    paths["out40_nocov"] = (out40 / "10_fixed_candidate_moderation_no_covariates.csv") if out40 else None
    paths["out40_cov"] = (out40 / "11_fixed_candidate_moderation_covariates.csv") if out40 else None
    paths["out40_final"] = (out40 / "40_final_core_edge_decision_table.csv") if out40 else None
    paths["out40_equiv_summary"] = (out40 / "00_v3_vs_v2_statistical_equivalence_summary.json") if out40 else None
    paths["out40_equiv_report"] = (out40 / "00_v3_vs_v2_statistical_equivalence_report.txt") if out40 else None
    paths["out40_pre_meanFD_audit"] = (out40 / "00_pre_meanFD_recovery_audit.csv") if out40 else None

    paths["brainnet_fix_dir"] = first_existing(root, [
        r"BrainNet_10_candidate_edges_plot_files_audit_and_fix",
    ]) or find_one(root, [
        "BrainNet_10_candidate_edges_plot_files_audit_and_fix",
        "**/BrainNet_10_candidate_edges_plot_files_audit_and_fix",
    ])
    bfix = paths.get("brainnet_fix_dir")
    paths["brainnet_corrected_edges"] = (bfix / "05_candidate_10_edges_clean_corrected.csv") if bfix else None

    return {k: str(v) if v and Path(v).exists() else None for k, v in paths.items()}


def choose_value_column_col(df: pd.DataFrame) -> Optional[str]:
    """从各种上游表中识别原始 value_column 所在列。"""
    priority = [
        "value_column",
        "canonical_value_column",
        "edge_value_column",
        "candidate_value_column",
        "fc_value_column",
        "original_value_column",
        "raw_value_column",
        "wide_value_column",
        "edge_column",
        "edge",
        "raw_edge",
        "original_edge",
        "connection",
    ]
    nmap = {norm_col(c): c for c in df.columns}
    for p in priority:
        if p in nmap:
            return nmap[p]
    # 模糊匹配：同时含 value 和 column 的列
    for c in df.columns:
        nc = norm_col(c)
        if "value" in nc and "column" in nc:
            return c
    return None


def split_value_column(vc: str) -> Tuple[str, str]:
    s = str(vc)
    for sep in ["__", "--", "_to_", " to ", "-"]:
        if sep in s:
            parts = s.split(sep)
            if len(parts) >= 2:
                return parts[0].strip(), parts[1].strip()
    return "", ""


def is_safe_value_column(vc: str, known_wide_cols: Optional[set] = None) -> Tuple[bool, str]:
    s = clean_value(vc)
    if not s:
        return False, "empty"
    if known_wide_cols is not None and s in known_wide_cols:
        # 如果它真实存在于宽表，优先认为是取值列；
        # 但如果明显是 Axx/OPC 等解剖名，仍给出警告而不是直接通过。
        if ANATOMICAL_TOKEN_RE.search(s) and not SAFE_VALUE_COLUMN_RE.search(s):
            return False, "exists_in_wide_but_looks_anatomical_label"
        return True, "exists_in_wide_table"
    if SAFE_VALUE_COLUMN_RE.search(s):
        return True, "matches_safe_value_column_pattern"
    if ANATOMICAL_TOKEN_RE.search(s):
        return False, "looks_like_anatomical_label_not_value_column"
    # 其他情况保守处理：可作为候选但要求在 36/40/宽表中审计通过
    return True, "non_anatomical_string_needs_presence_audit"


def extract_value_columns_from_table(
    path: Optional[Path],
    source_name: str,
    known_wide_cols: Optional[set] = None,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame(columns=["source_name", "source_file", "row_index", "detected_column", "value_column", "safe_value_column", "safety_reason"])
    try:
        df = read_table_any(path, nrows=max_rows)
        df = ensure_unique_columns(df, f"{source_name}::{path.name}")
    except Exception as e:
        log(f"读取失败，跳过 {source_name}: {path}; {e}")
        return pd.DataFrame(columns=["source_name", "source_file", "row_index", "detected_column", "value_column", "safe_value_column", "safety_reason"])

    vc_col = choose_value_column_col(df)
    if vc_col is None:
        # 如果没有标准列，但表头本身可能就是 subject-level 宽表，把非临床 edge-like 列作为 value_column。
        rows = []
        for c in df.columns:
            cs = str(c)
            if SAFE_VALUE_COLUMN_RE.search(cs):
                ok, reason = is_safe_value_column(cs, known_wide_cols)
                rows.append({
                    "source_name": source_name,
                    "source_file": str(path),
                    "row_index": np.nan,
                    "detected_column": "__column_header__",
                    "value_column": cs,
                    "safe_value_column": ok,
                    "safety_reason": reason,
                })
        return pd.DataFrame(rows)

    rows = []
    for i, x in enumerate(df[vc_col].tolist()):
        vc = clean_value(x)
        if not vc or vc.lower() == "nan":
            continue
        ok, reason = is_safe_value_column(vc, known_wide_cols)
        rows.append({
            "source_name": source_name,
            "source_file": str(path),
            "row_index": i,
            "detected_column": str(vc_col),
            "value_column": vc,
            "safe_value_column": bool(ok),
            "safety_reason": reason,
        })
    return pd.DataFrame(rows)


def collect_known_wide_cols(wide_path: Optional[Path]) -> set:
    if not wide_path or not wide_path.exists():
        return set()
    try:
        header = read_csv_any(wide_path, nrows=0)
        return set(map(str, header.columns))
    except Exception as e:
        log(f"宽表表头读取失败: {wide_path}; {e}")
        return set()


def load_stats_by_edge(path: Optional[Path], source_prefix: str) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame()
    try:
        df = read_table_any(path)
        df = ensure_unique_columns(df, f"stats::{source_prefix}::{path.name}")
    except Exception as e:
        log(f"统计表读取失败，跳过 {path}: {e}")
        return pd.DataFrame()

    vc_col = choose_value_column_col(df)
    if vc_col is None:
        return pd.DataFrame()
    df = df.copy()
    vc_series = df[vc_col]
    if isinstance(vc_series, pd.DataFrame):
        vc_series = vc_series.iloc[:, 0]
    # 先移除所有 value_column 同名/重复兼容列，再重建唯一 merge key。
    drop_like_value_column = [c for c in df.columns if norm_col(c) == "value_column" or norm_col(c).startswith("value_column__dup")]
    df = df.drop(columns=drop_like_value_column, errors="ignore")
    df.insert(0, "value_column", vc_series.astype(str).map(clean_value))

    # 只保留常用统计列，避免旧标签列进入
    keep = ["value_column"]
    stat_keywords = [
        "n_", "n", "beta", "b_", "se", "t_", "p_", "q_", "fdr", "r2", "slope",
        "selection_frequency", "selection_count", "direction", "model", "df",
    ]
    for c in df.columns:
        nc = norm_col(c)
        if c == vc_col:
            continue
        # 防止重复/兼容导出的 value_column.1、value_column__dup2 等身份列被当作统计列保留下来。
        if nc.startswith("value_column") or nc.startswith("canonical_value_column") or nc.startswith("edge_value_column"):
            continue
        if norm_col(c) in FORBIDDEN_LABEL_COLUMNS:
            continue
        if any(k in nc for k in stat_keywords):
            keep.append(c)
    out = df[keep].copy()
    rename = {c: f"{source_prefix}__{c}" for c in out.columns if c != "value_column"}
    out = out.rename(columns=rename)
    # 同一 value_column 保留第一行，避免横向合并膨胀
    out = out.drop_duplicates(subset=["value_column"], keep="first")
    return out


def build_canonical_from_sources(
    paths: Dict[str, Optional[str]],
    known_wide_cols: set,
    expected_n: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """优先使用当前 39_v3 已经生成的 canonical 文件；否则退回 40 映射。"""
    source_order = [
        ("39_v3_canonical", paths.get("out39_canonical")),
        ("39_v3_lineage", paths.get("out39_lineage")),
        ("40_v3_candidate_mapping", paths.get("out40_mapping")),
        ("40_v3_final_decision", paths.get("out40_final")),
        ("brainnet_corrected_edges_value_column_only_source_check", paths.get("brainnet_corrected_edges")),
    ]

    all_parts = []
    warnings: List[str] = []
    for source_name, pstr in source_order:
        p = Path(pstr) if pstr else None
        part = extract_value_columns_from_table(p, source_name, known_wide_cols=known_wide_cols)
        if len(part):
            part.insert(0, "source_priority", len(all_parts) + 1)
            all_parts.append(part)

    long_df = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame(
        columns=["source_priority", "source_name", "source_file", "row_index", "detected_column", "value_column", "safe_value_column", "safety_reason"]
    )

    # 选择 canonical 来源：优先 39_v3_canonical，且只接受安全 value_column
    canonical_source_name = None
    for source_name, _ in source_order:
        sub = long_df[(long_df["source_name"] == source_name) & (long_df["safe_value_column"] == True)].copy()
        sub = sub.drop_duplicates(subset=["value_column"], keep="first")
        if len(sub) == expected_n:
            canonical_source_name = source_name
            canonical = sub.copy()
            break

    if canonical_source_name is None:
        # 如果没有正好10条，则尝试从 40_v3_candidate_mapping 中取前10条安全列；
        # 但这属于风险状态，需硬停，避免悄悄改变候选集。
        counts = long_df[long_df["safe_value_column"] == True].groupby("source_name")["value_column"].nunique().to_dict() if len(long_df) else {}
        raise RuntimeError(
            f"未能从当前 Codex 文件夹中找到正好 {expected_n} 条安全 value_column 候选边。"
            f"各来源安全唯一边数: {counts}。请检查 39_v3/40_v3 结果是否完整。"
        )

    canonical = canonical.drop_duplicates(subset=["value_column"], keep="first").reset_index(drop=True)
    canonical.insert(0, "canonical_edge_id", [f"EDGE_{i:02d}" for i in range(1, len(canonical) + 1)])
    canonical["canonical_source"] = canonical_source_name

    # 拆分两个 ROI token，但不解释为脑区名
    node_tokens = canonical["value_column"].map(split_value_column)
    canonical["value_column_token_1"] = [x[0] for x in node_tokens]
    canonical["value_column_token_2"] = [x[1] for x in node_tokens]
    canonical["edge_identity_policy"] = "value_column_only_no_anatomical_label"
    canonical["anatomical_label_policy"] = "disabled_in_script39_v4"

    return canonical, long_df, warnings


def make_lineage_audit(
    canonical: pd.DataFrame,
    paths: Dict[str, Optional[str]],
    known_wide_cols: set,
) -> pd.DataFrame:
    values = canonical["value_column"].astype(str).tolist()
    audit = canonical[[
        "canonical_edge_id", "value_column", "value_column_token_1", "value_column_token_2",
        "canonical_source", "edge_identity_policy", "anatomical_label_policy"
    ]].copy()

    def value_set_from_table(pstr: Optional[str], source_name: str) -> set:
        p = Path(pstr) if pstr else None
        part = extract_value_columns_from_table(p, source_name, known_wide_cols=known_wide_cols)
        if not len(part):
            return set()
        return set(part.loc[part["safe_value_column"] == True, "value_column"].astype(str))

    sets = {
        "present_in_input_wide_table_header": set(known_wide_cols),
        "present_in_36_v3_detected_edge_audit": value_set_from_table(paths.get("out36_edge_audit"), "36_edge_audit"),
        "present_in_36_v3_fullbrain_model_results": value_set_from_table(paths.get("out36_all_edges"), "36_all_edges"),
        "present_in_39_v3_existing_canonical": value_set_from_table(paths.get("out39_canonical"), "39_canonical"),
        "present_in_39_v3_existing_lineage": value_set_from_table(paths.get("out39_lineage"), "39_lineage"),
        "present_in_40_v3_candidate_mapping": value_set_from_table(paths.get("out40_mapping"), "40_mapping"),
        "present_in_40_v3_subject_table_header": set(read_table_any(Path(paths["out40_subject"]), nrows=0).columns) if paths.get("out40_subject") else set(),
        "present_in_40_v3_covariate_results": value_set_from_table(paths.get("out40_cov"), "40_cov"),
        "present_in_40_v3_final_decision_table": value_set_from_table(paths.get("out40_final"), "40_final"),
    }

    for col, s in sets.items():
        audit[col] = audit["value_column"].map(lambda x: bool(str(x) in s))

    presence_cols = [c for c in audit.columns if c.startswith("present_in_")]
    audit["n_presence_checks_passed"] = audit[presence_cols].sum(axis=1).astype(int)

    # 核心硬标准：正式宽表、36_v3结果、40_v3候选映射必须存在。
    core_cols = [
        "present_in_input_wide_table_header",
        "present_in_36_v3_fullbrain_model_results",
        "present_in_40_v3_candidate_mapping",
    ]
    for c in core_cols:
        if c not in audit.columns:
            audit[c] = False
    audit["core_lineage_pass"] = audit[core_cols].all(axis=1)

    # 合并一些 36/40 统计信息供审计，但不加入旧标签
    stats_tables = [
        (paths.get("out36_all_edges"), "s36_fullbrain"),
        (paths.get("out40_cov"), "s40_cov"),
        (paths.get("out40_nocov"), "s40_nocov"),
        (paths.get("out40_final"), "s40_final"),
    ]
    out = audit.copy()
    for pstr, prefix in stats_tables:
        st = load_stats_by_edge(Path(pstr) if pstr else None, prefix)
        if len(st):
            st = ensure_unique_columns(st, f"merge_source::{prefix}")
            # 防御性检查：merge key 必须唯一且只出现一次。
            if list(st.columns).count("value_column") != 1:
                raise RuntimeError(f"{prefix} 统计表清洗后仍有重复 value_column 列，已硬停。")
            st = st.drop_duplicates(subset=["value_column"], keep="first")
            out = out.merge(st, on="value_column", how="left", validate="one_to_one")

    return out


def assert_no_forbidden_label_columns(df: pd.DataFrame, name: str) -> None:
    bad = [c for c in df.columns if norm_col(c) in FORBIDDEN_LABEL_COLUMNS]
    if bad:
        raise RuntimeError(f"{name} 中仍含旧标签/解剖命名列，已硬停: {bad}")


def assert_no_anatomical_value_columns(df: pd.DataFrame, name: str) -> None:
    if "value_column" not in df.columns:
        return
    bad_values = []
    for x in df["value_column"].astype(str):
        if ANATOMICAL_TOKEN_RE.search(x) and not SAFE_VALUE_COLUMN_RE.search(x):
            bad_values.append(x)
    if bad_values:
        raise RuntimeError(
            f"{name} 的 value_column 看起来像解剖标签而不是原始取值列，已硬停。示例: {bad_values[:10]}"
        )


def sanitize_for_formal_output(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [c for c in df.columns if norm_col(c) not in FORBIDDEN_LABEL_COLUMNS]
    out = df[keep_cols].copy()
    return out


def write_report(
    out_dir: Path,
    paths: Dict[str, Optional[str]],
    canonical: pd.DataFrame,
    audit: pd.DataFrame,
    source_long: pd.DataFrame,
) -> None:
    equiv = load_json_if_exists(Path(paths["out40_equiv_summary"]) if paths.get("out40_equiv_summary") else None)

    lines = []
    lines.append("39_v4 Codex 固定候选边 value_column 谱系审计报告")
    lines.append("=" * 100)
    lines.append(f"生成时间: {now()}")
    lines.append("")
    lines.append("一、脚本定位")
    lines.append("-" * 100)
    lines.append("本脚本不重新跑稳定性选择、不重新筛边、不重新跑 40 号治疗调节模型。")
    lines.append("本脚本只把当前 Codex 文件夹中的固定候选边身份标准化为原始 value_column。")
    lines.append("正式输出不包含 edge_label_mapped、edge_label_short、corrected BNA labels 或任何直接解剖命名。")
    lines.append("")
    lines.append("二、自动解析到的关键路径")
    lines.append("-" * 100)
    for k, v in paths.items():
        lines.append(f"{k}: {v if v else 'NOT_FOUND'}")
    lines.append("")
    lines.append("三、候选边审计结果")
    lines.append("-" * 100)
    lines.append(f"canonical candidate edges: {len(canonical)}")
    lines.append(f"core lineage pass: {int(audit['core_lineage_pass'].sum())}/{len(audit)}")
    if len(audit):
        lines.append("")
        lines.append("固定候选边 value_column 列表：")
        for _, r in canonical.iterrows():
            lines.append(f"- {r['canonical_edge_id']}: {r['value_column']}")
    lines.append("")
    lines.append("四、40_v3 preMeanFD 修复版等价审计状态")
    lines.append("-" * 100)
    if equiv:
        for k, v in equiv.items():
            lines.append(f"{k}: {v}")
    else:
        lines.append("未找到或无法读取 00_v3_vs_v2_statistical_equivalence_summary.json。")
        lines.append("这不影响本脚本生成 value_column 谱系表，但正式替代 40_v2 前仍需确认 40_v3 等价审计 PASS。")
    lines.append("")
    lines.append("五、结论口径")
    lines.append("-" * 100)
    if len(canonical) == 10 and bool(audit["core_lineage_pass"].all()):
        lines.append("PASS: 10 条固定候选边均通过核心 value_column 谱系审计。")
        lines.append("可以把本输出作为 39 号在 Codex 文件夹中的正式安全版。")
    else:
        lines.append("FAIL/WARN: 仍有候选边未通过核心谱系审计，不能直接用于正式论文结果。")
    lines.append("")
    lines.append("六、论文/记录中的建议表述")
    lines.append("-" * 100)
    lines.append("English:")
    lines.append(
        "Candidate connections were tracked using their original value-column identities throughout the full-brain screening, "
        "stability-selection audit, and fixed-candidate modeling workflow. Anatomical labels were not assigned during this "
        "lineage-audit step; anatomical naming and visualization were deferred to a separately audited 246-ROI mapping workflow."
    )
    lines.append("中文:")
    lines.append(
        "在全脑筛查、稳定性选择审计和固定候选边建模流程中，候选连接均以原始取值列身份进行追踪。"
        "本谱系审计步骤不分配解剖标签；解剖命名和可视化留待独立审计通过的 246 ROI 映射流程完成。"
    )
    lines.append("")
    lines.append("七、禁止事项")
    lines.append("-" * 100)
    lines.append("不要用本脚本输出直接解释脑区名称。")
    lines.append("不要用旧 BrainNet_10_candidate_edges_plot_files 中的 candidate10.node/edge。")
    lines.append("不要把 edge_label_mapped 或 edge_label_short 当作正式候选边身份。")

    (out_dir / "04_39_v4_value_column_lineage_report.txt").write_text("\n".join(lines), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="", help="Codex 文件夹根目录。默认使用当前工作目录。")
    ap.add_argument("--out_dir", default="", help="输出目录。默认 root 下 39_v4_codex...结果。")
    ap.add_argument("--expected_n", type=int, default=10, help="固定候选边数量，默认 10。")
    ap.add_argument("--allow_soft_fail", action="store_true", help="允许审计失败时仍写出结果；默认硬停。")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    root = Path(args.root).resolve() if args.root else Path.cwd().resolve()
    if not root.exists():
        raise FileNotFoundError(f"root 不存在: {root}")

    out_dir = ensure_dir(Path(args.out_dir).resolve() if args.out_dir else root / DEFAULT_OUT_NAME)

    log("=" * 100)
    log("39_v4 Codex：固定候选边 value_column 谱系审计启动（无解剖标签正式版）")
    log("=" * 100)
    log(f"root = {root}")
    log(f"out_dir = {out_dir}")

    paths = resolve_paths(root)
    (out_dir / "00_resolved_input_paths.json").write_text(
        json.dumps(paths, ensure_ascii=False, indent=2),
        encoding="utf-8-sig"
    )

    required = ["wide_table", "out36_all_edges", "out39_canonical", "out40_mapping"]
    missing = [k for k in required if not paths.get(k)]
    if missing:
        raise RuntimeError(
            "缺少关键输入，不能安全生成正式 39_v4。"
            f"missing={missing}。请确认你在第四步分析-codex 根目录运行，并且 36_v3/39_v3/40_v3 已完成。"
        )

    known_wide_cols = collect_known_wide_cols(Path(paths["wide_table"]) if paths.get("wide_table") else None)
    log(f"正式宽表列数: {len(known_wide_cols)}")

    canonical, source_long, warnings = build_canonical_from_sources(paths, known_wide_cols, expected_n=args.expected_n)
    assert_no_forbidden_label_columns(canonical, "canonical")
    assert_no_anatomical_value_columns(canonical, "canonical")

    source_long_safe = sanitize_for_formal_output(source_long)
    assert_no_forbidden_label_columns(source_long_safe, "source_long_safe")
    write_csv(source_long_safe, out_dir / "01_candidate_value_columns_from_sources_long.csv")

    audit = make_lineage_audit(canonical, paths, known_wide_cols)
    audit = sanitize_for_formal_output(audit)
    assert_no_forbidden_label_columns(audit, "audit")
    assert_no_anatomical_value_columns(audit, "audit")
    write_csv(audit, out_dir / "02_fixed10_value_column_lineage_audit.csv")

    # 正式 canonical 文件：只保留不含解剖标签的列
    canonical_out_cols = [
        "canonical_edge_id", "value_column", "value_column_token_1", "value_column_token_2",
        "canonical_source", "edge_identity_policy", "anatomical_label_policy",
    ]
    canonical_formal = canonical[canonical_out_cols].copy()
    assert_no_forbidden_label_columns(canonical_formal, "canonical_formal")
    assert_no_anatomical_value_columns(canonical_formal, "canonical_formal")
    write_csv(canonical_formal, out_dir / "03_canonical_fixed_10_candidate_edges_value_column_only.csv")

    # 兼容后续脚本的最小输入版：仍然不含解剖标签。若后续脚本需要 label，可使用 canonical_edge_id 作为中性显示名。
    compat = canonical_formal.copy()
    compat["display_edge_id"] = compat["canonical_edge_id"]
    compat["display_edge_name"] = compat["canonical_edge_id"]  # 中性名称，不是解剖标签
    write_csv(compat, out_dir / "03b_canonical_fixed_10_candidate_edges_for_downstream_compat.csv")

    # 安全状态 JSON
    status = {
        "script_version": "39_v4_codex_value_column_only_no_anatomical_labels",
        "root": str(root),
        "out_dir": str(out_dir),
        "n_canonical_edges": int(len(canonical_formal)),
        "all_core_lineage_pass": bool(audit["core_lineage_pass"].all()) if len(audit) else False,
        "n_core_lineage_pass": int(audit["core_lineage_pass"].sum()) if len(audit) else 0,
        "n_expected": int(args.expected_n),
        "forbidden_label_columns_policy": "hard_stop",
        "anatomical_value_column_policy": "hard_stop",
        "runtime_minutes": round((time.time() - t0) / 60, 3),
    }
    (out_dir / "00_39_v4_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8-sig")

    write_report(out_dir, paths, canonical_formal, audit, source_long_safe)

    if len(canonical_formal) != args.expected_n:
        msg = f"固定候选边数量不是 {args.expected_n}: got={len(canonical_formal)}"
        if args.allow_soft_fail:
            log("警告: " + msg)
        else:
            raise RuntimeError(msg)

    if not bool(audit["core_lineage_pass"].all()):
        failed = audit.loc[~audit["core_lineage_pass"], ["canonical_edge_id", "value_column"]].to_dict("records")
        msg = f"存在候选边未通过核心谱系审计: {failed}"
        if args.allow_soft_fail:
            log("警告: " + msg)
        else:
            raise RuntimeError(msg)

    log("=" * 100)
    log("完成：39_v4 Codex value_column-only 谱系审计 PASS。")
    log(f"输出目录: {out_dir}")
    log("=" * 100)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("运行失败: " + repr(e))
        log("建议：确认当前目录是 第四步分析-codex，且 36_v3、39_v3、40_v3 preMeanFD 修复版结果均已存在。")
        raise
