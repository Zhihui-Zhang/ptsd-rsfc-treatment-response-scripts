# -*- coding: utf-8 -*-
"""
48_v2_value_column_only_从40号协变量模型生成Table2同源DeltaR2_无旧标签正式版.py

用途：
  基于 40 号结果中的 subject-level core-edge table【被试层核心边表】和
  11_fixed_candidate_moderation_covariates.csv【协变量调节模型结果表】，
  重新拟合与 40 号一致的 fixed-candidate PCL-5 moderation model【固定候选 PCL-5 调节模型】，
  为 Table 2【表2】生成同源 ΔR²【增量解释率】。

v2 修复目标：
  1) 兼容 Codex 文件夹中当前 40_v3 结果目录；
  2) 允许内部匹配旧 40 表中的历史边列以复现数值；
  3) 正式输出只保留 EDGE_01–EDGE_10 + value_column【取值列】；
  4) 不输出旧 edge_short、旧 baseline_fc_z_col 或 descriptive anatomical label【描述性解剖标签】；
  5) 不改变模型公式，不重新筛选候选边，不改写 40 号原始结果；
  6) 输出旧标签残留审计，命中即报错。

运行：
  python -u 48_v2_value_column_only_从40号协变量模型生成Table2同源DeltaR2_无旧标签正式版.py

可选：
  python -u 48_v2_value_column_only_从40号协变量模型生成Table2同源DeltaR2_无旧标签正式版.py --root "D:\\...\\第四步分析：探索"
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
except Exception as exc:
    sm = None
    _STATSMODELS_IMPORT_ERROR = exc
else:
    _STATSMODELS_IMPORT_ERROR = None


SCRIPT_NAME = "48_v2_1_value_column_only_从40号协变量模型生成Table2同源DeltaR2_无旧标签正式版_修复40边名映射"
SUBJECT_FILE = "02_subject_level_core_edges.csv"
COV_RESULT_FILE = "11_fixed_candidate_moderation_covariates.csv"

Y_COL = "pcl_improvement"
TREAT_COL = "treatment_TMS"
COVARIATES = ["PCL_pre", "age", "sex", "pre_meanFD"]

SAFE_EDGE_MAP = [
    {"edge_id": "EDGE_01", "value_column": "unknown_roi_18__unknown_roi_24", "aliases": ["A45r–A11m", "A45r-A11m", "A45r__A11m", "A45r_A11m"]},
    {"edge_id": "EDGE_02", "value_column": "unknown_roi_65__unknown_roi_85", "aliases": ["A5l–vId/vIg", "A5l-vId/vIg", "A5l__vId_vIg", "A5l_vId_vIg"]},
    {"edge_id": "EDGE_03", "value_column": "unknown_roi_62__unknown_roi_105", "aliases": ["cpSTS–lsOccG", "cpSTS-lsOccG", "cpSTS__lsOccG", "cpSTS_lsOccG"]},
    {"edge_id": "EDGE_04", "value_column": "unknown_roi_64__unknown_roi_85", "aliases": ["A7c–vId/vIg", "A7c-vId/vIg", "A7c__vId_vIg", "A7c_vId_vIg"]},
    {"edge_id": "EDGE_05", "value_column": "unknown_roi_11__unknown_roi_67", "aliases": ["A9/46v–A7ip", "A9/46v-A7ip", "A9_46v__A7ip", "A9_46v_A7ip"]},
    {"edge_id": "EDGE_06", "value_column": "unknown_roi_56__unknown_roi_84", "aliases": ["A35/36c–dIa", "A35/36c-dIa", "A35_36c__dIa", "A35_36c_dIa"]},
    {"edge_id": "EDGE_07", "value_column": "unknown_roi_46__unknown_roi_95", "aliases": ["A37elv–cLinG", "A37elv-cLinG", "A37elv__cLinG", "A37elv_cLinG"]},
    {"edge_id": "EDGE_08", "value_column": "unknown_roi_89__unknown_roi_90", "aliases": ["A24rv–A32p", "A24rv-A32p", "A24rv__A32p", "A24rv_A32p"]},
    {"edge_id": "EDGE_09", "value_column": "unknown_roi_8__unknown_roi_92", "aliases": ["A9/46d–A24cd", "A9/46d-A24cd", "A9_46d__A24cd", "A9_46d_A24cd"]},
    {"edge_id": "EDGE_10", "value_column": "unknown_roi_23__unknown_roi_44", "aliases": ["A11l–aSTS", "A11l-aSTS", "A11l__aSTS", "A11l_aSTS"]},
]

LEGACY_PATTERNS = [
    "A45r", "A11m", "A5l", "vId", "vIg", "cpSTS", "lsOccG", "A7c",
    "A9/46v", "A7ip", "A35/36c", "dIa", "A37elv", "cLinG", "A24rv",
    "A32p", "A9/46d", "A24cd", "A11l", "aSTS",
    "a45ra11m", "a5lvidvig", "cpstslsoccg", "a7cvidvig", "a946va7ip",
    "a3536cdia", "a37elvcling", "a24rva32p", "a946da24cd", "a11lasts",
    "LEGACY_LABEL_SUPPRESSED",
]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def normalize_key(x: object) -> str:
    s = "" if x is None else str(x)
    s = s.replace("–", "_").replace("—", "_").replace("-", "_").replace("/", "_")
    s = re.sub(r"[^0-9A-Za-z]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s


def compressed_key(x: object) -> str:
    return normalize_key(x).replace("_", "")


ALIAS_TO_SAFE: Dict[str, Dict[str, str]] = {}
for item in SAFE_EDGE_MAP:
    for alias in item["aliases"] + [item["edge_id"], item["value_column"]]:
        safe = {"edge_id": item["edge_id"], "value_column": item["value_column"]}
        ALIAS_TO_SAFE[normalize_key(alias)] = safe
        ALIAS_TO_SAFE[compressed_key(alias)] = safe

# 40_v3 的 11_fixed_candidate_moderation_covariates.csv 通常按显著性排序，
# 不是 EDGE_01–EDGE_10 顺序。此列表只在文本匹配失败时作为最后兜底，
# 用于避免不同 Unicode 连字符/斜杠导致的假失败；不改变任何数值。
FALLBACK_40_V3_ROW_ORDER_EDGE_IDS = [
    "EDGE_02", "EDGE_01", "EDGE_08", "EDGE_04", "EDGE_09",
    "EDGE_10", "EDGE_05", "EDGE_06", "EDGE_03", "EDGE_07",
]


def safe_by_edge_id(eid: str) -> Dict[str, str]:
    for item in SAFE_EDGE_MAP:
        if item["edge_id"] == eid:
            return {"edge_id": item["edge_id"], "value_column": item["value_column"]}
    raise KeyError(eid)


def resolve_edge_identity(row: pd.Series, fallback_pos: Optional[int] = None) -> Dict[str, str]:
    for edge_col in ["edge_id", "canonical_edge_id", "edge", "edge_short"]:
        if edge_col in row and str(row.get(edge_col, "")).startswith("EDGE_"):
            out = safe_by_edge_id(str(row[edge_col]))
            out["identity_source"] = "safe_edge_id_from_input"
            return out

    candidate_values = []
    for c in ["edge_short", "edge", "feature", "connection", "baseline_fc_z_col"]:
        if c in row:
            candidate_values.append(row[c])
    candidate_values.extend([v for v in row.tolist() if isinstance(v, str)])

    for v in candidate_values:
        key1 = normalize_key(v)
        key2 = compressed_key(v)
        for key in (key1, key2):
            if key in ALIAS_TO_SAFE:
                out = dict(ALIAS_TO_SAFE[key])
                out["identity_source"] = "matched_internal_suppressed_legacy_40_label"
                return out
        for alias_key, safe in ALIAS_TO_SAFE.items():
            if alias_key and (alias_key in key1 or alias_key in key2):
                out = dict(safe)
                out["identity_source"] = "matched_internal_suppressed_legacy_40_label_contained"
                return out

    if fallback_pos is not None and 0 <= int(fallback_pos) < len(FALLBACK_40_V3_ROW_ORDER_EDGE_IDS):
        out = safe_by_edge_id(FALLBACK_40_V3_ROW_ORDER_EDGE_IDS[int(fallback_pos)])
        out["identity_source"] = "fallback_40_v3_saved_row_order_after_text_match_failed"
        return out

    raise KeyError("无法把输入行映射到 EDGE_01–EDGE_10。请检查 40 号结果是否为固定10候选边表。")


def read_csv_safely_from_bytes(csv_bytes: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(io.BytesIO(csv_bytes), encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(csv_bytes))


def read_csv_safely(path: Path) -> pd.DataFrame:
    return read_csv_safely_from_bytes(path.read_bytes())


def score_path(path: Path) -> int:
    s = str(path).lower()
    score = 0
    if path.name in [SUBJECT_FILE, COV_RESULT_FILE]:
        score += 50
    if "40_v3" in s:
        score += 100
    if "premeanfd" in s or "premeanfd修复" in s:
        score += 35
    if "稳定性选择候选边" in s:
        score += 30
    if "正式" in s:
        score += 15
    if "历史" in s or "（历史）" in s or "_历史" in s:
        score -= 200
    if "v2_a3536c_dia" in s or "a3536c" in s:
        score -= 20
    return score


def find_file_or_zip_member(root: Path, target_name: str, manual_path: str = "") -> Tuple[str, bytes]:
    if manual_path:
        p = Path(manual_path)
        if not p.exists():
            raise FileNotFoundError(f"指定文件不存在: {p}")
        return f"manual_csv: {p}", p.read_bytes()

    candidates = [p for p in root.rglob(target_name) if p.is_file()]
    if candidates:
        candidates = sorted(candidates, key=lambda p: score_path(p), reverse=True)
        chosen = candidates[0]
        log(f"[输入搜索] {target_name} 候选前5个：")
        for p in candidates[:5]:
            log(f"  score={score_path(p):>4} | {p}")
        return f"extracted_csv: {chosen}", chosen.read_bytes()

    zip_hits = []
    for zp in root.rglob("*.zip"):
        if not ("40" in zp.name or "稳定性选择" in zp.name):
            continue
        try:
            with zipfile.ZipFile(zp, "r") as zf:
                for name in zf.namelist():
                    if name.endswith(target_name):
                        zip_hits.append((score_path(zp) + score_path(Path(name)), zp, name))
        except zipfile.BadZipFile:
            continue
    if zip_hits:
        zip_hits.sort(key=lambda x: x[0], reverse=True)
        _, zp, name = zip_hits[0]
        with zipfile.ZipFile(zp, "r") as zf:
            return f"zip_csv: {zp} :: {name}", zf.read(name)

    raise FileNotFoundError(f"没有找到 {target_name}。请把脚本放在 Codex 第四步分析文件夹，或用参数手动指定。")


def fit_ols_statsmodels(y: pd.Series, x: pd.DataFrame):
    if sm is None:
        raise RuntimeError(
            "statsmodels 未成功导入，无法拟合 OLS。请先安装 statsmodels，或在原 40 号环境中运行本脚本。"
            f"\n原始导入错误: {_STATSMODELS_IMPORT_ERROR}"
        )
    x_const = sm.add_constant(x, has_constant="add")
    return sm.OLS(y.astype(float), x_const.astype(float)).fit()


def candidates_for_internal_baseline_col(edge_label: object, safe: Dict[str, str]) -> List[str]:
    # 正式输出不写这些内部候选列名；仅用于复现 40 号已有模型。
    raw = str(edge_label)
    vals = []
    for base in [safe["edge_id"], safe["value_column"], raw, raw.replace("-", "–"), raw.replace("–", "-")]:
        if not base or base == "nan":
            continue
        vals.extend([
            f"{base}__baseline_z",
            f"{base}_baseline_z",
            f"{base}__pre_z",
            f"{base}_pre_z",
        ])
    # alias 也加入内部匹配。
    for item in SAFE_EDGE_MAP:
        if item["edge_id"] == safe["edge_id"]:
            for alias in item["aliases"]:
                vals.extend([
                    f"{alias}__baseline_z",
                    f"{alias}_baseline_z",
                    f"{alias}__pre_z",
                    f"{alias}_pre_z",
                ])
    out = []
    for v in vals:
        if v not in out:
            out.append(v)
    return out


def get_edge_baseline_z_col(df: pd.DataFrame, edge_label: object, safe: Dict[str, str]) -> str:
    candidates = candidates_for_internal_baseline_col(edge_label, safe)
    for col in candidates:
        if col in df.columns:
            return col
    # 宽松匹配，但不写入正式输出。
    norm_cols = {normalize_key(c): c for c in df.columns}
    for col in candidates:
        key = normalize_key(col)
        if key in norm_cols:
            return norm_cols[key]
    raise KeyError(f"{safe['edge_id']} 找不到内部 baseline_z 列。请检查 02_subject_level_core_edges.csv。")


def fmt_float(x, nd=6):
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{nd}f}"


def scan_legacy_text(out_dir: Path) -> pd.DataFrame:
    rows = []
    files = list(out_dir.glob("*.csv")) + list(out_dir.glob("*.txt")) + list(out_dir.glob("*.json"))
    for p in files:
        text = p.read_text(encoding="utf-8-sig", errors="ignore")
        hits = []
        low_text = re.sub(r"[^0-9A-Za-z]+", "", text).lower()
        for pat in LEGACY_PATTERNS:
            if pat.startswith("a") and pat.islower():
                if pat in low_text:
                    hits.append(pat)
            else:
                if pat in text:
                    hits.append(pat)
        rows.append({
            "file": p.name,
            "n_hits": len(sorted(set(hits))),
            "hit_terms": "; ".join(sorted(set(hits))),
            "status": "PASS" if not hits else "FAIL",
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="从 40 号协变量模型生成 Table 2 同源 ΔR² 的 value_column-only 正式版。")
    ap.add_argument("--root", default=".", help="Codex 第四步分析文件夹或包含40号结果的目录。默认当前目录。")
    ap.add_argument("--subject_csv", default="", help="手动指定 02_subject_level_core_edges.csv。")
    ap.add_argument("--cov_csv", default="", help="手动指定 11_fixed_candidate_moderation_covariates.csv。")
    ap.add_argument("--out_dir", default="", help="输出目录。默认 root/48_v2_Table2_deltaR2_value_column_only")
    args = ap.parse_args()

    if sm is None:
        raise RuntimeError(
            "statsmodels 未成功导入。本脚本需要 statsmodels 来复现 40 号 OLS 模型。\n"
            f"原始导入错误: {_STATSMODELS_IMPORT_ERROR}"
        )

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else root / "48_v2_Table2_deltaR2_value_column_only"
    out_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 100)
    log(f"{SCRIPT_NAME} 启动")
    log(f"root = {root}")
    log(f"out_dir = {out_dir}")
    log("=" * 100)

    subject_desc, subject_bytes = find_file_or_zip_member(root, SUBJECT_FILE, args.subject_csv)
    cov_desc, cov_bytes = find_file_or_zip_member(root, COV_RESULT_FILE, args.cov_csv)
    log(f"[输入] 被试层数据: {subject_desc}")
    log(f"[输入] 协变量结果: {cov_desc}")

    subject_df = read_csv_safely_from_bytes(subject_bytes)
    cov_res = read_csv_safely_from_bytes(cov_bytes)

    required_subject_cols = [Y_COL, TREAT_COL] + COVARIATES
    missing_subject_cols = [c for c in required_subject_cols if c not in subject_df.columns]
    if missing_subject_cols:
        raise KeyError(f"被试层数据缺少必要列: {missing_subject_cols}")

    required_cov_cols = [
        "edge_short", "n", "n_TMS", "n_PSY",
        "beta_interaction_TMS_minus_PSY_per1SD", "p_interaction", "r2",
        "q_interaction_fdr_10edges",
    ]
    missing_cov_cols = [c for c in required_cov_cols if c not in cov_res.columns]
    if missing_cov_cols:
        raise KeyError(f"40号协变量结果表缺少必要列: {missing_cov_cols}")

    rows: List[Dict] = []
    map_rows: List[Dict] = []
    audit_lines: List[str] = [
        "Table 2 same-source Delta R2 audit - value_column-only version",
        "=============================================================",
        f"Subject-level source: {subject_desc}",
        f"Covariate result source: {cov_desc}",
        "",
        "Definition:",
        "Full model: pcl_improvement ~ treatment_TMS + baseline_FC_z + treatment_TMS:baseline_FC_z + PCL_pre + age + sex + pre_meanFD",
        "Reduced model: pcl_improvement ~ treatment_TMS + baseline_FC_z + PCL_pre + age + sex + pre_meanFD",
        "Delta R2 nested interaction = R2_full - R2_reduced_no_interaction",
        "Formal edge identity uses EDGE_01–EDGE_10 + value_column only.",
        "",
    ]

    for pos, (i, ref) in enumerate(cov_res.iterrows()):
        safe = resolve_edge_identity(ref, fallback_pos=pos)
        edge_internal_col = get_edge_baseline_z_col(subject_df, ref.get("edge_short", ""), safe)

        tmp_cols = [Y_COL, TREAT_COL, edge_internal_col] + COVARIATES
        dat = subject_df[tmp_cols].copy()
        dat = dat.replace([np.inf, -np.inf], np.nan).dropna(axis=0).copy()
        if dat.empty:
            raise ValueError(f"{safe['edge_id']}: dropna 后没有可分析样本。")

        dat["interaction"] = dat[TREAT_COL].astype(float) * dat[edge_internal_col].astype(float)

        y = dat[Y_COL].astype(float)
        x_full = dat[[TREAT_COL, edge_internal_col, "interaction"] + COVARIATES].astype(float)
        x_reduced = dat[[TREAT_COL, edge_internal_col] + COVARIATES].astype(float)

        m_full = fit_ols_statsmodels(y, x_full)
        m_reduced = fit_ols_statsmodels(y, x_reduced)

        n = int(m_full.nobs)
        n_tms = int((dat[TREAT_COL] == 1).sum())
        n_psy = int((dat[TREAT_COL] == 0).sum())
        r2_full = float(m_full.rsquared)
        r2_reduced = float(m_reduced.rsquared)
        delta_r2 = r2_full - r2_reduced

        beta_ref = float(ref["beta_interaction_TMS_minus_PSY_per1SD"])
        p_ref = float(ref["p_interaction"])
        r2_ref = float(ref["r2"])
        n_ref = int(ref["n"])
        q_ref = float(ref["q_interaction_fdr_10edges"])

        beta_full = float(m_full.params["interaction"])
        se_full = float(m_full.bse["interaction"])
        t_full = float(m_full.tvalues["interaction"])
        p_full = float(m_full.pvalues["interaction"])
        df_resid = float(m_full.df_resid)

        rows.append({
            "edge_id": safe["edge_id"],
            "value_column": safe["value_column"],
            "baseline_fc_z_safe_col": f"{safe['edge_id']}__baseline_z",
            "n": n,
            "n_TMS": n_tms,
            "n_PSY": n_psy,
            "r2_full_covariate_moderation_refit": r2_full,
            "r2_reduced_covariate_prognostic_refit": r2_reduced,
            "delta_r2_nested_interaction_from_script40_covariate_model": delta_r2,
            "beta_interaction_refit": beta_full,
            "se_interaction_refit": se_full,
            "t_interaction_refit": t_full,
            "p_interaction_refit": p_full,
            "df_resid_refit": df_resid,
            "beta_interaction_saved_40": beta_ref,
            "p_interaction_saved_40": p_ref,
            "q_interaction_fdr_10edges_saved_40": q_ref,
            "r2_full_saved_40": r2_ref,
            "n_saved_40": n_ref,
            "abs_diff_beta_refit_vs_saved": abs(beta_full - beta_ref),
            "abs_diff_p_refit_vs_saved": abs(p_full - p_ref),
            "abs_diff_r2_refit_vs_saved": abs(r2_full - r2_ref),
            "n_match_saved_40": bool(n == n_ref),
            "model_for_table2": "covariate_adjusted_script40_same_source_deltaR2",
            "edge_identity_policy": "EDGE_01_to_EDGE_10_plus_value_column_only_no_anatomical_label",
            "internal_column_match_policy": "matched_existing_40_column_but_suppressed_from_formal_output",
        })

        map_rows.append({
            "row_index": int(i),
            "edge_id": safe["edge_id"],
            "value_column": safe["value_column"],
            "identity_source": safe["identity_source"],
            "internal_baseline_column_found": True,
            "status": "PASS",
        })

        audit_lines.append(
            f"{safe['edge_id']}: n={n} (TMS={n_tms}, PSY={n_psy}); "
            f"R2_full={fmt_float(r2_full)}, R2_reduced={fmt_float(r2_reduced)}, "
            f"DeltaR2={fmt_float(delta_r2)}; "
            f"|beta_diff|={fmt_float(abs(beta_full - beta_ref), 10)}, "
            f"|p_diff|={fmt_float(abs(p_full - p_ref), 10)}, "
            f"|r2_diff|={fmt_float(abs(r2_full - r2_ref), 10)}, "
            f"n_match={n == n_ref}"
        )

    out = pd.DataFrame(rows)
    expected_ids = {f"EDGE_{i:02d}" for i in range(1, 11)}
    ids = set(out["edge_id"].astype(str))
    if ids != expected_ids:
        raise RuntimeError(f"输出 EDGE 编号不完整：{sorted(ids)}")

    max_beta_diff = float(out["abs_diff_beta_refit_vs_saved"].max())
    max_p_diff = float(out["abs_diff_p_refit_vs_saved"].max())
    max_r2_diff = float(out["abs_diff_r2_refit_vs_saved"].max())
    all_n_match = bool(out["n_match_saved_40"].all())

    audit_lines.extend([
        "",
        "Summary checks:",
        f"Rows: {len(out)}",
        f"All n match saved 40 table: {all_n_match}",
        f"Max |beta_refit - beta_saved|: {max_beta_diff:.12g}",
        f"Max |p_refit - p_saved|: {max_p_diff:.12g}",
        f"Max |r2_refit - r2_saved|: {max_r2_diff:.12g}",
        "Recommended Table 2 column:",
        "Use delta_r2_nested_interaction_from_script40_covariate_model if the manuscript wants Delta R2 from the same covariate-adjusted Script-40 model family as b/CI/p/q.",
    ])

    out_csv = out_dir / "Table2_deltaR2_same_source_from_script40_covariate_model_value_column_only.csv"
    out_xlsx = out_dir / "Table2_deltaR2_same_source_from_script40_covariate_model_value_column_only.xlsx"
    out_audit = out_dir / "Table2_deltaR2_same_source_from_script40_audit_value_column_only.txt"
    map_path = out_dir / "00_edge_identity_mapping_audit_v2.csv"
    legacy_path = out_dir / "00_output_legacy_label_text_audit_v2.csv"
    summary_path = out_dir / "00_run_summary_v2.json"

    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    try:
        out.to_excel(out_xlsx, index=False)
    except Exception as exc:
        audit_lines.append(f"Excel output failed: {exc}")
        out_xlsx = None

    out_audit.write_text("\n".join(audit_lines), encoding="utf-8")
    pd.DataFrame(map_rows).to_csv(map_path, index=False, encoding="utf-8-sig")

    legacy = scan_legacy_text(out_dir)
    legacy.to_csv(legacy_path, index=False, encoding="utf-8-sig")
    n_legacy_fail = int((legacy["status"] == "FAIL").sum())

    summary = {
        "script": SCRIPT_NAME,
        "subject_source": subject_desc,
        "covariate_result_source": cov_desc,
        "out_dir": str(out_dir),
        "n_rows": int(len(out)),
        "n_edges": int(out["edge_id"].nunique()),
        "edge_ids": sorted(out["edge_id"].unique().tolist()),
        "all_n_match_saved_40": all_n_match,
        "max_abs_beta_diff_refit_vs_saved": max_beta_diff,
        "max_abs_p_diff_refit_vs_saved": max_p_diff,
        "max_abs_r2_diff_refit_vs_saved": max_r2_diff,
        "n_legacy_fail_files": n_legacy_fail,
        "edge_identity_policy": "EDGE_01_to_EDGE_10_plus_value_column_only_no_anatomical_label",
        "safe_for_formal_use": bool(n_legacy_fail == 0 and ids == expected_ids and all_n_match),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if n_legacy_fail > 0:
        raise RuntimeError(f"旧标签审计失败：{legacy_path}")

    log("[完成] 48_v2 已输出并通过旧标签审计。")
    log(f"Rows = {len(out)}")
    log(f"All n match saved 40 table = {all_n_match}")
    log(f"Max |beta_refit - beta_saved| = {max_beta_diff:.12g}")
    log(f"Max |p_refit - p_saved| = {max_p_diff:.12g}")
    log(f"Max |r2_refit - r2_saved| = {max_r2_diff:.12g}")
    log(f"safe_for_formal_use = {summary['safe_for_formal_use']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[错误] {exc}", file=sys.stderr)
        raise
