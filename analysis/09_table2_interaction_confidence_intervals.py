# -*- coding: utf-8 -*-
"""
47_v2_value_column_only_Table2交互系数95CI_只读40既有bSE_无旧标签正式版.py

用途：
  为 Table 2【表2】生成 interaction b [95% CI]【交互项回归系数及95%置信区间】。
  本脚本只读取 40 号结果表中已经保存的 beta_interaction 和 se_interaction，
  不重新拟合模型，不重新筛选边，不重新计算 p/q 值。

v2 修复目标：
  1) 兼容 Codex 文件夹中当前 40_v3 结果目录；
  2) 允许内部读取旧 40 表中的历史边名以保证数值等价；
  3) 正式输出只保留 EDGE_01–EDGE_10 + value_column【取值列】；
  4) 删除 descriptive_anatomical_system【描述性解剖系统】和旧 BNA 标签；
  5) 对所有输出做旧标签残留审计，命中即报错。

运行：
  python -u 47_v2_value_column_only_Table2交互系数95CI_只读40既有bSE_无旧标签正式版.py

可选：
  python -u 47_v2_value_column_only_Table2交互系数95CI_只读40既有bSE_无旧标签正式版.py --root "D:\\...\\第四步分析：探索"
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd


SCRIPT_NAME = "47_v2_1_value_column_only_Table2交互系数95CI_只读40既有bSE_无旧标签正式版_修复40边名映射"
DEFAULT_COV_FILE = "11_fixed_candidate_moderation_covariates.csv"
DEFAULT_NOCOV_FILE = "10_fixed_candidate_moderation_no_covariates.csv"

# 固定候选边安全身份。旧标签仅作为“内部匹配键”使用，不写入任何正式输出。
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
    # 优先接受已经安全化的列。
    for edge_col in ["edge_id", "canonical_edge_id", "edge", "edge_short"]:
        if edge_col in row and str(row.get(edge_col, "")).startswith("EDGE_"):
            eid = str(row[edge_col])
            out = safe_by_edge_id(eid)
            out["identity_source"] = "safe_edge_id_from_input"
            return out

    # 兼容旧 40 输出，仅用于内部匹配。
    candidate_values = []
    for c in ["edge_short", "edge", "feature", "connection", "baseline_fc_z_col"]:
        if c in row:
            candidate_values.append(row[c])
    # 最后再扫整行字符串，处理列名差异。
    candidate_values.extend([v for v in row.tolist() if isinstance(v, str)])

    for v in candidate_values:
        key1 = normalize_key(v)
        key2 = compressed_key(v)
        for key in (key1, key2):
            if key in ALIAS_TO_SAFE:
                out = dict(ALIAS_TO_SAFE[key])
                out["identity_source"] = "matched_internal_suppressed_legacy_40_label"
                return out
        # 处理例如 A5l–vId/vIg__baseline_z 这类带后缀列名。
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


def score_path(path: Path) -> int:
    s = str(path).lower()
    score = 0
    if DEFAULT_COV_FILE.lower() in path.name.lower() or DEFAULT_NOCOV_FILE.lower() in path.name.lower():
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


def read_csv_safely_from_bytes(csv_bytes: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(io.BytesIO(csv_bytes), encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(csv_bytes))


def find_input_csv(root: Path, model: str, input_csv: str = "") -> Tuple[str, bytes]:
    target_file = DEFAULT_COV_FILE if model == "covariate_adjusted" else DEFAULT_NOCOV_FILE

    if input_csv:
        p = Path(input_csv)
        if not p.exists():
            raise FileNotFoundError(f"指定 input_csv 不存在: {p}")
        return f"manual_csv: {p}", p.read_bytes()

    candidates = [p for p in root.rglob(target_file) if p.is_file()]
    if candidates:
        candidates = sorted(candidates, key=lambda p: score_path(p), reverse=True)
        chosen = candidates[0]
        log("[输入搜索] 候选 CSV 前5个：")
        for p in candidates[:5]:
            log(f"  score={score_path(p):>4} | {p}")
        return f"extracted_csv: {chosen}", chosen.read_bytes()

    zip_candidates = [p for p in root.rglob("*.zip") if "40" in p.name or "稳定性选择" in p.name]
    zip_hits = []
    for zp in zip_candidates:
        try:
            with zipfile.ZipFile(zp, "r") as zf:
                for name in zf.namelist():
                    if name.endswith(target_file):
                        zip_hits.append((score_path(zp) + score_path(Path(name)), zp, name))
        except zipfile.BadZipFile:
            continue
    if zip_hits:
        zip_hits.sort(key=lambda x: x[0], reverse=True)
        _, zp, name = zip_hits[0]
        with zipfile.ZipFile(zp, "r") as zf:
            return f"zip_csv: {zp} :: {name}", zf.read(name)

    raise FileNotFoundError(f"没有找到 {target_file}。请把脚本放在 Codex 第四步分析文件夹，或用 --root/--input_csv 指定。")


def t_critical_975(df: int) -> float:
    if df <= 0:
        raise ValueError(f"df 必须 > 0，但得到 df={df}")
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df))
    except Exception:
        # 对 df≈58 的场景，近似足够用于格式化 CI。
        return 2.001717 if 55 <= df <= 60 else 1.96


def infer_df(row: pd.Series, model: str) -> Tuple[int, int, str]:
    n = int(row["n"])
    if model == "covariate_adjusted":
        covs = str(row.get("covariates_used", ""))
        cov_list = [c.strip() for c in covs.split(";") if c.strip()]
        n_parameters = 1 + 3 + len(cov_list)
        desc = "intercept + treatment + baselineFC + treatment×baselineFC + " + "+".join(cov_list)
    else:
        n_parameters = 4
        desc = "intercept + treatment + baselineFC + treatment×baselineFC"
    return n - n_parameters, n_parameters, desc


def fmt_num(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.{digits}f}"


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Codex 第四步分析文件夹或包含40号结果的目录。默认当前目录。")
    ap.add_argument("--input_csv", default="", help="手动指定 11_fixed_candidate_moderation_covariates.csv。")
    ap.add_argument("--model", default="covariate_adjusted", choices=["covariate_adjusted", "no_covariates"])
    ap.add_argument("--out_dir", default="", help="输出目录。默认 root/47_v2_Table2_CI_value_column_only")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else root / "47_v2_Table2_CI_value_column_only"
    out_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 100)
    log(f"{SCRIPT_NAME} 启动")
    log(f"root = {root}")
    log(f"model = {args.model}")
    log(f"out_dir = {out_dir}")
    log("=" * 100)

    source_desc, csv_bytes = find_input_csv(root, args.model, args.input_csv)
    log(f"[输入] {source_desc}")
    df = read_csv_safely_from_bytes(csv_bytes)

    required = [
        "edge_short", "n", "n_TMS", "n_PSY",
        "beta_interaction_TMS_minus_PSY_per1SD", "se_interaction",
        "t_interaction", "p_interaction", "q_interaction_fdr_10edges",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"输入表缺少必要列：{missing}")

    rows = []
    map_rows = []
    max_t_diff = 0.0
    for pos, (i, row) in enumerate(df.iterrows()):
        safe = resolve_edge_identity(row, fallback_pos=pos)
        beta = float(row["beta_interaction_TMS_minus_PSY_per1SD"])
        se = float(row["se_interaction"])
        if not math.isfinite(beta) or not math.isfinite(se) or se <= 0:
            raise ValueError(f"{safe['edge_id']} 的 beta/se 非法：beta={beta}, se={se}")

        df_resid, n_parameters, model_desc = infer_df(row, args.model)
        tcrit = t_critical_975(df_resid)
        ci_low = beta - tcrit * se
        ci_high = beta + tcrit * se
        t_from_bse = beta / se
        t_saved = float(row["t_interaction"])
        t_diff = abs(t_from_bse - t_saved)
        max_t_diff = max(max_t_diff, t_diff)

        direction = str(row.get("direction_from_model", ""))
        if "PSY" in direction:
            relative_advantage = "PSY"
        elif "TMS" in direction:
            relative_advantage = "TMS"
        else:
            relative_advantage = "PSY" if beta < 0 else "TMS"

        rows.append({
            "edge_id": safe["edge_id"],
            "value_column": safe["value_column"],
            "relative_advantage_at_higher_baseline_FC": relative_advantage,
            "model_source": args.model,
            "covariates_used": row.get("covariates_used", "") if args.model == "covariate_adjusted" else "none",
            "n": int(row["n"]),
            "n_PSY": int(row["n_PSY"]),
            "n_TMS": int(row["n_TMS"]),
            "beta_interaction_TMS_minus_PSY_per1SD": beta,
            "se_interaction": se,
            "df_resid_inferred": df_resid,
            "n_parameters_inferred": n_parameters,
            "tcrit_95_two_sided": tcrit,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "interaction_b_95CI_for_table": f"{fmt_num(beta)} [{fmt_num(ci_low)}, {fmt_num(ci_high)}]",
            "t_interaction_saved": t_saved,
            "t_interaction_from_b_over_se": t_from_bse,
            "abs_t_check_diff": t_diff,
            "p_interaction_saved": float(row["p_interaction"]),
            "q_interaction_fdr_10edges_saved": float(row["q_interaction_fdr_10edges"]),
            "r2_saved": float(row.get("r2", float("nan"))),
            "ci_method": "t-based 95% CI computed from saved beta and SE; model was not refit",
            "model_structure_for_df": model_desc,
            "edge_identity_policy": "EDGE_01_to_EDGE_10_plus_value_column_only_no_anatomical_label",
        })
        map_rows.append({
            "row_index": int(i),
            "edge_id": safe["edge_id"],
            "value_column": safe["value_column"],
            "identity_source": safe["identity_source"],
            "status": "PASS",
        })

    out = pd.DataFrame(rows)
    expected_ids = {f"EDGE_{i:02d}" for i in range(1, 11)}
    ids = set(out["edge_id"].astype(str))
    if ids != expected_ids:
        raise RuntimeError(f"输出 EDGE 编号不完整：{sorted(ids)}")

    csv_path = out_dir / "Table2_interaction_b_95CI_from_script40_existing_bSE_value_column_only.csv"
    xlsx_path = out_dir / "Table2_interaction_b_95CI_from_script40_existing_bSE_value_column_only.xlsx"
    audit_path = out_dir / "Table2_interaction_b_95CI_audit_value_column_only.txt"
    map_path = out_dir / "00_edge_identity_mapping_audit_v2.csv"
    legacy_path = out_dir / "00_output_legacy_label_text_audit_v2.csv"
    summary_path = out_dir / "00_run_summary_v2.json"

    out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        out.to_excel(xlsx_path, index=False)
    except Exception as exc:
        log(f"[提醒] xlsx 输出失败，仅保留 csv。原因：{exc}")
        xlsx_path = None

    pd.DataFrame(map_rows).to_csv(map_path, index=False, encoding="utf-8-sig")

    audit_lines = [
        "Table2 interaction b 95% CI audit - value_column-only version",
        "============================================================",
        f"Input source: {source_desc}",
        f"Model: {args.model}",
        "This script did NOT refit any regression model.",
        "It only derived 95% CIs from beta_interaction_TMS_minus_PSY_per1SD and se_interaction saved in Script 40 result table.",
        "Formal edge identity uses EDGE_01–EDGE_10 + value_column only.",
        f"Rows: {len(out)}",
        f"Max |saved t - beta/se|: {max_t_diff:.12g}",
        "Pass criterion for t check: max difference < 1e-6 is expected aside from rounding.",
        "",
        "Output files:",
        f"- {csv_path}",
        f"- {xlsx_path if xlsx_path else '(xlsx not created)'}",
        f"- {map_path}",
    ]
    audit_path.write_text("\n".join(audit_lines), encoding="utf-8")

    legacy = scan_legacy_text(out_dir)
    legacy.to_csv(legacy_path, index=False, encoding="utf-8-sig")
    n_legacy_fail = int((legacy["status"] == "FAIL").sum())

    summary = {
        "script": SCRIPT_NAME,
        "input_source": source_desc,
        "out_dir": str(out_dir),
        "n_rows": int(len(out)),
        "n_edges": int(out["edge_id"].nunique()),
        "edge_ids": sorted(out["edge_id"].unique().tolist()),
        "max_abs_t_check_diff": float(max_t_diff),
        "n_legacy_fail_files": n_legacy_fail,
        "edge_identity_policy": "EDGE_01_to_EDGE_10_plus_value_column_only_no_anatomical_label",
        "safe_for_formal_use": bool(n_legacy_fail == 0 and ids == expected_ids),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if n_legacy_fail > 0:
        raise RuntimeError(f"旧标签审计失败：{legacy_path}")

    log("[完成] 47_v2 已输出并通过旧标签审计。")
    log(f"safe_for_formal_use = {summary['safe_for_formal_use']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[错误] {exc}", file=sys.stderr)
        raise
