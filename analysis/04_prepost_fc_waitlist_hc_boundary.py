# -*- coding: utf-8 -*-
"""
42_v8_6_2_workstation
固定候选边 pre/post 绝对连接 + WL负控 + HC正常化/代偿
严格 value_column-only【仅取值列身份】连续编号终版：修复 v8.6 中 source_* / hc_edge_label 残余半标签。

用途：
- 优先读取已经生成的 42_v8_6 结果目录；
- 不重新计算任何统计量；
- 仅清理输出身份字段和审计/来源列文本；
- 输出 EDGE_01–EDGE_10 + unknown_roi_*__unknown_roi_*；
- 对所有 CSV/TXT/JSON 做旧标签硬审计。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd


SCRIPT_NAME = "42_v8_6_2_workstation_固定候选边prepost绝对连接_WL负控_HC正常化代偿分析_严格值列身份_全输出无旧标签连续编号终版_修复残余半标签.py"

SRC_DIR_CANDIDATES = [
    "42_v8_6_workstation_固定候选边prepost绝对连接_WL负控_HC正常化代偿分析结果_严格值列身份_全输出无旧标签连续编号终版",
    "42_v8_5_workstation_固定候选边prepost绝对连接_WL负控_HC正常化代偿分析结果_严格值列身份_全输出无旧标签版",
    "42_v8_4_workstation_固定候选边prepost绝对连接_WL负控_HC正常化代偿分析结果_严格值列身份_全输出无旧标签版",
]

OUT_DIR_NAME = "42_v8_6_2_workstation_固定候选边prepost绝对连接_WL负控_HC正常化代偿分析结果_严格值列身份_全输出无旧标签连续编号终版"

VALUE_COLUMN_ORDER = [
    "unknown_roi_18__unknown_roi_24",
    "unknown_roi_65__unknown_roi_85",
    "unknown_roi_62__unknown_roi_105",
    "unknown_roi_64__unknown_roi_85",
    "unknown_roi_11__unknown_roi_67",
    "unknown_roi_56__unknown_roi_84",
    "unknown_roi_46__unknown_roi_95",
    "unknown_roi_89__unknown_roi_90",
    "unknown_roi_8__unknown_roi_92",
    "unknown_roi_23__unknown_roi_44",
]

VALUE_TO_EDGE = {v: f"EDGE_{i:02d}" for i, v in enumerate(VALUE_COLUMN_ORDER, start=1)}
EDGE_TO_VALUE = {v: k for k, v in VALUE_TO_EDGE.items()}

OLD_EDGEKEY_TO_VALUE = {
    "a45ra11m": "unknown_roi_18__unknown_roi_24",
    "a5lvidvig": "unknown_roi_65__unknown_roi_85",
    "cpstslsoccg": "unknown_roi_62__unknown_roi_105",
    "a7cvidvig": "unknown_roi_64__unknown_roi_85",
    "a946va7ip": "unknown_roi_11__unknown_roi_67",
    "a3536cdia": "unknown_roi_56__unknown_roi_84",
    "a37elvcling": "unknown_roi_46__unknown_roi_95",
    "a24rva32p": "unknown_roi_89__unknown_roi_90",
    "a946da24cd": "unknown_roi_8__unknown_roi_92",
    "a11lasts": "unknown_roi_23__unknown_roi_44",
}
OLD_EDGE_TO_VALUE = {
    "A45r-A11m": "unknown_roi_18__unknown_roi_24",
    "A45r–A11m": "unknown_roi_18__unknown_roi_24",
    "A5l-vId/vIg": "unknown_roi_65__unknown_roi_85",
    "A5l–vId/vIg": "unknown_roi_65__unknown_roi_85",
    "cpSTS-lsOccG": "unknown_roi_62__unknown_roi_105",
    "cpSTS–lsOccG": "unknown_roi_62__unknown_roi_105",
    "A7c-vId/vIg": "unknown_roi_64__unknown_roi_85",
    "A7c–vId/vIg": "unknown_roi_64__unknown_roi_85",
    "A9/46v-A7ip": "unknown_roi_11__unknown_roi_67",
    "A9/46v–A7ip": "unknown_roi_11__unknown_roi_67",
    "A35/36c-dIa": "unknown_roi_56__unknown_roi_84",
    "A35/36c–dIa": "unknown_roi_56__unknown_roi_84",
    "A37elv-cLinG": "unknown_roi_46__unknown_roi_95",
    "A37elv–cLinG": "unknown_roi_46__unknown_roi_95",
    "A24rv-A32p": "unknown_roi_89__unknown_roi_90",
    "A24rv–A32p": "unknown_roi_89__unknown_roi_90",
    "A9/46d-A24cd": "unknown_roi_8__unknown_roi_92",
    "A9/46d–A24cd": "unknown_roi_8__unknown_roi_92",
    "A11l-aSTS": "unknown_roi_23__unknown_roi_44",
    "A11l–aSTS": "unknown_roi_23__unknown_roi_44",
}

LEGACY_TOKENS = [
    # full edge names
    "A45r-A11m", "A45r–A11m", "A5l-vId/vIg", "A5l–vId/vIg", "cpSTS-lsOccG", "cpSTS–lsOccG",
    "A7c-vId/vIg", "A7c–vId/vIg", "A9/46v-A7ip", "A9/46v–A7ip", "A35/36c-dIa", "A35/36c–dIa",
    "A37elv-cLinG", "A37elv–cLinG", "A24rv-A32p", "A24rv–A32p", "A9/46d-A24cd", "A9/46d–A24cd",
    "A11l-aSTS", "A11l–aSTS",
    # individual tokens / half-label residues
    "A45r", "A11m", "A5l", "vId/vIg", "vId", "vIg", "cpSTS", "lsOccG", "A7c", "A9/46v", "A7ip",
    "A35/36c", "dIa", "A37elv", "cLinG", "A24rv", "A32p", "A9/46d", "A24cd", "A11l", "aSTS",
    # corrected/alternate BNA residue tokens seen in propagated-label scripts
    "A40c", "A7m", "A32sg", "msOccG", "A23c", "dCa", "A7pc", "rHipp", "mAmyg", "lAmyg", "cHipp",
    # compressed keys
    "a45ra11m", "a5lvidvig", "cpstslsoccg", "a7cvidvig", "a946va7ip", "a3536cdia",
    "a37elvcling", "a24rva32p", "a946da24cd", "a11lasts",
    # explicit placeholder that still contains partial labels in v8.6
    "LEGACY_LABEL_SUPPRESSED",
]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gb18030", low_memory=False)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"输出 {path.name}: rows={len(df)}")


def infer_source_dir(cwd: Path, arg_source: str | None = None) -> Path:
    if arg_source:
        p = Path(arg_source)
        if not p.exists():
            raise FileNotFoundError(f"指定 source_dir 不存在：{p}")
        return p
    for name in SRC_DIR_CANDIDATES:
        p = cwd / name
        if p.exists() and p.is_dir():
            return p
    raise FileNotFoundError(
        "未找到可清理的 42_v8_6/v8_5/v8_4 结果目录。请先运行 42_v8_6_1，或用 --source_dir 手动指定结果目录。"
    )


def infer_value_column_from_row(row: pd.Series) -> str | None:
    for c in ["value_column", "edge_key", "selected_unknown_col", "hc_edge_label"]:
        if c in row.index and isinstance(row[c], str):
            s = str(row[c]).strip()
            m = re.search(r"unknown_roi_\d+__unknown_roi_\d+", s)
            if m:
                return m.group(0)
    if "canonical_edge_id" in row.index:
        eid = str(row["canonical_edge_id"]).strip()
        if eid in EDGE_TO_VALUE:
            return EDGE_TO_VALUE[eid]
    if "edge" in row.index:
        s = str(row["edge"]).strip()
        if s in EDGE_TO_VALUE:
            return EDGE_TO_VALUE[s]
        if s in OLD_EDGE_TO_VALUE:
            return OLD_EDGE_TO_VALUE[s]
    if "edge_key" in row.index:
        ek = str(row["edge_key"]).strip().lower()
        if ek in OLD_EDGEKEY_TO_VALUE:
            return OLD_EDGEKEY_TO_VALUE[ek]
    return None


def apply_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()

    values = []
    for _, row in out.iterrows():
        values.append(infer_value_column_from_row(row))
    value_series = pd.Series(values, index=out.index, dtype="object")

    if value_series.notna().any():
        if "value_column" not in out.columns:
            out["value_column"] = value_series
        else:
            out["value_column"] = value_series.where(value_series.notna(), out["value_column"].astype(str))
        out["canonical_edge_id"] = out["value_column"].map(VALUE_TO_EDGE).fillna(out.get("canonical_edge_id", ""))
        if "edge" in out.columns:
            out["edge"] = out["canonical_edge_id"]
        if "edge_key" in out.columns:
            out["edge_key"] = out["value_column"]

    # Clean source/provenance columns that caused v8.6 half-label residues.
    if "value_column" in out.columns:
        vc = out["value_column"].astype(str)
        for c, suffix in [
            ("source_pre_col", "__pre"),
            ("source_post_col", "__post"),
            ("source_delta_col", "__delta"),
        ]:
            if c in out.columns:
                out[c] = vc + suffix
        for c in ["source_value_col", "hc_edge_label", "selected_unknown_col", "selected_named_col", "selected_generic_col"]:
            if c in out.columns:
                out[c] = vc
        # remove legacy meta columns if present
        drop_cols = [c for c in out.columns if c.lower() in {
            "edge_short", "edge_short_x", "edge_short_y", "node_a", "node_b", "node_a_x", "node_b_x",
            "system", "system_cn", "system_x", "system_cn_x", "edge_map_label", "edge_map_label_x",
        }]
        if drop_cols:
            out = out.drop(columns=drop_cols)

    out = sanitize_text_dataframe(out)
    return out


def sanitize_text_value(x: Any) -> Any:
    if pd.isna(x):
        return x
    if not isinstance(x, str):
        return x
    s = x
    # First replace whole old edge names with generic non-anatomical placeholder.
    for old, vc in OLD_EDGE_TO_VALUE.items():
        s = s.replace(old, vc)
    for old_key, vc in OLD_EDGEKEY_TO_VALUE.items():
        s = s.replace(old_key, vc)
    # Then remove any residual single-token anatomical labels.
    for tok in sorted(LEGACY_TOKENS, key=len, reverse=True):
        if tok == "":
            continue
        s = s.replace(tok, "LABEL_REMOVED")
    # Collapse awkward placeholders, but keep meaning.
    s = re.sub(r"(LABEL_REMOVED[–\-_ /]*)+", "LABEL_REMOVED", s)
    s = s.replace("LABEL_REMOVED__pre", "value_column__pre")
    s = s.replace("LABEL_REMOVED__post", "value_column__post")
    s = s.replace("LABEL_REMOVED__delta", "value_column__delta")
    return s


def sanitize_text_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(sanitize_text_value)
    return out


def output_name_from_source(src_name: str) -> str:
    name = src_name
    name = re.sub(r"_v8_6(?=\.)", "_v8_6_2", name)
    name = re.sub(r"_v8_5(?=\.)", "_v8_6_2", name)
    name = re.sub(r"_v8_4(?=\.)", "_v8_6_2", name)
    return name


def scan_text_for_legacy(out_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(out_dir.iterdir()):
        if p.suffix.lower() not in [".csv", ".txt", ".json"]:
            continue
        try:
            txt = p.read_text(encoding="utf-8-sig", errors="ignore")
        except Exception as e:
            rows.append({"file": str(p), "status": "ERROR", "matched_token": "", "n_matches": 0, "error": repr(e)})
            continue
        matched = []
        for tok in LEGACY_TOKENS:
            if tok and tok in txt:
                matched.append(tok)
        rows.append({
            "file": str(p),
            "status": "FAIL" if matched else "PASS",
            "matched_token": " | ".join(matched[:20]),
            "n_matches": int(sum(txt.count(tok) for tok in matched)),
            "error": "",
        })
    audit = pd.DataFrame(rows)
    audit.to_csv(out_dir / "00_output_legacy_label_text_audit_v8_6_2.csv", index=False, encoding="utf-8-sig")
    return audit


def scan_edge_ids(out_dir: Path) -> pd.DataFrame:
    expected = [f"EDGE_{i:02d}" for i in range(1, 11)]
    rows = []
    for p in sorted(out_dir.glob("*.csv")):
        if p.name.startswith("00_output_legacy_label"):
            continue
        try:
            df = read_csv(p)
        except Exception as e:
            rows.append({"file": str(p), "edge_ids": "", "status": f"ERROR: {e}"})
            continue
        ids = set()
        for c in ["edge", "canonical_edge_id"]:
            if c in df.columns:
                ids |= set(df[c].dropna().astype(str).unique())
        if not ids:
            continue
        ids_sorted = sorted(ids)
        status = "PASS" if ids_sorted == expected else "FAIL"
        rows.append({"file": str(p), "edge_ids": " | ".join(ids_sorted), "status": status})
    audit = pd.DataFrame(rows)
    audit.to_csv(out_dir / "00_edge_id_continuity_audit_v8_6_2.csv", index=False, encoding="utf-8-sig")
    return audit


def write_report(out_dir: Path, source_dir: Path, legacy_fail: int, edge_fail: int) -> None:
    lines = [
        "=" * 100,
        "42号 v8.6.2 value_column-only 连续编号终版运行报告",
        "=" * 100,
        f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "一、来源",
        f"- source_dir = {source_dir}",
        "",
        "二、处理原则",
        "- 不重新计算任何统计量；只清理输出身份字段与来源/审计文本。",
        "- 正式边身份统一为 EDGE_01–EDGE_10。",
        "- 正式取值身份统一为 unknown_roi_*__unknown_roi_*。",
        "- source_pre_col/source_post_col/source_delta_col/hc_edge_label 等列不再保留旧脑区名或半截旧标签。",
        "",
        "三、审计结果",
        f"- legacy_label_fail_files = {legacy_fail}",
        f"- edge_id_fail_files = {edge_fail}",
        f"- status = {'PASS' if legacy_fail == 0 and edge_fail == 0 else 'FAIL'}",
        "",
        "四、解释原则",
        "- FC = resting-state functional connectivity【静息态功能连接】，不是 activation【激活】。",
        "- delta_fc = post_fc - pre_fc【后测连接减前测连接】。",
        "- distance_change = |post_fc - HC_mean| - |pre_fc - HC_mean|【距HC距离变化】。",
        "- HC-referenced normalization analysis【基于健康对照参考的正常化分析】应作为 exploratory interpretive analysis【探索性解释分析】。",
    ]
    (out_dir / "99_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description=SCRIPT_NAME)
    parser.add_argument("--source_dir", default=None, help="可选：手动指定 42_v8_6/v8_5/v8_4 结果目录")
    parser.add_argument("--out_dir", default=None, help="可选：手动指定输出目录")
    args = parser.parse_args()

    cwd = Path.cwd()
    source_dir = infer_source_dir(cwd, args.source_dir)
    out_dir = Path(args.out_dir) if args.out_dir else cwd / OUT_DIR_NAME

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 100)
    log("42号 v8.6.2 启动：修复 v8.6 source/hc 残余半标签，生成全输出无旧标签连续编号终版")
    log(f"source_dir = {source_dir}")
    log(f"out_dir = {out_dir}")
    log("=" * 100)

    for src in sorted(source_dir.iterdir()):
        if not src.is_file():
            continue
        if src.name.startswith("00_output_legacy_label_text_audit") or src.name.startswith("00_edge_id_continuity_audit"):
            continue
        out_name = output_name_from_source(src.name)

        if src.suffix.lower() == ".csv":
            df = read_csv(src)
            df2 = apply_identity_columns(df)
            write_csv(df2, out_dir / out_name)
        elif src.suffix.lower() == ".json":
            try:
                obj = json.loads(src.read_text(encoding="utf-8-sig", errors="ignore"))
            except Exception:
                obj = {}
            if isinstance(obj, dict):
                obj["script"] = SCRIPT_NAME
                obj["source_dir"] = str(source_dir)
                obj["out_dir"] = str(out_dir)
                obj["value_column_order"] = VALUE_COLUMN_ORDER
                obj["value_to_edge"] = VALUE_TO_EDGE
                obj["calculation_policy"] = "No numerical statistics were recomputed; only output identity/provenance text was standardized."
                obj["status"] = "PENDING_AUDIT"
            (out_dir / output_name_from_source(src.name)).write_text(
                sanitize_text_value(json.dumps(obj, ensure_ascii=False, indent=2)),
                encoding="utf-8-sig",
            )
        elif src.suffix.lower() == ".txt":
            # Do not copy old report because source paths can contain residual old labels.
            continue

    legacy_audit = scan_text_for_legacy(out_dir)
    edge_audit = scan_edge_ids(out_dir)
    legacy_fail = int((legacy_audit["status"] == "FAIL").sum()) if not legacy_audit.empty else 0
    edge_fail = int((edge_audit["status"] == "FAIL").sum()) if not edge_audit.empty else 0

    # update summary after audits
    summary_path = out_dir / "00_v8_6_run_summary.json"
    summary2_path = out_dir / "00_v8_6_2_run_summary.json"
    if summary_path.exists():
        try:
            obj = json.loads(summary_path.read_text(encoding="utf-8-sig", errors="ignore"))
        except Exception:
            obj = {}
        obj["script"] = SCRIPT_NAME
        obj["legacy_label_fail_files"] = legacy_fail
        obj["edge_id_fail_files"] = edge_fail
        obj["status"] = "PASS" if legacy_fail == 0 and edge_fail == 0 else "FAIL"
        summary_path.unlink(missing_ok=True)
        summary2_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    else:
        summary2_path.write_text(json.dumps({
            "script": SCRIPT_NAME,
            "source_dir": str(source_dir),
            "out_dir": str(out_dir),
            "value_column_order": VALUE_COLUMN_ORDER,
            "value_to_edge": VALUE_TO_EDGE,
            "legacy_label_fail_files": legacy_fail,
            "edge_id_fail_files": edge_fail,
            "status": "PASS" if legacy_fail == 0 and edge_fail == 0 else "FAIL",
            "calculation_policy": "No numerical statistics were recomputed; only output identity/provenance text was standardized."
        }, ensure_ascii=False, indent=2), encoding="utf-8-sig")

    write_report(out_dir, source_dir, legacy_fail, edge_fail)

    if legacy_fail or edge_fail:
        raise RuntimeError("v8.6.2 审计未通过：请查看 00_output_legacy_label_text_audit_v8_6_2.csv 和 00_edge_id_continuity_audit_v8_6_2.csv。")

    log("完成。v8.6.2 审计通过，可作为 42 号正式安全终版。")


if __name__ == "__main__":
    main()
