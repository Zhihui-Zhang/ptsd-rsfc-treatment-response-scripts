import argparse
import json
import math
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# =========================
# 基础工具
# =========================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x)


def extract_subject_token(text: str) -> str:
    s = safe_text(text)
    # 只抓真正的 sub- 片段，避免把 MNI152NLin2009cAsym 里的数字吃进去
    m = re.search(r"(sub-[^_\\/]+)", s, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return s


def normalize_digits(text: str) -> str:
    token = extract_subject_token(text)
    digits = "".join(re.findall(r"\d+", token))
    if not digits:
        tail = Path(safe_text(text)).name.split("_")[0]
        digits = "".join(re.findall(r"\d+", tail))
    if not digits:
        return ""
    return digits[-4:].zfill(4)


def first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# =========================
# 命名解析
# =========================

def parse_timepoint(text: str) -> str:
    s = extract_subject_token(text).lower()
    # HC 只有单次扫描，后续主分析默认按 baseline 处理
    if parse_group(text) == "HC":
        return "baseline"
    if "baseline" in s or "base" in s:
        return "baseline"
    if "pre" in s:
        return "pre"
    if "post" in s:
        return "post"
    return "unknown"


def parse_group(text: str) -> str:
    s = extract_subject_token(text).upper()
    if "TMS" in s:
        return "TMS"
    if "ACT" in s:
        return "ACT"
    if "MIN" in s:
        return "MIN"
    if "WL" in s:
        return "WL"
    if re.search(r"(^|[_\\/-])HC($|[_\\/-])", s) or s.startswith("SUB-HC") or "HC" in s:
        return "HC"
    return "UNKNOWN"


def canonical_subject_id(folder_name: str, bold_name: str) -> str:
    folder_group = parse_group(folder_name)
    bold_group = parse_group(bold_name)
    group = bold_group if bold_group != "UNKNOWN" else folder_group

    folder_time = parse_timepoint(folder_name)
    bold_time = parse_timepoint(bold_name)
    timepoint = bold_time if bold_time != "unknown" else folder_time

    digits_folder = normalize_digits(folder_name)
    digits_bold = normalize_digits(bold_name)
    digits = digits_bold if digits_bold else digits_folder

    if not digits:
        digits = "NA00"
    if group == "UNKNOWN":
        group = "UNKNOWN"
    if timepoint == "unknown":
        timepoint = "unknown"
    return f"sub-{timepoint}-{group}-{digits}"


def canonical_scan_id(folder_name: str, bold_name: str) -> str:
    return canonical_subject_id(folder_name, bold_name).replace("sub-", "scan-")


# =========================
# 扫描发现
# =========================

def list_preproc_bolds(func_dir: Path) -> List[Path]:
    candidates = []
    for p in func_dir.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if name.endswith(".json"):
            continue
        if "task-rest" not in name:
            continue
        if "desc-preproc_bold" not in name:
            continue
        if not (name.endswith(".nii") or name.endswith(".nii.gz")):
            continue
        candidates.append(p)

    # 优先真正的 sub- 文件；dssub-、ssub- 只作为兜底
    sub_files = [p for p in candidates if p.name.lower().startswith("sub-")]
    if sub_files:
        return sorted(sub_files)
    return sorted(candidates)


def infer_bold_json(func_dir: Path, stem_base: str) -> Optional[Path]:
    p = func_dir / f"{stem_base}.json"
    return p if p.exists() else None


def infer_brain_mask(func_dir: Path, stem_base: str) -> Optional[Path]:
    candidates = [
        func_dir / f"{stem_base.replace('_desc-preproc_bold', '_desc-brain_mask')}.nii.gz",
        func_dir / f"{stem_base.replace('_desc-preproc_bold', '_desc-brain_mask')}.nii",
    ]
    return first_existing(candidates)


def infer_confounds(func_dir: Path, bold_name: str, folder_name: str) -> Tuple[Optional[Path], Optional[Path], str]:
    stem_base = re.sub(r"\.nii(\.gz)?$", "", bold_name)
    digits = normalize_digits(folder_name + " " + bold_name)

    # 第一层：严格按 fMRIPrep 常见命名推断
    base_candidates = [
        re.sub(r"_space-[^_]+_desc-preproc_bold$", "_desc-confounds_timeseries", stem_base),
        re.sub(r"_desc-preproc_bold$", "_desc-confounds_timeseries", stem_base),
    ]
    tsv = first_existing([func_dir / f"{x}.tsv" for x in unique_preserve_order(base_candidates)])
    js = first_existing([func_dir / f"{x}.json" for x in unique_preserve_order(base_candidates)])
    if tsv:
        return tsv, js, "strict_filename_match"

    # 第二层：在当前 func 目录里做宽松匹配
    loose_tsvs = []
    loose_jsons = []
    for p in func_dir.iterdir():
        if not p.is_file():
            continue
        low = p.name.lower()
        if "task-rest" not in low or "desc-confounds_timeseries" not in low:
            continue
        if normalize_digits(p.name) != digits:
            continue
        if p.suffix.lower() == ".tsv":
            loose_tsvs.append(p)
        elif p.suffix.lower() == ".json":
            loose_jsons.append(p)

    if loose_tsvs:
        tsv = sorted(loose_tsvs)[0]
        js = sorted(loose_jsons)[0] if loose_jsons else None
        return tsv, js, "loose_func_dir_match"

    return None, None, "not_found"


def discover_one_func_dir(func_dir: Path) -> List[Dict]:
    rows: List[Dict] = []
    folder_name = func_dir.parent.name

    bolds = list_preproc_bolds(func_dir)
    if bolds:
        for bold in bolds:
            stem_base = re.sub(r"\.nii(\.gz)?$", "", bold.name)
            bold_json = infer_bold_json(func_dir, stem_base)
            brain_mask = infer_brain_mask(func_dir, stem_base)
            confounds_tsv, confounds_json, conf_note = infer_confounds(func_dir, bold.name, folder_name)

            status = "ready"
            note = conf_note
            if confounds_tsv is None:
                # GitHub release policy: only fMRIPrep outputs with matched confounds are eligible.
                status = "skip_no_fmriprep_confounds"
                note = "未找到 fMRIPrep desc-confounds_timeseries.tsv；按 fMRIPrep-only 上传策略跳过。"

            rows.append({
                "source_root": str(func_dir.parent.parent),
                "subject_folder": folder_name,
                "func_dir": str(func_dir),
                "bold_path": str(bold),
                "confounds_tsv": str(confounds_tsv) if confounds_tsv else "",
                "confounds_json": str(confounds_json) if confounds_json else "",
                "bold_json": str(bold_json) if bold_json else "",
                "brain_mask": str(brain_mask) if brain_mask else "",
                "raw_bold_fallback": "",
                "timepoint": parse_timepoint(folder_name + " " + bold.name),
                "group": parse_group(folder_name + " " + bold.name),
                "subject_code_normalized": normalize_digits(folder_name + " " + bold.name),
                "canonical_subject_id": canonical_subject_id(folder_name, bold.name),
                "canonical_scan_id": canonical_scan_id(folder_name, bold.name),
                "input_type": "fmriprep_preproc",
                "status": status,
                "note": note,
            })
        return rows

    # 没有标准预处理 bold 时，记录原始 bold 作为跳过项
    raw_bolds = []
    for p in func_dir.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if name.endswith(".json"):
            continue
        if "task-rest" not in name or "_bold" not in name:
            continue
        if "desc-preproc_bold" in name:
            continue
        if not (name.endswith(".nii") or name.endswith(".nii.gz")):
            continue
        raw_bolds.append(p)

    for bold in sorted(raw_bolds):
        raw_json = Path(str(bold).replace(".nii.gz", ".json").replace(".nii", ".json"))
        rows.append({
            "source_root": str(func_dir.parent.parent),
            "subject_folder": folder_name,
            "func_dir": str(func_dir),
            "bold_path": "",
            "confounds_tsv": "",
            "confounds_json": "",
            "bold_json": str(raw_json) if raw_json.exists() else "",
            "brain_mask": "",
            "raw_bold_fallback": str(bold),
            "timepoint": parse_timepoint(folder_name + " " + bold.name),
            "group": parse_group(folder_name + " " + bold.name),
            "subject_code_normalized": normalize_digits(folder_name + " " + bold.name),
            "canonical_subject_id": canonical_subject_id(folder_name, bold.name),
            "canonical_scan_id": canonical_scan_id(folder_name, bold.name),
            "input_type": "raw_bold_only",
            "status": "raw_only_skip",
            "note": "仅发现原始 task-rest_bold，未发现标准 desc-preproc_bold。",
        })
    return rows


def discover_scans(roots: List[Path]) -> pd.DataFrame:
    all_rows: List[Dict] = []
    seen_func_dirs = set()

    for root in roots:
        if not root.exists():
            print(f"[警告] 数据根目录不存在，跳过: {root}")
            continue
        for func_dir in sorted(root.rglob("func")):
            if not func_dir.is_dir():
                continue
            key = str(func_dir.resolve())
            if key in seen_func_dirs:
                continue
            seen_func_dirs.add(key)
            all_rows.extend(discover_one_func_dir(func_dir))

    cols = [
        "source_root", "subject_folder", "func_dir", "bold_path", "confounds_tsv",
        "confounds_json", "bold_json", "brain_mask", "raw_bold_fallback",
        "timepoint", "group", "subject_code_normalized", "canonical_subject_id",
        "canonical_scan_id", "input_type", "status", "note"
    ]
    if not all_rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(all_rows)
    priority = {"ready": 0, "skip_no_fmriprep_confounds": 1, "raw_only_skip": 2}
    df["_priority"] = df["status"].map(priority).fillna(99)
    df = df.sort_values(["canonical_scan_id", "_priority", "bold_path", "raw_bold_fallback"])
    df = df.drop_duplicates(subset=["canonical_scan_id"], keep="first")
    df = df.drop(columns=["_priority"]).reset_index(drop=True)
    return df[cols]


# =========================
# 图谱解析
# =========================

def load_brainnetome_labels(xlsx_path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(xlsx_path)
    best = None
    best_score = -1
    for sheet in xls.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
        if df.empty:
            continue
        cols = [str(c).lower() for c in df.columns]
        score = 0
        if any(k in c for c in cols for k in ["label", "id", "index", "roi"]):
            score += 1
        if any(k in c for c in cols for k in ["name", "region", "subregion"]):
            score += 1
        if len(df) >= 100:
            score += 1
        if score > best_score:
            best_score = score
            best = df.copy()

    if best is None or best.empty:
        raise ValueError(f"无法从 Brainnetome 标签表读取内容: {xlsx_path}")

    label_col = None
    name_col = None
    for c in best.columns:
        cl = str(c).lower()
        if label_col is None and any(k in cl for k in ["label", "id", "index", "roi"]):
            label_col = c
        if name_col is None and any(k in cl for k in ["name", "region", "subregion"]):
            name_col = c

    if label_col is None:
        label_col = best.columns[0]
    if name_col is None:
        name_col = best.columns[min(1, len(best.columns) - 1)]

    out = best[[label_col, name_col]].copy()
    out.columns = ["roi_value", "roi_name"]
    out["roi_value"] = pd.to_numeric(out["roi_value"], errors="coerce")
    out = out.dropna(subset=["roi_value"]).copy()
    out["roi_value"] = out["roi_value"].astype(int)
    out = out[out["roi_value"] > 0].drop_duplicates(subset=["roi_value"]).sort_values("roi_value")
    out["roi_name"] = out["roi_name"].astype(str).str.strip()
    if out.empty:
        raise ValueError(f"Brainnetome 标签表解析后为空: {xlsx_path}")
    return out.reset_index(drop=True)


def parse_aal_xml(xml_path: Path) -> pd.DataFrame:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rows = []

    # 兼容两种常见格式：
    # 1) <label index="1">Precentral_L</label>
    # 2) <label><index>1</index><name>Precentral_L</name></label>
    for elem in root.iter():
        tag = elem.tag.lower().split("}")[-1]
        if tag != "label":
            continue

        idx = elem.attrib.get("index") or elem.attrib.get("id") or elem.attrib.get("value")
        name = (elem.text or "").strip()

        if idx is None:
            child_map = {ch.tag.lower().split("}")[-1]: (ch.text or "").strip() for ch in elem}
            idx = child_map.get("index") or child_map.get("id") or child_map.get("value")
            name = child_map.get("name") or child_map.get("label") or name

        if idx is None:
            continue
        try:
            idx_int = int(idx)
        except Exception:
            continue
        if idx_int <= 0:
            continue
        rows.append({"roi_value": idx_int, "roi_name": name if name else f"AAL_{idx_int:03d}"})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"AAL XML 解析为空: {xml_path}")
    df = df.drop_duplicates(subset=["roi_value"]).sort_values("roi_value").reset_index(drop=True)
    return df


def load_atlas_positive_values(atlas_img_path: Path) -> set:
    atlas_img = nib.load(str(atlas_img_path))
    vals = np.unique(np.asarray(atlas_img.dataobj))
    out = set()
    for v in vals.tolist():
        try:
            iv = int(round(float(v)))
        except Exception:
            continue
        if iv > 0 and abs(float(v) - iv) < 1e-6:
            out.add(iv)
    return out


def parse_aal_txt(txt_path: Path, atlas_values: Optional[set] = None) -> pd.DataFrame:
    raw_rows = []
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('%'):
                continue

            parts = re.split(r"[\s,;]+", line)
            parts = [p for p in parts if p]
            if not parts:
                continue

            int_tokens = []
            name_tokens = []
            for p in parts:
                if re.fullmatch(r"[-+]?\d+", p):
                    try:
                        int_tokens.append(int(p))
                    except Exception:
                        pass
                else:
                    name_tokens.append(p)

            if not int_tokens:
                continue

            name = "_".join(name_tokens).strip('_')
            if not name:
                name = f"AAL_line_{line_no}"

            positive_ints = [x for x in int_tokens if x > 0]
            raw_rows.append({
                "line_no": line_no,
                "roi_name": name,
                "candidate_values": positive_ints,
            })

    if not raw_rows:
        raise ValueError(f"AAL TXT 无可解析内容: {txt_path}")

    rows = []
    for r in raw_rows:
        cands = r["candidate_values"]
        roi_value = None
        if atlas_values:
            matched = [x for x in cands if x in atlas_values]
            if len(matched) == 1:
                roi_value = matched[0]
            elif len(matched) > 1:
                roi_value = max(matched)

        if roi_value is None:
            if len(cands) == 1:
                roi_value = cands[0]
            elif len(cands) >= 2:
                roi_value = cands[-1]

        if roi_value is None or roi_value <= 0:
            continue

        rows.append({"roi_value": int(roi_value), "roi_name": r["roi_name"]})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"AAL TXT 解析为空: {txt_path}")

    if atlas_values:
        df = df[df["roi_value"].isin(atlas_values)].copy()

    df = df.drop_duplicates(subset=["roi_value"]).sort_values("roi_value").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"AAL TXT 与图谱值无法对齐: {txt_path}")
    return df


def resolve_aal_files(aal_dirs: List[Path]) -> Tuple[Path, Optional[Path], pd.DataFrame]:
    img_names = {"aal.nii", "aal.nii.gz", "roi_mni_v4.nii", "roi_mni_v4.nii.gz", "roi_mni_v5.nii", "roi_mni_v5.nii.gz"}
    xml_names = {"aal.xml", "roi_mni_v4.xml", "roi_mni_v5.xml"}
    txt_names = {"aal.txt", "roi_mni_v4.txt", "roi_mni_v5.txt"}

    found_img = None
    found_label = None
    labels_df = None

    for d in aal_dirs:
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if not p.is_file():
                continue
            low = p.name.lower()
            if found_img is None and low in img_names:
                found_img = p
            if found_label is None and low in xml_names:
                found_label = p
            elif found_label is None and low in txt_names:
                found_label = p
        if found_img and found_label:
            break

    if found_img is None:
        raise FileNotFoundError("未找到 AAL 图谱影像文件（如 AAL.nii / ROI_MNI_V4.nii）。")
    if found_label is None:
        raise FileNotFoundError("未找到 AAL 标签文件（如 AAL.xml / ROI_MNI_V4.txt）。")

    atlas_values = load_atlas_positive_values(found_img)

    if found_label.suffix.lower() == ".xml":
        labels_df = parse_aal_xml(found_label)
    else:
        labels_df = parse_aal_txt(found_label, atlas_values=atlas_values)

    return found_img, found_label, labels_df


# =========================
# confounds / 元信息
# =========================

def choose_confounds(confounds_path: Optional[Path]) -> Optional[pd.DataFrame]:
    if confounds_path is None or not confounds_path.exists():
        return None

    df = pd.read_csv(confounds_path, sep="\t")
    keep_cols = []
    base_exact = {
        "trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z",
        "csf", "white_matter", "global_signal", "framewise_displacement",
        "dvars", "std_dvars"
    }
    base_prefix = ("cosine", "motion_outlier", "non_steady_state_outlier", "a_comp_cor", "t_comp_cor")

    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in base_exact or cl.startswith(base_prefix):
            keep_cols.append(c)

    if not keep_cols:
        return None

    out = df[keep_cols].copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.fillna(0.0)


def read_tr_from_json(json_path: Optional[Path]) -> Optional[float]:
    if json_path is None or not json_path.exists():
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        tr = obj.get("RepetitionTime")
        return float(tr) if tr is not None else None
    except Exception:
        return None


# =========================
# ROI 提取
# =========================

def fisher_z_matrix(r: np.ndarray) -> np.ndarray:
    clipped = np.clip(r, -0.999999, 0.999999)
    z = np.arctanh(clipped)
    np.fill_diagonal(z, 0.0)
    return z


def sanitize_filename(text: str) -> str:
    text = re.sub(r'[\\/:*?"<>|]+', '_', safe_text(text))
    text = re.sub(r'\s+', '_', text).strip('_')
    return text or 'NA'


def align_timeseries_and_labels(timeseries: np.ndarray, labels_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n_cols = timeseries.shape[1]
    labels = labels_df.copy().reset_index(drop=True)

    if len(labels) < n_cols:
        extra_n = n_cols - len(labels)
        extra = pd.DataFrame({
            "roi_value": np.arange(len(labels) + 1, len(labels) + extra_n + 1),
            "roi_name": [f"unknown_roi_{i+1}" for i in range(extra_n)],
        })
        labels = pd.concat([labels, extra], ignore_index=True)
    elif len(labels) > n_cols:
        labels = labels.iloc[:n_cols].copy()

    raw_columns = [sanitize_filename(x) for x in labels["roi_name"].tolist()]
    deduped = []
    counts: Dict[str, int] = {}
    for c in raw_columns:
        counts[c] = counts.get(c, 0) + 1
        deduped.append(c if counts[c] == 1 else f"{c}__{counts[c]}")

    ts_df = pd.DataFrame(timeseries, columns=deduped)
    label_out = labels.copy()
    label_out["roi_output_name"] = deduped
    return ts_df, label_out


def extract_one_atlas(
    atlas_name_cn: str,
    atlas_prefix: str,
    atlas_img: Path,
    labels_df: pd.DataFrame,
    scan_row: pd.Series,
    atlas_output_root: Path,
) -> Dict:
    scan_id = scan_row["canonical_scan_id"]
    scan_dir = atlas_output_root / scan_id
    ensure_dir(scan_dir)

    bold_path = Path(scan_row["bold_path"])
    confounds_path = Path(scan_row["confounds_tsv"]) if safe_text(scan_row["confounds_tsv"]) else None
    bold_json_path = Path(scan_row["bold_json"]) if safe_text(scan_row["bold_json"]) else None
    mask_path = Path(scan_row["brain_mask"]) if safe_text(scan_row["brain_mask"]) else None

    tr = read_tr_from_json(bold_json_path)
    confounds = choose_confounds(confounds_path)

    masker = NiftiLabelsMasker(
        labels_img=str(atlas_img),
        mask_img=str(mask_path) if mask_path and mask_path.exists() else None,
        standardize="zscore_sample",
        detrend=True,
        t_r=tr,
        verbose=0,
    )

    timeseries = masker.fit_transform(str(bold_path), confounds=confounds)
    ts_df, label_out = align_timeseries_and_labels(timeseries, labels_df)

    cm = ConnectivityMeasure(kind="correlation", standardize="zscore_sample")
    fc_r = cm.fit_transform([timeseries])[0]
    fc_z = fisher_z_matrix(fc_r)

    colnames = label_out["roi_output_name"].tolist()
    fc_r_df = pd.DataFrame(fc_r, index=colnames, columns=colnames)
    fc_z_df = pd.DataFrame(fc_z, index=colnames, columns=colnames)

    label_file = scan_dir / f"{atlas_prefix}_ROI标签表.csv"
    ts_file = scan_dir / f"{atlas_prefix}_ROI时序.csv"
    r_file = scan_dir / f"{atlas_prefix}_Pearson相关矩阵.csv"
    z_file = scan_dir / f"{atlas_prefix}_FisherZ矩阵.csv"
    meta_file = scan_dir / "提取元信息.json"

    label_out.to_csv(label_file, index=False, encoding="utf-8-sig")
    ts_df.to_csv(ts_file, index=False, encoding="utf-8-sig")
    fc_r_df.to_csv(r_file, encoding="utf-8-sig")
    fc_z_df.to_csv(z_file, encoding="utf-8-sig")

    meta = {
        "atlas_name_cn": atlas_name_cn,
        "atlas_prefix": atlas_prefix,
        "canonical_scan_id": scan_id,
        "canonical_subject_id": scan_row["canonical_subject_id"],
        "subject_folder": scan_row["subject_folder"],
        "group": scan_row["group"],
        "timepoint": scan_row["timepoint"],
        "subject_code_normalized": scan_row["subject_code_normalized"],
        "bold_path": str(bold_path),
        "confounds_tsv": str(confounds_path) if confounds_path and confounds_path.exists() else "",
        "brain_mask": str(mask_path) if mask_path and mask_path.exists() else "",
        "tr": tr,
        "n_timepoints": int(timeseries.shape[0]),
        "n_rois_output": int(timeseries.shape[1]),
        "atlas_img": str(atlas_img),
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "canonical_scan_id": scan_id,
        "canonical_subject_id": scan_row["canonical_subject_id"],
        "subject_folder": scan_row["subject_folder"],
        "group": scan_row["group"],
        "timepoint": scan_row["timepoint"],
        "subject_code_normalized": scan_row["subject_code_normalized"],
        "status": "success",
        "n_timepoints": int(timeseries.shape[0]),
        "n_rois_output": int(timeseries.shape[1]),
        "used_confounds": bool(confounds is not None),
        "output_dir": str(scan_dir),
        "label_file": str(label_file),
        "timeseries_file": str(ts_file),
        "fc_r_file": str(r_file),
        "fc_z_file": str(z_file),
        "meta_file": str(meta_file),
        "error": "",
    }


# =========================
# 主流程
# =========================

def save_group_outputs(prefix: str, atlas_root: Path, success_rows: List[Dict], fail_rows: List[Dict]) -> None:
    success_df = pd.DataFrame(success_rows)
    fail_df = pd.DataFrame(fail_rows)

    success_csv = atlas_root / f"{prefix}_受试者提取清单.csv"
    fail_csv = atlas_root / f"{prefix}_失败或跳过清单.csv"
    summary_json = atlas_root / f"{prefix}_提取汇总.json"

    success_df.to_csv(success_csv, index=False, encoding="utf-8-sig")
    fail_df.to_csv(fail_csv, index=False, encoding="utf-8-sig")

    summary = {
        "atlas_prefix": prefix,
        "n_success": int(len(success_df)),
        "n_failed_or_skipped": int(len(fail_df)),
        "success_csv": str(success_csv),
        "fail_csv": str(fail_csv),
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    default_script_dir = Path(r"E:\E_zhangzhihui\从yv那边提取\脚本")
    default_output_root = default_script_dir / "HC双图谱提取结果"

    parser = argparse.ArgumentParser(description="批量提取 HC 的 Brainnetome246 + AAL116 ROI 时序和 FC")
    parser.add_argument(
        "--data_roots", nargs="+",
        default=[r"F:\zhangzhihui1026\zzh_resting_state_analysis\fmri_study_1\fmri_prep_worksite1\hc"],
        help="一个或多个待扫描的 HC fMRIPrep 根目录"
    )
    parser.add_argument("--output_root", type=str, default=str(default_output_root), help="输出根目录")
    parser.add_argument(
        "--brainnetome_img", type=str,
        default=r"E:\E_zhangzhihui\提取roi\脑网络组246图谱\BNA_MPM_thr25_1.25mm.nii.gz",
        help="Brainnetome 图谱影像路径"
    )
    parser.add_argument(
        "--brainnetome_labels", type=str,
        default=r"E:\E_zhangzhihui\提取roi\脑网络组246图谱\BNA_subregions.xlsx",
        help="Brainnetome 标签表 xlsx 路径"
    )
    parser.add_argument(
        "--aal_dirs", nargs="+",
        default=[
            r"atlas\AAL_local",
            r"atlas\AAL_reference",
        ],
        help="AAL 图谱目录候选列表"
    )
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 个 ready 扫描，0 表示全量")
    parser.add_argument("--skip_brainnetome", action="store_true", help="跳过 Brainnetome 提取")
    parser.add_argument("--skip_aal", action="store_true", help="跳过 AAL 提取")
    parser.add_argument("--discover_only", action="store_true", help="仅扫描并生成清单，不做提取")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_roots = [Path(x) for x in args.data_roots]
    output_root = Path(args.output_root)
    ensure_dir(output_root)

    print("=" * 90)
    print("开始扫描目录并建立清单")
    for i, root in enumerate(data_roots, 1):
        print(f"数据根目录 {i}: {root}")
    print("=" * 90)
    print("说明：HC 扫描将固定标记为 group=HC, timepoint=baseline")

    scans_df = discover_scans(data_roots)
    discover_csv = output_root / "00_扫描发现总清单.csv"
    ready_csv = output_root / "00_可提取扫描清单.csv"
    skip_csv = output_root / "00_缺失或跳过扫描清单.csv"

    scans_df.to_csv(discover_csv, index=False, encoding="utf-8-sig")
    ready_df = scans_df[scans_df["status"].eq("ready")].copy().reset_index(drop=True)
    skip_df = scans_df[~scans_df["status"].eq("ready")].copy().reset_index(drop=True)
    ready_df.to_csv(ready_csv, index=False, encoding="utf-8-sig")
    skip_df.to_csv(skip_csv, index=False, encoding="utf-8-sig")

    print(f"总发现扫描数: {len(scans_df)}")
    print(f"可提取扫描数（含无 confounds 但可跑）: {len(ready_df)}")
    print(f"其中严格标准 ready 数: {(scans_df['status'] == 'ready').sum()}")
    print(f"其中因缺少 fMRIPrep confounds 跳过数: {(scans_df['status'] == 'skip_no_fmriprep_confounds').sum()}")
    print(f"缺失/跳过扫描数: {len(skip_df)}")
    print(f"发现清单: {discover_csv}")

    if args.discover_only:
        print("已按 discover_only 模式结束。")
        return

    if args.limit and args.limit > 0:
        ready_df = ready_df.head(args.limit).copy()
        print(f"按 limit 仅处理前 {len(ready_df)} 个扫描")

    if ready_df.empty:
        print("没有可提取扫描，结束。")
        return

    atlas_jobs = []
    if not args.skip_brainnetome:
        bna_img = Path(args.brainnetome_img)
        bna_labels = Path(args.brainnetome_labels)
        if not bna_img.exists():
            raise FileNotFoundError(f"Brainnetome 图谱不存在: {bna_img}")
        if not bna_labels.exists():
            raise FileNotFoundError(f"Brainnetome 标签表不存在: {bna_labels}")
        bna_df = load_brainnetome_labels(bna_labels)
        atlas_jobs.append({
            "atlas_name_cn": "Brainnetome 246【脑网络组图谱 246】",
            "prefix": "Brainnetome246",
            "img": bna_img,
            "labels_df": bna_df,
            "root": output_root / "01_Brainnetome246_提取结果",
        })

    if not args.skip_aal:
        aal_img, aal_label_file, aal_df = resolve_aal_files([Path(x) for x in args.aal_dirs])
        atlas_jobs.append({
            "atlas_name_cn": "AAL 116【自动解剖标注图谱 116】",
            "prefix": "AAL116",
            "img": aal_img,
            "labels_df": aal_df,
            "root": output_root / "02_AAL116_提取结果",
            "label_file": aal_label_file,
        })

    for atlas in atlas_jobs:
        ensure_dir(atlas["root"])
        atlas["labels_df"].to_csv(atlas["root"] / f"{atlas['prefix']}_标签映射表.csv", index=False, encoding="utf-8-sig")

    for atlas in atlas_jobs:
        print("=" * 90)
        print(f"开始提取: {atlas['atlas_name_cn']}")
        print(f"图谱影像: {atlas['img']}")
        print(f"输出目录: {atlas['root']}")
        print(f"待处理扫描数: {len(ready_df)}")
        print("=" * 90)

        success_rows: List[Dict] = []
        fail_rows: List[Dict] = []
        total = len(ready_df)

        for idx, row in ready_df.iterrows():
            scan_id = row["canonical_scan_id"]
            print(f"[{idx + 1}/{total}] {scan_id}")
            try:
                out = extract_one_atlas(
                    atlas_name_cn=atlas["atlas_name_cn"],
                    atlas_prefix=atlas["prefix"],
                    atlas_img=atlas["img"],
                    labels_df=atlas["labels_df"],
                    scan_row=row,
                    atlas_output_root=atlas["root"],
                )
                success_rows.append(out)
                print(f"    -> 成功，timepoints={out['n_timepoints']}, rois={out['n_rois_output']}, used_confounds={out['used_confounds']}")
            except Exception as e:
                fail_rows.append({
                    "canonical_scan_id": scan_id,
                    "canonical_subject_id": row["canonical_subject_id"],
                    "subject_folder": row["subject_folder"],
                    "group": row["group"],
                    "timepoint": row["timepoint"],
                    "subject_code_normalized": row["subject_code_normalized"],
                    "status": "failed",
                    "error": repr(e),
                    "bold_path": row["bold_path"],
                    "confounds_tsv": row["confounds_tsv"],
                })
                print(f"    -> 失败: {e}")

        save_group_outputs(atlas["prefix"], atlas["root"], success_rows, fail_rows)
        print("=" * 90)
        print(f"{atlas['atlas_name_cn']} 提取完成")
        print(f"成功: {len(success_rows)}")
        print(f"失败: {len(fail_rows)}")
        print("=" * 90)

    print("全部任务结束。")
    print(f"总输出目录: {output_root}")


if __name__ == "__main__":
    main()
