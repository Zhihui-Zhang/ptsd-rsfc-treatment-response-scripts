# -*- coding: utf-8 -*-
"""Build Supplementary Tables S1-S2 Word file and fixed supplementary results workbook.

Inputs are the script-46 sensitivity/design-audit result files. Outputs are:
- 论文写作/Supplementary Online Content_PAI_TableS1S2.docx
- 论文写作/固定补充材料结果.xlsx

The Word document is intentionally limited to the requested supplement content.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


ROOT = Path(__file__).resolve().parent
WRITING_DIR = ROOT / "论文写作"
RESULT_DIR = ROOT / "46_PSY_TMS非头对头设计风险审计_敏感性模型_v5_显式站点混杂修正版_结果"

DOCX_OUT = WRITING_DIR / "Supplementary Online Content_PAI_TableS1S2.docx"
XLSX_OUT = WRITING_DIR / "固定补充材料结果.xlsx"
TEX_OUT = ROOT / "latex_template" / "supplementary_online_content_template" / "supplementary_online_content_PAI_TableS1S2.tex"

ORDER = [
    "A40c_L--A7m_L__pre",
    "A32sg_R--msOccG_R__pre",
    "mAmyg_R--lAmyg_L__pre",
    "A23c_L--dCa_R__pre",
    "A32sg_L--msOccG_R__pre",
    "A7ip_R--cLinG_R__pre",
    "A7pc_L--rHipp_L__pre",
    "A40rv_R--dIa_L__pre",
    "A32p_L--msOccG_L__pre",
    "vId_vIg_L--cHipp_R__pre",
]


def fmt_num(x: object, digits: int = 3) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(v):
        return ""
    return f"{v:.{digits}f}"


def fmt_p(x: object) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(v):
        return ""
    if v < 0.001:
        return f"{v:.2e}"
    return f"{v:.4f}"


def clean_edge(edge: str) -> str:
    return str(edge).replace("__pre", "")


def direction(beta: float) -> str:
    return "PSY-favoring" if beta < 0 else "TMS-favoring"


def interpretation(row: dict) -> str:
    m1_q = float(row["M1 FDR q"])
    m3_q = float(row["M3 IPW FDR q"])
    m3_p = float(row["M3 IPW p value"])
    same_sign = float(row["M1 beta interaction"]) * float(row["M3 IPW beta interaction"]) > 0
    if m1_q < 0.05 and m3_q < 0.05:
        return "FDR-supported in both models; strongest sensitivity-supported signal"
    if m1_q < 0.05 and m3_p < 0.05 and same_sign:
        return "Covariate-supported; IPW nominal and sign-consistent"
    if m1_q < 0.05 and same_sign:
        return "Covariate-supported only; IPW weakened"
    return "Not supported in sensitivity models"


def load_table_s1() -> pd.DataFrame:
    res = pd.read_csv(RESULT_DIR / "04b_模型结果紧凑摘要.csv", encoding="utf-8-sig")
    m1 = res[res["model_type"].eq("M1_covariate_adjusted_no_site_trial")].set_index("edge")
    m3 = res[res["model_type"].eq("M3_IPW_observed_covariate_weighted_no_site_trial")].set_index("edge")
    rows = []
    for edge in ORDER:
        r1 = m1.loc[edge]
        r3 = m3.loc[edge]
        row = {
            "Candidate connection": clean_edge(edge),
            "Direction": direction(float(r1["beta_interaction"])),
            "M1 beta interaction": float(r1["beta_interaction"]),
            "M1 SE": float(r1["se_interaction"]),
            "M1 p value": float(r1["p_interaction"]),
            "M1 FDR q": float(r1["q_fdr_within_model"]),
            "M3 IPW beta interaction": float(r3["beta_interaction"]),
            "M3 IPW SE": float(r3["se_interaction"]),
            "M3 IPW p value": float(r3["p_interaction"]),
            "M3 IPW FDR q": float(r3["q_fdr_within_model"]),
        }
        row["Sensitivity interpretation"] = interpretation(row)
        rows.append(row)
    return pd.DataFrame(rows)


def load_sy_panels() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    site = pd.read_csv(RESULT_DIR / "01b_site_by_pathway交叉表.csv", encoding="utf-8-sig")
    site = site.rename(columns={site.columns[0]: "Site", "PSY": "PSY", "TMS": "TMS"})

    balance_src = pd.read_csv(RESULT_DIR / "02_PSY_TMS基线协变量平衡_SMD.csv", encoding="utf-8-sig")
    balance_rows = []
    label_map = {
        "baseline PCL": ("Baseline PCL-5", "moderate imbalance"),
        "age": ("Age", "small imbalance"),
        "mean FD": ("Mean FD", "notable imbalance"),
        "PCL improvement / outcome": ("PCL-5 improvement", "outcome difference"),
    }
    for src_label, (label, interp) in label_map.items():
        r = balance_src[balance_src["variable"].eq(src_label)].iloc[0]
        balance_rows.append({
            "Variable": label,
            "PSY": fmt_num(r["mean_PSY"], 2 if label != "Mean FD" else 3),
            "TMS": fmt_num(r["mean_TMS"], 2 if label != "Mean FD" else 3),
            "SMD / effect size": f"SMD={float(r['SMD_TMS_minus_PSY']):.3f}",
            "Interpretation": interp,
        })
    balance_rows.append({
        "Variable": "Sex",
        "PSY": "12/11",
        "TMS": "7/36",
        "SMD / effect size": "Cramer's V=0.378",
        "Interpretation": "categorical imbalance",
    })
    balance = pd.DataFrame(balance_rows)

    estimability = pd.DataFrame([
        {
            "Model": "M1",
            "Formula": "treatment pathway x baseline FC + baseline PCL-5 + age + sex + mean FD",
            "Estimable?": "Yes",
            "Reason": "primary covariate-adjusted sensitivity model",
        },
        {
            "Model": "M2",
            "Formula": "treatment pathway x baseline FC + baseline PCL-5 + age + sex + mean FD + site",
            "Estimable?": "No",
            "Reason": "0/10 candidate connections estimable; treatment pathway and site were completely aligned; design matrix not full rank (rank=8, columns=9)",
        },
        {
            "Model": "M3",
            "Formula": "IPW-weighted treatment pathway x baseline FC model",
            "Estimable?": "Yes",
            "Reason": "observed-covariate-weighted sensitivity model",
        },
    ])
    return site, balance, estimability


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_border(cell, color: str = "D9D9D9") -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_borders = tc_pr.first_child_found_in("w:tcBorders")
    if tc_borders is None:
        tc_borders = OxmlElement("w:tcBorders")
        tc_pr.append(tc_borders)
    for edge in ("top", "left", "bottom", "right"):
        tag = "w:" + edge
        element = tc_borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            tc_borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "4")
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def format_doc_table(table, header_fill: str = "F2F2F2", font_size: float = 8.0) -> None:
    table.style = "Table Grid"
    table.autofit = True
    for row_idx, row in enumerate(table.rows):
        for cell in row.cells:
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_border(cell)
            for para in cell.paragraphs:
                para.alignment = WD_ALIGN_PARAGRAPH.LEFT
                for run in para.runs:
                    run.font.name = "Arial"
                    run.font.size = Pt(font_size)
                    if row_idx == 0:
                        run.font.bold = True
            if row_idx == 0:
                set_cell_shading(cell, header_fill)


def add_df_table(doc: Document, df: pd.DataFrame, formats: dict[str, str] | None = None, font_size: float = 8.0):
    formats = formats or {}
    table = doc.add_table(rows=1, cols=len(df.columns))
    for j, col in enumerate(df.columns):
        table.rows[0].cells[j].text = col
    for _, row in df.iterrows():
        cells = table.add_row().cells
        for j, col in enumerate(df.columns):
            val = row[col]
            if col in formats and formats[col] == "p":
                text = fmt_p(val)
            elif col in formats and formats[col] == "num":
                text = fmt_num(val)
            else:
                text = str(val)
            cells[j].text = text
    format_doc_table(table, font_size=font_size)
    return table


def set_landscape(section) -> None:
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = section.page_height, section.page_width
    section.left_margin = Inches(0.45)
    section.right_margin = Inches(0.45)
    section.top_margin = Inches(0.6)
    section.bottom_margin = Inches(0.6)


def set_portrait(section) -> None:
    section.orientation = WD_ORIENT.PORTRAIT
    section.page_width = Inches(8.27)
    section.page_height = Inches(11.69)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.7)


def add_note(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    run = p.add_run("Note. ")
    run.bold = True
    run.font.name = "Arial"
    run.font.size = Pt(9)
    rest = p.add_run(text)
    rest.font.name = "Arial"
    rest.font.size = Pt(9)
    p.paragraph_format.space_after = Pt(8)


def build_docx(table_s1: pd.DataFrame, site: pd.DataFrame, balance: pd.DataFrame, estimability: pd.DataFrame) -> None:
    doc = Document()
    set_portrait(doc.sections[0])
    styles = doc.styles
    styles["Normal"].font.name = "Arial"
    styles["Normal"].font.size = Pt(10)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run("Supplementary Materials")
    r.bold = True
    r.font.name = "Arial"
    r.font.size = Pt(16)

    doc.add_paragraph("Content of Supplementary Materials").runs[0].bold = True
    for line in [
        "Supplementary Table S1. Sensitivity analyses of treatment-pathway interaction effects for fixed baseline candidate connections.",
        "Supplementary Table S2. Design audit of site-treatment-pathway alignment and model estimability.",
    ]:
        p = doc.add_paragraph(line)
        p.style = styles["Normal"]

    doc.add_section(WD_SECTION.NEW_PAGE)
    set_landscape(doc.sections[-1])

    p = doc.add_paragraph()
    run = p.add_run("Supplementary Table S1. Sensitivity analyses of treatment-pathway interaction effects for fixed baseline candidate connections.")
    run.bold = True
    run.font.size = Pt(10)
    run.font.name = "Arial"
    add_df_table(
        doc,
        table_s1,
        formats={
            "M1 beta interaction": "num",
            "M1 SE": "num",
            "M1 p value": "p",
            "M1 FDR q": "p",
            "M3 IPW beta interaction": "num",
            "M3 IPW SE": "num",
            "M3 IPW p value": "p",
            "M3 IPW FDR q": "p",
        },
        font_size=7.0,
    )
    add_note(
        doc,
        "M1 adjusted for baseline PCL-5 severity, age, sex, and mean framewise displacement. "
        "M3 used observed-covariate inverse probability weighting based on baseline PCL-5 severity, age, sex, and mean framewise displacement. "
        "FDR correction was applied across the 10 fixed candidate baseline connections. "
        "Positive interaction coefficients indicate stronger baseline connectivity-improvement association in the TMS pathway relative to the PSY pathway; "
        "negative coefficients indicate stronger association in the PSY pathway.",
    )

    doc.add_section(WD_SECTION.NEW_PAGE)
    set_portrait(doc.sections[-1])
    p = doc.add_paragraph()
    run = p.add_run("Supplementary Table S2. Design audit of site-treatment-pathway alignment and model estimability.")
    run.bold = True
    run.font.size = Pt(10)
    run.font.name = "Arial"

    for subtitle in [
        "Panel A. Site x treatment pathway.",
    ]:
        p = doc.add_paragraph(subtitle)
        p.runs[0].bold = True
    add_df_table(doc, site, font_size=9.0)

    p = doc.add_paragraph("Panel B. Baseline covariate balance.")
    p.runs[0].bold = True
    add_df_table(doc, balance, font_size=8.5)

    p = doc.add_paragraph("Panel C. Site-adjusted model estimability.")
    p.runs[0].bold = True
    add_df_table(doc, estimability, font_size=8.0)
    add_note(
        doc,
        "Site was derived from the known data structure, with TMS participants assigned to site_TMS and PSY participants assigned to site_nonTMS_ACT_MIN_WL. "
        "Because site and treatment pathway were completely aligned, models including both treatment pathway and site were not estimable. "
        "Therefore, site-adjusted models were treated as a design audit rather than as primary inferential models.",
    )

    doc.save(DOCX_OUT)


def style_sheet(ws, freeze: str = "A2") -> None:
    header_fill = PatternFill("solid", fgColor="EAEAEA")
    thin = Side(style="thin", color="D9D9D9")
    ws.freeze_panes = freeze
    for row in ws.iter_rows():
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = Border(left=thin, right=thin, top=thin, bottom=thin)
            if cell.row == 1:
                cell.fill = header_fill
                cell.font = Font(bold=True)
    for col in ws.columns:
        max_len = max(len(str(c.value)) if c.value is not None else 0 for c in col)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max(max_len + 2, 10), 42)


def write_df(ws, df: pd.DataFrame) -> None:
    ws.append(list(df.columns))
    for _, row in df.iterrows():
        ws.append(list(row))


def build_xlsx(table_s1: pd.DataFrame, site: pd.DataFrame, balance: pd.DataFrame, estimability: pd.DataFrame) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "README"
    ws.append(["Workbook", "固定补充材料结果"])
    ws.append(["Purpose", "Central result workbook for fixed supplementary tables; future supplementary results can be added as new sheets."])
    ws.append(["Source", str(RESULT_DIR)])
    ws.append(["Current sheets", "S1_sensitivity, S2A_site_pathway, S2B_covariate_balance, S2C_estimability"])
    style_sheet(ws, freeze="A1")

    for title, df in [
        ("S1_sensitivity", table_s1),
        ("S2A_site_pathway", site),
        ("S2B_covariate_balance", balance),
        ("S2C_estimability", estimability),
    ]:
        ws = wb.create_sheet(title)
        write_df(ws, df)
        style_sheet(ws)
        if title == "S1_sensitivity":
            for col in ["C", "D", "G", "H"]:
                for cell in ws[col][1:]:
                    cell.number_format = "0.000"
            for col in ["E", "F", "I", "J"]:
                for cell in ws[col][1:]:
                    cell.number_format = "0.0000"
    wb.save(XLSX_OUT)


def latex_escape(x: object) -> str:
    s = str(x)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def df_to_latex_rows(df: pd.DataFrame, formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    lines = []
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            val = row[col]
            if col in formats and formats[col] == "p":
                text = fmt_p(val)
            elif col in formats and formats[col] == "num":
                text = fmt_num(val)
            else:
                text = str(val)
            vals.append(latex_escape(text))
        lines.append(" & ".join(vals) + r" \\")
    return "\n".join(lines)


def build_tex(table_s1: pd.DataFrame, site: pd.DataFrame, balance: pd.DataFrame, estimability: pd.DataFrame) -> None:
    TEX_OUT.parent.mkdir(parents=True, exist_ok=True)
    s1_rows = df_to_latex_rows(
        table_s1,
        formats={
            "M1 beta interaction": "num",
            "M1 SE": "num",
            "M1 p value": "p",
            "M1 FDR q": "p",
            "M3 IPW beta interaction": "num",
            "M3 IPW SE": "num",
            "M3 IPW p value": "p",
            "M3 IPW FDR q": "p",
        },
    )
    site_rows = df_to_latex_rows(site)
    balance_rows = df_to_latex_rows(balance)
    estimability_rows = df_to_latex_rows(estimability)
    tex = rf"""% Auto-generated from script 46 results by build_supplementary_tables_s1_s2.py
\documentclass[11pt]{{article}}
\usepackage[margin=0.75in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{pdflscape}}
\usepackage{{array}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{6pt}}
\renewcommand{{\arraystretch}}{{1.15}}
\begin{{document}}

\begin{{center}}
\Large\textbf{{Supplementary Materials}}
\end{{center}}

\textbf{{Content of Supplementary Materials}}

Supplementary Table S1. Sensitivity analyses of treatment-pathway interaction effects for fixed baseline candidate connections.

Supplementary Table S2. Design audit of site-treatment-pathway alignment and model estimability.

\clearpage
\begin{{landscape}}
\textbf{{Supplementary Table S1. Sensitivity analyses of treatment-pathway interaction effects for fixed baseline candidate connections.}}

\scriptsize
\begin{{longtable}}{{p{{0.09\linewidth}}p{{0.08\linewidth}}rrrrrrrrp{{0.22\linewidth}}}}
\toprule
Candidate connection & Direction & M1 beta interaction & M1 SE & M1 p value & M1 FDR q & M3 IPW beta interaction & M3 IPW SE & M3 IPW p value & M3 IPW FDR q & Sensitivity interpretation \\
\midrule
\endfirsthead
\toprule
Candidate connection & Direction & M1 beta interaction & M1 SE & M1 p value & M1 FDR q & M3 IPW beta interaction & M3 IPW SE & M3 IPW p value & M3 IPW FDR q & Sensitivity interpretation \\
\midrule
\endhead
{s1_rows}
\bottomrule
\end{{longtable}}
\normalsize
\textit{{Note.}} M1 adjusted for baseline PCL-5 severity, age, sex, and mean framewise displacement. M3 used observed-covariate inverse probability weighting based on baseline PCL-5 severity, age, sex, and mean framewise displacement. FDR correction was applied across the 10 fixed candidate baseline connections. Positive interaction coefficients indicate stronger baseline connectivity-improvement association in the TMS pathway relative to the PSY pathway; negative coefficients indicate stronger association in the PSY pathway.
\end{{landscape}}

\clearpage
\textbf{{Supplementary Table S2. Design audit of site-treatment-pathway alignment and model estimability.}}

\textbf{{Panel A. Site x treatment pathway.}}

\begin{{longtable}}{{lrr}}
\toprule
Site & PSY & TMS \\
\midrule
{site_rows}
\bottomrule
\end{{longtable}}

\textbf{{Panel B. Baseline covariate balance.}}

\begin{{longtable}}{{p{{0.22\linewidth}}p{{0.13\linewidth}}p{{0.13\linewidth}}p{{0.20\linewidth}}p{{0.22\linewidth}}}}
\toprule
Variable & PSY & TMS & SMD / effect size & Interpretation \\
\midrule
{balance_rows}
\bottomrule
\end{{longtable}}

\textbf{{Panel C. Site-adjusted model estimability.}}

\begin{{longtable}}{{p{{0.08\linewidth}}p{{0.35\linewidth}}p{{0.12\linewidth}}p{{0.35\linewidth}}}}
\toprule
Model & Formula & Estimable? & Reason \\
\midrule
{estimability_rows}
\bottomrule
\end{{longtable}}

\textit{{Note.}} Site was derived from the known data structure, with TMS participants assigned to site\_TMS and PSY participants assigned to site\_nonTMS\_ACT\_MIN\_WL. Because site and treatment pathway were completely aligned, models including both treatment pathway and site were not estimable. Therefore, site-adjusted models were treated as a design audit rather than as primary inferential models.

\end{{document}}
"""
    TEX_OUT.write_text(tex, encoding="utf-8")


def main() -> None:
    WRITING_DIR.mkdir(parents=True, exist_ok=True)
    table_s1 = load_table_s1()
    site, balance, estimability = load_sy_panels()
    build_docx(table_s1, site, balance, estimability)
    build_xlsx(table_s1, site, balance, estimability)
    build_tex(table_s1, site, balance, estimability)

    # Keep a source snapshot beside the Word/Excel outputs for reproducibility.
    tex_dir = ROOT / "latex_template" / "supplementary_online_content_template"
    tex_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), tex_dir / "build_supplementary_tables_s1_s2.py")
    print(f"Wrote {DOCX_OUT}")
    print(f"Wrote {XLSX_OUT}")
    print(f"Wrote {TEX_OUT}")


if __name__ == "__main__":
    main()
