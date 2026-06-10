import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = "D:/自科＋脑中心论文选题/PAI选题/工作站传输/第四步分析-codex";
const outDir = "D:/自科＋脑中心论文选题/nature-skill/outputs/s8_submission_clean";
const outputPath = path.join(outDir, "Supplementary_Table_S8_candidate_generation_and_locking_submission_clean.xlsx");

const files = {
  manifest: path.join(root, "论文写作/补充材料/S4_S6_symptom_outcome_locked_outputs/00_locked_10_candidate_edges_manifest.csv"),
  s5: path.join(root, "论文写作/补充材料/S4_S6_symptom_outcome_locked_outputs/Supplementary_Table_S5_candidate_identity_source_column_lineage_Brainnetome_mapping.csv"),
  final40: path.join(root, "40_v3_稳定性选择候选边_正式固定候选治疗调节和可塑性分析结果_修正BNA标签_preMeanFD修复版/40_final_core_edge_decision_table.csv"),
  cov40: path.join(root, "40_v3_稳定性选择候选边_正式固定候选治疗调节和可塑性分析结果_修正BNA标签_preMeanFD修复版/11_fixed_candidate_moderation_covariates.csv"),
  audit40: path.join(root, "40_v3_稳定性选择候选边_正式固定候选治疗调节和可塑性分析结果_修正BNA标签_preMeanFD修复版/00_run_audit.json"),
  canonical39v4: path.join(root, "39_v4_codex_固定候选边_value_column谱系审计_无解剖标签正式版结果/03_canonical_fixed_10_candidate_edges_value_column_only.csv"),
  historyConsensus: "D:/自科＋脑中心论文选题/PAI选题/工作站传输/第四步分析：探索/39_（历史）稳定性选择结果_论文备份和汇总包_v2_纳入K10K30/03_跨版本候选边_consensus_table.csv",
  historyOverview: "D:/自科＋脑中心论文选题/PAI选题/工作站传输/第四步分析：探索/39_（历史）稳定性选择结果_论文备份和汇总包_v2_纳入K10K30/01_纳入结果文件夹_overview.csv",
};

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    const next = text[i + 1];
    if (inQuotes) {
      if (ch === '"' && next === '"') {
        cell += '"';
        i++;
      } else if (ch === '"') {
        inQuotes = false;
      } else {
        cell += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ",") {
      row.push(cell);
      cell = "";
    } else if (ch === "\n") {
      row.push(cell.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += ch;
    }
  }
  if (cell.length || row.length) {
    row.push(cell.replace(/\r$/, ""));
    rows.push(row);
  }
  const headers = rows.shift().map((h, i) => i === 0 ? h.replace(/^\uFEFF/, "") : h);
  return rows.filter(r => r.some(v => v !== "")).map(r => Object.fromEntries(headers.map((h, i) => [h, r[i] ?? ""])));
}

async function readCsv(file) {
  return parseCsv(await fs.readFile(file, "utf8"));
}

function asNumber(value, digits = 4) {
  if (value === undefined || value === null || value === "") return "";
  const n = Number(value);
  if (!Number.isFinite(n)) return value;
  if (Math.abs(n) < 0.001 && n !== 0) return n.toExponential(3);
  return Number(n.toFixed(digits));
}

function colLetter(n) {
  let s = "";
  while (n > 0) {
    const m = (n - 1) % 26;
    s = String.fromCharCode(65 + m) + s;
    n = Math.floor((n - 1) / 26);
  }
  return s;
}

function addSheet(workbook, name, headers, rows) {
  const sheet = workbook.worksheets.add(name);
  const values = [headers, ...rows.map(r => headers.map(h => r[h] ?? ""))];
  const range = `A1:${colLetter(headers.length)}${values.length}`;
  sheet.getRange(range).values = values;
  return sheet;
}

function recoverCovariates(text) {
  if (!text || text === "possible_covariate_run") {
    return "Not recoverable from archived stability-selection summary";
  }
  return text.replace(/[\[\]']/g, "").replace(/,/g, ";");
}

function stabilityTier(h) {
  const ge020 = Number(h.n_runs_selection_ge_0p20 || 0);
  const ge010 = Number(h.n_runs_selection_ge_0p10 || 0);
  if (ge020 >= 2) return "recurrent_stability_signal";
  if (ge010 >= 2) return "moderate_stability_signal";
  return "limited_stability_signal";
}

const [manifest, s5, final40, cov40, audit40, canonical39v4, historyConsensus, historyOverview] = await Promise.all([
  readCsv(files.manifest),
  readCsv(files.s5),
  readCsv(files.final40),
  readCsv(files.cov40),
  fs.readFile(files.audit40, "utf8").then(JSON.parse),
  readCsv(files.canonical39v4),
  readCsv(files.historyConsensus),
  readCsv(files.historyOverview),
]);

const s5ById = new Map(s5.map(r => [r.candidate_id, r]));
const finalByShort = new Map(final40.map(r => [r.edge_short, r]));
const covByShort = new Map(cov40.map(r => [r.edge_short, r]));
const canonicalByValue = new Map(canonical39v4.map(r => [r.value_column, r]));
const historyByValue = new Map(historyConsensus.map(r => [r.edge, r]));

const panelA = [
  {
    panel: "A",
    item: "Analysis universe",
    submission_text: "The upstream value-column scan covered 7,503 analyzable baseline rsFC value columns available in the harmonized analysis table, rather than all theoretical Brainnetome-246 ROI-to-ROI pairs.",
    boundary_note: "This wording avoids implying that all 30,135 theoretical Brainnetome-246 pairs entered the archived value-column scan.",
  },
  {
    panel: "A",
    item: "Candidate generation",
    submission_text: "Candidate connections were identified through exploratory whole-brain stability-selection runs over analyzable baseline rsFC value columns.",
    boundary_note: "This is hypothesis-generating candidate generation, not a whole-brain FDR-confirmatory discovery claim.",
  },
  {
    panel: "A",
    item: "Stability-selection settings",
    submission_text: "Archived stability-selection runs used repeated stratified subsampling of the TMS and psychotherapy groups, sample fraction 0.80, and top-k retention settings of 10, 20, or 30 edges across sensitivity runs.",
    boundary_note: "Some archived stability-selection summaries do not fully recover covariate specification; the formal fixed-candidate model used the final covariate set listed in Panel B.",
  },
  {
    panel: "A",
    item: "Candidate identity lock",
    submission_text: "The final 10 candidates were locked by canonical source-column identity and then mapped to corrected Brainnetome anatomical labels for reporting.",
    boundary_note: "Machine-readable unknown_roi value-column names are source-column identifiers and are not anatomical labels.",
  },
  {
    panel: "A",
    item: "Formal fixed-candidate model",
    submission_text: `The formal fixed-candidate model used n=${audit40.n_tms_psy_subjects} participants (${audit40.n_TMS} TMS, ${audit40.n_PSY} psychotherapy) and adjusted for baseline PCL-5, age, sex, and pre-treatment mean framewise displacement.`,
    boundary_note: "This formal model, rather than exploratory stability-run covariate variants, provides the reported fixed-candidate estimates.",
  },
];

const panelB = manifest
  .sort((a, b) => Number(a.table2_order) - Number(b.table2_order))
  .map(r => {
    const s5r = s5ById.get(r.candidate_id) ?? {};
    const f = finalByShort.get(r.candidate_connection) ?? {};
    const c = covByShort.get(r.candidate_connection) ?? {};
    const h = historyByValue.get(r.value_column) ?? {};
    const canon = canonicalByValue.get(r.value_column) ?? {};
    return {
      panel: "B",
      table2_order: r.table2_order,
      candidate_id: r.candidate_id,
      candidate_connection: r.candidate_connection,
      canonical_edge_id: r.edge_label_44 || canon.canonical_edge_id,
      canonical_value_column: r.value_column,
      endpoint_1: r.endpoint1_anatomical_label,
      endpoint_2: r.endpoint2_anatomical_label,
      anatomical_endpoint_pair: r.anatomical_endpoints_for_table2 || r.anatomical_endpoints,
      favored_pathway: r.primary_favored_pathway,
      interaction_sign: r.sign_of_interaction,
      baseline_fc_n: (r.n_note || "").startsWith("n=") ? r.n_note : s5r.baseline_fc_nonmissing_n || r.n_note,
      formal_model: c.model || "covariate_adjusted",
      formal_covariates: "baseline PCL-5; age; sex; pre-treatment mean FD",
      beta_interaction: asNumber(r.table2_interaction_b || f.cov_beta_interaction || c.beta_interaction_TMS_minus_PSY_per1SD, 3),
      confidence_interval: r.table2_ci,
      p_value: asNumber(r.table2_p || f.cov_p_interaction || c.p_interaction, 6),
      q_value_10_candidates: asNumber(r.table2_q || f.cov_q_interaction_10edges || c.q_interaction_fdr_10edges, 6),
      delta_r2: asNumber(r.table2_delta_r2, 3),
      stability_evidence_status: h.edge ? "value_column_matched_archived_stability_metric" : "not_recovered",
      stability_runs_available: h.n_runs_available,
      stability_runs_recommended: h.n_recommended_runs,
      stability_runs_frequency_ge_0p20: h.n_runs_selection_ge_0p20,
      max_selection_frequency: asNumber(h.max_selection_frequency, 4),
      mean_selection_frequency: asNumber(h.mean_selection_frequency, 4),
      direction_consensus: h.dominant_direction_consensus,
      direction_consistency: asNumber(h.direction_cross_run_consistency, 4),
      stability_tier_no_downstream_information: h.edge ? stabilityTier(h) : "not_recovered",
      locked_status: "locked_by_final_Table_2_and_canonical_value_column",
      downstream_use_boundary: "Not selected or retained based on downstream analyses",
    };
  });

const panelC = [
  ["Primary PCL-5 moderation", "Yes", "No", "Primary fixed-candidate test"],
  ["General prognostic models", "Yes", "No", "Treatment-selection versus prognosis contrast"],
  ["PCL/GAD/PHQ symptom-outcome matrix", "Yes", "No", "Post-lock symptom convergence check"],
  ["PSY longitudinal deltaFC follow-up", "Yes", "No", "Post-lock process-level follow-up"],
  ["WL deltaFC checks", "Yes", "No", "Post-lock boundary check"],
  ["HC-referenced analyses", "Yes", "No", "Post-lock reference analysis"],
  ["TMS process checks", "Yes", "No", "Exploratory post-lock process check"],
  ["Figure visualization", "Yes", "No", "Anatomical display only"],
].map(([analysis, used, selection, purpose]) => ({
  panel: "C",
  downstream_analysis: analysis,
  used_locked_candidates: used,
  used_for_candidate_selection_or_locking: selection,
  purpose,
  boundary_statement: "Downstream analysis was performed after the candidate list was locked.",
}));

const sourceNotes = [
  {
    source_id: "v36v3",
    analysis_archive_or_output: "Full-brain value-column scan output",
    permitted_use: "Upstream analyzable value-column screening context",
    submission_boundary: "Do not describe as all theoretical Brainnetome-246 ROI pairs or as whole-brain FDR-confirmatory discovery.",
  },
  {
    source_id: "v38_v39_archived",
    analysis_archive_or_output: "Archived exploratory stability-selection summaries",
    permitted_use: "Value-column-matched stability metrics and sensitivity-run settings",
    submission_boundary: "Do not use archived anatomical labels; do not use downstream follow-up results as candidate-generation evidence.",
  },
  {
    source_id: "v39v4",
    analysis_archive_or_output: "Formal value-column lock",
    permitted_use: "Canonical identity source for the locked 10 candidates",
    submission_boundary: "Use value_column identity; anatomical labels are assigned only after corrected Brainnetome mapping.",
  },
  {
    source_id: "v40v3",
    analysis_archive_or_output: "Fixed-candidate model with corrected Brainnetome labels",
    permitted_use: "Final model estimates, corrected labels, and formal covariate set",
    submission_boundary: "Use for submitted Table 2 linkage and candidate labels.",
  },
  {
    source_id: "S5_manifest",
    analysis_archive_or_output: "Locked candidate manifest and S5 identity mapping",
    permitted_use: "Candidate ID order and cross-table consistency",
    submission_boundary: "Use this source as the submitted C01-C10 order.",
  },
];

const stabilityRows = historyOverview.map(r => ({
  run_label: r.run_label,
  n_rows_input: r.n_rows_input,
  n_TMS: r.n_TMS,
  n_PSY: r.n_PSY,
  n_subsamples: r.n_subsamples,
  sample_fraction: r.sample_frac,
  top_k: r.top_k,
  covariate_specification: recoverCovariates(r.covariates_used),
  n_decision_rows: r.n_decision_rows,
  n_recommended_edges: r.n_recommended_edges,
  submission_interpretation: "exploratory stability-selection sensitivity run",
}));

const wb = Workbook.create();
addSheet(wb, "Panel_A_protocol", Object.keys(panelA[0]), panelA);
addSheet(wb, "Panel_B_locked_candidates", Object.keys(panelB[0]), panelB);
addSheet(wb, "Panel_C_downstream_boundary", Object.keys(panelC[0]), panelC);
addSheet(wb, "Stability_run_settings", Object.keys(stabilityRows[0]), stabilityRows);
addSheet(wb, "Source_notes", Object.keys(sourceNotes[0]), sourceNotes);

await fs.mkdir(outDir, { recursive: true });
await wb.render({ sheetName: "Panel_A_protocol", range: "A1:D6", scale: 1 });
await wb.render({ sheetName: "Panel_B_locked_candidates", range: "A1:J11", scale: 1 });

const errors = await wb.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "formula error scan",
});
console.log(errors.ndjson);

const output = await SpreadsheetFile.exportXlsx(wb);
await output.save(outputPath);
console.log(outputPath);
