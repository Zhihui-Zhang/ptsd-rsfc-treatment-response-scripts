import fs from "node:fs/promises";
import { SpreadsheetFile } from "@oai/artifact-tool";

const outputPath = "D:/自科＋脑中心论文选题/nature-skill/outputs/s8_submission_clean/Supplementary_Table_S8_candidate_generation_and_locking_submission_clean.xlsx";
const workbook = await SpreadsheetFile.importXlsx(await fs.readFile(outputPath));

const sheets = await workbook.inspect({ kind: "sheet", include: "id,name" });
console.log(sheets.ndjson);

const panelB = await workbook.inspect({
  kind: "table",
  range: "Panel_B_locked_candidates!A1:AC11",
  include: "values",
  tableMaxRows: 12,
  tableMaxCols: 29,
});
console.log(panelB.ndjson);

for (const [label, pattern] of [
  ["local_path_or_drive_scan", "D:|自科|PAI选题|工作站传输|nature-skill"],
  ["han_character_scan", "[\\u4E00-\\u9FFF]"],
  ["plasticity_scan", "plasticity|可塑"],
  ["possible_covariate_scan", "possible_covariate_run|possible_covariate|possible"],
  ["old_historical_label_scan", "A45r|A5l|cpSTS|A7c|A9 46v|A35 36c|A37elv|A24rv|A11l|A9 46d"],
  ["bad_candidate_id_scan", "Cundefined|EDGE_undefined"],
]) {
  const result = await workbook.inspect({
    kind: "match",
    searchTerm: pattern,
    options: { useRegex: true, maxResults: 100 },
    summary: label,
  });
  console.log(result.ndjson);
}

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "formula error scan",
});
console.log(errors.ndjson);
