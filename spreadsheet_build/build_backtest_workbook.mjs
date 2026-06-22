import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  SpreadsheetFile,
  Workbook,
} from "@oai/artifact-tool";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(scriptDir, "..");
const outputDir = path.join(root, "output", "research_v8");
const previewDir = path.join(outputDir, "workbook_previews");
await fs.mkdir(previewDir, { recursive: true });

const workbook = Workbook.create();

const palette = {
  navy: "#16324F",
  blue: "#2B6F93",
  teal: "#2A9D8F",
  pale: "#EAF2F8",
  green: "#DDF3E4",
  red: "#FCE1E1",
  amber: "#FFF1CC",
  gray: "#F3F5F7",
  border: "#CED6DE",
  text: "#1F2933",
  white: "#FFFFFF",
};

function styleTitle(sheet, range, text) {
  range.merge();
  range.values = [[text]];
  range.format = {
    fill: palette.navy,
    font: { bold: true, color: palette.white, size: 18 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };
  range.format.rowHeight = 34;
}

function styleHeader(range) {
  range.format = {
    fill: palette.blue,
    font: { bold: true, color: palette.white },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: palette.border },
  };
}

function styleBody(range) {
  range.format = {
    font: { color: palette.text },
    borders: { preset: "all", style: "thin", color: palette.border },
    verticalAlignment: "center",
  };
}

async function addCsvSheet(name, relativePath) {
  const csvText = await fs.readFile(path.join(root, relativePath), "utf8");
  const imported = await Workbook.fromCSV(csvText, { sheetName: name });
  const sourceSheet = imported.worksheets.getItem(name);
  const values = sourceSheet.getUsedRange().values;
  const sheet = workbook.worksheets.add(name);
  sheet.getRangeByIndexes(0, 0, values.length, values[0].length).values = values;
  const used = sheet.getUsedRange();
  styleBody(used);
  styleHeader(sheet.getRangeByIndexes(0, 0, 1, values[0].length));
  sheet.freezePanes.freezeRows(1);
  sheet.showGridLines = false;
  used.format.autofitColumns();
  for (let col = 0; col < values[0].length; col += 1) {
    const current = sheet.getRangeByIndexes(0, col, values.length, 1);
    if (current.format.columnWidth > 24) current.format.columnWidth = 24;
  }
  return sheet;
}

const summary = workbook.worksheets.add("摘要");
summary.showGridLines = false;
styleTitle(summary, summary.getRange("A1:H2"), "科技主题基金量化回测摘要");
summary.getRange("A3:H3").merge();
summary.getRange("A3").values = [[
  "数据截至2026-06-12；近半年首次查看后，后续结果均为迭代对照。未通过验收的模型只做影子跟踪。",
]];
summary.getRange("A3:H3").format = {
  fill: palette.amber,
  font: { color: "#7A4E00", italic: true },
  wrapText: true,
  horizontalAlignment: "left",
};
summary.getRange("A3:H3").format.rowHeight = 30;

const fundSheet = await addCsvSheet(
  "基金近半年",
  "output/research_v8/theme_locked_comparison.csv",
);
const rotationSheet = await addCsvSheet(
  "组合近半年",
  "output/research_v8/rotation_locked_comparison.csv",
);
const walkSheet = await addCsvSheet(
  "历史模拟实盘",
  "output/research_v8/rotation_walk_forward.csv",
);
const navSheet = await addCsvSheet(
  "净值校验",
  "output/research_v8/nav_validation.csv",
);
const selectionSheet = await addCsvSheet(
  "训练期选择",
  "output/research_v8/theme_training_selection.csv",
);
const riskBudgetSheet = await addCsvSheet(
  "风险预算模型",
  "output/research_v8/risk_budget_locked_comparison.csv",
);
const correlationSheet = await addCsvSheet(
  "中美短窗相关",
  "output/research_v8/short_window_correlations.csv",
);
const estimateSheet = await addCsvSheet(
  "估值覆盖层",
  "output/research_v8/estimate_overlay_locked_comparison.csv",
);

fundSheet.getRange("D2:F10").format.numberFormat = "0.00%;[Red](0.00%);-";
fundSheet.getRange("G2:G10").format.numberFormat = "0.000";
fundSheet.getRange("H2:H10").format.numberFormat = "0.00%;[Red](0.00%);-";
rotationSheet.getRange("B2:D3").format.numberFormat = "0.00%;[Red](0.00%);-";
rotationSheet.getRange("E2:E3").format.numberFormat = "0.000";
rotationSheet.getRange("F2:F3").format.numberFormat = "0.00%;[Red](0.00%);-";
walkSheet.getRange("D2:I20").format.numberFormat = "0.00%;[Red](0.00%);-";
riskBudgetSheet.getRange("B2:D3").format.numberFormat = "0.00%;[Red](0.00%);-";
riskBudgetSheet.getRange("F2:F3").format.numberFormat = "0.00%;[Red](0.00%);-";
correlationSheet.getRange("E2:J30").format.numberFormat = "0.0%;[Red](0.0%);-";
estimateSheet.getRange("C2:E10").format.numberFormat = "0.00%;[Red](0.00%);-";
estimateSheet.getRange("G2:G10").format.numberFormat = "0.00%;[Red](0.00%);-";
walkSheet.getRange("C:C").format.columnWidth = 32;
walkSheet.getRange("L:M").format.columnWidth = 24;
walkSheet.getRange("1:1").format.rowHeight = 34;

summary.getRange("A5:B5").values = [["研究状态", "结论"]];
styleHeader(summary.getRange("A5:B5"));
summary.getRange("A6:B9").values = [
  ["单基金统一模型", "未通过"],
  ["逐主题模型", "未通过"],
  ["科技组合超配", "未通过"],
  ["当前用途", "仓位辅助 / 影子跟踪"],
];
styleBody(summary.getRange("A6:B9"));
summary.getRange("B6:B8").format = {
  fill: palette.red,
  font: { bold: true, color: "#8B1E1E" },
  horizontalAlignment: "center",
};
summary.getRange("B9").format = {
  fill: palette.amber,
  font: { bold: true, color: "#7A4E00" },
  horizontalAlignment: "center",
};

summary.getRange("D5:E5").values = [["关键统计", "数值"]];
styleHeader(summary.getRange("D5:E5"));
summary.getRange("D6:D10").values = [
  ["完整净值基金数"],
  ["单基金模型PBO"],
  ["组合模型PBO"],
  ["历史模拟跑赢比例"],
  ["历史模拟超额中位数"],
];
summary.getRange("E6").formulas = [["=COUNTIF('净值校验'!H2:H4,TRUE)"]];
summary.getRange("E7:E10").values = [[0.5476190476], [0.2857142857], [0.25], [-0.0185711461]];
styleBody(summary.getRange("D6:E10"));
summary.getRange("E7:E10").format.numberFormat = "0.0%;[Red](0.0%);-";
summary.getRange("E7:E10").conditionalFormats.add("cellIs", {
  operator: "greaterThan",
  formula: 0.5,
  format: { fill: palette.red, font: { color: "#8B1E1E", bold: true } },
});

summary.getRange("A12:D12").values = [[
  "基金",
  "买入持有",
  "20/60均线",
  "逐主题模型",
]];
styleHeader(summary.getRange("A12:D12"));
summary.getRange("A13:A15").values = [["CPO/通信"], ["半导体代理"], ["人工智能"]];
summary.getRange("B13:D15").formulas = [
  ["='基金近半年'!D2", "='基金近半年'!D3", "='基金近半年'!D4"],
  ["='基金近半年'!D5", "='基金近半年'!D6", "='基金近半年'!D7"],
  ["='基金近半年'!D8", "='基金近半年'!D9", "='基金近半年'!D10"],
];
styleBody(summary.getRange("A13:D15"));
summary.getRange("B13:D15").format.numberFormat = "0.0%;[Red](0.0%);-";
summary.getRange("B13:D15").conditionalFormats.add("colorScale", {
  colors: ["#F4A6A6", "#FFF4CC", "#A8DDB5"],
  thresholds: ["min", "50%", "max"],
});

const fundChart = summary.charts.add("bar", summary.getRange("A12:D15"));
fundChart.title = "近半年基金策略收益对比";
fundChart.hasLegend = true;
fundChart.yAxis = { numberFormatCode: "0%" };
fundChart.setPosition("F5", "N17");

summary.getRange("A18:C18").values = [["科技组合", "收益率", "最大回撤"]];
styleHeader(summary.getRange("A18:C18"));
summary.getRange("A19:A20").values = [["三基金等权买入持有"], ["CPO超配候选"]];
summary.getRange("B19:C20").formulas = [
  ["='组合近半年'!B2", "='组合近半年'!F2"],
  ["='组合近半年'!B3", "='组合近半年'!F3"],
];
styleBody(summary.getRange("A19:C20"));
summary.getRange("B19:C20").format.numberFormat = "0.0%;[Red](0.0%);-";

summary.getRange("A22:H22").merge();
summary.getRange("A22").values = [["审计解释"]];
summary.getRange("A22:H22").format = {
  fill: palette.teal,
  font: { bold: true, color: palette.white },
};
summary.getRange("A23:H26").merge();
summary.getRange("A23").values = [[
  "组合候选近半年收益较高，主要来自测试开始前已经确定的CPO超配。期间没有再次换仓，"
  + "因此不能把差额解释为成功择时。历史模拟实盘仅25%的区间取得正超额，"
  + "单基金模型PBO约54.8%，组合模型PBO约28.6%；当前全部模型均未通过真实部署验收。",
]];
summary.getRange("A23:H26").format = {
  fill: palette.pale,
  font: { color: palette.text },
  wrapText: true,
  verticalAlignment: "top",
};

summary.getRange("A28:H28").merge();
summary.getRange("A28").values = [["当前影子配置（科技主题资金内部）"]];
summary.getRange("A28:H28").format = {
  fill: palette.teal,
  font: { bold: true, color: palette.white },
};
summary.getRange("A29:D29").values = [["CPO/通信", "半导体代理", "人工智能", "模型验收"]];
styleHeader(summary.getRange("A29:D29"));
summary.getRange("A30:D30").values = [[0.6, 0.2, 0.2, "未通过，仅影子跟踪"]];
styleBody(summary.getRange("A30:D30"));
summary.getRange("A30:C30").format.numberFormat = "0%";
summary.getRange("D30").format = { fill: palette.red, font: { bold: true, color: "#8B1E1E" } };

const checks = workbook.worksheets.add("检查");
checks.showGridLines = false;
styleTitle(checks, checks.getRange("A1:F2"), "模型与数据检查");
checks.getRange("A4:F4").values = [[
  "检查项", "实际值", "期望值", "差异/状态", "结论", "说明",
]];
styleHeader(checks.getRange("A4:F4"));
checks.getRange("A5:A9").values = [
  ["净值校验通过数"],
  ["净值基金总数"],
  ["单基金部署"],
  ["组合部署"],
  ["测试期执行延迟"],
];
checks.getRange("B5").formulas = [["=COUNTIF('净值校验'!H2:H4,TRUE)"]];
checks.getRange("B6").formulas = [["=COUNTA('净值校验'!A2:A4)"]];
checks.getRange("B7:B9").values = [["FALSE"], ["FALSE"], ["下一净值日"]];
checks.getRange("C5:C9").values = [[3], [3], ["FALSE"], ["FALSE"], ["下一净值日"]];
checks.getRange("D5").formulas = [["=B5-C5"]];
checks.getRange("D6").formulas = [["=B6-C6"]];
checks.getRange("D7:D9").formulas = [["=IF(B7=C7,\"匹配\",\"不匹配\")"], ["=IF(B8=C8,\"匹配\",\"不匹配\")"], ["=IF(B9=C9,\"匹配\",\"不匹配\")"]];
checks.getRange("E5:E6").formulas = [["=IF(D5=0,\"OK\",\"FAIL\")"], ["=IF(D6=0,\"OK\",\"FAIL\")"]];
checks.getRange("E7:E9").formulas = [["=IF(D7=\"匹配\",\"OK\",\"FAIL\")"], ["=IF(D8=\"匹配\",\"OK\",\"FAIL\")"], ["=IF(D9=\"匹配\",\"OK\",\"FAIL\")"]];
checks.getRange("F5:F9").values = [
  ["三只净值序列均需通过"],
  ["CPO、半导体代理、AI"],
  ["未通过跨基金验收"],
  ["历史模拟实盘不稳定"],
  ["避免使用未知当日净值"],
];
styleBody(checks.getRange("A5:F9"));
checks.getRange("E5:E9").conditionalFormats.add("containsText", {
  text: "OK",
  format: { fill: palette.green, font: { color: "#176B36", bold: true } },
});
checks.getRange("E5:E9").conditionalFormats.add("containsText", {
  text: "FAIL",
  format: { fill: palette.red, font: { color: "#8B1E1E", bold: true } },
});
checks.freezePanes.freezeRows(4);

const sources = workbook.worksheets.add("来源与口径");
sources.showGridLines = false;
styleTitle(sources, sources.getRange("A1:F2"), "来源、口径与限制");
sources.getRange("A4:F4").values = [[
  "项目", "值/口径", "单位", "截至日期", "来源", "备注",
]];
styleHeader(sources.getRange("A4:F4"));
sources.getRange("A5:F12").values = [
  ["完整净值", "成立以来单位净值", "日", "2026-06-12", "https://akshare.akfamily.xyz/data/fund/fund_public.html", "AKShare，东财公开数据"],
  ["备用净值", "Data_netWorthTrend", "日", "2026-06-12", "https://fund.eastmoney.com/", "网页公开接口，不保证长期稳定"],
  ["盘中估值", "fund_value_estimation_em / fundgz", "%", "盘中", "https://akshare.akfamily.xyz/data/fund/fund_public.html", "估值不是最终净值"],
  ["ETF确认", "价格、IOPV、折溢价", "%", "盘中", "https://akshare.akfamily.xyz/data/fund/fund_public.html", "用于ETF联接基金交叉确认"],
  ["模型论文", "Time Series Momentum", "", "2012", "https://doi.org/10.1016/j.jfineco.2011.11.003", "多周期趋势依据"],
  ["模型论文", "Volatility-Managed Portfolios", "", "2017", "https://doi.org/10.1111/jofi.12513", "波动率管理依据"],
  ["过拟合", "Probability of Backtest Overfitting", "", "", "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253", "CSCV/PBO依据"],
  ["中美相关", "20个交易日短窗口", "日", "2026-06-12", "https://www.investing.com/", "低置信，只作方向确认"],
];
styleBody(sources.getRange("A5:F12"));
sources.getRange("A4:F12").format.wrapText = true;
sources.freezePanes.freezeRows(4);

summary.freezePanes.freezeRows(3);
summary.getRange("A1:N30").format.font = { name: "Microsoft YaHei" };
summary.getRange("A:A").format.columnWidth = 19;
summary.getRange("B:E").format.columnWidth = 15;
summary.getRange("F:N").format.columnWidth = 12;
checks.getRange("A:A").format.columnWidth = 20;
checks.getRange("B:E").format.columnWidth = 14;
checks.getRange("F:F").format.columnWidth = 30;
sources.getRange("A:A").format.columnWidth = 16;
sources.getRange("B:B").format.columnWidth = 28;
sources.getRange("C:D").format.columnWidth = 14;
sources.getRange("E:E").format.columnWidth = 48;
sources.getRange("F:F").format.columnWidth = 30;

const inspect = await workbook.inspect({
  kind: "table",
  range: "摘要!A1:H30",
  include: "values,formulas",
  tableMaxRows: 30,
  tableMaxCols: 8,
});
await fs.writeFile(path.join(previewDir, "summary_inspect.ndjson"), inspect.ndjson, "utf8");

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
await fs.writeFile(path.join(previewDir, "formula_errors.ndjson"), errors.ndjson, "utf8");

for (const sheetName of ["摘要", "基金近半年", "组合近半年", "风险预算模型", "估值覆盖层", "历史模拟实盘", "中美短窗相关", "净值校验", "检查", "来源与口径"]) {
  const preview = await workbook.render({
    sheetName,
    autoCrop: "all",
    scale: 1,
    format: "png",
  });
  await fs.writeFile(
    path.join(previewDir, `${sheetName}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(outputDir, "科技主题基金量化回测汇总.xlsx"));
console.log(path.join(outputDir, "科技主题基金量化回测汇总.xlsx"));
