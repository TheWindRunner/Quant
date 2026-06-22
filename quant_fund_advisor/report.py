"""Small self-contained HTML research report."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


def _table(frame: pd.DataFrame) -> str:
    return frame.to_html(border=0, classes="data", float_format=lambda x: f"{x:.4f}")


def write_html_report(
    output: str | Path,
    recommendations: pd.DataFrame,
    relationships: pd.DataFrame,
    equity: pd.DataFrame,
    metrics: dict[str, float],
    as_of: str,
) -> Path:
    metric_frame = pd.DataFrame([metrics]).T.rename(columns={0: "value"})
    recent_equity = equity.tail(20).copy()
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>基金量化研究报告</title>
<style>
body{{font-family:Arial,"Microsoft YaHei",sans-serif;max-width:1100px;margin:32px auto;color:#17202a}}
h1,h2{{color:#143d59}} .note{{background:#fff4ce;padding:12px;border-left:4px solid #e0a800}}
table.data{{border-collapse:collapse;width:100%;margin:12px 0 28px}}
.data th,.data td{{padding:7px;border-bottom:1px solid #ddd;text-align:right}}
.data th:first-child,.data td:first-child{{text-align:left}}
</style></head><body>
<h1>基金量化研究报告</h1>
<p>数据截止：{escape(as_of)}</p>
<p class="note">仅供研究，不构成投资建议。基金净值滞后、申赎限制、费率与支付宝实际上架状态须在交易前复核。</p>
<h2>最新信号</h2>{_table(recommendations)}
<h2>美股到A股行业传导</h2>{_table(relationships)}
<h2>回测指标</h2>{_table(metric_frame)}
<h2>最近20期净值</h2>{_table(recent_equity)}
</body></html>"""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path

