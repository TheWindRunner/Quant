# 基金量化研究工具

这是一个研究级、可回测的基金行业轮动框架。它支持 CSV 离线数据，也支持通过
AKShare 免费获取 A 股、A 股 ETF、开放式基金净值和美股历史行情。

## 安装

仅运行离线演示：

```powershell
pip install -e .
python -m quant_fund_advisor.cli demo --output output
```

使用免费在线行情：

```powershell
pip install -e ".[live]"
```

AKShare 不需要 API Key。它封装公开网页数据源，适合研究和个人使用，但上游页面变化时
接口可能暂时失效。每次形成投资判断前，都应检查数据截止日期和基金公告。

## 查看和下载历史走势

```powershell
# A 股
fund-advisor history --type cn_stock --symbol 600519 --start 2024-01-01

# A 股 ETF
fund-advisor history --type cn_etf --symbol 159995 --start 2024-01-01 --adjust qfq

# 开放式基金净值
fund-advisor history --type open_fund --symbol 000001 --start 2024-01-01

# 美股或美股 ETF
fund-advisor history --type us_stock --symbol XLK --start 2024-01-01 --adjust qfq
```

默认保存到 `data/<类型>_<代码>.csv`，字段统一为日期索引和
`open/high/low/close/volume` 等可用列；开放式基金以单位净值作为 `close`。

## 接入完整 Quant 工作流

编辑 `data/market_manifest.csv`，每行配置一个行业代理：

```text
group,sector,asset_type,symbol,adjust
us,technology,us_stock,XLK,qfq
cn,technology,cn_etf,159995,qfq
```

`group` 必须包含 `us` 和 `cn`，相同行业名应在两组中对应。运行：

```powershell
fund-advisor live --manifest data/market_manifest.csv --start 2023-01-01 --output output
```

流程会：

1. 免费拉取清单内所有标的历史数据。
2. 保存 `output/us_prices.csv` 和 `output/cn_prices.csv`，方便复核和离线重跑。
3. 计算跨市场滞后相关性、行业评分、推荐结果和回测。
4. 生成 `output/report.html`。

网络接口不可用时，可以继续使用落盘数据：

```powershell
fund-advisor csv --us output/us_prices.csv --cn output/cn_prices.csv --output output
```

## 数据源选择

- **AKShare（默认）**：免费、免密钥、覆盖中国股票/ETF/基金和美股，最适合当前项目。
- **Tushare Pro**：结构化接口较稳定，但很多历史接口需要积分或更高权限，不作为免费默认源。
- **Alpha Vantage**：有免费额度，适合少量美股查询，但需要 API Key 且有调用频率限制。

这里的信号和回测不构成投资建议。历史相关性不代表因果关系，回测也无法覆盖申赎延迟、
暂停交易、额度限制、税费和全部冲击成本。
