from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "output" / "ai_supply_chain_potential"
OUT.mkdir(parents=True, exist_ok=True)


SECTOR_BASKETS = {
    "PCB/高速板": {
        "physical_logic": "AI服务器功耗、层数、信号完整性和背板密度上升，PCB单机价值量被动提升。",
        "heat_seed": 0.90,
        "bottleneck_score": 0.85,
        "funds": "720001/024481/021523/021528",
        "stocks": [
            ("300476", "胜宏科技"),
            ("002463", "沪电股份"),
            ("002916", "深南电路"),
            ("600183", "生益科技"),
            ("688183", "生益电子"),
        ],
    },
    "CPO/光模块": {
        "physical_logic": "GPU集群规模扩大后，跨节点通信带宽成为系统吞吐瓶颈，光模块与CPO受益。",
        "heat_seed": 0.88,
        "bottleneck_score": 0.90,
        "funds": "007817/008326/008327",
        "stocks": [
            ("300308", "中际旭创"),
            ("300502", "新易盛"),
            ("300394", "天孚通信"),
            ("002281", "光迅科技"),
            ("688498", "源杰科技"),
        ],
    },
    "存储/HBM": {
        "physical_logic": "AI训练和推理受显存容量、带宽和堆叠良率约束，HBM与高端DRAM景气延长。",
        "heat_seed": 0.72,
        "bottleneck_score": 0.88,
        "funds": "018816/008887/006502/006503",
        "stocks": [
            ("603986", "兆易创新"),
            ("301308", "江波龙"),
            ("688525", "佰维存储"),
            ("300223", "北京君正"),
            ("688041", "海光信息"),
        ],
    },
    "半导体设备": {
        "physical_logic": "HBM、先进封装和国产晶圆扩产最终受设备、工艺和良率爬坡约束。",
        "heat_seed": 0.55,
        "bottleneck_score": 0.80,
        "funds": "017811/006502/006503",
        "stocks": [
            ("002371", "北方华创"),
            ("688012", "中微公司"),
            ("688120", "华海清科"),
            ("300604", "长川科技"),
            ("688072", "拓荆科技"),
        ],
    },
    "液冷/温控": {
        "physical_logic": "芯片功耗密度上升，风冷散热能力接近约束，液冷从可选项变成高密度算力底座。",
        "heat_seed": 0.45,
        "bottleneck_score": 0.86,
        "funds": "暂无纯基金，优先通过AI/数字经济或设备基金间接暴露",
        "stocks": [
            ("002837", "英维克"),
            ("300249", "依米康"),
            ("603912", "佳力图"),
            ("301018", "申菱环境"),
            ("300990", "同飞股份"),
        ],
    },
    "电力/电网/配电": {
        "physical_logic": "AI数据中心持续耗电，新增算力受供电容量、配电设备、变压器和并网节奏约束。",
        "heat_seed": 0.38,
        "bottleneck_score": 0.82,
        "funds": "018034或电力设备/电网相关基金",
        "stocks": [
            ("600406", "国电南瑞"),
            ("000400", "许继电气"),
            ("002028", "思源电气"),
            ("601179", "中国西电"),
            ("600312", "平高电气"),
        ],
    },
    "高速铜连接/连接器": {
        "physical_logic": "机柜内短距互连需要更低功耗和更低成本方案，铜连接在部分距离段替代光互连。",
        "heat_seed": 0.42,
        "bottleneck_score": 0.76,
        "funds": "暂无纯基金，可能映射到电子/通信/AI硬件基金",
        "stocks": [
            ("002475", "立讯精密"),
            ("300115", "长盈精密"),
            ("002130", "沃尔核材"),
            ("300351", "永贵电器"),
            ("603328", "依顿电子"),
        ],
    },
    "光纤/光缆": {
        "physical_logic": "数据中心东西向流量和长距互联放大，光纤需求跟随AI集群网络扩张。",
        "heat_seed": 0.40,
        "bottleneck_score": 0.70,
        "funds": "007817/008326间接覆盖",
        "stocks": [
            ("601869", "长飞光纤"),
            ("600487", "亨通光电"),
            ("600522", "中天科技"),
            ("600105", "永鼎股份"),
            ("300265", "通光线缆"),
        ],
    },
    "服务器/算力整机": {
        "physical_logic": "GPU、交换机、供电和散热最终集成为服务器和机柜，订单弹性强但热度通常较高。",
        "heat_seed": 0.82,
        "bottleneck_score": 0.65,
        "funds": "008585/019829",
        "stocks": [
            ("000977", "浪潮信息"),
            ("603019", "中科曙光"),
            ("601138", "工业富联"),
            ("000938", "紫光股份"),
            ("300442", "润泽科技"),
        ],
    },
    "电子材料/先进封装材料": {
        "physical_logic": "先进封装、HBM和高速PCB都依赖材料稳定性，材料扩产慢于下游需求时会形成隐性瓶颈。",
        "heat_seed": 0.36,
        "bottleneck_score": 0.74,
        "funds": "半导体/电子材料相关基金间接覆盖",
        "stocks": [
            ("300346", "南大光电"),
            ("688126", "沪硅产业"),
            ("688019", "安集科技"),
            ("300054", "鼎龙股份"),
            ("300655", "晶瑞电材"),
        ],
    },
}


def fetch_stock_nav(symbol: str, start_date: str) -> pd.Series:
    import akshare as ak

    end_date = pd.Timestamp.today().strftime("%Y%m%d")
    prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
    loaders = [
        lambda: ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        ).rename(columns={"日期": "date", "收盘": "close"}),
        lambda: ak.stock_zh_a_daily(
            symbol=f"{prefix}{symbol}",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        ),
        lambda: ak.stock_zh_a_hist_tx(
            symbol=f"{prefix}{symbol}",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        ),
    ]
    errors = []
    frame = None
    for loader in loaders:
        try:
            candidate = loader()
            if isinstance(candidate, pd.DataFrame) and not candidate.empty:
                frame = candidate
                break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")
    if frame is None:
        raise RuntimeError("；".join(errors))
    frame = frame.rename(columns={"日期": "date", "收盘": "close"})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    return frame.dropna(subset=["date", "close"]).set_index("date")["close"].sort_index()


def basket_metrics(name: str, meta: dict[str, object]) -> dict[str, object]:
    start = (pd.Timestamp.today() - pd.Timedelta(days=260)).strftime("%Y%m%d")
    series = []
    used = []
    errors = []
    for code, stock_name in meta["stocks"]:
        try:
            nav = fetch_stock_nav(code, start)
            if len(nav) >= 40:
                series.append(nav.rename(stock_name))
                used.append(f"{code} {stock_name}")
            time.sleep(0.15)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{code} {stock_name}: {type(exc).__name__}: {exc}")
    if not series:
        raise RuntimeError(f"{name} 没有可用股票行情: {'; '.join(errors)}")
    prices = pd.concat(series, axis=1).dropna().sort_index()
    ret = prices.pct_change(fill_method=None)
    basket = prices.div(prices.iloc[0]).mean(axis=1)
    out = {
        "板块": name,
        "样本股票": "；".join(used),
        "可用股票数": len(used),
        "最新日期": prices.index[-1].date().isoformat(),
        "近1周涨跌幅": float(basket.iloc[-1] / basket.iloc[-6] - 1) if len(basket) > 6 else np.nan,
        "近1月涨跌幅": float(basket.iloc[-1] / basket.iloc[-21] - 1) if len(basket) > 21 else np.nan,
        "近3月涨跌幅": float(basket.iloc[-1] / basket.iloc[-61] - 1) if len(basket) > 61 else np.nan,
        "近半年涨跌幅": float(basket.iloc[-1] / basket.iloc[0] - 1),
        "近1月上涨广度": float((prices.iloc[-1] / prices.iloc[-21] - 1 > 0).mean()) if len(prices) > 21 else np.nan,
        "20日年化波动": float(ret.tail(20).mean(axis=1).std() * np.sqrt(252)),
        "热度代理": float(meta["heat_seed"]),
        "物理瓶颈分": float(meta["bottleneck_score"]),
        "基金映射": str(meta["funds"]),
        "物理逻辑": str(meta["physical_logic"]),
        "错误": "；".join(errors),
    }
    return out


def zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def score(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["动量持续性分"] = (
        0.50 * zscore(result["近1月涨跌幅"])
        + 0.25 * zscore(result["近3月涨跌幅"])
        + 0.15 * zscore(result["近半年涨跌幅"])
        + 0.10 * zscore(result["近1月上涨广度"])
    )
    result["低热度分"] = 1.0 - result["热度代理"]
    result["高潜综合分"] = (
        0.55 * result["动量持续性分"]
        + 0.25 * result["物理瓶颈分"]
        + 0.20 * result["低热度分"]
        - 0.10 * zscore(result["20日年化波动"])
    )
    result["是否低热度高涨"] = (
        (result["近1月涨跌幅"] > result["近1月涨跌幅"].median())
        & (result["热度代理"] <= result["热度代理"].median())
    )
    return result.sort_values("高潜综合分", ascending=False)


def pct(value: float) -> str:
    return "" if pd.isna(value) else f"{value * 100:.2f}%"


def write_report(frame: pd.DataFrame) -> None:
    top = frame.head(6).copy()
    low_heat = frame.loc[frame["是否低热度高涨"]].copy()
    lines = [
        "# AI供应链高潜板块扫描",
        "",
        "## 口径",
        "",
        "- 不再使用单日涨跌幅判断板块强弱，核心动量改为近1月涨跌幅。",
        "- 同时保留近3月、近半年，用来判断趋势是否只是短期脉冲。",
        "- 热门不等于好：综合分会惩罚高热度板块，寻找“已经上涨但讨论热度相对低”的方向。",
        "- 物理瓶颈来自AI算力链条约束：电力、散热、带宽互连、PCB、存储、设备、材料。",
        "",
        "## 高潜综合排名",
        "",
        top[[
            "板块",
            "近1周涨跌幅",
            "近1月涨跌幅",
            "近3月涨跌幅",
            "近半年涨跌幅",
            "近1月上涨广度",
            "热度代理",
            "物理瓶颈分",
            "高潜综合分",
            "基金映射",
        ]].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 低热度但近1月已经上涨的候选",
        "",
        (low_heat[[
            "板块",
            "近1月涨跌幅",
            "近3月涨跌幅",
            "热度代理",
            "物理瓶颈分",
            "基金映射",
            "物理逻辑",
        ]].to_markdown(index=False, floatfmt=".4f") if not low_heat.empty else "暂无。"),
        "",
        "## 解释",
        "",
        "- 如果只看热度，容易追到PCB、CPO这种已经充分拥挤的方向。",
        "- 如果只看低热度，容易买到还没被验证的弱板块。",
        "- 本表优先找“近1月涨幅已确认 + 物理瓶颈明确 + 热度未完全打满”的环节。",
    ]
    (OUT / "ai_supply_chain_potential_report_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = []
    for name, meta in SECTOR_BASKETS.items():
        rows.append(basket_metrics(name, meta))
    frame = score(pd.DataFrame(rows))
    frame.to_csv(OUT / "ai_supply_chain_potential_cn.csv", index=False, encoding="utf-8-sig")
    write_report(frame)
    print(frame[["板块", "近1月涨跌幅", "近3月涨跌幅", "近半年涨跌幅", "热度代理", "物理瓶颈分", "高潜综合分", "是否低热度高涨", "基金映射"]].to_string(index=False))


if __name__ == "__main__":
    main()
