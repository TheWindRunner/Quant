from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".deps"
if DEPS.exists() and str(DEPS) not in sys.path:
    sys.path.insert(0, str(DEPS))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools.backtest_pcb_purchase_limit import setup_chinese_font
from tools.optimize_rotation_with_pcb_limit import benchmark_cache, evaluate_choice
from tools.optimize_sector_rotation_with_c_fee import ModelSpec, confirm, fee_protect
from tools.optimize_short_core_satellite_60day import evaluate_locked_test
from tools.sector_level_rotation_5themes import load_sector_navs
from tools.short_term_60day_holdout import ONE_MONTH_DAYS, boundaries


OUT = ROOT / "output" / "ml_relative_satellite_60day"
VALIDATION_DAYS = 84
VALIDATION_ENDPOINT_SPACING = 5


def feature_panel(navs: pd.DataFrame) -> pd.DataFrame:
    daily = navs.pct_change(fill_method=None)
    frames = []
    non_pcb = [sector for sector in navs.columns if sector != "PCB"]
    momentum = {
        days: navs.pct_change(days, fill_method=None)
        for days in [1, 2, 5, 10, 20, 40, 60]
    }
    dd20 = navs / navs.rolling(20).max() - 1.0
    vol10 = daily.rolling(10).std() * np.sqrt(252.0)
    vol20 = daily.rolling(20).std() * np.sqrt(252.0)
    for sector in non_pcb:
        frame = pd.DataFrame(index=navs.index)
        for days in [1, 2, 5, 10, 20, 40]:
            frame[f"相对动量{days}"] = momentum[days][sector] - momentum[days]["PCB"]
        frame["板块动量5"] = momentum[5][sector]
        frame["板块动量20"] = momentum[20][sector]
        frame["板块回撤20"] = dd20[sector]
        frame["板块波动10"] = vol10[sector]
        frame["板块波动20"] = vol20[sector]
        frame["PCB动量5"] = momentum[5]["PCB"]
        frame["PCB动量20"] = momentum[20]["PCB"]
        frame["PCB动量60"] = momentum[60]["PCB"]
        frame["PCB回撤20"] = dd20["PCB"]
        frame["板块"] = sector
        frame["日期"] = navs.index
        frames.append(frame.reset_index(drop=True))
    return pd.concat(frames, ignore_index=True)


def add_target(panel: pd.DataFrame, navs: pd.DataFrame, horizon: int) -> pd.DataFrame:
    future = navs.shift(-horizon) / navs - 1.0
    target_map = {}
    for sector in navs.columns:
        if sector == "PCB":
            continue
        target_map[sector] = future[sector] - future["PCB"]
    target_frame = pd.DataFrame(target_map)
    indexed = panel.set_index(["日期", "板块"])
    stacked_target = target_frame.stack().rename("目标相对收益")
    stacked_target.index.names = ["日期", "板块"]
    return indexed.join(stacked_target, how="left").reset_index()


class RidgeModel:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.columns: list[str] = []
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.coef: np.ndarray | None = None
        self.intercept = 0.0

    def fit(self, frame: pd.DataFrame) -> None:
        self.columns = [
            column
            for column in frame.columns
            if column not in {"日期", "板块", "目标相对收益"}
        ]
        clean = frame.dropna(subset=self.columns + ["目标相对收益"])
        x = clean[self.columns].to_numpy(dtype=float)
        y = clean["目标相对收益"].to_numpy(dtype=float)
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.0
        z = (x - self.mean) / self.scale
        y_mean = float(y.mean())
        centered = y - y_mean
        penalty = self.alpha * np.eye(z.shape[1])
        self.coef = np.linalg.solve(z.T @ z + penalty, z.T @ centered)
        self.intercept = y_mean

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        assert self.mean is not None and self.scale is not None and self.coef is not None
        valid = frame[self.columns].notna().all(axis=1)
        result = pd.Series(np.nan, index=frame.index, dtype=float)
        x = frame.loc[valid, self.columns].to_numpy(dtype=float)
        z = (x - self.mean) / self.scale
        result.loc[valid] = self.intercept + z @ self.coef
        return result


def fit_predict(
    panel: pd.DataFrame,
    navs: pd.DataFrame,
    horizon: int,
    alpha: float,
    fit_end: pd.Timestamp,
) -> tuple[pd.DataFrame, RidgeModel]:
    targeted = add_target(panel, navs, horizon)
    fit_end_pos = int(navs.index.get_loc(fit_end))
    matured_target_date = navs.index[max(0, fit_end_pos - horizon)]
    training = targeted.loc[targeted["日期"] <= matured_target_date].copy()
    model = RidgeModel(alpha)
    model.fit(training)
    prediction = panel[["日期", "板块"]].copy()
    prediction["预测相对收益"] = model.predict(panel)
    matrix = prediction.pivot(index="日期", columns="板块", values="预测相对收益")
    return matrix.reindex(navs.index), model


def predicted_choice(
    navs: pd.DataFrame,
    prediction: pd.DataFrame,
    threshold: float,
    confirm_days: int,
    min_hold_days: int,
) -> pd.Series:
    raw = []
    for date, row in prediction.iterrows():
        valid = row.dropna()
        if valid.empty or float(valid.max()) <= threshold:
            raw.append("PCB")
        else:
            raw.append(str(valid.idxmax()))
    confirmed = confirm(pd.Series(raw, index=navs.index), confirm_days)
    return fee_protect(
        navs,
        confirmed,
        min_hold_days=min_hold_days,
        advantage_window=20,
        advantage_threshold=0.20,
        stop_window=5,
        stop_loss=-0.08,
    )


def validation_endpoints(train_end: int) -> tuple[int, list[int]]:
    validation_start = train_end - VALIDATION_DAYS + 1
    fit_end = validation_start - 1
    first_endpoint = validation_start + ONE_MONTH_DAYS - 1
    endpoints = list(range(first_endpoint, train_end + 1, VALIDATION_ENDPOINT_SPACING))
    if endpoints[-1] != train_end:
        endpoints.append(train_end)
    return fit_end, sorted(set(endpoints))


def choose_hyperparameters(navs: pd.DataFrame, panel: pd.DataFrame, train_end: int):
    fit_end_pos, endpoints = validation_endpoints(train_end)
    fit_end_date = navs.index[fit_end_pos]
    benchmark = benchmark_cache(navs, endpoints, ONE_MONTH_DAYS)
    rows = []
    prediction_cache = {}
    choice_cache = {}
    for horizon in [3, 5, 10]:
        for alpha in [0.1, 1.0, 10.0]:
            prediction, _ = fit_predict(panel, navs, horizon, alpha, fit_end_date)
            prediction_cache[(horizon, alpha)] = prediction
            for threshold in [0.0, 0.0025, 0.005, 0.01]:
                for confirm_days in [2, 3, 5]:
                    for min_hold_days in [7, 14, 30]:
                        key = (horizon, alpha, threshold, confirm_days, min_hold_days)
                        choice = predicted_choice(
                            navs, prediction, threshold, confirm_days, min_hold_days
                        )
                        choice_cache[key] = choice
                        for core_weight in [0.80, 0.90, 0.95]:
                            detail, summary = evaluate_choice(
                                navs,
                                choice,
                                endpoints,
                                ONE_MONTH_DAYS,
                                benchmark,
                                core_weight,
                            )
                            excess = detail.iloc[:, 7].astype(float)
                            wins = detail.iloc[:, 9].astype(bool)
                            dd_improvement = detail.iloc[:, 8].astype(float)
                            score = (
                                5.0 * float(wins.mean())
                                + 50.0 * float(excess.mean())
                                + 25.0 * float(excess.median())
                                + 5.0 * float(dd_improvement.mean())
                            )
                            rows.append(
                                {
                                    "预测周期": horizon,
                                    "岭惩罚": alpha,
                                    "预测阈值": threshold,
                                    "确认天数": confirm_days,
                                    "最短持有天数": min_hold_days,
                                    "PCB核心权重": core_weight,
                                    "验证窗口数": len(endpoints),
                                    "验证胜出数": int(wins.sum()),
                                    "验证胜率": float(wins.mean()),
                                    "验证平均收益率": float(list(summary.values())[1]),
                                    "验证平均超额": float(excess.mean()),
                                    "验证超额中位数": float(excess.median()),
                                    "验证回撤改善": float(dd_improvement.mean()),
                                    "验证评分": score,
                                }
                            )
    ranking = pd.DataFrame(rows).sort_values("验证评分", ascending=False).reset_index(drop=True)
    best = ranking.iloc[0]
    return best, ranking, fit_end_date, endpoints


def refit_selected(navs, panel, best, train_end):
    horizon = int(best["预测周期"])
    alpha = float(best["岭惩罚"])
    prediction, model = fit_predict(panel, navs, horizon, alpha, navs.index[train_end])
    choice = predicted_choice(
        navs,
        prediction,
        float(best["预测阈值"]),
        int(best["确认天数"]),
        int(best["最短持有天数"]),
    )
    spec = ModelSpec(
        name="ridge_relative_satellite_locked",
        family="岭回归相对PCB核心卫星",
        params={
            "pcb_core_weight": float(best["PCB核心权重"]),
            "prediction_horizon": horizon,
            "alpha": alpha,
            "threshold": float(best["预测阈值"]),
            "confirm_days": int(best["确认天数"]),
            "min_hold_days": int(best["最短持有天数"]),
        },
    )
    coefficients = pd.DataFrame(
        {"因子": model.columns, "标准化系数": model.coef}
    ).sort_values("标准化系数", key=lambda values: values.abs(), ascending=False)
    return spec, choice, coefficients


def make_chart(detail: pd.DataFrame) -> None:
    setup_chinese_font()
    pivot = detail.pivot(index="窗口", columns="策略", values="收益率")
    excess = detail.loc[detail["策略"] == "训练锁定核心卫星"].set_index("窗口")[
        "核心卫星相对全仓超额"
    ]
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), dpi=160, sharex=True)
    pivot.plot(ax=axes[0], linewidth=2.0, marker="o", markersize=3)
    axes[0].set_title("多因子岭回归核心卫星：30个一月封存窗口")
    axes[0].set_ylabel("收益率")
    axes[0].grid(alpha=0.25)
    axes[1].bar(
        excess.index,
        excess * 100,
        color=np.where(excess >= 0, "#d62728", "#2ca02c"),
    )
    axes[1].axhline(0.0, color="gray", linewidth=1.0)
    axes[1].set_xlabel("连续入场窗口编号")
    axes[1].set_ylabel("相对全仓PCB超额（百分点）")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "多因子岭回归30窗口测试_cn.png")
    plt.close(fig)


def make_report(
    bounds,
    fit_end_date,
    fit_label_end_date,
    refit_label_end_date,
    val_endpoints,
    spec,
    ranking,
    summary,
    coefficients,
):
    strategy = summary.loc[summary["策略"] == "训练锁定核心卫星"].iloc[0]
    passed = int(strategy["核心卫星跑赢全仓次数"]) >= 16
    lines = [
        "# 多因子岭回归核心卫星60日封存验证",
        "",
        "## 严格时间顺序",
        "",
        f"- 回归拟合截止：{fit_end_date.date().isoformat()}",
        f"- 内部验证模型最后成熟标签日：{fit_label_end_date.date().isoformat()}",
        f"- 训练区内部验证窗口数：{len(val_endpoints)}",
        f"- 最终重拟合截止：{bounds['训练结束日期']}",
        f"- 最终重拟合最后成熟标签日：{refit_label_end_date.date().isoformat()}",
        f"- 最近60日封存：{bounds['封存区开始日期']} 至 {bounds['封存区结束日期']}",
        "- 封存测试结果未进入回归、阈值或仓位参数选择。",
        "",
        "## 锁定模型",
        "",
        f"- 参数：`{spec.params}`",
        f"- 是否达到30窗口过半目标：{'是' if passed else '否'}",
        "",
        "## 测试结果",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 因子系数",
        "",
        coefficients.to_markdown(index=False, floatfmt=".6f"),
        "",
        "执行口径继续使用FIFO C类赎回费、基金当日转换、PCB四通道合计每日4000元限购、现金赎回T+2可用，以及2个净值日信号延迟。",
    ]
    (OUT / "多因子岭回归60日封存报告_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    navs = load_sector_navs()
    bounds = boundaries(navs)
    train_end = int(bounds["训练结束位置"])
    panel = feature_panel(navs)
    best, ranking, fit_end_date, val_endpoints = choose_hyperparameters(navs, panel, train_end)
    spec, choice, coefficients = refit_selected(navs, panel, best, train_end)
    horizon = int(best["预测周期"])
    fit_end_pos = int(navs.index.get_loc(fit_end_date))
    fit_label_end_date = navs.index[fit_end_pos - horizon]
    refit_label_end_date = navs.index[train_end - horizon]
    detail, summary = evaluate_locked_test(navs, spec, choice)
    pd.DataFrame([bounds]).to_csv(OUT / "训练测试边界_cn.csv", index=False, encoding="utf-8-sig")
    ranking.to_csv(OUT / "训练区内部验证排名_cn.csv", index=False, encoding="utf-8-sig")
    coefficients.to_csv(OUT / "岭回归因子系数_cn.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(OUT / "封存测试30窗口明细_cn.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "封存测试汇总_cn.csv", index=False, encoding="utf-8-sig")
    make_chart(detail)
    make_report(
        bounds,
        fit_end_date,
        fit_label_end_date,
        refit_label_end_date,
        val_endpoints,
        spec,
        ranking,
        summary,
        coefficients,
    )
    print("内部验证第一名:")
    print(best.to_string())
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
