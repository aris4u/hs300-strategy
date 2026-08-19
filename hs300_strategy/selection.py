"""Selection proof with T+1 open execution. Survivorship bias is explicit.

MFE/MAE/quality ratings are NOT used here for live filters.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter, ScalarFormatter

from hs300_strategy.data import DATA_DIR
from hs300_strategy.execution import EXECUTION_NOTE
from hs300_strategy.strategy_backtest import portfolio_from_position

OUTPUT_DIR = DATA_DIR.parent / "output"


def evaluate_selection(
    open_px: pd.DataFrame,
    close_px: pd.DataFrame,
    pos: pd.DataFrame,
    idx_open: pd.Series,
    idx_close: pd.Series,
    cal: pd.DatetimeIndex,
    bt_start: str,
    plot_path: Path | None = None,
) -> dict:
    """Holdings from T-close position, filled T+1 open. Three benchmarks."""
    daily = portfolio_from_position(pos, open_px, close_px, idx_open, idx_close)
    bt0 = pd.Timestamp(bt_start)
    work = daily.loc[daily.index >= bt0].copy()
    if work.empty:
        return {"ok": False, "error": "empty window"}

    held = (pos.shift(1).reindex(work.index).fillna(0.0) > 1e-9)
    valid = open_px.reindex(work.index).notna() & close_px.reindex(work.index).notna()
    n_long = held.sum(axis=1)
    n_rest = (valid & ~held).sum(axis=1)
    mask = (n_long > 0) & (n_rest > 0)

    out = {
        "ok": True,
        "start": work.index.min().strftime("%Y-%m-%d"),
        "end": work.index.max().strftime("%Y-%m-%d"),
        "days_with_both_legs": int(mask.sum()),
        "avg_long": float(n_long.loc[mask].mean()) if mask.any() else 0.0,
        "avg_rest": float(n_rest.loc[mask].mean()) if mask.any() else 0.0,
        "long_vs_rest": _series_stats(work.loc[mask, "gross_vs_unselected"], "持仓等权 − 未入选等权"),
        "long_vs_universe": _series_stats(work.loc[mask, "gross_vs_uni"], "持仓等权 − 成分股等权"),
        "long_vs_hs300": _series_stats(work.loc[mask, "gross_vs_hs300"], "持仓等权 − 沪深300"),
        "satellite_vs_hs300": _series_stats(work.loc[mask, "net_vs_hs300"], "仓位加权净超额 − 沪深300"),
        "events": _event_study(bt0),
        "verdict": "",
        "execution": EXECUTION_NOTE,
        "survivorship": (
            "当前沪深300成分名单（非 point-in-time），基准与持仓都有幸存者偏差，"
            "不得称为无偏基准。"
        ),
        "method": (
            "T日收盘信号，T+1开盘成交。主检验是持仓对未入选等权；"
            "相对沪深300是市场基准口径。事后MFE/质量评级不参与本检验仓位。"
            "JCTREND=JC_EVENT且QQS，只报告统计关联，不是趋势确认因果。"
        ),
    }
    t = out["long_vs_rest"]["t_stat"]
    p = out["long_vs_rest"]["p_value"]
    mu = out["long_vs_rest"]["mean_daily"]
    p_hs = out["long_vs_hs300"]["p_value"]
    mu_hs = out["long_vs_hs300"]["mean_daily"]
    if p < 0.05 and mu > 0:
        out["verdict"] = "选股有效：持仓股显著强于未入选股。"
    elif p_hs < 0.05 and mu_hs > 0 and (p >= 0.05 or mu <= 0):
        out["verdict"] = (
            "策略相对市场基准存在超额，但暂不能区分选股能力与风格暴露。"
        )
    elif p < 0.05 and mu < 0:
        out["verdict"] = "选股反向：持仓股显著弱于未入选股。"
    else:
        out["verdict"] = "选股未证明有效：持仓与未入选的日均价差不显著异于 0。"

    _plot_spread(
        work["held_cc_ret"] if "held_cc_ret" in work.columns else work["gross_ret"],
        work["unselected_ew_ret"],
        work["hs300_ret"],
        out,
        plot_path,
    )
    return out


def save_selection(report: dict, path: Path | None = None) -> Path:
    path = path or (OUTPUT_DIR / "selection.json")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def format_selection(report: dict) -> str:
    if not report.get("ok"):
        return "[选股有效性检验] 无结果"
    lines = [
        "[选股有效性检验]",
        report.get("method", ""),
        report.get("execution", ""),
        report.get("survivorship", ""),
        f"区间 {report['start']} ~ {report['end']}  双边都有股票的交易日 {report['days_with_both_legs']}",
        f"平均持仓 {report['avg_long']:.1f} 只  未入选 {report['avg_rest']:.1f} 只",
        "",
        f"结论：{report['verdict']}",
        "",
    ]
    for key in ("long_vs_rest", "long_vs_universe", "long_vs_hs300", "satellite_vs_hs300"):
        s = report.get(key) or {}
        if not s:
            continue
        lines.append(
            f"{s['label']}  日均 {s['mean_daily']:+.3%}  年化 {s['annual']:+.2%}  "
            f"t={s['t_stat']:.2f}  p={s['p_value']:.3f}  胜率 {s['win_rate']:.1%}"
        )
    ev = report.get("events") or {}
    if ev.get("ok"):
        lines += [
            "",
            f"启动事件研究（标准窗口N=20，事后统计，非交易过滤）{ev['n']} 次",
            f"  中位超额收益 {ev['med_excess_ret']:+.1%}  t={ev['t_excess_ret']:.2f}  p={ev['p_excess_ret']:.3f}",
            "  质量评级/MFE 仅作事后描述，不得进入实时筛选。",
        ]
    return "\n".join(lines)


def _series_stats(x: pd.Series, label: str) -> dict:
    v = pd.to_numeric(x, errors="coerce").dropna()
    v = v.iloc[1:] if len(v) else v
    n = int(len(v))
    if n < 5 or float(v.std(ddof=1) or 0) == 0:
        return {
            "label": label,
            "n": n,
            "mean_daily": 0.0,
            "annual": 0.0,
            "t_stat": 0.0,
            "p_value": 1.0,
            "win_rate": 0.0,
        }
    mu = float(v.mean())
    sd = float(v.std(ddof=1))
    t = mu / (sd / math.sqrt(n))
    p = 2.0 * _norm_sf(abs(t))
    return {
        "label": label,
        "n": n,
        "mean_daily": mu,
        "annual": mu * 252,
        "t_stat": t,
        "p_value": p,
        "win_rate": float((v > 0).mean()),
    }


def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _event_study(bt0: pd.Timestamp) -> dict:
    path = OUTPUT_DIR / "research" / "event_blotter_n20.csv"
    if not path.exists():
        path = OUTPUT_DIR / "stock_launches.csv"
    if not path.exists():
        return {"ok": False}
    ev = pd.read_csv(path)
    date_col = "signal_date" if "signal_date" in ev.columns else "date"
    ev[date_col] = pd.to_datetime(ev[date_col])
    ev = ev[ev[date_col] >= bt0].copy()
    ret_col = "excess_ret" if "excess_ret" in ev.columns else "excess_ret_20"
    if ret_col not in ev.columns:
        return {"ok": False}
    if "complete" in ev.columns:
        ev = ev[ev["complete"] == 1]
    x = pd.to_numeric(ev[ret_col], errors="coerce").dropna()
    n = int(len(x))
    if n < 3:
        return {"ok": False}
    t = float(x.mean() / (x.std(ddof=1) / math.sqrt(n))) if x.std(ddof=1) else 0.0
    return {
        "ok": True,
        "n": n,
        "start": ev[date_col].min().strftime("%Y-%m-%d"),
        "med_excess_ret": float(x.median()),
        "t_excess_ret": t,
        "p_excess_ret": 2.0 * _norm_sf(abs(t)),
        "note": "标准事件窗口N=20，事后统计。",
    }


def _plot_spread(
    long_ew: pd.Series,
    rest_ew: pd.Series,
    idx_ret: pd.Series,
    report: dict,
    plot_path: Path | None = None,
) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    if long_ew is None or long_ew.empty:
        return
    long_ew = long_ew.copy().fillna(0.0)
    rest_ew = rest_ew.copy().fillna(0.0)
    idx_ret = idx_ret.copy().fillna(0.0)
    long_ew.iloc[0] = rest_ew.iloc[0] = idx_ret.iloc[0] = 0.0
    nav_l = (1 + long_ew).cumprod()
    nav_r = (1 + rest_ew).cumprod()
    nav_i = (1 + idx_ret).cumprod()
    fig, (ax, ax2) = plt.subplots(
        2,
        1,
        figsize=(13.2, 12.4),
        gridspec_kw={"height_ratios": [1.2, 1.45]},
        sharex=True,
    )
    ax.plot(nav_l.index, nav_l, color="#1f6feb", lw=2.0, label="公式持仓（等权，T+1开盘）")
    ax.plot(nav_r.index, nav_r, color="#e67e22", lw=1.7, label="未入选成分股等权")
    ax.plot(nav_i.index, nav_i, color="#8b949e", lw=1.6, label="沪深300")
    ax.set_yscale("log")
    hi = float(max(nav_l.max(), nav_r.max(), nav_i.max()))
    ax.set_ylim(0.75, hi * 1.18)
    ticks = [1, 2, 4, 8, 16, 32]
    ax.set_yticks([t for t in ticks if t <= hi * 1.2])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title(f"选股检验    {report.get('verdict', '')}", fontsize=13, pad=10)
    ax.set_ylabel("净值（期初=1，对数轴）", fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, which="major", alpha=0.32)
    ax.tick_params(labelsize=10)

    rel_rest = (nav_l / nav_r.replace(0, pd.NA) - 1.0) * 100.0
    rel_idx = (nav_l / nav_i.replace(0, pd.NA) - 1.0) * 100.0
    ax2.plot(rel_rest.index, rel_rest, color="#e67e22", lw=1.8, label="持仓 / 未入选 − 1")
    ax2.plot(rel_idx.index, rel_idx, color="#8b949e", lw=1.8, label="持仓 / 沪深300 − 1")
    ax2.axhline(0.0, color="#5c6573", lw=1.0)
    ax2.fill_between(rel_rest.index, rel_rest, 0.0, color="#e67e22", alpha=0.10)
    ax2.set_ylabel("相对净值差（%）", fontsize=11)
    ax2.set_xlabel("日期", fontsize=11)
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, alpha=0.32)
    ax2.tick_params(labelsize=10)
    lo = float(min(rel_rest.min(), rel_idx.min()))
    hi2 = float(max(rel_rest.max(), rel_idx.max()))
    span = max(hi2 - lo, 8.0)
    ax2.set_ylim(lo - 0.08 * span, hi2 + 0.08 * span)
    fig.tight_layout(h_pad=1.2)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = plot_path or (OUTPUT_DIR / "selection.png")
    fig.savefig(out, dpi=160, facecolor="white")
    eq = pd.DataFrame(
        {
            "nav_long": nav_l,
            "nav_rest": nav_r,
            "nav_hs300": nav_i,
            "rel_vs_rest_pct": rel_rest,
            "rel_vs_hs300_pct": rel_idx,
        }
    )
    eq.to_csv(Path(out).with_name(Path(out).stem + "_equity.csv"), encoding="utf-8-sig")
    plt.close(fig)
