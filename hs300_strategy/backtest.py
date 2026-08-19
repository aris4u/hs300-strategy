"""Full strategy backtest on one OHLC series. T close signal, T+1 open fill.

This is the state-machine path, not an event study.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hs300_strategy.charts import LOOKBACK_BARS, plot_kline_signals
from hs300_strategy.config import WARMUP_BARS
from hs300_strategy.execution import EXECUTION_NOTE, single_blotter, single_t1_open
from hs300_strategy.formula import STATE_LABELS

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
WARMUP = WARMUP_BARS
KEY_EVENTS = (
    ("entry_ok", "entry"),
    ("exit_ok", "exit"),
    ("launch_turn", "launch"),
    ("f_signal", "f_alert"),
    ("reduce_trend", "reduce_trend"),
    ("reduce_band", "reduce_band"),
    ("escape_top", "escape"),
)


@dataclass
class BacktestResult:
    signals: pd.DataFrame
    metrics_state: dict
    metrics_event: dict
    equity_path: Path
    chart_path: Path
    kline_path: Path


def run_backtest(signals: pd.DataFrame) -> BacktestResult:
    df = signals.copy().sort_values("date").reset_index(drop=True)
    if len(df) <= WARMUP + 5:
        raise ValueError("K 线太短，至少需要约 130 个交易日。")
    if "open" not in df.columns:
        raise ValueError("回测需要 open 列：T+1 开盘成交，不能用收盘价代替。")

    pos = df["position"].astype(float) if "position" in df.columns else _event_position(df)
    pnl = single_t1_open(pos, df["open"], df["close"])
    sample = _equity_from_pnl(df, pnl, pos, label_from="state")
    # ui_state is display-only; same T+1 engine, not a second strategy.
    ui_state = df["ui_state"] if "ui_state" in df.columns else df["state"]
    pos_ui = ui_state.map({0: 0.0, 1: 0.0, 2: 1.0, 3: 1.0, 4: 0.5, 5: 0.0}).astype(float)
    pnl_ui = single_t1_open(pos_ui, df["open"], df["close"])
    sample_ui = _equity_from_pnl(df, pnl_ui, pos_ui, label_from="ui_state")

    merged = sample_ui[["date", "close", "label", "position", "equity", "benchmark"]].rename(
        columns={"position": "pos_state", "equity": "eq_state"}
    )
    merged["pos_event"] = sample["position"].to_numpy()
    merged["eq_event"] = sample["equity"].to_numpy()
    merged["gross_ret"] = sample["strategy_ret"].to_numpy()

    metrics_event = _metrics(sample)
    metrics_state = _metrics(sample_ui)
    metrics_event["execution"] = EXECUTION_NOTE
    metrics_state["execution"] = EXECUTION_NOTE
    metrics_state["note"] = "ui_state 覆盖状态对照，不是仓位。"
    n_launch = int((df["launch_turn"] == 1).sum()) if "launch_turn" in df.columns else 0
    metrics_state["trades"] = metrics_event["trades"] = n_launch
    metrics_state["take_profits"] = metrics_event["take_profits"] = (
        int((df["take_profit"] == 1).sum()) if "take_profit" in df.columns else 0
    )
    blotter = single_blotter(df["date"], pos, df["open"], df["close"])
    metrics_state["entries"] = metrics_event["entries"] = int(len(blotter))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "signals.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(OUTPUT_DIR / "equity.csv", index=False, encoding="utf-8-sig")
    if not blotter.empty:
        blotter.to_csv(OUTPUT_DIR / "trades.csv", index=False, encoding="utf-8-sig")
    key = key_signal_frame(df)
    key.to_csv(OUTPUT_DIR / "key_signals.csv", index=False, encoding="utf-8-sig")
    chart_path = OUTPUT_DIR / "equity.png"
    kline_path = OUTPUT_DIR / "charts" / "hs300_kline.png"
    _plot(merged, chart_path, df)
    plot_kline_signals(df, kline_path, title="HS300 K-line signals", bars=LOOKBACK_BARS)
    return BacktestResult(
        signals=df,
        metrics_state=metrics_state,
        metrics_event=metrics_event,
        equity_path=OUTPUT_DIR / "equity.csv",
        chart_path=chart_path,
        kline_path=kline_path,
    )


def _event_position(df: pd.DataFrame) -> pd.Series:
    pos = 0.0
    out = np.zeros(len(df))
    for i, row in enumerate(df.itertuples(index=False)):
        if int(getattr(row, "launch_turn", 0)) == 1:
            pos = 1.0
        if int(getattr(row, "reduce_band", 0)) == 1:
            pos = 0.5 if pos > 0 else pos
        elif int(getattr(row, "reduce_trend", 0)) == 1:
            pos = min(pos, 0.7)
        if int(getattr(row, "take_profit", 0)) == 1 or int(getattr(row, "escape_top", 0)) == 1:
            pos = 0.0
        out[i] = pos
    return pd.Series(out, index=df.index)


def _equity_from_pnl(df: pd.DataFrame, pnl: pd.DataFrame, position: pd.Series, label_from: str) -> pd.DataFrame:
    strat_ret = pnl["net_ret"].astype(float)
    bh = (df["close"].astype(float) / df["open"].astype(float).iloc[WARMUP] - 1)  # unused shape
    bench_ret = df["close"].astype(float).pct_change().fillna(0.0)
    equity = (1 + strat_ret).cumprod()
    bh_equity = (1 + bench_ret).cumprod()
    valid = df.index >= WARMUP
    eq = equity / equity.iloc[WARMUP]
    bh = bh_equity / bh_equity.iloc[WARMUP]
    raw = df[label_from] if label_from in df.columns else df["state"]
    labels = raw.map(STATE_LABELS) if label_from == "ui_state" else df.get("label", raw.map(STATE_LABELS))
    work = pd.DataFrame(
        {
            "date": df["date"],
            "close": df["close"],
            "state": df["state"],
            "label": labels,
            "position": pnl["position_held"].to_numpy(),
            "strategy_ret": strat_ret,
            "gross_ret": pnl["gross_ret"].to_numpy(),
            "equity": eq.where(valid, other=pd.NA),
            "benchmark": bh.where(valid, other=pd.NA),
        }
    )
    return work.loc[valid].copy()


def _metrics(sample: pd.DataFrame) -> dict:
    n = len(sample)
    years = n / 252 if n else 0
    eq = sample["equity"].astype(float)
    bh = sample["benchmark"].astype(float)
    ret = sample["strategy_ret"].astype(float)
    total = float(eq.iloc[-1] / eq.iloc[0] - 1) if n else 0.0
    bh_total = float(bh.iloc[-1] / bh.iloc[0] - 1) if n else 0.0
    ann = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else 0.0
    bh_ann = (1 + bh_total) ** (1 / years) - 1 if years > 0 and bh_total > -1 else 0.0
    vol = float(ret.std() * (252 ** 0.5)) if n > 1 else 0.0
    sharpe = float(ret.mean() / ret.std() * (252 ** 0.5)) if ret.std() else 0.0
    dd = float((eq / eq.cummax() - 1).min()) if n else 0.0
    bh_dd = float((bh / bh.cummax() - 1).min()) if n else 0.0
    days_in = float((sample["position"] > 0).mean())
    return {
        "start": sample["date"].iloc[0].strftime("%Y-%m-%d"),
        "end": sample["date"].iloc[-1].strftime("%Y-%m-%d"),
        "days": n,
        "total_return": total,
        "annual_return": ann,
        "max_drawdown": dd,
        "sharpe": sharpe,
        "volatility": vol,
        "time_in_market": days_in,
        "buyhold_return": bh_total,
        "buyhold_annual": bh_ann,
        "buyhold_drawdown": bh_dd,
        "excess": total - bh_total,
        "state_counts": sample["label"].value_counts().to_dict(),
    }


def key_signal_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.sort_values("date").reset_index(drop=True)
    close = work["close"].astype(float)
    rows: list[dict] = []
    for col, name in KEY_EVENTS:
        if col not in work.columns:
            continue
        hits = work.index[work[col] == 1]
        for i in hits:
            row = work.iloc[int(i)]
            if name in ("reduce_trend", "reduce_band") and "position" in work.columns and float(row["position"]) <= 0:
                continue
            rows.append(
                {
                    "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                    "signal": name,
                    "close": round(float(row["close"]), 2),
                    "open": round(float(row["open"]), 2) if "open" in work.columns else "",
                    "env_level": int(row["env_level"]) if "env_level" in work.columns else "",
                    "l2_flow": round(float(row["l2_flow"]), 4) if pd.notna(row.get("l2_flow")) else "",
                    "note": "信号日收盘价不是成交价；成交在下一交易日开盘。",
                    "post_ret_20_expost": _fwd_pct(close, int(i), 20),
                    "post_ret_60_expost": _fwd_pct(close, int(i), 60),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["date", "signal"]).reset_index(drop=True)


def format_key_signals(df: pd.DataFrame) -> str:
    key = key_signal_frame(df)
    lines = ["Key events (position 0→>0 = entry)"]
    launches = key[key["signal"] == "entry"] if not key.empty else key
    if launches.empty:
        lines.append("（无）")
    else:
        lines.append(launches.to_string(index=False))
        last = launches.iloc[-1]
        lines.append(f"最近一次 entry {last['date']}  收盘 {last['close']}  共 {len(launches)} 次")
    raw_launch = key[key["signal"] == "launch"] if not key.empty else key
    if not raw_launch.empty:
        lines.append(f"launch_turn {len(raw_launch)} 次（含持仓中重复触发）")
        lines.append("")
        lines.append("exit")
        exits = key[key["signal"] == "exit"]
        if exits.empty:
            lines.append("（无）")
        else:
            lines.append(exits.to_string(index=False))
        n_bot = int((key["signal"] == "f_alert").sum())
        n_red = int(key["signal"].isin(["reduce_trend", "reduce_band"]).sum())
        n_esc = int((key["signal"] == "escape").sum())
        lines.append("")
        lines.append(f"f_signal {n_bot}  减仓 {n_red}  CT {n_esc}  （明细 output/key_signals.csv）")
    return "\n".join(lines)


def _fwd_pct(close: pd.Series, i: int, horizon: int) -> str:
    j = i + horizon
    if j >= len(close):
        return ""
    base = float(close.iloc[i])
    if base == 0:
        return ""
    return f"{close.iloc[j] / base - 1:.1%}"


def _plot(sample: pd.DataFrame, path: Path, signals: pd.DataFrame | None = None) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax_px, ax_eq) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [1.15, 1]})

    ax_px.plot(sample["date"], sample["close"], color="#8b949e", linewidth=1.1, label="HS300 close")
    if signals is not None and not signals.empty:
        sig = signals.copy()
        sig["date"] = pd.to_datetime(sig["date"])
        start = pd.Timestamp(sample["date"].min())
        sig = sig[sig["date"] >= start]
        launch = sig[sig["entry_ok"] == 1] if "entry_ok" in sig.columns else sig[sig["launch_turn"] == 1]
        exit_ = sig[sig["exit_ok"] == 1] if "exit_ok" in sig.columns else sig[sig["take_profit"] == 1]
        wash = sig[sig["washout"] == 1] if "washout" in sig.columns else sig.iloc[0:0]
        if not wash.empty:
            ax_px.scatter(wash["date"], wash["close"], marker=".", s=18, c="#8b949e", alpha=0.55, zorder=4, label="washout")
        if not launch.empty:
            ax_px.scatter(launch["date"], launch["close"], marker="^", s=56, c="#f4d03f", zorder=5, label="entry")
        if not exit_.empty:
            ax_px.scatter(exit_["date"], exit_["close"], marker="v", s=48, c="#cf222e", zorder=5, label="exit")
    ax_px.set_title("HS300 · price and key events")
    ax_px.set_ylabel("Close")
    ax_px.legend(loc="upper left")
    ax_px.grid(True, alpha=0.3)

    ax_eq.plot(sample["date"], sample["eq_event"], label="sequential position", color="#1f6feb")
    ax_eq.plot(sample["date"], sample["eq_state"], label="ui_state overlay (not size)", color="#bc8cff")
    ax_eq.plot(sample["date"], sample["benchmark"], label="HS300 buy&hold", color="#8b949e")
    ax_eq.set_title("Equity (post warmup = 1)")
    ax_eq.set_xlabel("Date")
    ax_eq.set_ylabel("NAV")
    ax_eq.legend(loc="upper left")
    ax_eq.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def format_metrics(title: str, m: dict) -> str:
    states = m.get("state_counts", {})
    state_line = "  ".join(f"{k}:{v}" for k, v in states.items())
    extra = ""
    if "trades" in m:
        extra = f"launch_turn {m['trades']}  take_profit {m.get('take_profits', 0)}  entry {m.get('entries', m['trades'])}\n"
    exec_line = m.get("execution", EXECUTION_NOTE)
    return "\n".join(
        [
            f"[{title}]",
            extra.rstrip(),
            exec_line,
            f"区间 {m['start']} ~ {m['end']}  （{m['days']} 个交易日）",
            f"策略收益 {m['total_return']:.2%}  年化 {m['annual_return']:.2%}  最大回撤 {m['max_drawdown']:.2%}  夏普 {m['sharpe']:.2f}",
            f"沪深300  {m['buyhold_return']:.2%}  年化 {m['buyhold_annual']:.2%}  最大回撤 {m['buyhold_drawdown']:.2%}",
            f"超额 {m['excess']:.2%}  持仓时间占比 {m['time_in_market']:.1%}",
            f"状态天数 {state_line}",
        ]
    )
