"""Candlestick overlay: launch / reduce / escape on the price chart."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from hs300_strategy.advise import make_advice
from hs300_strategy.data import DATA_DIR, fetch_hs300
from hs300_strategy.formula import compute_signals
from hs300_strategy.moneyflow import read_cached_l2
from hs300_strategy.stock_data import fetch_constituents, fetch_stock_kline

OUTPUT_DIR = DATA_DIR.parent / "output"
CHART_DIR = OUTPUT_DIR / "charts"
LOOKBACK_BARS = 252 * 3  # 3 trading years

LABEL_CN = {
    "watch": "观察",
    "bottom_watch": "底部观察",
    "entry": "开始建仓",
    "hold": "趋势持有",
    "reduce": "风险控制",
    "exit": "离场",
}

# Only action points on the 3-year chart. Wash/other internals still compute, not drawn.
MARKERS = (
    ("mark_opp", "机会点", "#9b59b6", "o", 55, "low", 0.014),
    ("mark_entry", "建仓点", "#1abc9c", "^", 95, "low", 0.008),
    ("mark_reduce", "减仓点", "#e74c3c", "v", 85, "high", 0.008),
    ("mark_risk", "风险点", "#3498db", "d", 60, "high", 0.028),
    ("mark_exit", "清仓点", "#c0392b", "x", 70, "high", 0.020),
)

_ENV_CACHE: dict[tuple[str, str], pd.Series] = {}
_RANK_MAP: dict[str, dict] | None = None


def invalidate_caches() -> None:
    global _RANK_MAP
    _ENV_CACHE.clear()
    _RANK_MAP = None


def load_stock_signals(
    ts_code: str,
    start: str = "20100101",
    end: str | None = None,
    use_cache: bool = True,
    with_flow: bool = True,
) -> pd.DataFrame:
    end = end or date.today().strftime("%Y%m%d")
    k = fetch_stock_kline(ts_code, start=start, end=end, use_cache=use_cache)
    work = k.copy()
    work["date"] = pd.to_datetime(work["date"])
    work["market_env"] = work["date"].map(_market_env(start, end))
    if with_flow:
        flow = read_cached_l2(ts_code, start, end)
        if flow is not None and not flow.empty:
            work = work.merge(flow[["date", "l2jbl"]], on="date", how="left")
    return compute_signals(work, asset="stock")


def _chart_marks(df: pd.DataFrame) -> pd.DataFrame:
    """Map formula flags to the five trade points shown on the chart."""
    out = df.copy()

    def flag(name: str) -> pd.Series:
        if name not in out.columns:
            return pd.Series(False, index=out.index)
        return out[name].fillna(0).astype(int).eq(1)

    out["mark_opp"] = flag("f_signal").astype(int)
    out["mark_entry"] = flag("launch_turn").astype(int)
    out["mark_reduce"] = (flag("reduce_band") | flag("reduce_trend")).astype(int)
    out["mark_risk"] = flag("caution").astype(int)
    out["mark_exit"] = (flag("escape_top") | flag("take_profit")).astype(int)
    return out


def plot_kline_signals(
    df: pd.DataFrame,
    path: Path | str,
    title: str = "",
    bars: int = LOOKBACK_BARS,
    rank_row: dict | None = None,
    ts_code: str | None = None,
    dpi: int = 160,
) -> Path:
    full = df.copy().sort_values("date").reset_index(drop=True)
    full["date"] = pd.to_datetime(full["date"])
    work = _chart_marks(_select_window(full, bars))
    n = len(work)
    width = 0.82 if n < 250 else 0.72 if n < 500 else 0.62
    fig_w = 16 if n < 250 else 18 if n < 500 else 20

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax, ax_vol) = plt.subplots(
        2,
        1,
        figsize=(fig_w, 8.2),
        gridspec_kw={"height_ratios": [3.2, 0.9]},
        facecolor="white",
    )
    ax_vol.sharex(ax)
    x = np.arange(n, dtype=float)
    _shade_position(ax, work, x)
    _draw_candles(ax, work, x, width=width)
    _draw_markers(ax, work, x)
    _set_price_ylim(ax, work)
    _legend_all(ax, work)

    ax.set_ylabel("价格", fontsize=11)
    first = work["date"].iloc[0].strftime("%Y-%m-%d")
    last_d = work["date"].iloc[-1].strftime("%Y-%m-%d")
    ax.set_title(f"{title or 'K线信号'}    {first} ~ {last_d}", fontsize=13, pad=8, loc="left")
    ax.grid(True, axis="y", alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)

    vol = work["volume"].astype(float).to_numpy()
    colors = np.where(work["close"] >= work["open"], "#e74c3c", "#1abc9c")
    ax_vol.bar(x, vol, color=colors, width=width, alpha=0.9, linewidth=0)
    ax_vol.set_ylabel("成交量", fontsize=11)
    ax_vol.grid(True, axis="y", alpha=0.18, linewidth=0.6)
    ax_vol.set_xlabel("")

    ticks = _date_ticks(n)
    ax_vol.set_xticks(ticks)
    fmt = "%Y-%m" if n >= 400 else "%Y-%m-%d"
    ax_vol.set_xticklabels([work["date"].iloc[i].strftime(fmt) for i in ticks], fontsize=10)
    ax.set_xlim(-1, n)
    ax_vol.set_xlim(-1, n)

    fig.subplots_adjust(hspace=0.08, left=0.045, right=0.99, top=0.93, bottom=0.08)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return path


def plot_stock(
    ts_code: str,
    bars: int = LOOKBACK_BARS,
    start: str = "20100101",
    end: str | None = None,
    use_cache: bool = True,
    with_flow: bool = True,
) -> tuple[pd.DataFrame, Path]:
    sig = load_stock_signals(ts_code, start=start, end=end, use_cache=use_cache, with_flow=with_flow)
    members = fetch_constituents(use_cache=True)
    name = ""
    hit = members[members["ts_code"] == ts_code]
    if not hit.empty:
        name = str(hit.iloc[0]["name"])
    last = sig.iloc[-1]
    label = LABEL_CN.get(str(last["label"]), last["label"])
    title = (
        f"{name}  {ts_code}    {pd.Timestamp(last['date']).strftime('%Y-%m-%d')}  "
        f"收盘 {float(last['close']):.2f}    仓位 {float(last['position']):.0%}    {label}"
    )
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    path = CHART_DIR / f"{ts_code.replace('.', '_')}.png"
    plot_kline_signals(sig, path, title=title, bars=bars, ts_code=ts_code)
    return sig, path


def _rank_map() -> dict[str, dict]:
    global _RANK_MAP
    if _RANK_MAP is None:
        path = OUTPUT_DIR / "stock_rank.csv"
        if path.exists():
            df = pd.read_csv(path)
            _RANK_MAP = {str(r["ts_code"]): r.to_dict() for _, r in df.iterrows()}
        else:
            _RANK_MAP = {}
    return _RANK_MAP


def _market_env(start: str, end: str) -> pd.Series:
    key = (start, end)
    cached = _ENV_CACHE.get(key)
    if cached is not None:
        return cached
    hs300 = fetch_hs300(start=start, end=end, use_cache=True)
    sig = compute_signals(hs300.copy(), asset="index")
    env = sig.set_index("date")["env_level"]
    _ENV_CACHE[key] = env
    return env


def _select_window(df: pd.DataFrame, bars: int) -> pd.DataFrame:
    """Always the most recent `bars` sessions (default 3 years)."""
    if not bars or len(df) <= bars:
        return df.reset_index(drop=True)
    return df.tail(bars).reset_index(drop=True)


def _date_ticks(n: int) -> np.ndarray:
    if n <= 1:
        return np.array([0])
    if n >= 500:
        count = 12
    elif n >= 200:
        count = 10
    else:
        count = 8
    return np.unique(np.linspace(0, n - 1, count).astype(int))


def _draw_candles(ax, df: pd.DataFrame, x: np.ndarray, width: float = 0.7) -> None:
    o = df["open"].astype(float).to_numpy()
    h = df["high"].astype(float).to_numpy()
    l = df["low"].astype(float).to_numpy()
    c = df["close"].astype(float).to_numpy()
    up = c >= o
    down = ~up
    ax.vlines(x[up], l[up], h[up], color="#c0392b", linewidth=1.05, zorder=2)
    ax.vlines(x[down], l[down], h[down], color="#1e8449", linewidth=1.05, zorder=2)
    body = np.maximum(np.abs(c - o), (h - l) * 0.018 + 1e-6)
    low_body = np.minimum(o, c)
    ax.bar(x[up], body[up], bottom=low_body[up], width=width, color="#e74c3c", linewidth=0, zorder=3)
    ax.bar(x[down], body[down], bottom=low_body[down], width=width, color="#1abc9c", linewidth=0, zorder=3)


def _set_price_ylim(ax, df: pd.DataFrame) -> None:
    lo = float(np.nanmin(df["low"]))
    hi = float(np.nanmax(df["high"]))
    span = (hi - lo) or 1.0
    ax.set_ylim(lo - span * 0.10, hi + span * 0.12)


def _shade_position(ax, df: pd.DataFrame, x: np.ndarray) -> None:
    if "position" not in df.columns:
        return
    pos = df["position"].astype(float).to_numpy()
    lo = float(np.nanmin(df["low"])) * 0.985
    hi = float(np.nanmax(df["high"])) * 1.015
    ax.fill_between(x, lo, hi, where=pos >= 0.99, color="#1abc9c", alpha=0.14, zorder=0, linewidth=0)
    ax.fill_between(
        x,
        lo,
        hi,
        where=(pos >= 0.45) & (pos < 0.99),
        color="#e67e22",
        alpha=0.14,
        zorder=0,
        linewidth=0,
    )


def _legend_all(ax, df: pd.DataFrame) -> None:
    """Always list every drawable signal, including those with count 0."""
    handles = []
    for col, name, color, marker, *_rest in MARKERS:
        n = int((df[col].astype(int) == 1).sum()) if col in df.columns else 0
        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="none",
                markerfacecolor=color,
                markeredgecolor="#1c2833",
                markeredgewidth=0.8,
                markersize=7,
                linestyle="None",
                label=f"{name}  {n}",
            )
        )
    handles.append(Patch(facecolor="#1abc9c", alpha=0.35, edgecolor="none", label="满仓区间"))
    handles.append(Patch(facecolor="#e67e22", alpha=0.35, edgecolor="none", label="减仓区间"))
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.96, ncol=4, borderpad=0.4)


def _draw_markers(ax, df: pd.DataFrame, x: np.ndarray) -> None:
    low = df["low"].astype(float).to_numpy()
    high = df["high"].astype(float).to_numpy()
    span = np.nanmax(high) - np.nanmin(low)
    for col, name, color, marker, size, loc, pad_frac in MARKERS:
        if col not in df.columns:
            continue
        mask = df[col].astype(int) == 1
        if not mask.any():
            continue
        pad = span * pad_frac if span else 0.01
        xi = x[mask.to_numpy()]
        yi = (low if loc == "low" else high)[mask.to_numpy()]
        yi = yi - pad if loc == "low" else yi + pad
        ax.scatter(
            xi,
            yi,
            marker=marker,
            s=size,
            c=color,
            zorder=7,
            linewidths=0.6,
            edgecolors="#1c2833",
        )


def _plot_one_code(ts_code: str) -> tuple:
    try:
        sig, path = plot_stock(ts_code)
        adv = make_advice(sig, _rank_map().get(ts_code))
        return ts_code, str(path), adv
    except Exception as exc:
        return ts_code, f"ERR {exc}", None


def _init_plot_worker() -> None:
    _market_env("20100101", date.today().strftime("%Y%m%d"))
    _rank_map()


def plot_universe(codes: list[str] | None = None, workers: int | None = None) -> tuple[int, int]:
    """Draw a 3-year chart for every HS300 constituent (or the given codes)."""
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from time import perf_counter

    if codes is None:
        members = fetch_constituents(use_cache=True)
        codes = members["ts_code"].tolist()
        names = dict(zip(members["ts_code"], members["name"]))
    else:
        members = fetch_constituents(use_cache=True)
        names = dict(zip(members["ts_code"], members["name"]))
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    n_workers = workers or max(1, min(8, (os.cpu_count() or 4) - 1))
    t0 = perf_counter()
    ok = fail = 0
    advice_rows = []
    print(f"画K线 {len(codes)} 只  {n_workers} 进程 …")
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_plot_worker) as pool:
        futs = [pool.submit(_plot_one_code, code) for code in codes]
        done = 0
        for fut in as_completed(futs):
            code, msg, adv = fut.result()
            done += 1
            if adv is None or str(msg).startswith("ERR"):
                fail += 1
                print(f"  {code} {msg}")
            else:
                ok += 1
                advice_rows.append(
                    {
                        "ts_code": code,
                        "name": names.get(code, code),
                        "action": adv["action"],
                        "position_hint": adv["position_hint"],
                        "confidence": adv["confidence"],
                        "position": adv.get("position", 0),
                        "headline": adv["headline"],
                        "signal_date": adv.get("signal_date"),
                        "entry_date": adv.get("entry_date"),
                        "entry_price": adv.get("entry_price"),
                        "exit_date": adv.get("exit_date"),
                        "exit_price": adv.get("exit_price"),
                        "execution": adv.get("execution"),
                    }
                )
            if done % 30 == 0 or done == len(codes):
                print(f"  {done}/{len(codes)}  成功 {ok}  失败 {fail}  {perf_counter() - t0:.1f}s", flush=True)
    if advice_rows:
        out = pd.DataFrame(advice_rows)
        order = {"试多": 0, "持有": 1, "减仓": 2, "清仓": 3, "观望": 4}
        out["ord"] = out["action"].map(order).fillna(9)
        out = out.sort_values(["ord", "position"], ascending=[True, False]).drop(columns=["ord"])
        path = OUTPUT_DIR / "today_advice.csv"
        out.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"当日建议表 {path}")
        print("分布：", out["action"].value_counts().to_dict())
        live = out[out["action"].isin(["试多", "持有", "减仓", "清仓"])]
        if live.empty:
            print("今天没有处于持仓/开平仓状态的股票")
        else:
            print("今天有动作或仍持仓的股票：")
            print(live[["ts_code", "name", "action", "position_hint", "confidence"]].to_string(index=False))
    print(f"K线图目录 {CHART_DIR}  共 {ok} 张")
    return ok, fail

