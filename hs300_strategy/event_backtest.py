"""Event study: launch at T close, buy T+1 open, hold N days, sell exit close.

This is a standardized validity test. It is not the live trading strategy.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from hs300_strategy.config import (
    EVENT_PRIMARY_N,
    FULL_START,
    HOLD_PERIODS,
    LAUNCH_DEBOUNCE,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    WARMUP_BARS,
)
from hs300_strategy.events import debounce_launches


def build_launch_events(
    launch: pd.DataFrame,
    warmup: int = WARMUP_BARS,
    gap: int = LAUNCH_DEBOUNCE,
) -> pd.DataFrame:
    """One row per debounced launch: ts_code, signal_date."""
    rows = []
    cal = launch.index
    for code in launch.columns:
        hits = launch.index[launch[code].fillna(0) > 0]
        if len(hits) == 0:
            continue
        kept = debounce_launches(pd.Series(hits), cal, gap=gap)
        loc = {d: i for i, d in enumerate(cal)}
        for d in pd.to_datetime(kept):
            i = loc.get(pd.Timestamp(d))
            if i is None or i < warmup:
                continue
            rows.append({"ts_code": code, "signal_date": pd.Timestamp(d)})
    return pd.DataFrame(rows)


def event_trades(
    events: pd.DataFrame,
    open_px: pd.DataFrame,
    high_px: pd.DataFrame,
    low_px: pd.DataFrame,
    close_px: pd.DataFrame,
    idx_open: pd.Series,
    idx_high: pd.Series,
    idx_low: pd.Series,
    idx_close: pd.Series,
    launch: pd.DataFrame,
    horizons: tuple[int, ...] = HOLD_PERIODS,
) -> pd.DataFrame:
    """T+1 open entry, N-day close exit. Same calendar for stock and HS300."""
    cal = close_px.index
    loc = {pd.Timestamp(d): i for i, d in enumerate(cal)}
    idx_open = idx_open.reindex(cal)
    idx_high = idx_high.reindex(cal)
    idx_low = idx_low.reindex(cal)
    idx_close = idx_close.reindex(cal)
    out_rows: list[dict] = []
    codes = list(close_px.columns)
    for ev in events.itertuples(index=False):
        code = ev.ts_code
        sd = pd.Timestamp(ev.signal_date)
        si = loc.get(sd)
        if si is None or code not in close_px.columns:
            continue
        entry_i = si + 1
        if entry_i >= len(cal):
            continue
        if code not in open_px.columns:
            continue
        entry_px = open_px.iloc[entry_i][code]
        b_entry = idx_open.iloc[entry_i]
        if not _pos(entry_px) or not _pos(b_entry):
            continue
        entry_date = cal[entry_i]
        base = {
            "ts_code": code,
            "signal_date": sd,
            "entry_date": pd.Timestamp(entry_date),
            "entry_price": float(entry_px),
            "hs300_entry_price": float(b_entry),
        }
        # universe / unselected on the entry morning
        uni_open = open_px.iloc[entry_i]
        launched_today = set(launch.columns[launch.iloc[si].fillna(0) > 0]) if si < len(launch) else set()
        for n in horizons:
            exit_i = entry_i + n - 1
            row = dict(base)
            row["hold_n"] = n
            if exit_i >= len(cal):
                row["complete"] = 0
                out_rows.append(row)
                continue
            s_exit = close_px.iloc[exit_i][code]
            b_exit = idx_close.iloc[exit_i]
            if not _pos(s_exit) or not _pos(b_exit):
                row["complete"] = 0
                out_rows.append(row)
                continue
            hi = high_px.iloc[entry_i : exit_i + 1][code]
            lo = low_px.iloc[entry_i : exit_i + 1][code]
            b_hi = idx_high.iloc[entry_i : exit_i + 1]
            b_lo = idx_low.iloc[entry_i : exit_i + 1]
            stock_ret = float(s_exit / entry_px - 1)
            hs_ret = float(b_exit / b_entry - 1)
            mfe = float(hi.max() / entry_px - 1) if hi.notna().any() else np.nan
            mae = float(lo.min() / entry_px - 1) if lo.notna().any() else np.nan
            hs_mfe = float(b_hi.max() / b_entry - 1) if b_hi.notna().any() else np.nan
            hs_mae = float(b_lo.min() / b_entry - 1) if b_lo.notna().any() else np.nan
            uni_close = close_px.iloc[exit_i]
            both = uni_open.notna() & uni_close.notna() & (uni_open > 0)
            uni_rets = (uni_close[both] / uni_open[both] - 1).astype(float)
            unsel_mask = both.copy()
            for c in launched_today:
                if c in unsel_mask.index:
                    unsel_mask[c] = False
            unsel_rets = (uni_close[unsel_mask] / uni_open[unsel_mask] - 1).astype(float)
            row.update(
                {
                    "complete": 1,
                    "exit_date": pd.Timestamp(cal[exit_i]),
                    "exit_price": float(s_exit),
                    "hs300_exit_price": float(b_exit),
                    "stock_ret": stock_ret,
                    "hs300_ret": hs_ret,
                    "excess_ret": stock_ret - hs_ret,
                    "mfe": mfe,
                    "mae": mae,
                    "hs300_mfe": hs_mfe,
                    "hs300_mae": hs_mae,
                    "excess_mfe": mfe - hs_mfe if pd.notna(mfe) and pd.notna(hs_mfe) else np.nan,
                    "excess_mae": mae - hs_mae if pd.notna(mae) and pd.notna(hs_mae) else np.nan,
                    "uni_ew_ret": float(uni_rets.mean()) if len(uni_rets) else np.nan,
                    "unselected_ew_ret": float(unsel_rets.mean()) if len(unsel_rets) else np.nan,
                    "n_universe": int(both.sum()),
                    "n_unselected": int(unsel_mask.sum()),
                    "bars": n,
                }
            )
            row["excess_vs_uni"] = (
                stock_ret - row["uni_ew_ret"] if pd.notna(row["uni_ew_ret"]) else np.nan
            )
            row["excess_vs_unselected"] = (
                stock_ret - row["unselected_ew_ret"] if pd.notna(row["unselected_ew_ret"]) else np.nan
            )
            out_rows.append(row)
    return pd.DataFrame(out_rows)


def summarize_events(trades: pd.DataFrame, sample: str = "full") -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    work = trades[trades.get("complete", 1) == 1].copy() if "complete" in trades.columns else trades.copy()
    work = _filter_sample(work, sample)
    rows = []
    horizons = sorted(work["hold_n"].dropna().unique().astype(int)) if "hold_n" in work.columns else [EVENT_PRIMARY_N]
    for n in horizons:
        g = work[work["hold_n"] == n] if "hold_n" in work.columns else work
        rows.append(_horizon_stats(g, n, sample))
    return pd.DataFrame(rows)


def _horizon_stats(g: pd.DataFrame, n: int, sample: str) -> dict:
    r = pd.to_numeric(g["stock_ret"], errors="coerce").dropna()
    x = pd.to_numeric(g["excess_ret"], errors="coerce").dropna()
    xu = pd.to_numeric(g.get("excess_vs_unselected"), errors="coerce").dropna()
    mfe = pd.to_numeric(g["mfe"], errors="coerce")
    mae = pd.to_numeric(g["mae"], errors="coerce")
    t_x, p_x = _ttest_mean(x)
    t_u, p_u = _ttest_mean(xu)
    return {
        "sample": sample,
        "hold_n": int(n),
        "n_events": int(len(g)),
        "n_complete": int(len(r)),
        "mean_ret": _mean(r),
        "median_ret": _median(r),
        "mean_mfe": _mean(mfe),
        "median_mfe": _median(mfe),
        "mean_mae": _mean(mae),
        "median_mae": _median(mae),
        "win_rate": float((r > 0).mean()) if len(r) else np.nan,
        "beat_hs300": float((x > 0).mean()) if len(x) else np.nan,
        "mean_excess_hs300": _mean(x),
        "median_excess_hs300": _median(x),
        "t_excess_hs300": t_x,
        "p_excess_hs300": p_x,
        "mean_excess_unselected": _mean(xu),
        "t_excess_unselected": t_u,
        "p_excess_unselected": p_u,
        "mean_hs300_ret": _mean(g.get("hs300_ret")),
        "mean_uni_ew_ret": _mean(g.get("uni_ew_ret")),
        "mean_unselected_ew_ret": _mean(g.get("unselected_ew_ret")),
        "mean_excess_mfe": _mean(g.get("excess_mfe")),
        "mean_hs300_mfe": _mean(g.get("hs300_mfe")),
        "mean_hs300_mae": _mean(g.get("hs300_mae")),
        "mean_excess_mae": _mean(g.get("excess_mae")),
        "start": _min_date(g, "signal_date"),
        "end": _max_date(g, "signal_date"),
        "note": (
            "持有期超额 = 个股(exit_close/entry_open-1) − 沪深300(exit_close/entry_open-1)。"
            "MFE/MAE 是路径极值，不与持有期超额混用。"
            "事件窗口重叠，t 统计量为描述性，不是独立样本。"
        ),
    }


def walk_forward_events(trades: pd.DataFrame, hold_n: int = EVENT_PRIMARY_N) -> pd.DataFrame:
    """Expanding train through year-end Y-1, test calendar year Y. Params frozen."""
    g = trades[(trades.get("complete", 1) == 1) & (trades["hold_n"] == hold_n)].copy()
    if g.empty:
        return pd.DataFrame()
    g["signal_date"] = pd.to_datetime(g["signal_date"])
    years = sorted(g["signal_date"].dt.year.unique())
    rows = []
    for y in years:
        if y < 2015:
            continue
        train_end = pd.Timestamp(f"{y - 1}-12-31")
        test = g[(g["signal_date"] >= pd.Timestamp(f"{y}-01-01")) & (g["signal_date"] <= pd.Timestamp(f"{y}-12-31"))]
        train = g[(g["signal_date"] >= pd.Timestamp(TRAIN_START)) & (g["signal_date"] <= train_end)]
        rows.append(_horizon_stats(train, hold_n, f"wf_train_to_{y - 1}"))
        rows.append(_horizon_stats(test, hold_n, f"wf_test_{y}"))
    return pd.DataFrame(rows)


def _filter_sample(df: pd.DataFrame, sample: str) -> pd.DataFrame:
    if df.empty or "signal_date" not in df.columns:
        return df
    d = pd.to_datetime(df["signal_date"])
    if sample == "train":
        return df[(d >= pd.Timestamp(TRAIN_START)) & (d <= pd.Timestamp(TRAIN_END))]
    if sample == "test":
        return df[(d >= pd.Timestamp(TEST_START)) & (d <= pd.Timestamp(TEST_END))]
    if sample == "full":
        return df[(d >= pd.Timestamp(FULL_START)) & (d <= pd.Timestamp(TEST_END))]
    return df


def _ttest_mean(x: pd.Series) -> tuple[float, float]:
    v = pd.to_numeric(x, errors="coerce").dropna()
    n = int(len(v))
    if n < 5 or float(v.std(ddof=1) or 0) == 0:
        return float("nan"), float("nan")
    t = float(v.mean() / (v.std(ddof=1) / math.sqrt(n)))
    p = 2.0 * (0.5 * math.erfc(abs(t) / math.sqrt(2.0)))
    return t, p


def _mean(x) -> float:
    v = pd.to_numeric(x, errors="coerce")
    return float(v.mean()) if v.notna().any() else float("nan")


def _median(x) -> float:
    v = pd.to_numeric(x, errors="coerce")
    return float(v.median()) if v.notna().any() else float("nan")


def _min_date(g: pd.DataFrame, col: str) -> str:
    if col not in g.columns or g.empty:
        return ""
    return pd.to_datetime(g[col]).min().strftime("%Y-%m-%d")


def _max_date(g: pd.DataFrame, col: str) -> str:
    if col not in g.columns or g.empty:
        return ""
    return pd.to_datetime(g[col]).max().strftime("%Y-%m-%d")


def _pos(x) -> bool:
    try:
        return pd.notna(x) and float(x) > 0
    except (TypeError, ValueError):
        return False
