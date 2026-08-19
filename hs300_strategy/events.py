"""Launch vs HS300 MFE / MAE / excess return."""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (10, 20, 60)
DEBOUNCE_BARS = 10

QUALITY_CN = {
    "watching": "观察中",
    "high_value": "高价值",
    "low_value": "低价值",
    "neutral": "中性",
}


def debounce_launches(dates: pd.Series, index: pd.DatetimeIndex, gap: int = DEBOUNCE_BARS) -> pd.Series:
    """Keep the first launch when two hits are closer than `gap` bars."""
    loc = {d: i for i, d in enumerate(index)}
    keep = []
    last_i = -10**9
    for d in pd.to_datetime(dates):
        i = loc.get(pd.Timestamp(d), None)
        if i is None:
            continue
        if i - last_i >= gap:
            keep.append(pd.Timestamp(d))
            last_i = i
    return pd.Series(keep, dtype="datetime64[ns]")


def excursion_row(stock: pd.DataFrame, bench: pd.DataFrame, signal_date, horizons: tuple[int, ...] = HORIZONS) -> dict:
    """Legacy close-to-close path. Do not use for research conclusions.

    Research event study is hs300_strategy.event_backtest (T+1 open, N-day close).
    """
    s = stock.sort_values("date").reset_index(drop=True)
    b = bench.sort_values("date").reset_index(drop=True)
    s["date"] = pd.to_datetime(s["date"])
    b["date"] = pd.to_datetime(b["date"])
    sd = pd.Timestamp(signal_date)
    si = s.index[s["date"] == sd]
    bi = b.index[b["date"] == sd]
    if len(si) == 0 or len(bi) == 0:
        return {}
    i = int(si[0])
    j = int(bi[0])
    entry = float(s.loc[i, "close"])
    b_entry = float(b.loc[j, "close"])
    if entry <= 0 or b_entry <= 0:
        return {}
    row = {"date": sd, "entry": entry, "hs300_entry": b_entry}
    for h in horizons:
        s_win = s.iloc[i + 1 : i + 1 + h]
        b_win = b.iloc[j + 1 : j + 1 + h]
        n = min(len(s_win), len(b_win), h)
        row[f"bars_{h}"] = n
        if n == 0:
            continue
        s_win = s_win.iloc[:n]
        b_win = b_win.iloc[:n]
        mfe = float(s_win["high"].max() / entry - 1)
        mae = float(s_win["low"].min() / entry - 1)
        ret = float(s_win["close"].iloc[-1] / entry - 1)
        bm_mfe = float(b_win["high"].max() / b_entry - 1)
        bm_mae = float(b_win["low"].min() / b_entry - 1)
        bm_ret = float(b_win["close"].iloc[-1] / b_entry - 1)
        s_cum = s_win["close"].to_numpy() / entry
        b_cum = b_win["close"].to_numpy() / b_entry
        rel = s_cum / np.where(b_cum == 0, np.nan, b_cum)
        row[f"mfe_{h}"] = mfe
        row[f"mae_{h}"] = mae
        row[f"ret_{h}"] = ret
        row[f"hs300_mfe_{h}"] = bm_mfe
        row[f"hs300_mae_{h}"] = bm_mae
        row[f"hs300_ret_{h}"] = bm_ret
        row[f"excess_mfe_{h}"] = mfe - bm_mfe
        row[f"excess_mae_{h}"] = mae - bm_mae
        row[f"excess_ret_{h}"] = ret - bm_ret
        row[f"rel_mfe_{h}"] = float(np.nanmax(rel) - 1)
        row[f"rel_mae_{h}"] = float(np.nanmin(rel) - 1)
        row[f"efficiency_{h}"] = mfe / abs(mae) if mae < 0 else np.nan
    return row


def label_quality(events: pd.DataFrame) -> pd.DataFrame:
    """Subjective/semi-quantitative rating from ex-post 20-day MFE/MAE.

    Not an objective label. Must not be used as a live buy filter or as
    supervised training targets. Uses future path after the signal.
    """
    from hs300_strategy.config import RATING_KIND

    out = events.copy()
    out["rating_kind"] = RATING_KIND
    complete = out.get("bars_20", 0) >= 20
    if isinstance(complete, int):
        complete = pd.Series(True, index=out.index)
    mfe = out.get("excess_mfe_20")
    mae = out.get("mae_20")
    eff = out.get("efficiency_20")
    if mfe is None:
        out["quality"] = "watching"
        out["is_low_value"] = 0
        return out

    high = complete & (mfe >= 0.05) & (mae >= -0.12) & (eff.fillna(0) >= 1.5)
    low = complete & ((mfe < 0.02) | (eff.fillna(0) < 1.0) | (mae < -0.18))
    out["quality"] = np.where(
        ~complete,
        "watching",
        np.where(high, "high_value", np.where(low, "low_value", "neutral")),
    )
    out["is_low_value"] = (out["quality"] == "low_value").astype(int)
    return out
