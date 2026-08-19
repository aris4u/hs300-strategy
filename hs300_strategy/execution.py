"""Canonical execution: T close signal, T+1 open fill. Never fill at T close."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hs300_strategy.config import COMMISSION, SLIPPAGE_BUY, SLIPPAGE_SELL, STAMP_TAX

EXECUTION_NOTE = (
    "T日收盘生成信号，T+1日开盘成交。禁止使用T日收盘价作为成交价。"
    "账本字段：signal_date, entry_date, entry_price, exit_date, exit_price。"
)


def single_t1_open(
    pos: pd.Series,
    open_: pd.Series,
    close: pd.Series,
    *,
    commission: float = COMMISSION,
    stamp_tax: float = STAMP_TAX,
    slip_buy: float = SLIPPAGE_BUY,
    slip_sell: float = SLIPPAGE_SELL,
) -> pd.DataFrame:
    """pos[t] is the target known at close t, filled at open t+1."""
    pos = pos.astype(float).fillna(0.0)
    open_ = open_.astype(float)
    close = close.astype(float)
    overnight = (open_ / close.shift(1) - 1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    intraday = (close / open_ - 1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    w_intraday = pos.shift(1).fillna(0.0)
    w_overnight = pos.shift(2).fillna(0.0)
    gross = w_overnight * overnight + w_intraday * intraday
    delta = w_intraday - w_overnight
    buy = delta.clip(lower=0.0)
    sell = (-delta.clip(upper=0.0))
    cost = buy * (commission + slip_buy) + sell * (commission + slip_sell + stamp_tax)
    return pd.DataFrame(
        {
            "position_signal": pos,
            "position_held": w_intraday,
            "gross_ret": gross,
            "net_ret": gross - cost,
            "cost": cost,
            "turnover": buy + sell,
            "open": open_,
            "close": close,
        }
    )


def single_blotter(dates: pd.Series, pos: pd.Series, open_: pd.Series, close: pd.Series) -> pd.DataFrame:
    """One row per round-trip. signal_date = close when size becomes >0."""
    dates = pd.to_datetime(dates).reset_index(drop=True)
    pos = pd.Series(pos).astype(float).fillna(0.0).reset_index(drop=True)
    open_ = pd.Series(open_).astype(float).reset_index(drop=True)
    close = pd.Series(close).astype(float).reset_index(drop=True)
    prev = pos.shift(1).fillna(0.0)
    starts = [i for i in range(len(pos)) if pos.iloc[i] > 1e-12 and prev.iloc[i] <= 1e-12]
    rows = []
    for i in starts:
        if i + 1 >= len(dates):
            rows.append(
                {
                    "signal_date": dates.iloc[i],
                    "entry_date": pd.NaT,
                    "entry_price": np.nan,
                    "exit_date": pd.NaT,
                    "exit_price": np.nan,
                    "signal_pos": float(pos.iloc[i]),
                    "status": "pending_next_open",
                }
            )
            continue
        entry_i = i + 1
        rest = pos.iloc[i + 1 :]
        zeros = rest.index[rest <= 1e-12]
        if len(zeros) == 0:
            exit_i = len(dates) - 1
            exit_px = float(close.iloc[exit_i]) if pd.notna(close.iloc[exit_i]) else np.nan
            status = "open"
        else:
            z = int(zeros[0])
            if z + 1 < len(dates):
                exit_i = z + 1
                exit_px = float(open_.iloc[exit_i]) if pd.notna(open_.iloc[exit_i]) else np.nan
                status = "closed_next_open"
            else:
                exit_i = z
                exit_px = float(close.iloc[z]) if pd.notna(close.iloc[z]) else np.nan
                status = "flatten_last_bar"
        ep = float(open_.iloc[entry_i]) if pd.notna(open_.iloc[entry_i]) else np.nan
        rows.append(
            {
                "signal_date": dates.iloc[i],
                "entry_date": dates.iloc[entry_i],
                "entry_price": ep,
                "exit_date": dates.iloc[exit_i],
                "exit_price": exit_px,
                "signal_pos": float(pos.iloc[i]),
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def last_bar_execution(dates: pd.Series, pos: pd.Series, open_: pd.Series, close: pd.Series, launch: pd.Series | None = None) -> dict:
    """Live hint for the last bar: do not treat close as a fill."""
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    pos = pd.Series(pos).astype(float).fillna(0.0).reset_index(drop=True)
    open_ = pd.Series(open_).astype(float).reset_index(drop=True)
    close = pd.Series(close).astype(float).reset_index(drop=True)
    i = len(dates) - 1
    prev = float(pos.iloc[i - 1]) if i else 0.0
    cur = float(pos.iloc[i])
    launched = False
    if launch is not None and len(launch):
        launched = int(pd.Series(launch).iloc[i] or 0) == 1
    out = {
        "asof": dates.iloc[i],
        "close_not_a_fill": float(close.iloc[i]) if pd.notna(close.iloc[i]) else np.nan,
        "signal_date": pd.NaT,
        "entry_date": pd.NaT,
        "entry_price": np.nan,
        "exit_date": pd.NaT,
        "exit_price": np.nan,
        "note": EXECUTION_NOTE,
    }
    if launched or (cur > 1e-12 and prev <= 1e-12):
        out["signal_date"] = dates.iloc[i]
        out["entry_date"] = None
        out["note"] = "今日收盘产生买入/加仓信号，下一交易日开盘成交，今日收盘价不是成交价。"
        return out
    if cur <= 1e-12 and prev > 1e-12:
        out["signal_date"] = dates.iloc[i]
        out["note"] = "今日收盘产生平仓信号，下一交易日开盘卖出，今日收盘价不是成交价。"
        return out
    blot = single_blotter(dates, pos, open_, close)
    if blot.empty:
        return out
    last = blot.iloc[-1]
    if str(last.get("status")) == "open" or (cur > 1e-12):
        out["signal_date"] = last["signal_date"]
        out["entry_date"] = last["entry_date"]
        out["entry_price"] = last["entry_price"]
        out["note"] = "当前持仓来自既往信号的T+1开盘成交。"
    return out
