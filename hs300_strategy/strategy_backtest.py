"""Full state-machine strategy: T close signal, T+1 open fill.

Not an event study. Costs apply. Gross and net are both reported.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from hs300_strategy.config import (
    COMMISSION,
    FULL_START,
    SLIPPAGE_BUY,
    SLIPPAGE_SELL,
    STAMP_TAX,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
)


def position_blotter(pos: pd.DataFrame, open_px: pd.DataFrame, close_px: pd.DataFrame) -> pd.DataFrame:
    """signal_date = day position becomes >0; entry_date = next bar open."""
    rows = []
    cal = list(pos.index)
    loc = {d: i for i, d in enumerate(cal)}
    for code in pos.columns:
        p = pos[code].fillna(0.0)
        prev = p.shift(1).fillna(0.0)
        starts = p.index[(p > 1e-12) & (prev <= 1e-12)]
        for sd in starts:
            i = loc[sd]
            if i + 1 >= len(cal):
                continue
            entry_d = cal[i + 1]
            ep = open_px.loc[entry_d, code] if code in open_px.columns else np.nan
            rest = p.iloc[i + 1 :]
            end_idx = rest.index[rest <= 1e-12]
            if len(end_idx) == 0:
                exit_d = cal[-1]
                xp = close_px.loc[exit_d, code] if code in close_px.columns else np.nan
            else:
                # flatten known at close of first zero day; sell next open
                z = end_idx[0]
                zi = loc[z]
                if zi + 1 < len(cal):
                    exit_d = cal[zi + 1]
                    xp = open_px.loc[exit_d, code] if code in open_px.columns else np.nan
                else:
                    exit_d = z
                    xp = close_px.loc[z, code] if code in close_px.columns else np.nan
            rows.append(
                {
                    "ts_code": code,
                    "signal_date": pd.Timestamp(sd),
                    "entry_date": pd.Timestamp(entry_d),
                    "entry_price": float(ep) if pd.notna(ep) else np.nan,
                    "exit_date": pd.Timestamp(exit_d),
                    "exit_price": float(xp) if pd.notna(xp) else np.nan,
                    "signal_pos": float(p.loc[sd]),
                }
            )
    return pd.DataFrame(rows)


def session_parts(open_px: pd.DataFrame, close_px: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overnight = open_px / close_px.shift(1) - 1
    intraday = close_px / open_px - 1
    return overnight, intraday


def portfolio_from_position(
    pos: pd.DataFrame,
    open_px: pd.DataFrame,
    close_px: pd.DataFrame,
    idx_open: pd.Series,
    idx_close: pd.Series,
    *,
    commission: float = COMMISSION,
    stamp_tax: float = STAMP_TAX,
    slip_buy: float = SLIPPAGE_BUY,
    slip_sell: float = SLIPPAGE_SELL,
    normalize: bool = True,
) -> pd.DataFrame:
    """pos[t] = target known at close t, filled at open t+1.

    Overnight P&L uses yesterday's filled weights; intraday uses this morning's fill.
    """
    cal = close_px.index
    pos = pos.reindex(cal).fillna(0.0)
    open_px = open_px.reindex(cal)
    close_px = close_px.reindex(cal)
    overnight, intraday = session_parts(open_px, close_px)
    overnight = overnight.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    intraday = intraday.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    valid = open_px.notna() & close_px.notna() & close_px.shift(1).notna() & (open_px > 0)
    w_sig = pos.where(pos > 1e-12, 0.0)
    if normalize:
        score = w_sig.sum(axis=1)
        w_sig = w_sig.div(score.where(score > 1e-12, np.nan), axis=0).fillna(0.0)
    else:
        # equal-weight among names with pos>0, ignore size other than 0/positive
        held = (w_sig > 1e-12) & valid
        n = held.sum(axis=1).replace(0, np.nan)
        w_sig = held.astype(float).div(n, axis=0).fillna(0.0)

    w_intraday = w_sig.shift(1).fillna(0.0)   # filled this open
    w_overnight = w_sig.shift(2).fillna(0.0)  # held into this open

    gross = (w_overnight * overnight).sum(axis=1) + (w_intraday * intraday).sum(axis=1)
    delta = w_intraday - w_overnight
    buy = delta.clip(lower=0.0).sum(axis=1)
    sell = (-delta.clip(upper=0.0)).sum(axis=1)
    cost = buy * (commission + slip_buy) + sell * (commission + slip_sell + stamp_tax)
    turnover = (buy + sell)
    net = gross - cost

    held = w_intraday > 1e-12
    rest = valid & ~held
    n_long = held.sum(axis=1)
    n_rest = rest.sum(axis=1)
    n_uni = valid.sum(axis=1)

    # Benchmarks: fully invested close-to-close on the same calendar day.
    cc = (close_px / close_px.shift(1) - 1).replace([np.inf, -np.inf], np.nan)
    uni_ew = _ew(cc, valid, n_uni)
    rest_ew = _ew(cc, rest, n_rest)
    long_cc = _ew(cc, held, n_long)

    idx_open = idx_open.reindex(cal)
    idx_close = idx_close.reindex(cal)
    idx_cc = (idx_close / idx_close.shift(1) - 1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    idx_open_to_close = (idx_close / idx_open - 1).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out = pd.DataFrame(
        {
            "gross_ret": gross,
            "net_ret": net,
            "cost": cost,
            "turnover": turnover,
            "n_hold": n_long,
            "n_rest": n_rest,
            "uni_ew_ret": uni_ew,
            "unselected_ew_ret": rest_ew,
            "held_cc_ret": long_cc,
            "hs300_ret": idx_cc,
            "hs300_open_to_close": idx_open_to_close,
            "gross_vs_hs300": gross - idx_cc,
            "net_vs_hs300": net - idx_cc,
            "gross_vs_uni": gross - uni_ew,
            "net_vs_uni": net - uni_ew,
            "gross_vs_unselected": gross - rest_ew,
            "net_vs_unselected": net - rest_ew,
        },
        index=cal,
    )
    return out


def reconstruct_positions(
    launch: pd.DataFrame,
    reduce_band: pd.DataFrame,
    reduce_trend: pd.DataFrame,
    take_profit: pd.DataFrame,
    escape: pd.DataFrame,
    env: pd.DataFrame,
    mode: str,
    band_level: float = 0.50,
    trend_level: float = 0.70,
) -> pd.DataFrame:
    """Rebuild size from the same daily flags. Does not change formula.py.

    always_100: launch → 100% until TP / CT / env==1. Ignore JC.
    state_machine: same sequential rules as formula._sequential_position.
    restore_100: 100% while in trade; JC cuts apply only on the event day, then restore.
    """
    cal = launch.index
    codes = launch.columns
    frames = []
    for code in codes:
        frames.append(
            _one_pos(
                launch[code].fillna(0).to_numpy(),
                reduce_band[code].fillna(0).to_numpy() if code in reduce_band else np.zeros(len(cal)),
                reduce_trend[code].fillna(0).to_numpy() if code in reduce_trend else np.zeros(len(cal)),
                take_profit[code].fillna(0).to_numpy() if code in take_profit else np.zeros(len(cal)),
                escape[code].fillna(0).to_numpy() if code in escape else np.zeros(len(cal)),
                env[code].fillna(0).to_numpy() if code in env else np.zeros(len(cal)),
                mode,
                band_level,
                trend_level,
            )
        )
    return pd.DataFrame(np.column_stack(frames), index=cal, columns=codes)


def metrics_from_daily(daily: pd.DataFrame, sample: str, label: str, ret_col: str = "net_ret") -> dict:
    work = _slice_dates(daily, sample)
    if work.empty:
        return {"sample": sample, "label": label, "n": 0}
    r = work[ret_col].astype(float).fillna(0.0)
    g = work["gross_ret"].astype(float).fillna(0.0) if "gross_ret" in work.columns else r
    nav = (1 + r).cumprod()
    nav_g = (1 + g).cumprod()
    bench = (1 + work["hs300_ret"].astype(float).fillna(0.0)).cumprod()
    n = len(work)
    years = n / 252 if n else 0
    total = float(nav.iloc[-1] / nav.iloc[0] - 1) if n else 0.0
    total_g = float(nav_g.iloc[-1] / nav_g.iloc[0] - 1) if n else 0.0
    bh = float(bench.iloc[-1] / bench.iloc[0] - 1) if n else 0.0
    dd = float((nav / nav.cummax() - 1).min()) if n else 0.0
    x = work["net_vs_hs300"] if ret_col == "net_ret" and "net_vs_hs300" in work.columns else (r - work["hs300_ret"])
    xu = work["net_vs_unselected"] if ret_col == "net_ret" and "net_vs_unselected" in work.columns else work.get("gross_vs_unselected")
    t_x, p_x = _ttest(x.iloc[1:] if len(x) > 1 else x)
    t_u, p_u = _ttest(xu.iloc[1:] if xu is not None and len(xu) > 1 else xu)
    t_uni, p_uni = _ttest(work.get("net_vs_uni", work.get("gross_vs_uni")))
    return {
        "sample": sample,
        "label": label,
        "ret_col": ret_col,
        "start": work.index.min().strftime("%Y-%m-%d"),
        "end": work.index.max().strftime("%Y-%m-%d"),
        "n": n,
        "gross_return": total_g,
        "net_return": total,
        "hs300_return": bh,
        "excess_hs300": total - bh,
        "max_drawdown": dd,
        "avg_turnover": float(work["turnover"].mean()) if "turnover" in work.columns else np.nan,
        "avg_hold": float(work["n_hold"].mean()) if "n_hold" in work.columns else np.nan,
        "cost_drag": total_g - total,
        "t_vs_hs300": t_x,
        "p_vs_hs300": p_x,
        "t_vs_unselected": t_u,
        "p_vs_unselected": p_u,
        "t_vs_uni": t_uni,
        "p_vs_uni": p_uni,
        "mean_daily_vs_hs300": float(pd.to_numeric(x, errors="coerce").mean()),
        "mean_daily_vs_unselected": float(pd.to_numeric(xu, errors="coerce").mean()) if xu is not None else np.nan,
        "mean_daily_vs_uni": float(pd.to_numeric(work.get("net_vs_uni"), errors="coerce").mean()),
        "sharpe_net": _sharpe(r),
        "verdict_style": _style_verdict(p_x, p_u, float(pd.to_numeric(x, errors="coerce").mean()), float(pd.to_numeric(xu, errors="coerce").mean()) if xu is not None else 0),
    }


def enhance_blend(
    sat_daily: pd.DataFrame,
    idx_ret: pd.Series,
    satellite: float,
    heat: pd.Series | None = None,
    heat_scale: float = 0.40,
) -> pd.DataFrame:
    """Core index + satellite sleeve. heat uses lagged index move (already shifted by caller)."""
    sat = sat_daily["gross_ret"].reindex(idx_ret.index).fillna(0.0)
    sat_net = sat_daily["net_ret"].reindex(idx_ret.index).fillna(0.0)
    idx_ret = idx_ret.fillna(0.0)
    has = sat_daily["n_hold"].reindex(idx_ret.index).fillna(0) > 0
    expo = pd.Series(np.where(has.to_numpy(), float(satellite), 0.0), index=idx_ret.index)
    if heat is not None:
        h = heat.reindex(idx_ret.index).fillna(False).astype(bool)
        expo = expo.where(~h, expo * heat_scale)
    gross = (1.0 - expo) * idx_ret + expo * sat
    net = (1.0 - expo) * idx_ret + expo * sat_net
    to = sat_daily["turnover"].reindex(idx_ret.index).fillna(0.0) * expo
    out = pd.DataFrame(
        {
            "gross_ret": gross,
            "net_ret": net,
            "hs300_ret": idx_ret,
            "turnover": to,
            "satellite_w": expo,
            "n_hold": sat_daily["n_hold"].reindex(idx_ret.index).fillna(0),
            "unselected_ew_ret": sat_daily["unselected_ew_ret"].reindex(idx_ret.index).fillna(0),
            "uni_ew_ret": sat_daily["uni_ew_ret"].reindex(idx_ret.index).fillna(0),
        },
        index=idx_ret.index,
    )
    out["gross_vs_hs300"] = out["gross_ret"] - out["hs300_ret"]
    out["net_vs_hs300"] = out["net_ret"] - out["hs300_ret"]
    out["net_vs_unselected"] = out["net_ret"] - out["unselected_ew_ret"]
    out["net_vs_uni"] = out["net_ret"] - out["uni_ew_ret"]
    out["cost"] = out["gross_ret"] - out["net_ret"]
    return out


def _one_pos(launch, band, trend, tp, esc, env, mode, band_level, trend_level) -> np.ndarray:
    n = len(launch)
    pos = 0.0
    out = np.zeros(n)
    in_trade = False
    for i in range(n):
        if launch[i] > 0:
            pos = 1.0
            in_trade = True
        flatten = in_trade and (tp[i] > 0 or esc[i] > 0 or env[i] == 1)
        if flatten:
            pos = 0.0
            in_trade = False
            out[i] = 0.0
            continue
        if not in_trade:
            out[i] = 0.0
            continue
        if mode == "always_100":
            pos = 1.0
        elif mode == "restore_100":
            pos = 1.0
            if band[i] > 0:
                pos = band_level
            elif trend[i] > 0:
                pos = trend_level
        else:  # state_machine
            if band[i] > 0:
                pos = band_level
            elif trend[i] > 0:
                pos = min(pos, trend_level)
        out[i] = pos
    return out


def _ew(ret: pd.DataFrame, mask: pd.DataFrame, n: pd.Series) -> pd.Series:
    raw = ret.where(mask, 0.0).sum(axis=1)
    return raw.div(n.replace(0, np.nan)).fillna(0.0)


def _slice_dates(daily: pd.DataFrame, sample: str) -> pd.DataFrame:
    idx = pd.DatetimeIndex(daily.index)
    work = daily.copy()
    work.index = idx
    if sample == "train":
        return work[(idx >= pd.Timestamp(TRAIN_START)) & (idx <= pd.Timestamp(TRAIN_END))]
    if sample == "test":
        return work[(idx >= pd.Timestamp(TEST_START)) & (idx <= pd.Timestamp(TEST_END))]
    if sample == "enhance":
        return work[(idx >= pd.Timestamp("2024-09-02")) & (idx <= pd.Timestamp(TEST_END))]
    return work[(idx >= pd.Timestamp(FULL_START)) & (idx <= pd.Timestamp(TEST_END))]


def _ttest(x) -> tuple[float, float]:
    if x is None:
        return float("nan"), float("nan")
    v = pd.to_numeric(x, errors="coerce").dropna()
    v = v.replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(v))
    if n < 5 or float(v.std(ddof=1) or 0) == 0:
        return float("nan"), float("nan")
    t = float(v.mean() / (v.std(ddof=1) / math.sqrt(n)))
    p = 2.0 * (0.5 * math.erfc(abs(t) / math.sqrt(2.0)))
    return t, p


def _sharpe(r: pd.Series) -> float:
    v = pd.to_numeric(r, errors="coerce").dropna()
    if len(v) < 5 or float(v.std(ddof=1) or 0) == 0:
        return float("nan")
    return float(v.mean() / v.std(ddof=1) * math.sqrt(252))


def _style_verdict(p_hs, p_unsel, mu_hs, mu_unsel) -> str:
    hs_sig = pd.notna(p_hs) and p_hs < 0.05 and mu_hs > 0
    un_sig = pd.notna(p_unsel) and p_unsel < 0.05 and mu_unsel > 0
    if hs_sig and not un_sig:
        return "策略相对市场基准存在超额，但暂不能区分选股能力与风格暴露。"
    if hs_sig and un_sig:
        return "相对沪深300与未入选等权均显著，选股方向的证据更强（仍非因果）。"
    return "相对沪深300或未入选等权未同时达到显著正超额。"
