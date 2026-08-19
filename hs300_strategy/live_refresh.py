"""Intraday K-line + signal preview for the currently viewed stock.

Patches today's forming bar from Tongdaxin quotes, recomputes signals, redraws
one PNG. This is a preview: confirmed signals remain T close / T+1 open.
Does not overwrite daily K-line CSV (volume units differ: TDX 手 vs BaoStock 股).
"""

from __future__ import annotations

import threading
import time
from datetime import date, datetime, time as dtime, timedelta

import pandas as pd

from hs300_strategy.advise import ENV_CN, make_advice
from hs300_strategy.charts import CHART_DIR, LOOKBACK_BARS, LABEL_CN, plot_kline_signals
from hs300_strategy.config import LIVE_PLOT_DPI, LIVE_POLL_CLOSED_SECONDS, LIVE_POLL_SECONDS, LIVE_PREVIEW_NOTE
from hs300_strategy.data import fetch_hs300, read_csv_cached
from hs300_strategy.formula import compute_signals
from hs300_strategy.moneyflow import read_cached_l2
from hs300_strategy.stock_data import fetch_constituents, fetch_stock_kline
from hs300_strategy.tdx_l2 import snap_quotes

_PLOT_LOCK = threading.Lock()
_CACHE_LOCK = threading.Lock()
INDEX_CODE = "000300.SH"
_LAST_BAR: dict[str, tuple] = {}
_PAYLOAD: dict[str, dict] = {}
_QUOTE_FP: dict[str, tuple] = {}
_IDX_ENV: tuple | None = None
_LAST_PLOT_AT: dict[str, float] = {}
_PLOT_MIN_SECONDS = 5.0


def _now_cn() -> datetime:
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("Asia/Shanghai"))
    except Exception:
        return datetime.utcnow() + timedelta(hours=8)


def _session_open(now: datetime | None = None) -> bool:
    now = now or _now_cn()
    if now.weekday() >= 5:
        return False
    t = now.time()
    return dtime(9, 15) <= t <= dtime(15, 5)


def _poll_seconds(now: datetime | None = None) -> int:
    return LIVE_POLL_SECONDS if _session_open(now) else LIVE_POLL_CLOSED_SECONDS


def _patch_today(df: pd.DataFrame, ohlcv: dict, today: pd.Timestamp) -> pd.DataFrame:
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    work = work.sort_values("date").reset_index(drop=True)
    if work.empty:
        raise ValueError("没有可用的日K缓存")
    last = work["date"].iloc[-1]
    vol = ohlcv.get("volume")
    if vol is None:
        vol = float(work["volume"].iloc[-1]) if last == today else 0.0
    fields = {
        "open": float(ohlcv["open"]),
        "high": float(ohlcv["high"]),
        "low": float(ohlcv["low"]),
        "close": float(ohlcv["close"]),
        "volume": float(vol),
    }
    if last == today:
        for col, val in fields.items():
            if col in work.columns:
                work.loc[work.index[-1], col] = val
        return work
    new = {col: work.iloc[-1][col] if col in work.columns else pd.NA for col in work.columns}
    new["date"] = today
    new.update(fields)
    return pd.concat([work, pd.DataFrame([new])], ignore_index=True)


def _quote_row(quotes: pd.DataFrame, ts_code: str) -> pd.Series | None:
    if quotes is None or quotes.empty or "ts_code" not in quotes.columns:
        return None
    hit = quotes[quotes["ts_code"] == ts_code]
    if hit.empty:
        return None
    return hit.iloc[0]


def _ohlcv_from_snap(snap: dict, yesterday_volume: float | None) -> dict:
    vol = snap.get("vol")
    if vol is None:
        vol = yesterday_volume
    return {
        "open": float(snap["open"]),
        "high": float(snap["high"]),
        "low": float(snap["low"]),
        "close": float(snap["price"]),
        "volume": None if vol is None else float(vol),
    }


def _quote_fp(ohlcv: dict) -> tuple:
    return (
        round(float(ohlcv["open"]), 4),
        round(float(ohlcv["high"]), 4),
        round(float(ohlcv["low"]), 4),
        round(float(ohlcv["close"]), 4),
        int(float(ohlcv["volume"] or 0)),
    )


def refresh_stock(ts_code: str, *, redraw: bool = True, force_plot: bool = False) -> dict:
    """Refresh one stock from TDX quote. Thread-safe for matplotlib."""
    ts_code = ts_code.strip().upper()
    today = pd.Timestamp(date.today()).normalize()
    now = _now_cn()
    try:
        snaps = snap_quotes([ts_code, INDEX_CODE])
    except Exception as exc:
        return {"ok": False, "error": f"通达信行情失败：{exc}", "ts_code": ts_code}

    stk = snaps.get(ts_code)
    if not stk or stk.get("price") is None:
        return {
            "ok": False,
            "error": "通达信没有这只股票的盘口。确认本机通达信/mootdx 能连行情服务器。",
            "ts_code": ts_code,
        }
    ohlcv = _ohlcv_from_snap(stk, None)
    quote_fp = _quote_fp(ohlcv)
    with _CACHE_LOCK:
        cached = _PAYLOAD.get(ts_code)
        if cached and _QUOTE_FP.get(ts_code) == quote_fp:
            out = dict(cached)
            out["quote_time"] = now.strftime("%Y-%m-%d %H:%M:%S")
            out["chart_changed"] = False
            out["session_open"] = _session_open(now)
            out["poll_seconds"] = _poll_seconds(now)
            out["tdx_price"] = ohlcv["close"]
            return out

    k = fetch_stock_kline(ts_code, use_cache=True)
    if k is None or k.empty:
        return {"ok": False, "error": f"没有 {ts_code} 的日K缓存。请先运行 python plot_all.py"}

    yesterday_vol = None
    kdates = pd.to_datetime(k["date"]).dt.normalize()
    prev = k.loc[kdates < today]
    if not prev.empty:
        yesterday_vol = float(prev.iloc[-1]["volume"])
    elif not k.empty:
        yesterday_vol = float(k.iloc[-1]["volume"])
    if ohlcv["volume"] is None:
        ohlcv["volume"] = yesterday_vol
    work = _patch_today(k, ohlcv, today)

    idx = snaps.get(INDEX_CODE)
    hs = fetch_hs300(use_cache=True)
    idx_fp = None
    if idx and idx.get("price") is not None:
        try:
            y_idx = float(hs.iloc[-1]["volume"]) if not hs.empty else None
            idx_ohlcv = _ohlcv_from_snap(idx, y_idx)
            hs = _patch_today(hs, idx_ohlcv, today)
            idx_fp = _quote_fp(idx_ohlcv)
        except Exception:
            idx_fp = None
    global _IDX_ENV
    if _IDX_ENV is not None and _IDX_ENV[0] == idx_fp:
        env_map = _IDX_ENV[1]
    else:
        idx_sig = compute_signals(hs.copy(), asset="index")
        env_map = idx_sig.set_index(pd.to_datetime(idx_sig["date"]).dt.normalize())["env_level"]
        _IDX_ENV = (idx_fp, env_map)
    work["market_env"] = pd.to_datetime(work["date"]).dt.normalize().map(env_map)

    start = pd.Timestamp(work["date"].min()).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")
    flow = read_cached_l2(ts_code, start, end)
    if flow is not None and not flow.empty:
        flow = flow.copy()
        flow["date"] = pd.to_datetime(flow["date"]).dt.normalize()
        work = work.merge(flow[["date", "l2jbl"]], on="date", how="left")
    l2_live = None
    if "l2jbl" in work.columns:
        raw = pd.to_numeric(work["l2jbl"].iloc[-1], errors="coerce")
        if pd.notna(raw):
            l2_live = float(raw)

    sig = compute_signals(work, asset="stock")
    last = sig.iloc[-1]
    last_date = pd.Timestamp(last["date"]).normalize()
    preview = last_date == today
    members = fetch_constituents(use_cache=True)
    name = ""
    hit = members[members["ts_code"] == ts_code]
    if not hit.empty:
        name = str(hit.iloc[0]["name"])
    rank_row = None
    rank_path = CHART_DIR.parent / "stock_rank.csv"
    if rank_path.exists():
        rank = read_csv_cached(rank_path)
        rhit = rank[rank["ts_code"] == ts_code]
        if not rhit.empty:
            rank_row = rhit.iloc[0].to_dict()
    advice = make_advice(sig, rank_row)
    flags = advice.get("flags") or []
    sig_fp = (
        tuple(flags),
        str(advice["action"]),
        round(float(advice.get("position") or 0), 4),
    )
    sig_changed = _LAST_BAR.get(ts_code + ":sig") != sig_fp
    _LAST_BAR[ts_code + ":sig"] = sig_fp
    now_t = time.time()
    session = _session_open(now)
    stale_plot = ts_code not in _LAST_PLOT_AT or (now_t - _LAST_PLOT_AT[ts_code] >= _PLOT_MIN_SECONDS)
    do_plot = bool(redraw and (force_plot or session) and (sig_changed or stale_plot))
    if not session and not force_plot:
        do_plot = False
    chart_ts = int(now_t) if do_plot else int(_LAST_BAR.get(ts_code + ":ts") or now_t)
    if do_plot:
        _LAST_BAR[ts_code + ":ts"] = chart_ts
        _LAST_PLOT_AT[ts_code] = now_t
        label = LABEL_CN.get(str(last.get("label", "")), last.get("label", ""))
        kind = "盘中预览" if preview else "收盘快照"
        title = (
            f"{name}  {ts_code}    {kind} {last_date.strftime('%Y-%m-%d')}  "
            f"现价 {float(last['close']):.2f}    仓位 {float(last['position']):.0%}    {label}"
        )
        path = CHART_DIR / f"{ts_code.replace('.', '_')}.png"
        with _PLOT_LOCK:
            plot_kline_signals(
                sig, path, title=title, bars=LOOKBACK_BARS, ts_code=ts_code, dpi=LIVE_PLOT_DPI
            )
    io = stk.get("inner_outer")
    result = {
        "ok": True,
        "ts_code": ts_code,
        "name": name,
        "live": True,
        "preview": bool(preview),
        "session_open": _session_open(now),
        "source": "tdx",
        "quote_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date": last_date.strftime("%Y-%m-%d"),
        "open": float(last["open"]),
        "high": float(last["high"]),
        "low": float(last["low"]),
        "close": float(last["close"]),
        "volume": float(last["volume"]),
        "tdx_price": ohlcv["close"],
        "inner_outer": None if io is None else float(io),
        "l2jbl_today": l2_live,
        "action": advice["action"],
        "position_hint": advice["position_hint"],
        "confidence": advice["confidence"],
        "headline": advice["headline"],
        "detail": advice["detail"],
        "flags": flags,
        "position": advice["position"],
        "color": advice["color"],
        "env": ENV_CN.get(int(last["env_level"]) if pd.notna(last.get("env_level")) else 0, str(last.get("env_level"))),
        "dist_score": float(last.get("dist_score", 0) or 0),
        "signal_date": advice.get("signal_date"),
        "entry_date": advice.get("entry_date"),
        "entry_price": advice.get("entry_price"),
        "exit_date": advice.get("exit_date"),
        "exit_price": advice.get("exit_price"),
        "execution": advice.get("execution"),
        "has_chart": True,
        "chart": f"/charts/{ts_code.replace('.', '_')}.png",
        "chart_ts": chart_ts,
        "chart_changed": bool(do_plot),
        "poll_seconds": _poll_seconds(now),
        "note": LIVE_PREVIEW_NOTE,
    }
    with _CACHE_LOCK:
        _QUOTE_FP[ts_code] = quote_fp
        _PAYLOAD[ts_code] = result
    return result
