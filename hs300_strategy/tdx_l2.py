"""Read live L2-style fields from a local Tongdaxin install via mootdx."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pandas as pd

from hs300_strategy.data import disable_http_proxy

DEFAULT_TDX = Path(r"C:\new_tdx")
LARGE_YUAN = 200_000  # 单笔 ≥ 20 万视作大单，近似 LARGEINTRDVOL


from hs300_strategy.secrets import load_env


def tdx_dir() -> Path:
    load_env()
    raw = os.environ.get("TDX_DIR", "").strip()
    p = Path(raw) if raw else DEFAULT_TDX
    return p


def tdx_ready() -> bool:
    root = tdx_dir()
    return (root / "TdxW.exe").exists() and (root / "l2plugin.cfg").exists()


def to_tdx_symbol(ts_code: str) -> str:
    return ts_code.split(".")[0]


def to_quote_symbol(ts_code: str) -> str:
    """mootdx 把 000300 当成个股；上证指数要用 sh000300。"""
    num, mkt = ts_code.split(".")
    if mkt.upper() == "SH" and num.startswith("000"):
        return f"sh{num}"
    return num


def _first_num(row: pd.Series, names: tuple[str, ...]) -> float | None:
    for name in names:
        if name not in row.index:
            continue
        v = pd.to_numeric(row[name], errors="coerce")
        try:
            if pd.isna(v):
                continue
        except (TypeError, ValueError):
            continue
        return float(v)
    return None


def quote_ohlcv(row: pd.Series, yesterday_volume: float | None = None) -> dict:
    """Map a mootdx quotes row to daily OHLCV. Volume is converted to 股."""
    price = _first_num(row, ("price", "last_price", "now", "close"))
    o = _first_num(row, ("open", "open_price"))
    h = _first_num(row, ("high", "high_price"))
    l = _first_num(row, ("low", "low_price"))
    tdx_vol = _first_num(row, ("vol", "volume", "zongshou"))
    if price is None:
        raise ValueError("行情没有最新价")
    o = o if o and o > 0 else price
    h = h if h and h > 0 else max(o, price)
    l = l if l and l > 0 else min(o, price)
    h = max(h, o, price)
    l = min(l, o, price)
    vol = None
    if tdx_vol is not None and tdx_vol >= 0:
        as_shares = tdx_vol * 100.0  # 通达信成交量通常是手
        y = yesterday_volume if yesterday_volume and yesterday_volume > 0 else None
        if y is None:
            vol = as_shares
        else:
            vol = as_shares if abs(as_shares - y) <= abs(tdx_vol - y) else tdx_vol
    return {
        "open": o,
        "high": h,
        "low": l,
        "close": price,
        "volume": vol,
        "last_close": _first_num(row, ("last_close", "pre_close", "yes_close")),
        "circ_share": _first_num(row, ("ltg", "circulation", "capital", "ltsz")),
    }


def live_quotes(ts_codes: list[str]) -> pd.DataFrame:
    """Five-level book + 内外盘. Requires network to TDX quote servers."""
    client = _quotes_client()
    symbols = [to_quote_symbol(c) for c in ts_codes]
    frames = []
    for i in range(0, len(symbols), 80):
        chunk = symbols[i : i + 80]
        part = client.quotes(symbol=chunk)
        if part is not None and not part.empty:
            frames.append(part)
    if not frames:
        return pd.DataFrame()
    q = pd.concat(frames, ignore_index=True)
    q["code"] = q["code"].astype(str).str.zfill(6)
    lookup = {to_tdx_symbol(c): c for c in ts_codes}
    q["ts_code"] = q["code"].map(lookup)
    b = pd.to_numeric(q.get("b_vol"), errors="coerce").fillna(0)
    s = pd.to_numeric(q.get("s_vol"), errors="coerce").fillna(0)
    q["inner_outer"] = b - s
    tot = (b + s).replace(0, pd.NA)
    q["inner_outer_ratio"] = (b - s) / tot
    return q


_CLIENT = None
_SNAP_LOCK = threading.Lock()
_SNAP: dict = {"t": 0.0, "rows": {}}


def _quotes_client():
    global _CLIENT
    disable_http_proxy()
    if _CLIENT is None:
        from mootdx.quotes import Quotes

        _CLIENT = Quotes.factory(market="std", timeout=5)
    return _CLIENT


def snap_quotes(ts_codes: list[str]) -> dict[str, dict]:
    """Latest bid/ask/price. Cached ~1s so the UI can poll every second."""
    codes: list[str] = []
    seen: set[str] = set()
    for raw in ts_codes:
        c = str(raw).strip().upper().replace("_", ".")
        if not c or c in seen:
            continue
        if "." not in c and c.isdigit():
            c = f"{c}.SH" if c.startswith(("6", "9")) else f"{c}.SZ"
        seen.add(c)
        codes.append(c)
    if not codes:
        return {}
    now = time.time()
    with _SNAP_LOCK:
        fresh = now - float(_SNAP["t"]) < 0.85
        have = _SNAP["rows"]
        missing = [c for c in codes if c not in have]
        if fresh and not missing:
            return {c: have[c] for c in codes if c in have}
        try:
            client = _quotes_client()
            symbols = [to_quote_symbol(c) for c in codes]
            frames = []
            for i in range(0, len(symbols), 80):
                part = client.quotes(symbol=symbols[i : i + 80])
                if part is not None and not part.empty:
                    frames.append(part)
            if not frames:
                return {c: have[c] for c in codes if c in have}
            q = pd.concat(frames, ignore_index=True)
            q["code"] = q["code"].astype(str).str.zfill(6)
            lookup = {to_tdx_symbol(c): c for c in codes}
            q["ts_code"] = q["code"].map(lookup)
            b = pd.to_numeric(q.get("b_vol"), errors="coerce").fillna(0)
            s = pd.to_numeric(q.get("s_vol"), errors="coerce").fillna(0)
            q["inner_outer"] = b - s
            rows = dict(have)
            for _, row in q.iterrows():
                ts = row.get("ts_code")
                if not ts or pd.isna(ts):
                    continue
                try:
                    ohlc = quote_ohlcv(row)
                except ValueError:
                    continue
                prev = ohlc.get("last_close")
                price = ohlc["close"]
                pct = None
                if prev and prev > 0 and price is not None:
                    pct = (price - prev) / prev
                rows[str(ts)] = {
                    "price": price,
                    "open": ohlc["open"],
                    "high": ohlc["high"],
                    "low": ohlc["low"],
                    "last_close": prev,
                    "pct": pct,
                    "bid1": _first_num(row, ("bid1",)),
                    "ask1": _first_num(row, ("ask1",)),
                    "vol": ohlc.get("volume"),
                    "inner_outer": float(row["inner_outer"]) if pd.notna(row.get("inner_outer")) else None,
                    "servertime": str(row["servertime"]) if "servertime" in row.index and row["servertime"] else "",
                }
            _SNAP["t"] = now
            _SNAP["rows"] = rows
            return {c: rows[c] for c in codes if c in rows}
        except Exception:
            global _CLIENT
            _CLIENT = None
            return {c: have[c] for c in codes if c in have}


def ticks_l2jbl(ts_code: str, circ_share: float, offset: int = 2000) -> dict:
    """Today's large-trade imbalance / circulating shares * 100 ≈ L2JBL."""
    disable_http_proxy()
    from mootdx.quotes import Quotes

    client = Quotes.factory(market="std", timeout=10)
    ticks = client.transactions(symbol=to_tdx_symbol(ts_code), start=0, offset=offset)
    if ticks is None or ticks.empty or not circ_share or circ_share <= 0:
        return {"l2jbl": None, "n_ticks": 0, "large_in": 0.0, "large_out": 0.0}
    price = pd.to_numeric(ticks["price"], errors="coerce")
    vol = pd.to_numeric(ticks["vol"], errors="coerce").fillna(0)  # 手
    side = pd.to_numeric(ticks["buyorsell"], errors="coerce").fillna(1)
    yuan = price * vol * 100.0
    large = yuan >= LARGE_YUAN
    # TDX 分笔：0 买入，1 卖出
    buy = large & (side == 0)
    sell = large & (side == 1)
    large_in = float(vol[buy].sum() * 100)
    large_out = float(vol[sell].sum() * 100)
    l2jbl = (large_in - large_out) / float(circ_share) * 100.0
    return {
        "l2jbl": l2jbl,
        "n_ticks": int(len(ticks)),
        "large_in": large_in,
        "large_out": large_out,
    }
