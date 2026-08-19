"""Python版本对Level-2资金行为采用日度大单净额近似映射，不是通达信
LARGEINTRDVOL / LARGEOUTTRDVOL / L2_AMO 的 100% 复刻。对照：docs/l2_mapping.md。

指数和 ETF 在该数据源上没有 moneyflow。用沪深300权重靠前的成分股加总，
作为指数级大单净流入代理。接口走 .env 里的 TUSHARE_HTTP_URL（默认 jiaoch.site）。
"""

from __future__ import annotations

from datetime import date, timedelta
from time import sleep

import pandas as pd

from hs300_strategy.data import DATA_DIR, cache_covers, official_end_key, read_csv_cached
from hs300_strategy.secrets import tushare_pro

CACHE_FILE = DATA_DIR / "hs300_moneyflow.csv"
STOCK_CACHE_DIR = DATA_DIR / "mf_stocks"
ETF_CODE = "510300.SH"
TOP_N = 20
INDEX_CODE = "000300.SH"

FALLBACK_CODES = [
    "600519.SH",
    "300750.SZ",
    "601318.SH",
    "600036.SH",
    "000858.SZ",
    "000333.SZ",
    "601166.SH",
    "600900.SH",
    "601398.SH",
    "601288.SH",
    "000001.SZ",
    "600030.SH",
    "601012.SH",
    "002594.SZ",
    "600276.SH",
    "000651.SZ",
    "601888.SH",
    "300059.SZ",
    "601127.SH",
    "002475.SZ",
]


def fetch_moneyflow(start: str = "20100101", end: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    end = end or date.today().strftime("%Y%m%d")
    start_key = start.replace("-", "")
    end_key = end.replace("-", "")

    if use_cache and CACHE_FILE.exists():
        cached = pd.read_csv(CACHE_FILE, parse_dates=["date"])
        last = pd.Timestamp(cached["date"].max())
        first = pd.Timestamp(cached["date"].min()).strftime("%Y%m%d")
        if cache_covers(last, end_key, honor_stamp=False) and first <= start_key:
            return _slice(cached, start_key, end_key)

    pro = tushare_pro()
    codes = _hs300_top_codes(pro)
    frames = []
    for i, code in enumerate(codes):
        part = _stock_flow(pro, code, start_key, end_key)
        if part is not None and not part.empty:
            frames.append(part)
        if i + 1 < len(codes):
            sleep(0.15)
    if not frames:
        raise RuntimeError("资金流接口可连，但成分股 moneyflow 都为空。")

    raw = pd.concat(frames, ignore_index=True)
    df = _aggregate(raw)
    df.to_csv(CACHE_FILE, index=False, encoding="utf-8-sig")
    return _slice(df, start_key, end_key)


def merge_moneyflow(kline: pd.DataFrame, flow: pd.DataFrame) -> pd.DataFrame:
    out = kline.copy()
    out["date"] = pd.to_datetime(out["date"])
    flow = flow.copy()
    flow["date"] = pd.to_datetime(flow["date"])
    cols = ["date", "l2jbl", "主力净额", "净流入额"]
    return out.merge(flow[cols], on="date", how="left")


def _hs300_top_codes(pro) -> list[str]:
    today = date.today().replace(day=1)
    for months_back in range(0, 24):
        month = _shift_month(today, -months_back)
        start = month.strftime("%Y%m%d")
        end = _month_end(month).strftime("%Y%m%d")
        try:
            weights = pro.index_weight(index_code=INDEX_CODE, start_date=start, end_date=end)
        except Exception:
            continue
        if weights is None or weights.empty:
            continue
        last = str(weights["trade_date"].max())
        snap = weights[weights["trade_date"].astype(str) == last].copy()
        code_col = "con_code" if "con_code" in snap.columns else "ts_code"
        snap["weight"] = pd.to_numeric(snap["weight"], errors="coerce")
        codes = (
            snap.sort_values("weight", ascending=False)[code_col]
            .astype(str)
            .head(TOP_N)
            .tolist()
        )
        if codes:
            print(f"资金流成分 {len(codes)} 只，权重日 {last}")
            return codes
    print("指数权重不可用，改用固定权重股名单")
    return list(FALLBACK_CODES)


def read_cached_l2(code: str, start: str, end: str) -> pd.DataFrame | None:
    """Read local moneyflow only. Never hits the network."""
    path = STOCK_CACHE_DIR / f"{code.replace('.', '_')}.csv"
    if not path.exists():
        return None
    cached = read_csv_cached(path, parse_dates=["date"])
    if cached.empty:
        return None
    start_key = start.replace("-", "")
    end_key = end.replace("-", "")
    return _slice(cached, start_key, end_key)


def fetch_stock_l2(code: str, start: str, end: str, pro=None, use_cache: bool = True) -> pd.DataFrame | None:
    """单只股票的 L2JBL（大单净额/流通市值*100），不缩放。"""
    start_key = start.replace("-", "")
    end_key = end.replace("-", "")
    if use_cache:
        hit = read_cached_l2(code, start_key, end_key)
        if hit is not None and not hit.empty:
            last = pd.Timestamp(hit["date"].max())
            if cache_covers(last, end_key, honor_stamp=False):
                return hit
    if pro is None:
        pro = tushare_pro()
    return _stock_flow(pro, code, start_key, end_key, use_cache=True)


def _stock_flow(pro, code: str, start: str, end: str, use_cache: bool = True) -> pd.DataFrame | None:
    STOCK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = STOCK_CACHE_DIR / f"{code.replace('.', '_')}.csv"
    fetch_end = official_end_key(end)
    cached = None
    fetch_start = start
    if use_cache and path.exists():
        cached = pd.read_csv(path, parse_dates=["date"])
        if not cached.empty:
            last = pd.Timestamp(cached["date"].max())
            if cache_covers(last, end, honor_stamp=False):
                return _slice(cached, start, end)
            fetch_start = (last + timedelta(days=1)).strftime("%Y%m%d")
            if fetch_start > fetch_end:
                return _slice(cached, start, end)

    try:
        flow = pro.moneyflow(ts_code=code, start_date=fetch_start, end_date=fetch_end)
    except Exception:
        return _slice(cached, start, end) if cached is not None else None
    if flow is None or flow.empty or "trade_date" not in flow.columns:
        if cached is not None and not cached.empty:
            return _slice(cached, start, end)
        return None

    basic = _daily_basic(pro, code, fetch_start, fetch_end)
    df = _to_l2_proxy(flow, basic)
    df["ts_code"] = code
    if cached is not None and not cached.empty:
        df = pd.concat([cached, df], ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    cap = pd.Timestamp(fetch_end)
    df = df[pd.to_datetime(df["date"]) <= cap].reset_index(drop=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return _slice(df, start, end)


def _daily_basic(pro, code: str, start: str, end: str) -> pd.DataFrame:
    try:
        basic = pro.daily_basic(
            ts_code=code,
            start_date=start,
            end_date=end,
            fields="trade_date,circ_mv,total_mv",
        )
    except Exception:
        return pd.DataFrame()
    if basic is None or basic.empty:
        return pd.DataFrame()
    basic = basic.rename(columns={"trade_date": "date"})
    basic["date"] = pd.to_datetime(basic["date"])
    return basic


def _to_l2_proxy(flow: pd.DataFrame, basic: pd.DataFrame) -> pd.DataFrame:
    df = flow.copy()
    df["date"] = pd.to_datetime(df["trade_date"])
    for col in (
        "buy_lg_amount",
        "sell_lg_amount",
        "buy_elg_amount",
        "sell_elg_amount",
        "net_mf_amount",
    ):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["主力净额"] = (df["buy_lg_amount"] + df["buy_elg_amount"]) - (df["sell_lg_amount"] + df["sell_elg_amount"])
    df["净流入额"] = df["net_mf_amount"]
    if not basic.empty:
        df = df.merge(basic[["date", "circ_mv"]], on="date", how="left")
    else:
        df["circ_mv"] = pd.NA
    df["circ_mv"] = pd.to_numeric(df["circ_mv"], errors="coerce")
    df["l2jbl"] = df["主力净额"] / df["circ_mv"] * 100
    return df.sort_values("date")[["date", "l2jbl", "主力净额", "净流入额", "circ_mv"]].reset_index(drop=True)


def _aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    g = raw.groupby("date", as_index=False).agg(
        主力净额=("主力净额", "sum"),
        净流入额=("净流入额", "sum"),
        circ_mv=("circ_mv", "sum"),
        n=("主力净额", "size"),
    )
    g["l2jbl"] = g["主力净额"] / g["circ_mv"] * 100
    g.loc[g["circ_mv"].fillna(0) <= 0, "l2jbl"] = pd.NA
    return g.sort_values("date")[["date", "l2jbl", "主力净额", "净流入额"]].reset_index(drop=True)


def _slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    out = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))].copy()
    return out.reset_index(drop=True)


def _shift_month(d: date, delta: int) -> date:
    month = d.month - 1 + delta
    year = d.year + month // 12
    month = month % 12 + 1
    return date(year, month, 1)


def _month_end(d: date) -> date:
    nxt = _shift_month(d, 1)
    return nxt - timedelta(days=1)
