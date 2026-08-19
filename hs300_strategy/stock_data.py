"""沪深300 成分股日K（前复权）与成分列表。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from hs300_strategy.data import (
    DATA_DIR,
    _normalize,
    _ymd,
    cache_covers,
    clip_official_bars,
    disable_http_proxy,
    official_end_key,
    read_csv_cached,
)

STOCK_DIR = DATA_DIR / "stocks"
CONSTITUENT_FILE = DATA_DIR / "hs300_members.csv"
INDUSTRY_FILE = DATA_DIR / "hs300_industry.csv"


def to_bao(ts_code: str) -> str:
    num, mkt = ts_code.split(".")
    return f"{mkt.lower()}.{num}"


def to_ts(bao_code: str) -> str:
    mkt, num = bao_code.split(".")
    return f"{num}.{mkt.upper()}"


def fetch_constituents(use_cache: bool = True) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if use_cache and CONSTITUENT_FILE.exists():
        return read_csv_cached(CONSTITUENT_FILE).copy()
    disable_http_proxy()
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(lg.error_msg)
    try:
        rs = bs.query_hs300_stocks()
        if rs.error_code != "0":
            raise RuntimeError(rs.error_msg)
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        raw = pd.DataFrame(rows, columns=rs.fields)
    finally:
        bs.logout()
    out = pd.DataFrame(
        {
            "ts_code": [to_ts(c) for c in raw["code"]],
            "name": raw["code_name"],
            "bao_code": raw["code"],
        }
    )
    out.to_csv(CONSTITUENT_FILE, index=False, encoding="utf-8-sig")
    return out


def fetch_industries(codes: list[str] | None = None, use_cache: bool = True) -> pd.DataFrame:
    """BaoStock 行业分类，缓存到 data/hs300_industry.csv。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if use_cache and INDUSTRY_FILE.exists():
        cached = pd.read_csv(INDUSTRY_FILE)
        if not codes:
            return cached
        have = set(cached["ts_code"].astype(str))
        if set(codes) <= have:
            return cached[cached["ts_code"].isin(codes)].reset_index(drop=True)
    disable_http_proxy()
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(lg.error_msg)
    try:
        rs = bs.query_stock_industry()
        if rs.error_code != "0":
            raise RuntimeError(rs.error_msg)
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        raw = pd.DataFrame(rows, columns=rs.fields)
    finally:
        bs.logout()
    out = pd.DataFrame(
        {
            "ts_code": [to_ts(str(c)) for c in raw["code"]],
            "name": raw["code_name"],
            "industry": raw["industry"].astype(str),
        }
    )
    if codes:
        out = out[out["ts_code"].isin(codes)].copy()
    out.to_csv(INDUSTRY_FILE, index=False, encoding="utf-8-sig")
    return out.reset_index(drop=True)


def fetch_stock_kline(ts_code: str, start: str = "20100101", end: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    STOCK_DIR.mkdir(parents=True, exist_ok=True)
    end = end or date.today().strftime("%Y%m%d")
    start_key = start.replace("-", "")
    end_key = end.replace("-", "")
    fetch_end = official_end_key(end_key)
    path = STOCK_DIR / f"{ts_code.replace('.', '_')}.csv"
    if use_cache and path.exists():
        cached = read_csv_cached(path, parse_dates=["date"])
        if not cached.empty:
            last = pd.Timestamp(cached["date"].max())
            if cache_covers(last, end_key):
                return cached[(cached["date"] >= pd.Timestamp(start_key)) & (cached["date"] <= pd.Timestamp(end_key))].reset_index(drop=True)

    disable_http_proxy()
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(lg.error_msg)
    try:
        rs = bs.query_history_k_data_plus(
            to_bao(ts_code),
            "date,open,high,low,close,volume",
            start_date=_ymd(start_key),
            end_date=_ymd(fetch_end),
            frequency="d",
            adjustflag="2",
        )
        if rs.error_code != "0":
            raise RuntimeError(rs.error_msg)
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        raw = pd.DataFrame(rows, columns=rs.fields)
    finally:
        bs.logout()
    df = clip_official_bars(_normalize(raw), end_key)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df[(df["date"] >= pd.Timestamp(start_key)) & (df["date"] <= pd.Timestamp(end_key))].reset_index(drop=True)


def fetch_many_klines(codes: list[str], start: str, end: str, use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """一次登录批量拉 K 线。缺了已收盘交易日才重拉；前复权全量覆盖以保证除权后历史价正确。"""
    STOCK_DIR.mkdir(parents=True, exist_ok=True)
    start_key = start.replace("-", "")
    end_key = end.replace("-", "")
    fetch_end = official_end_key(end_key)
    out: dict[str, pd.DataFrame] = {}
    need: list[str] = []
    for code in codes:
        path = STOCK_DIR / f"{code.replace('.', '_')}.csv"
        if use_cache and path.exists():
            cached = read_csv_cached(path, parse_dates=["date"])
            if cached.empty:
                need.append(code)
                continue
            last = pd.Timestamp(cached["date"].max())
            if cache_covers(last, end_key):
                out[code] = cached[
                    (cached["date"] >= pd.Timestamp(start_key)) & (cached["date"] <= pd.Timestamp(end_key))
                ].reset_index(drop=True)
                continue
        need.append(code)
    if not need:
        return out

    disable_http_proxy()
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(lg.error_msg)
    try:
        for i, code in enumerate(need, start=1):
            rs = bs.query_history_k_data_plus(
                to_bao(code),
                "date,open,high,low,close,volume",
                start_date=_ymd(start_key),
                end_date=_ymd(fetch_end),
                frequency="d",
                adjustflag="2",
            )
            if rs.error_code != "0":
                print(f"  {code} K线失败：{rs.error_msg}")
                continue
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                continue
            df = clip_official_bars(_normalize(pd.DataFrame(rows, columns=rs.fields)), end_key)
            path = STOCK_DIR / f"{code.replace('.', '_')}.csv"
            df.to_csv(path, index=False, encoding="utf-8-sig")
            out[code] = df[
                (df["date"] >= pd.Timestamp(start_key)) & (df["date"] <= pd.Timestamp(end_key))
            ].reset_index(drop=True)
            if i % 10 == 0 or i == len(need):
                print(f"  K线 {i}/{len(need)}", flush=True)
    finally:
        bs.logout()
    return out
