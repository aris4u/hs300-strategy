"""拉取沪深300日K并缓存。

本机 Windows 用户代理是 127.0.0.1:17891（Clash/V2Ray 一类）。
AKShare 默认走东财：走这个代理会 ProxyError，直连也会被 RST。
BaoStock 用自己的端口协议，腾讯财经可直连，都不依赖东财。
"""

from __future__ import annotations

import json
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_FILE = DATA_DIR / "hs300.csv"
STAMP_FILE = DATA_DIR / "kline_fetch_stamp.json"
INDEX_SYMBOL = "000300"
_CSV_MEM: dict[str, tuple[float, pd.DataFrame]] = {}

# Official daily bars are requested only after this clock (vendor lag after 15:00 close).
BAR_READY = dtime(15, 20)
# If today's bar still missing by this time, treat it as holiday / vendor empty.
VENDOR_GIVE_UP = dtime(16, 30)


def now_cn() -> datetime:
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("Asia/Shanghai"))
    except Exception:
        return datetime.utcnow() + timedelta(hours=8)


def last_closed_session(now: datetime | None = None) -> date:
    """Last session whose daily bar may be written to CSV (not a forming intraday bar)."""
    now = now or now_cn()
    d = now.date()
    if now.weekday() >= 5:
        return d - timedelta(days=now.weekday() - 4)
    if now.time() < BAR_READY:
        d = d - timedelta(days=1)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        return d
    return d


def official_end_key(end: str | None = None) -> str:
    end_key = (end or date.today().strftime("%Y%m%d")).replace("-", "")
    return min(end_key, last_closed_session().strftime("%Y%m%d"))


def vendor_likely_done(now: datetime | None = None) -> bool:
    now = now or now_cn()
    if now.weekday() >= 5:
        return True
    return now.time() >= VENDOR_GIVE_UP


def write_kline_stamp(closed_key: str | None = None) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "closed": closed_key or official_end_key(),
        "at": now_cn().strftime("%Y-%m-%d %H:%M:%S"),
    }
    STAMP_FILE.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def clear_kline_stamp() -> None:
    if STAMP_FILE.exists():
        STAMP_FILE.unlink()


def stamp_covers(end: str | None = None) -> bool:
    if not STAMP_FILE.exists():
        return False
    try:
        payload = json.loads(STAMP_FILE.read_text(encoding="utf-8"))
    except Exception:
        return False
    return str(payload.get("closed") or "") >= official_end_key(end)


def cache_covers(last, end: str | None = None, *, honor_stamp: bool = True) -> bool:
    """True if cache already has the last official session (or holiday stamp)."""
    if last is None:
        return False
    try:
        if pd.isna(last):
            return False
    except (TypeError, ValueError):
        pass
    need = official_end_key(end)
    if pd.Timestamp(last).strftime("%Y%m%d") >= need:
        return True
    return bool(honor_stamp and stamp_covers(end))


def clip_official_bars(df: pd.DataFrame, end: str | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    cap = pd.Timestamp(official_end_key(end))
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out[out["date"] <= cap]
    return out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def cache_asof_date() -> date | None:
    if not CACHE_FILE.exists():
        return None
    dates = pd.read_csv(CACHE_FILE, usecols=["date"])
    if dates.empty:
        return None
    return pd.to_datetime(dates["date"]).max().date()


def read_csv_cached(path: Path, parse_dates=None) -> pd.DataFrame:
    """Reread a CSV only when its mtime changes. Callers must not mutate the frame."""
    key = str(path)
    mtime = path.stat().st_mtime
    hit = _CSV_MEM.get(key)
    if hit and hit[0] == mtime:
        return hit[1]
    df = pd.read_csv(path, parse_dates=parse_dates)
    _CSV_MEM[key] = (mtime, df)
    return df


def disable_http_proxy() -> None:
    """忽略环境变量和 Windows 注册表里的系统代理。"""
    import os
    import urllib.request

    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    urllib.request.getproxies = lambda: {}  # type: ignore[method-assign]
    urllib.request.getproxies_environment = lambda: {}  # type: ignore[method-assign]
    if hasattr(urllib.request, "getproxies_registry"):
        urllib.request.getproxies_registry = lambda: {}  # type: ignore[method-assign]


def _http_session():
    import requests

    disable_http_proxy()
    session = requests.Session()
    session.trust_env = False
    session.proxies = {"http": None, "https": None}
    return session


def fetch_hs300(start: str = "20100101", end: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    """返回按日期升序、列为 open/high/low/close/volume 的日K。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    end = end or date.today().strftime("%Y%m%d")
    start_key = start.replace("-", "")
    end_key = end.replace("-", "")
    fetch_end = official_end_key(end_key)

    if use_cache and CACHE_FILE.exists():
        cached = read_csv_cached(CACHE_FILE, parse_dates=["date"])
        cached = cached.sort_values("date")
        if not cached.empty:
            last = pd.Timestamp(cached["date"].max())
            if cache_covers(last, end_key):
                return _slice(cached, start_key, end_key)

    disable_http_proxy()
    errors: list[str] = []
    df = None
    for name, loader in (
        ("baostock", _from_baostock),
        ("tencent", _from_tencent),
        ("akshare", _from_akshare),
    ):
        try:
            df = loader(start_key, fetch_end)
            if df is not None and not df.empty:
                print(f"数据来源：{name}")
                break
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            df = None

    if df is None or df.empty:
        raise RuntimeError("无法获取沪深300数据。\n" + "\n".join(errors))

    df = clip_official_bars(df, end_key)
    df.to_csv(CACHE_FILE, index=False, encoding="utf-8-sig")
    return _slice(df, start_key, end_key)


def _from_baostock(start: str, end: str) -> pd.DataFrame:
    import baostock as bs

    start_d = _ymd(start)
    end_d = _ymd(end)
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(lg.error_msg)
    try:
        rs = bs.query_history_k_data_plus(
            "sh.000300",
            "date,open,high,low,close,volume",
            start_date=start_d,
            end_date=end_d,
            frequency="d",
            adjustflag="3",
        )
        if rs.error_code != "0":
            raise RuntimeError(rs.error_msg)
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        raw = pd.DataFrame(rows, columns=rs.fields)
    finally:
        bs.logout()
    return _normalize(raw)


def _from_tencent(start: str, end: str) -> pd.DataFrame:
    """按年切片拉取，避开东财。"""
    session = _http_session()
    frames = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        beg = f"{year}-01-01"
        stop = f"{year}-12-31"
        url = (
            "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
            f"?param=sh000300,day,{beg},{stop},320,"
        )
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        days = payload.get("data", {}).get("sh000300", {}).get("day") or []
        if days:
            part = pd.DataFrame(days, columns=["date", "open", "close", "high", "low", "volume"][: len(days[0])])
            frames.append(part)
    if not frames:
        raise RuntimeError("腾讯接口无数据")
    raw = pd.concat(frames, ignore_index=True)
    return _normalize(raw)


def _from_akshare(start: str, end: str) -> pd.DataFrame:
    """最后兜底。东财在本机直连和代理下都不通，一般走不到这里。"""
    disable_http_proxy()
    import akshare as ak

    raw = ak.index_zh_a_hist(symbol=INDEX_SYMBOL, period="daily", start_date=start, end_date=end)
    if raw is None or raw.empty:
        raise RuntimeError("AKShare 返回空表")
    return _normalize(raw)


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    }
    df = raw.rename(columns=rename)
    missing = [c for c in ("date", "open", "high", "low", "close", "volume") if c not in df.columns]
    if missing:
        raise ValueError(f"指数数据缺列: {missing}；实际列={list(raw.columns)}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)


def _slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    out = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))].copy()
    return out.reset_index(drop=True)


def _ymd(yyyymmdd: str) -> str:
    s = yyyymmdd.replace("-", "")
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
