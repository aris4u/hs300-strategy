"""Load OHLC + formula flags for research backtests (no new factors)."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import perf_counter

import pandas as pd

from hs300_strategy.data import fetch_hs300
from hs300_strategy.formula import compute_signals
from hs300_strategy.moneyflow import read_cached_l2
from hs300_strategy.stock_data import STOCK_DIR, fetch_constituents, fetch_many_klines

_WORKER: dict = {}


def load_research_universe(
    start: str = "20100101",
    end: str | None = None,
    use_cache: bool = True,
    with_flow: bool = True,
    limit: int | None = None,
    workers: int | None = None,
) -> dict:
    t0 = perf_counter()
    members = fetch_constituents(use_cache=use_cache)
    if limit:
        members = members.head(limit).copy()
    codes = members["ts_code"].tolist()
    names = dict(zip(members["ts_code"], members["name"]))
    print(f"成分股 {len(codes)} 只（当前名单，存在幸存者偏差）")

    hs300 = fetch_hs300(start=start, end=end, use_cache=True)
    hs300["date"] = pd.to_datetime(hs300["date"])
    hs300 = hs300.sort_values("date").reset_index(drop=True)
    hs300_sig = compute_signals(hs300.copy(), asset="index")
    market_env = hs300_sig.set_index("date")["env_level"]

    klines = fetch_many_klines(codes, start, end or "20260817", use_cache=use_cache)
    ready = [c for c in codes if c in klines and len(klines[c]) >= 160]
    print(f"可用K线 {len(ready)} 只  {perf_counter() - t0:.1f}s")

    n_workers = workers or max(1, min(8, (os.cpu_count() or 4) - 1))
    print(f"计算信号  {n_workers} 进程 …")
    bags: dict[str, pd.DataFrame] = {}
    leak_sample = None
    jobs = [(c, names.get(c, c)) for c in ready]
    done = 0
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(market_env, start, end, with_flow),
    ) as pool:
        futs = [pool.submit(_one, job) for job in jobs]
        for fut in as_completed(futs):
            code, frame, raw = fut.result()
            done += 1
            if frame is not None:
                bags[code] = frame
                if leak_sample is None and raw is not None and len(raw) > 400:
                    leak_sample = raw
            if done % 30 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}  {perf_counter() - t0:.1f}s", flush=True)

    cal = pd.DatetimeIndex(hs300["date"].sort_values().unique())

    def panel(col: str, fill=None) -> pd.DataFrame:
        m = {c: bags[c][col] for c in bags if col in bags[c].columns}
        df = pd.DataFrame(m).reindex(cal)
        if fill is not None:
            df = df.fillna(fill)
        return df

    idx = hs300.set_index("date")
    return {
        "cal": cal,
        "names": names,
        "n_stocks": len(bags),
        "open": panel("open"),
        "high": panel("high"),
        "low": panel("low"),
        "close": panel("close"),
        "launch": panel("launch_turn", 0.0),
        "position": panel("position", 0.0),
        "position_overlay": panel("position_overlay", 0.0),
        "reduce_band": panel("reduce_band", 0.0),
        "reduce_trend": panel("reduce_trend", 0.0),
        "take_profit": panel("take_profit", 0.0),
        "escape": panel("escape_top", 0.0),
        "f_signal": panel("f_signal", 0.0),
        "env": panel("env_level", 0.0),
        "idx_open": idx["open"].astype(float).reindex(cal),
        "idx_high": idx["high"].astype(float).reindex(cal),
        "idx_low": idx["low"].astype(float).reindex(cal),
        "idx_close": idx["close"].astype(float).reindex(cal),
        "leak_sample": leak_sample,
        "elapsed": perf_counter() - t0,
    }


def _init_worker(market_env: pd.Series, start: str, end: str, with_flow: bool) -> None:
    _WORKER["env"] = market_env
    _WORKER["start"] = start
    _WORKER["end"] = end
    _WORKER["with_flow"] = with_flow


def _one(job: tuple[str, str]):
    code, _name = job
    path = STOCK_DIR / f"{code.replace('.', '_')}.csv"
    if not path.exists():
        return code, None, None
    work = pd.read_csv(path, parse_dates=["date"])
    if len(work) < 160:
        return code, None, None
    if _WORKER["with_flow"]:
        flow = read_cached_l2(code, _WORKER["start"], _WORKER["end"] or "20260817")
        if flow is not None and not flow.empty:
            work = work.merge(flow[["date", "l2jbl"]], on="date", how="left")
    work["market_env"] = work["date"].map(_WORKER["env"])
    try:
        sig = compute_signals(work, asset="stock", overlay=False)
        ov = compute_signals(work, asset="stock", overlay=True)
    except Exception:
        return code, None, None
    sig["date"] = pd.to_datetime(sig["date"])
    ov["date"] = pd.to_datetime(ov["date"])
    frame = sig.set_index("date")[
        [
            "open",
            "high",
            "low",
            "close",
            "launch_turn",
            "position",
            "reduce_band",
            "reduce_trend",
            "take_profit",
            "escape_top",
            "f_signal",
            "env_level",
        ]
    ].copy()
    frame["position_overlay"] = ov.set_index("date")["position"]
    return code, frame, work
