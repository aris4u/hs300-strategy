"""Live launch screen: signal-only. No ex-post MFE/quality in decisions.

Event-study holding (fixed N) is for proof stats only — not called \"full strategy\".
Execution: T close signal, T+1 open intended fill.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from time import perf_counter

import numpy as np
import pandas as pd

from hs300_strategy.config import EVENT_PRIMARY_N, HOLD_PERIODS
from hs300_strategy.data import DATA_DIR, fetch_hs300
from hs300_strategy.execution import EXECUTION_NOTE
from hs300_strategy.formula import compute_signals
from hs300_strategy.moneyflow import read_cached_l2
from hs300_strategy.stock_data import STOCK_DIR, fetch_constituents, fetch_many_klines

OUTPUT_DIR = DATA_DIR.parent / "output"
BT_START = "20100701"
LAUNCH_LOOKBACK = 20  # display window for \"in event hold\"; not an optimized N
_WORKER: dict = {}


def run_screen(
    start: str = "20100101",
    end: str | None = None,
    bt_start: str = BT_START,
    use_cache: bool = True,
    with_flow: bool = True,
    with_live_tdx: bool = True,
    limit: int | None = None,
    workers: int | None = None,
) -> dict:
    t0 = perf_counter()
    end = end or date.today().strftime("%Y%m%d")
    members = fetch_constituents(use_cache=use_cache)
    if limit:
        members = members.head(limit).copy()
    codes = members["ts_code"].tolist()
    names = dict(zip(members["ts_code"], members["name"]))
    print(f"成分股 {len(codes)} 只（当前名单，幸存者偏差）")

    hs300 = fetch_hs300(start=start, end=end, use_cache=True)
    hs300["date"] = pd.to_datetime(hs300["date"])
    hs300 = hs300.sort_values("date").reset_index(drop=True)
    market_env = compute_signals(hs300.copy(), asset="index").set_index("date")["env_level"]

    klines = fetch_many_klines(codes, start, end, use_cache=use_cache)
    ready = [c for c in codes if c in klines and len(klines[c]) >= 160]
    print(f"可用K线 {len(ready)} 只")

    n_workers = workers or max(1, min(8, (os.cpu_count() or 4) - 1))
    print(f"计算信号  {n_workers} 进程 …")
    open_map: dict[str, pd.Series] = {}
    close_map: dict[str, pd.Series] = {}
    pos_map: dict[str, pd.Series] = {}
    launch_map: dict[str, pd.Series] = {}
    l2_map: dict[str, pd.Series] = {}
    jobs = [(c, names.get(c, c)) for c in ready]
    done = 0
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(market_env, start, end, with_flow),
    ) as pool:
        futs = [pool.submit(_one, job) for job in jobs]
        for fut in as_completed(futs):
            code, o, c, pos, launch, l2 = fut.result()
            done += 1
            if c is not None:
                open_map[code] = o
                close_map[code] = c
                pos_map[code] = pos
                launch_map[code] = launch
                l2_map[code] = l2
            if done % 30 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}  {perf_counter() - t0:.1f}s", flush=True)

    cal = pd.DatetimeIndex(hs300["date"].sort_values().unique())
    open_px = pd.DataFrame(open_map).reindex(cal)
    close_px = pd.DataFrame(close_map).reindex(cal)
    pos = pd.DataFrame(pos_map).reindex(cal).fillna(0.0)
    launch = pd.DataFrame(launch_map).reindex(cal).fillna(0.0)
    l2 = pd.DataFrame(l2_map).reindex(cal)

    # Live decision: launch in lookback OR state-machine position > 0. No quality/MFE.
    recent = _recent_launch(launch, LAUNCH_LOOKBACK)
    valid = open_px.notna() & close_px.notna()
    in_pool = recent & valid
    watch = (launch.fillna(0) > 0).astype(bool) & valid & ~in_pool

    idx_open = hs300.set_index("date")["open"].astype(float).reindex(cal)
    idx_close = hs300.set_index("date")["close"].astype(float).reindex(cal)
    from hs300_strategy.selection import evaluate_selection, format_selection, save_selection

    # Event-study proof: fixed hold EVENT_PRIMARY_N (standard window), not full strategy.
    event_hold = _recent_launch(launch, EVENT_PRIMARY_N) & valid
    proof = evaluate_selection(
        open_px,
        close_px,
        event_hold.astype(float),
        idx_open,
        idx_close,
        cal,
        bt_start,
        plot_path=OUTPUT_DIR / "screen_selection.png",
    )
    proof["object"] = "event_study_fixed_hold"
    proof["standard_event_window"] = EVENT_PRIMARY_N
    proof["hold_periods_reported"] = list(HOLD_PERIODS)
    proof["screen_rule"] = (
        f"实时推荐只看黄三角启动（近{LAUNCH_LOOKBACK}日），不用事后质量/MFE/MAE。"
        f"检验对象是事件研究：启动后固定持有{EVENT_PRIMARY_N}日（标准事件窗口），"
        "不是完整状态机策略。T日收盘信号，T+1开盘成交。"
        + " " + EXECUTION_NOTE
    )
    save_selection(proof, OUTPUT_DIR / "screen_selection.json")

    today = _today_table(in_pool, watch, close_px, open_px, pos, launch, l2, names, cal)
    if with_live_tdx:
        today = _attach_live(today)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    today.to_csv(OUTPUT_DIR / "screen_today.csv", index=False, encoding="utf-8-sig")
    print(f"总耗时 {perf_counter() - t0:.1f}s")
    return {
        "today": today,
        "proof": proof,
        "proof_text": format_selection(proof),
        "n_pick": int((today["bucket"] == "推荐").sum()) if not today.empty else 0,
        "n_watch": int((today["bucket"] == "观察").sum()) if not today.empty else 0,
    }


def _init_worker(market_env, start, end, with_flow) -> None:
    _WORKER["env"] = market_env
    _WORKER["start"] = start
    _WORKER["end"] = end
    _WORKER["with_flow"] = with_flow


def _one(job: tuple[str, str]):
    code, _name = job
    path = STOCK_DIR / f"{code.replace('.', '_')}.csv"
    if not path.exists():
        return code, None, None, None, None, None
    work = pd.read_csv(path, parse_dates=["date"])
    if len(work) < 160:
        return code, None, None, None, None, None
    if _WORKER["with_flow"]:
        flow = read_cached_l2(code, _WORKER["start"], _WORKER["end"])
        if flow is not None and not flow.empty:
            work = work.merge(flow[["date", "l2jbl"]], on="date", how="left")
    work["market_env"] = work["date"].map(_WORKER["env"])
    try:
        sig = compute_signals(work, asset="stock", overlay=False)
    except Exception:
        return code, None, None, None, None, None
    sig["date"] = pd.to_datetime(sig["date"])
    idx = sig.set_index("date")
    l2 = idx["l2_flow"] if "l2_flow" in idx.columns else pd.Series(np.nan, index=idx.index)
    return (
        code,
        idx["open"].astype(float),
        idx["close"].astype(float),
        idx["position"].astype(float),
        idx["launch_turn"].astype(float),
        l2.astype(float),
    )


def _recent_launch(launch: pd.DataFrame, lookback: int) -> pd.DataFrame:
    hit = (launch.fillna(0) > 0).astype(int)
    return hit.rolling(lookback, min_periods=1).max().fillna(0).astype(bool)


def _today_table(in_pool, watch, close_px, open_px, pos, launch, l2, names, cal) -> pd.DataFrame:
    last = cal[-1]
    rows = []
    for code in close_px.columns:
        rec = bool(in_pool.loc[last, code]) if last in in_pool.index else False
        w = bool(watch.loc[last, code]) if last in watch.index else False
        if not rec and not w:
            continue
        series_l = launch[code].fillna(0)
        hits = series_l.index[series_l > 0]
        signal_date = pd.Timestamp(hits[-1]) if len(hits) else None
        bars_ago = int((cal.get_loc(last) - cal.get_loc(hits[-1]))) if len(hits) else None
        # Intended fill: next open after signal; if signal was earlier, entry already happened.
        entry_date = None
        entry_price = None
        if signal_date is not None:
            si = cal.get_loc(signal_date)
            if isinstance(si, slice):
                si = si.start
            if si + 1 < len(cal):
                entry_date = cal[si + 1]
                ep = open_px.loc[entry_date, code] if code in open_px.columns else np.nan
                entry_price = float(ep) if pd.notna(ep) else None
            else:
                entry_date = None  # pending next open
        rows.append(
            {
                "ts_code": code,
                "name": names.get(code, code),
                "bucket": "推荐" if rec else "观察",
                "signal_date": signal_date.strftime("%Y-%m-%d") if signal_date is not None else "",
                "entry_date": entry_date.strftime("%Y-%m-%d") if entry_date is not None else "待下一交易日开盘",
                "entry_price": entry_price,
                "exit_date": "",
                "exit_price": None,
                "date": last.strftime("%Y-%m-%d"),
                "close": float(close_px.loc[last, code]) if pd.notna(close_px.loc[last, code]) else None,
                "position": float(pos.loc[last, code]),
                "l2jbl": float(l2.loc[last, code]) if pd.notna(l2.loc[last, code]) else None,
                "last_launch": signal_date.strftime("%Y-%m-%d") if signal_date is not None else "",
                "bars_ago": bars_ago,
                "execution": EXECUTION_NOTE,
                "note": "收盘价不是成交价；质量评级不参与本表筛选。",
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = {"推荐": 0, "观察": 1}
    out["ord"] = out["bucket"].map(order)
    return out.sort_values(["ord", "bars_ago", "l2jbl"], ascending=[True, True, False]).drop(columns=["ord"])


def _attach_live(today: pd.DataFrame) -> pd.DataFrame:
    if today.empty:
        return today
    try:
        from hs300_strategy.tdx_l2 import live_quotes, tdx_ready

        if not tdx_ready():
            today["tdx"] = "未找到 C:\\new_tdx"
            return today
        q = live_quotes(today["ts_code"].tolist())
        if q.empty:
            today["tdx"] = "行情服务器无数据"
            return today
        keep = q[["ts_code", "price", "bid1", "ask1", "inner_outer", "inner_outer_ratio"]].copy()
        keep = keep.rename(columns={"price": "tdx_price"})
        today = today.merge(keep, on="ts_code", how="left")
        today["tdx"] = "五档+内外盘"
    except Exception as exc:
        today["tdx"] = f"通达信未接入：{exc}"
    return today
