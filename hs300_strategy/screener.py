"""Scan HS300 constituents: filter weak launches, rank by excess MFE/MAE."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from time import perf_counter, sleep

import numpy as np
import pandas as pd

from hs300_strategy.data import DATA_DIR, fetch_hs300
from hs300_strategy.events import debounce_launches, excursion_row, label_quality
from hs300_strategy.formula import compute_signals
from hs300_strategy.moneyflow import STOCK_CACHE_DIR, fetch_stock_l2, read_cached_l2
from hs300_strategy.stock_data import STOCK_DIR, fetch_constituents, fetch_many_klines

OUTPUT_DIR = DATA_DIR.parent / "output"
CHART_DIR = OUTPUT_DIR / "charts"
RECENT_BARS = 50
MIN_HIST = 4

_WORKER: dict = {}


def run_screener(
    start: str = "20100101",
    end: str | None = None,
    use_cache: bool = True,
    limit: int | None = None,
    with_flow: bool = True,
    plot_top: int = 8,
    workers: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    t0 = perf_counter()
    end = end or date.today().strftime("%Y%m%d")
    members = fetch_constituents(use_cache=use_cache)
    if limit:
        members = members.head(limit).copy()
    codes = members["ts_code"].tolist()
    names = dict(zip(members["ts_code"], members["name"]))
    print(f"成分股 {len(codes)} 只")

    print("沪深300 指数 …")
    hs300 = fetch_hs300(start=start, end=end, use_cache=True)
    hs300_sig = compute_signals(hs300.copy(), asset="index")
    market_env = hs300_sig.set_index("date")["env_level"]

    print("成分股日K …")
    klines = fetch_many_klines(codes, start, end, use_cache=use_cache)
    ready = [c for c in codes if c in klines and len(klines[c]) >= 160]
    print(f"可用K线 {len(ready)} 只  {perf_counter() - t0:.1f}s")

    if with_flow:
        missing = [c for c in ready if not (STOCK_CACHE_DIR / f"{c.replace('.', '_')}.csv").exists()]
        if missing:
            print(f"资金流缺缓存 {len(missing)} 只，联网补齐 …")
            try:
                from hs300_strategy.secrets import tushare_pro

                pro = tushare_pro()
                for i, code in enumerate(missing, start=1):
                    try:
                        fetch_stock_l2(code, start, end, pro=pro, use_cache=False)
                    except Exception:
                        pass
                    sleep(0.08)
                    if i % 20 == 0:
                        print(f"  资金流 {i}/{len(missing)}", flush=True)
            except Exception as exc:
                print(f"资金流客户端不可用，改用量价代理：{exc}")
                with_flow = False
        else:
            print("资金流全部走本地缓存")

    n_workers = workers or max(1, min(8, (os.cpu_count() or 4) - 1))
    print(f"计算信号  {n_workers} 进程 …")
    events_rows: list[dict] = []
    jobs = [(code, names.get(code, code)) for code in ready]
    done = 0
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(hs300, market_env, start, end, with_flow),
    ) as pool:
        futs = [pool.submit(_scan_one, job) for job in jobs]
        for fut in as_completed(futs):
            rows = fut.result()
            if rows:
                events_rows.extend(rows)
            done += 1
            if done % 30 == 0 or done == len(jobs):
                print(f"  信号 {done}/{len(jobs)}  累计启动 {len(events_rows)}  {perf_counter() - t0:.1f}s", flush=True)

    events = pd.DataFrame(events_rows)
    if events.empty:
        raise RuntimeError("没有任何个股打出 launch_turn。")
    events = label_quality(events)
    ranked = rank_stocks(events, hs300)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    events.to_csv(OUTPUT_DIR / "stock_launches.csv", index=False, encoding="utf-8-sig")
    ranked.to_csv(OUTPUT_DIR / "stock_rank.csv", index=False, encoding="utf-8-sig")
    if plot_top:
        _plot_ranked(ranked, klines, names, market_env, start, end, with_flow, plot_top)
    print(f"总耗时 {perf_counter() - t0:.1f}s")
    return events, ranked


def _init_worker(hs300: pd.DataFrame, market_env: pd.Series, start: str, end: str, with_flow: bool) -> None:
    _WORKER["hs300"] = hs300
    _WORKER["env"] = market_env
    _WORKER["start"] = start
    _WORKER["end"] = end
    _WORKER["with_flow"] = with_flow


def _scan_one(job: tuple[str, str]) -> list[dict]:
    code, name = job
    start = _WORKER["start"]
    end = _WORKER["end"]
    path = STOCK_DIR / f"{code.replace('.', '_')}.csv"
    if not path.exists():
        return []
    work = pd.read_csv(path, parse_dates=["date"])
    if len(work) < 160:
        return []
    if _WORKER["with_flow"]:
        flow = read_cached_l2(code, start, end)
        if flow is not None and not flow.empty:
            work = work.merge(flow[["date", "l2jbl"]], on="date", how="left")
    work["market_env"] = work["date"].map(_WORKER["env"])
    try:
        sig = compute_signals(work, asset="stock")
    except Exception:
        return []
    hits = sig.loc[sig["launch_turn"] == 1, "date"]
    if hits.empty:
        return []
    idx = pd.DatetimeIndex(pd.to_datetime(sig["date"]))
    kept = debounce_launches(hits, idx)
    hs300 = _WORKER["hs300"]
    rows = []
    for d in kept:
        row = excursion_row(sig, hs300, d)
        if not row:
            continue
        row["ts_code"] = code
        row["name"] = name
        hit = sig.loc[sig["date"] == d]
        row["l2_flow"] = float(hit["l2_flow"].iloc[0]) if "l2_flow" in hit.columns else np.nan
        row["env_level"] = int(hit["env_level"].iloc[0])
        rows.append(row)
    return rows


def rank_stocks(events: pd.DataFrame, hs300: pd.DataFrame) -> pd.DataFrame:
    """Archive ranking from ex-post quality. NOT a live buy filter.

    recommend_rank / hist_ok / score are research-archive fields only.
    Live screening must use launch_turn + state machine, never these columns.
    """
    from hs300_strategy.config import RATING_KIND

    cal = pd.DatetimeIndex(pd.to_datetime(hs300["date"]).sort_values().unique())
    last_day = cal[-1]
    loc = {d: i for i, d in enumerate(cal)}
    hist = events[events["quality"] != "watching"]
    rows = []
    for code, g in events.groupby("ts_code"):
        h = hist[hist["ts_code"] == code]
        n_hist = len(h)
        n_high = int((h["quality"] == "high_value").sum()) if n_hist else 0
        n_low = int((h["quality"] == "low_value").sum()) if n_hist else 0
        last = g.sort_values("date").iloc[-1]
        last_i = loc.get(pd.Timestamp(last["date"]))
        now_i = loc.get(last_day, len(cal) - 1)
        bars_ago = (now_i - last_i) if last_i is not None else 10**9
        recent = bars_ago <= RECENT_BARS
        med_ex = float(h["excess_mfe_20"].median()) if n_hist else np.nan
        med_mae = float(h["mae_20"].median()) if n_hist else np.nan
        hit = n_high / n_hist if n_hist else np.nan
        last_complete = last.get("bars_20", 0) >= 20
        last_ex = float(last["excess_mfe_20"]) if last_complete and pd.notna(last.get("excess_mfe_20")) else np.nan
        last_eff = float(last["efficiency_20"]) if last_complete and pd.notna(last.get("efficiency_20")) else np.nan
        score_hist = 0.0
        if n_hist >= MIN_HIST and pd.notna(med_ex) and pd.notna(hit):
            score_hist = 0.55 * med_ex + 0.35 * hit + 0.10 * max(med_mae, -0.2)
        score_last = last_ex if pd.notna(last_ex) else score_hist
        score = 0.6 * score_hist + 0.4 * (score_last if pd.notna(score_last) else 0.0)
        proven = n_hist >= MIN_HIST and pd.notna(hit) and (hit >= 0.25 or (pd.notna(med_ex) and med_ex >= 0.05))
        last_junk = last["quality"] == "low_value"
        rows.append(
            {
                "ts_code": code,
                "name": last["name"],
                "last_launch": last["date"].strftime("%Y-%m-%d") if hasattr(last["date"], "strftime") else str(last["date"]),
                "bars_ago": bars_ago,
                "last_quality": last["quality"],
                "last_excess_mfe_20": last_ex,
                "last_mae_20": float(last["mae_20"]) if last_complete and pd.notna(last.get("mae_20")) else np.nan,
                "last_excess_ret_20": float(last["excess_ret_20"]) if last_complete and pd.notna(last.get("excess_ret_20")) else np.nan,
                "last_efficiency_20": last_eff,
                "n_hist": n_hist,
                "n_high": n_high,
                "n_low": n_low,
                "hit_rate": hit,
                "med_excess_mfe_20": med_ex,
                "med_mae_20": med_mae,
                "recent_launch": int(recent),
                "hist_ok": int(proven),
                "last_is_low": int(last_junk),
                "score": score,
                "l2_flow": last.get("l2_flow", np.nan),
                "env_level": last.get("env_level", np.nan),
                "rating_kind": RATING_KIND,
                "live_filter": 0,
                "archive_only": 1,
                "note": "事后主观评级档案；禁止进入实时筛选与生产决策。",
            }
        )
    ranked = pd.DataFrame(rows)
    # Archive sort only — do NOT invent recommend_rank as a live buy list.
    ranked["recommend_rank"] = np.nan
    return ranked.sort_values(["recent_launch", "bars_ago", "score"], ascending=[False, True, False])


def _plot_ranked(
    ranked: pd.DataFrame,
    klines: dict[str, pd.DataFrame],
    names: dict[str, str],
    market_env: pd.Series,
    start: str,
    end: str,
    with_flow: bool,
    n: int,
) -> None:
    from hs300_strategy.charts import LOOKBACK_BARS, plot_kline_signals
    from hs300_strategy.events import QUALITY_CN

    rec = ranked[ranked["recent_launch"] == 1].head(n)
    if rec.empty:
        rec = ranked.sort_values("bars_ago").head(n)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    for _, row in rec.iterrows():
        code = row["ts_code"]
        k = klines.get(code)
        if k is None or k.empty:
            continue
        work = k.copy()
        work["date"] = pd.to_datetime(work["date"])
        work["market_env"] = work["date"].map(market_env)
        if with_flow:
            flow = read_cached_l2(code, start, end)
            if flow is not None and not flow.empty:
                work = work.merge(flow[["date", "l2jbl"]], on="date", how="left")
        try:
            sig = compute_signals(work, asset="stock")
            q = QUALITY_CN.get(str(row["last_quality"]), row["last_quality"])
            title = f"{names.get(code, '')}  {code}    最近启动 {row['last_launch']}    {q}"
            path = CHART_DIR / f"{code.replace('.', '_')}.png"
            plot_kline_signals(sig, path, title=title, bars=LOOKBACK_BARS, ts_code=code)
            print(f"  K线图 {path}")
        except Exception as exc:
            print(f"  {code} 画图失败：{exc}")
