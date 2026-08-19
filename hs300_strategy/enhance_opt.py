"""Index-enhancement sleeve grid. Does not overwrite output/enhance_*.

Baseline stays in output/enhance_opt/baseline/.
Ranking uses T-day fields only. Fill at T+1 open.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from time import perf_counter

import numpy as np
import pandas as pd

from hs300_strategy.config import (
    ENHANCE_HEAT_BARS,
    ENHANCE_HEAT_SCALE,
    ENHANCE_HEAT_TH,
    ENHANCE_WINDOW_START,
    OUTPUT_DIR,
    STOCK_FORMULA,
)
from hs300_strategy.data import fetch_hs300
from hs300_strategy.formula import compute_signals
from hs300_strategy.moneyflow import read_cached_l2
from hs300_strategy.stock_data import STOCK_DIR, fetch_constituents, fetch_many_klines
from hs300_strategy.strategy_backtest import enhance_blend, portfolio_from_position, reconstruct_positions

OPT_DIR = OUTPUT_DIR / "enhance_opt"
BASELINE_DIR = OPT_DIR / "baseline"
CACHE_PATH = OPT_DIR / "universe_v1.pkl"
_WORKER: dict = {}

FACTORS = [
    ("fzqd", "FZQD"),
    ("accum_score", "吸筹强度"),
    ("l2_flow", "L2JBL"),
    ("trend_score", "启动强度"),
    ("live_chip", "活筹强度"),
    ("env_score", "环境评分"),
]
PANEL_COLS = [
    "open",
    "close",
    "launch_turn",
    "position",
    "reduce_band",
    "reduce_trend",
    "escape_top",
    "env_level",
    "fzqd",
    "accum_score",
    "l2_flow",
    "trend_score",
    "live_chip",
    "env_score",
]
TOPNS = [5, 10, 15, 20, 30, None]


def snapshot_baseline() -> None:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    for name in (
        "enhance_metrics.json",
        "enhance_equity.csv",
        "enhance_monthly.csv",
        "enhance.png",
        "enhance_trades.csv",
        "selection.png",
        "selection.json",
    ):
        src = OUTPUT_DIR / name
        if src.exists():
            shutil.copy2(src, BASELINE_DIR / name)


def load_universe(use_cache: bool = True, workers: int | None = None) -> dict:
    if use_cache and CACHE_PATH.exists():
        print(f"读取缓存 {CACHE_PATH}")
        return pickle.loads(CACHE_PATH.read_bytes())
    t0 = perf_counter()
    members = fetch_constituents(use_cache=True)
    codes = members["ts_code"].tolist()
    names = dict(zip(members["ts_code"], members["name"]))
    end = date.today().strftime("%Y%m%d")
    hs300 = fetch_hs300(start="20100101", end=end, use_cache=True)
    hs300["date"] = pd.to_datetime(hs300["date"])
    hs300 = hs300.sort_values("date").reset_index(drop=True)
    hs300_sig = compute_signals(hs300.copy(), asset="index")
    market_env = hs300_sig.set_index("date")["env_level"]
    fetch_many_klines(codes, "20100101", end, use_cache=True)
    ready = [c for c in codes if (STOCK_DIR / f"{c.replace('.', '_')}.csv").exists()]
    n_workers = workers or max(1, min(8, (os.cpu_count() or 4) - 1))
    print(f"计算 overlay 信号 {len(ready)} 只  {n_workers} 进程")
    bags: dict[str, pd.DataFrame] = {}
    jobs = [(c, names.get(c, c)) for c in ready]
    done = 0
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(market_env, "20100101", end, True),
    ) as pool:
        futs = [pool.submit(_one, job) for job in jobs]
        for fut in as_completed(futs):
            code, frame = fut.result()
            done += 1
            if frame is not None:
                bags[code] = frame
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
    uni = {
        "cal": cal,
        "names": names,
        "n_stocks": len(bags),
        "open": panel("open"),
        "close": panel("close"),
        "launch": panel("launch_turn", 0.0),
        "position_overlay": panel("position", 0.0),
        "reduce_band": panel("reduce_band", 0.0),
        "reduce_trend": panel("reduce_trend", 0.0),
        "escape": panel("escape_top", 0.0),
        "env": panel("env_level", 0.0),
        "fzqd": panel("fzqd"),
        "accum_score": panel("accum_score"),
        "l2_flow": panel("l2_flow"),
        "trend_score": panel("trend_score"),
        "live_chip": panel("live_chip"),
        "env_score": panel("env_score"),
        "idx_open": idx["open"].astype(float).reindex(cal),
        "idx_close": idx["close"].astype(float).reindex(cal),
    }
    OPT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_bytes(pickle.dumps(uni, protocol=pickle.HIGHEST_PROTOCOL))
    print(f"已缓存 {CACHE_PATH}  {perf_counter() - t0:.1f}s")
    return uni


def _init_worker(market_env: pd.Series, start: str, end: str, with_flow: bool) -> None:
    _WORKER["env"] = market_env
    _WORKER["start"] = start
    _WORKER["end"] = end
    _WORKER["with_flow"] = with_flow


def _one(job: tuple[str, str]):
    code, _name = job
    path = STOCK_DIR / f"{code.replace('.', '_')}.csv"
    if not path.exists():
        return code, None
    work = pd.read_csv(path, parse_dates=["date"])
    if len(work) < 160:
        return code, None
    if _WORKER["with_flow"]:
        flow = read_cached_l2(code, _WORKER["start"], _WORKER["end"] or "20260817")
        if flow is not None and not flow.empty:
            work = work.merge(flow[["date", "l2jbl"]], on="date", how="left")
    work["market_env"] = work["date"].map(_WORKER["env"])
    try:
        sig = compute_signals(work, asset="stock", overlay=True)
    except Exception:
        return code, None
    sig["date"] = pd.to_datetime(sig["date"])
    missing = [c for c in PANEL_COLS if c not in sig.columns]
    if missing:
        return code, None
    return code, sig.set_index("date")[PANEL_COLS].copy()


def cs_z(df: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    x = df.where(mask)
    mu = x.mean(axis=1)
    sd = x.std(axis=1, ddof=0).replace(0, np.nan)
    return x.sub(mu, axis=0).div(sd, axis=0)


def composite_score(uni: dict, mask: pd.DataFrame) -> pd.DataFrame:
    zs = []
    for col, _ in FACTORS:
        zs.append(cs_z(uni[col], mask))
    stacked = np.nanmean(np.stack([z.to_numpy(dtype=float) for z in zs], axis=0), axis=0)
    return pd.DataFrame(stacked, index=mask.index, columns=mask.columns).fillna(0.0)


def select_topn(pos: pd.DataFrame, score: pd.DataFrame, n: int | None) -> pd.DataFrame:
    cand = pos > 1e-12
    if n is None:
        return cand.astype(float)
    sc = score.where(cand)
    rnk = sc.rank(axis=1, ascending=False, method="first")
    return ((rnk <= float(n)) & cand).astype(float)


def pos_ct_only(launch: pd.DataFrame, escape: pd.DataFrame) -> pd.DataFrame:
    la = (launch.fillna(0).to_numpy() > 0).astype(np.int8)
    es = (escape.fillna(0).to_numpy() > 0).astype(np.int8)
    n, m = la.shape
    out = np.zeros((n, m), dtype=float)
    pos = np.zeros(m, dtype=float)
    for i in range(n):
        pos = np.where(la[i] > 0, 1.0, pos)
        pos = np.where(es[i] > 0, 0.0, pos)
        out[i] = pos
    return pd.DataFrame(out, index=launch.index, columns=launch.columns)


def pos_fixed_hold(launch: pd.DataFrame, hold_days: int) -> pd.DataFrame:
    la = (launch.fillna(0).to_numpy() > 0).astype(np.int8)
    n, m = la.shape
    out = np.zeros((n, m), dtype=float)
    last = np.full(m, -10**9, dtype=np.int32)
    for i in range(n):
        last = np.where(la[i] > 0, i, last)
        out[i] = ((i - last) < hold_days) & (last >= 0)
    return pd.DataFrame(out.astype(float), index=launch.index, columns=launch.columns)


def pos_no_flag(uni: dict, which: str) -> pd.DataFrame:
    band = uni["reduce_band"]
    trend = uni["reduce_trend"]
    zeros = band * 0.0
    tp = zeros
    if which == "no_trend":
        trend = zeros
    elif which == "no_band":
        band = zeros
    return reconstruct_positions(
        uni["launch"],
        band,
        trend,
        tp,
        uni["escape"],
        uni["env"],
        mode="state_machine",
        band_level=STOCK_FORMULA.overlay_pos_band,
        trend_level=STOCK_FORMULA.overlay_pos_trend,
    )


def eval_sleeve(
    pos: pd.DataFrame,
    uni: dict,
    *,
    satellite: float,
    use_heat: bool,
    bt_start: str = ENHANCE_WINDOW_START,
) -> dict:
    idx_close = uni["idx_close"]
    sat_daily = portfolio_from_position(pos, uni["open"], uni["close"], uni["idx_open"], idx_close)
    idx_ret = sat_daily["hs300_ret"]
    heat = None
    if use_heat:
        idx_5 = idx_close.pct_change(ENHANCE_HEAT_BARS).shift(1)
        heat = (idx_5 > ENHANCE_HEAT_TH).fillna(False)
    blended = enhance_blend(sat_daily, idx_ret, satellite, heat, ENHANCE_HEAT_SCALE)
    mask = blended.index >= pd.Timestamp(bt_start)
    sample = blended.loc[mask].copy()
    if sample.empty:
        raise RuntimeError("empty sample")
    for col in ("gross_ret", "net_ret", "hs300_ret", "turnover"):
        if col in sample.columns:
            sample.iloc[0, sample.columns.get_loc(col)] = 0.0
    sample["nav"] = (1 + sample["net_ret"]).cumprod()
    sample["bench"] = (1 + sample["hs300_ret"]).cumprod()
    total = float(sample["nav"].iloc[-1] / sample["nav"].iloc[0] - 1)
    bh = float(sample["bench"].iloc[-1] / sample["bench"].iloc[0] - 1)
    dd = float((sample["nav"] / sample["nav"].cummax() - 1).min())
    n_hold = sample["n_hold"].astype(float)
    live = n_hold > 0
    heat_days = 0
    if use_heat and heat is not None:
        heat_days = int(heat.reindex(sample.index).fillna(False).astype(bool).sum())
    return {
        "start": sample.index[0].strftime("%Y-%m-%d"),
        "end": sample.index[-1].strftime("%Y-%m-%d"),
        "days": int(len(sample)),
        "combo": total,
        "hs300": bh,
        "excess_add": total - bh,
        "max_dd": dd,
        "avg_hold": float(n_hold.mean()),
        "avg_hold_active": float(n_hold.loc[live].mean()) if live.any() else 0.0,
        "heat_days": heat_days,
        "satellite": satellite,
        "use_heat": use_heat,
    }


def _row(
    version: str,
    core: str,
    sat: str,
    n_label: str,
    pick: str,
    exit_rule: str,
    m: dict,
) -> dict:
    return {
        "version": version,
        "core": core,
        "sat": sat,
        "n_hold_rule": n_label,
        "pick": pick,
        "exit": exit_rule,
        "combo": m["combo"],
        "hs300": m["hs300"],
        "excess_add": m["excess_add"],
        "max_dd": m["max_dd"],
        "avg_hold_active": m["avg_hold_active"],
        "avg_hold": m["avg_hold"],
    }


def n_label(n: int | None) -> str:
    return "全部启动" if n is None else f"Top{n}启动"


def run_grid(uni: dict) -> pd.DataFrame:
    overlay = uni["position_overlay"].fillna(0.0)
    zeros = overlay * 0.0
    cand = overlay > 1e-12
    score = composite_score(uni, cand)
    rows: list[dict] = []

    frozen = json.loads((BASELINE_DIR / "enhance_metrics.json").read_text(encoding="utf-8"))
    rows.append(
        {
            "version": "Baseline",
            "core": "70%",
            "sat": "30%",
            "n_hold_rule": "全部启动",
            "pick": "全部启动(原仓位权重)",
            "exit": "当前退出",
            "combo": float(frozen["total_return"]),
            "hs300": float(frozen["benchmark_return"]),
            "excess_add": float(frozen["excess_additive"]),
            "max_dd": float(frozen["max_drawdown"]),
            "avg_hold_active": float(frozen["avg_holdings_when_active"]),
            "avg_hold": float(frozen["avg_holdings"]),
        }
    )

    print("方向一/二：综合分 Top N（当前退出）")
    for n in TOPNS:
        pos = select_topn(overlay, score, n)
        m = eval_sleeve(pos, uni, satellite=0.30, use_heat=True)
        rows.append(_row(n_label(n), "70%", "30%", n_label(n), "综合分等权", "当前退出", m))
        print(
            f"  {n_label(n):8s}  combo={m['combo']:+.2%}  hs300={m['hs300']:+.2%}  "
            f"diff={m['excess_add']:+.2%}  dd={m['max_dd']:.2%}  n={m['avg_hold_active']:.1f}"
        )

    print("方向一：单因子排序 Top N（当前退出）")
    for col, label in FACTORS:
        for n in TOPNS:
            pos = select_topn(overlay, uni[col], n)
            m = eval_sleeve(pos, uni, satellite=0.30, use_heat=True)
            rows.append(
                _row(f"{label}-{n_label(n)}", "70%", "30%", n_label(n), f"按{label}排序", "当前退出", m)
            )

    books = {
        "A_current": (overlay, "当前退出"),
        "B_no_jctrend": (pos_no_flag(uni, "no_trend"), "去掉JCTREND减仓"),
        "C_no_jcband": (pos_no_flag(uni, "no_band"), "去掉JCBAND减仓"),
        "E_ct_only": (pos_ct_only(uni["launch"], uni["escape"]), "只保留CT退出"),
        "F_hold20": (pos_fixed_hold(uni["launch"], 20), "启动后固定20日"),
        "G_hold30": (pos_fixed_hold(uni["launch"], 30), "启动后固定30日"),
    }

    print("方向三：退出规则（全部 + 综合Top10/20）")
    for book_key, (book, exit_name) in books.items():
        for n, ver_prefix in ((None, "全部"), (10, "Top10"), (20, "Top20")):
            mask = book > 1e-12
            sc = composite_score(uni, mask)
            pos = select_topn(book, sc, n)
            m = eval_sleeve(pos, uni, satellite=0.30, use_heat=True)
            rows.append(
                _row(
                    f"{ver_prefix}+{exit_name}",
                    "70%",
                    "30%",
                    n_label(n),
                    "综合分等权" if n is not None else "全部启动",
                    exit_name,
                    m,
                )
            )
            print(
                f"  {ver_prefix:4s} {exit_name:16s}  diff={m['excess_add']:+.2%}  "
                f"combo={m['combo']:+.2%}  dd={m['max_dd']:.2%}  n={m['avg_hold_active']:.1f}"
            )
        if book_key == "A_current":
            m = eval_sleeve(overlay, uni, satellite=0.30, use_heat=False)
            rows.append(_row("全部+去掉热度降仓", "70%", "30%", "全部启动", "全部启动", "去掉指数5日>4%降仓", m))
            print(
                f"  D heat-off          diff={m['excess_add']:+.2%}  combo={m['combo']:+.2%}  "
                f"dd={m['max_dd']:.2%}"
            )
            for n in (10, 20):
                pos = select_topn(overlay, score, n)
                m = eval_sleeve(pos, uni, satellite=0.30, use_heat=False)
                rows.append(
                    _row(f"Top{n}+去掉热度降仓", "70%", "30%", f"Top{n}启动", "综合分等权", "去掉指数5日>4%降仓", m)
                )

    df = pd.DataFrame(rows)
    print("方向四：70/30 75/25 80/20（在已转正或最接近的选股上）")
    candidates = df.sort_values("excess_add", ascending=False)
    seeds: list[tuple[str, pd.DataFrame, int | None, bool, str, str]] = []
    pos_ok = df[df["excess_add"] > 0].copy()
    if not pos_ok.empty:
        best = pos_ok.sort_values(["excess_add"], ascending=False).iloc[0]
        print(f"  已有正超额，用最优选股再测权重：{best['version']}")
    else:
        best = candidates.iloc[0]
        print(f"  尚未转正，用最接近的选股再测权重：{best['version']}")

    # Rebuild a few high-value books for weight grid: composite top10/20 current, and best exit.
    weight_books = [
        ("Top10", select_topn(overlay, score, 10), 10, True, "综合分等权", "当前退出"),
        ("Top20", select_topn(overlay, score, 20), 20, True, "综合分等权", "当前退出"),
        ("Top10无热度", select_topn(overlay, score, 10), 10, False, "综合分等权", "去掉指数5日>4%降仓"),
        ("Top10+CT", select_topn(pos_ct_only(uni["launch"], uni["escape"]), composite_score(uni, pos_ct_only(uni["launch"], uni["escape"]) > 1e-12), 10), 10, True, "综合分等权", "只保留CT退出"),
        ("Top10+20日", select_topn(pos_fixed_hold(uni["launch"], 20), composite_score(uni, pos_fixed_hold(uni["launch"], 20) > 1e-12), 10), 10, True, "综合分等权", "启动后固定20日"),
        ("全部无热度", overlay, None, False, "全部启动", "去掉指数5日>4%降仓"),
    ]
    for name, pos, n, heat, pick, exit_name in weight_books:
        for sat, core_s, sat_s in ((0.30, "70%", "30%"), (0.25, "75%", "25%"), (0.20, "80%", "20%")):
            if sat == 0.30 and name in ("Top10", "Top20", "Top10无热度", "全部无热度"):
                continue
            m = eval_sleeve(pos, uni, satellite=sat, use_heat=heat)
            rows.append(_row(f"{name} {core_s}/{sat_s[0:2]}", core_s, sat_s, n_label(n), pick, exit_name, m))
            print(
                f"  {name:12s} {core_s}/{sat_s}  diff={m['excess_add']:+.2%}  "
                f"combo={m['combo']:+.2%}  dd={m['max_dd']:.2%}"
            )

    # extra CT/fixed 70/30 already in rows; keep
    _ = zeros
    return pd.DataFrame(rows)


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def pick_winner(df: pd.DataFrame) -> pd.Series | None:
    dd_floor = -0.1483 - 0.025  # do not worsen DD by more than ~2.5pp
    ok = df[(df["excess_add"] > 0) & (df["max_dd"] >= dd_floor) & (df["version"] != "Baseline")].copy()
    if ok.empty:
        ok = df[(df["excess_add"] > 0) & (df["version"] != "Baseline")].copy()
    if ok.empty:
        return None
    moderate = ok[ok["avg_hold_active"] <= 40]
    if not moderate.empty:
        ok = moderate

    def simplicity(r: pd.Series) -> tuple:
        pick_pen = 0 if r["pick"] in ("按环境评分排序", "综合分等权", "全部启动(原仓位权重)", "全部启动") else 1
        n_pen = {"全部启动": 3, "Top30启动": 2, "Top20启动": 0, "Top15启动": 0, "Top10启动": 0, "Top5启动": 0}.get(
            r["n_hold_rule"], 2
        )
        if float(r["avg_hold_active"]) > 80:
            n_pen += 10
        exit_pen = {
            "当前退出": 0,
            "去掉JCTREND减仓": 1,
            "去掉JCBAND减仓": 1,
            "去掉指数5日>4%降仓": 1,
            "只保留CT退出": 2,
            "启动后固定20日": 2,
            "启动后固定30日": 2,
        }.get(r["exit"], 3)
        sat_pen = 0 if r["sat"] == "30%" else 1
        dd_pen = 0 if r["max_dd"] >= -0.155 else 1
        return (pick_pen, exit_pen, n_pen, sat_pen, dd_pen, -float(r["excess_add"]))

    ok = ok.copy()
    ok["_key"] = ok.apply(simplicity, axis=1)
    return ok.sort_values("_key").iloc[0]


def write_outputs(df: pd.DataFrame) -> pd.Series | None:
    OPT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OPT_DIR / "results.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    simple = df.drop_duplicates(subset=["version", "pick", "exit", "sat"]).copy()
    lines = [
        "版本 | 核心仓 | 增强仓 | 持股数 | 选股规则 | 退出规则 | 组合收益 | 沪深300 | 累计收益差 | 最大回撤 | 平均持股",
        "---|---|---|---|---|---|---|---|---|---|---",
    ]
    seen = set()
    for _, r in simple.iterrows():
        key = (r["version"], r["exit"], r["sat"], r["pick"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"{r['version']} | {r['core']} | {r['sat']} | {r['n_hold_rule']} | {r['pick']} | {r['exit']} | "
            f"{fmt_pct(r['combo'])} | {fmt_pct(r['hs300'])} | {fmt_pct(r['excess_add'])} | "
            f"{fmt_pct(r['max_dd'])} | {r['avg_hold_active']:.1f}"
        )
    (OPT_DIR / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    note = {
        "live_chip": "活筹强度用当日成交量/MA20 作为 T 日可获得代理（原公式无 WINNER/筹码活筹字段）。",
        "trend_score": "启动强度用公式已计算的 qsqd（趋势强度）作为 T 日代理。",
        "fzqd": "公式原已计算，仅导出到信号表，不改信号逻辑。",
        "ranking": "每日在增强仓候选（应用层仓位>0）内做截面标准化后等权加总，再取 Top N，T+1 开盘等权。",
        "baseline": "output/enhance_opt/baseline/ 为原 70/30 结果，未覆盖 output/enhance_*。",
    }
    (OPT_DIR / "params.json").write_text(json.dumps(note, ensure_ascii=False, indent=2), encoding="utf-8")
    winner = pick_winner(df)
    if winner is not None:
        (OPT_DIR / "winner.json").write_text(
            json.dumps(winner.drop(labels=["_key"], errors="ignore").to_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    print(f"结果 {out_csv}")
    return winner


def run_env_followup(uni: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Cross the only Top-N factor that turned additive excess positive."""
    overlay = uni["position_overlay"].fillna(0.0)
    env = uni["env_score"]
    books = {
        "当前退出": (overlay, True),
        "去掉JCTREND减仓": (pos_no_flag(uni, "no_trend"), True),
        "去掉JCBAND减仓": (pos_no_flag(uni, "no_band"), True),
        "去掉指数5日>4%降仓": (overlay, False),
        "只保留CT退出": (pos_ct_only(uni["launch"], uni["escape"]), True),
        "启动后固定20日": (pos_fixed_hold(uni["launch"], 20), True),
        "启动后固定30日": (pos_fixed_hold(uni["launch"], 30), True),
    }
    rows = []
    print("环境评分 Top N × 退出/权重")
    for n in (5, 10, 15, 20):
        for exit_name, (book, heat) in books.items():
            pos = select_topn(book, env, n)
            m = eval_sleeve(pos, uni, satellite=0.30, use_heat=heat)
            rows.append(_row(f"环境评分-Top{n}+{exit_name}", "70%", "30%", f"Top{n}启动", "按环境评分排序", exit_name, m))
            print(
                f"  Top{n} {exit_name:16s}  diff={m['excess_add']:+.2%}  "
                f"combo={m['combo']:+.2%}  dd={m['max_dd']:.2%}  n={m['avg_hold_active']:.1f}"
            )
    for n in (5, 10):
        pos = select_topn(overlay, env, n)
        for sat, core_s, sat_s in ((0.25, "75%", "25%"), (0.20, "80%", "20%")):
            m = eval_sleeve(pos, uni, satellite=sat, use_heat=True)
            rows.append(_row(f"环境评分-Top{n} {core_s}/{sat_s[:2]}", core_s, sat_s, f"Top{n}启动", "按环境评分排序", "当前退出", m))
            print(
                f"  Top{n} {core_s}/{sat_s}  diff={m['excess_add']:+.2%}  "
                f"combo={m['combo']:+.2%}  dd={m['max_dd']:.2%}"
            )
    extra = pd.DataFrame(rows)
    return pd.concat([df, extra], ignore_index=True)


def save_winner_equity(uni: dict, winner: pd.Series) -> None:
    overlay = uni["position_overlay"].fillna(0.0)
    n = {"Top5启动": 5, "Top10启动": 10, "Top15启动": 15, "Top20启动": 20, "Top30启动": 30, "全部启动": None}.get(
        winner["n_hold_rule"]
    )
    heat = winner["exit"] != "去掉指数5日>4%降仓"
    if winner["exit"] == "去掉JCTREND减仓":
        book = pos_no_flag(uni, "no_trend")
    elif winner["exit"] == "去掉JCBAND减仓":
        book = pos_no_flag(uni, "no_band")
    elif winner["exit"] == "只保留CT退出":
        book = pos_ct_only(uni["launch"], uni["escape"])
    elif winner["exit"] == "启动后固定20日":
        book = pos_fixed_hold(uni["launch"], 20)
    elif winner["exit"] == "启动后固定30日":
        book = pos_fixed_hold(uni["launch"], 30)
    else:
        book = overlay
    score = uni["env_score"] if "环境评分" in str(winner["pick"]) else composite_score(uni, book > 1e-12)
    if "吸筹" in str(winner["pick"]):
        score = uni["accum_score"]
    pos = select_topn(book, score, n)
    sat = 0.30 if winner["sat"] == "30%" else (0.25 if winner["sat"] == "25%" else 0.20)
    idx_close = uni["idx_close"]
    sat_daily = portfolio_from_position(pos, uni["open"], uni["close"], uni["idx_open"], idx_close)
    heat_s = None
    if heat:
        heat_s = (idx_close.pct_change(ENHANCE_HEAT_BARS).shift(1) > ENHANCE_HEAT_TH).fillna(False)
    blended = enhance_blend(sat_daily, sat_daily["hs300_ret"], sat, heat_s, ENHANCE_HEAT_SCALE)
    sample = blended.loc[blended.index >= pd.Timestamp(ENHANCE_WINDOW_START)].copy()
    sample.iloc[0, sample.columns.get_loc("net_ret")] = 0.0
    sample.iloc[0, sample.columns.get_loc("hs300_ret")] = 0.0
    sample["nav"] = (1 + sample["net_ret"]).cumprod()
    sample["bench"] = (1 + sample["hs300_ret"]).cumprod()
    sample = sample.reset_index(names="date")
    sample.to_csv(OPT_DIR / "winner_equity.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    snapshot_baseline()
    uni = load_universe(use_cache=True)
    df = run_grid(uni)
    df = run_env_followup(uni, df)
    winner = write_outputs(df)
    if winner is not None:
        save_winner_equity(uni, winner)
    print("\n===== 简表（含 Baseline）=====")
    show = df[
        df["version"].isin(
            ["Baseline", "全部启动", "Top5启动", "Top10启动", "Top15启动", "Top20启动", "Top30启动"]
        )
        | df["version"].str.contains(r"去掉|CT|固定|热度|70|75|80", regex=True)
    ]
    cols = ["version", "sat", "n_hold_rule", "pick", "exit", "combo", "hs300", "excess_add", "max_dd", "avg_hold_active"]
    with pd.option_context("display.max_rows", 200, "display.width", 160):
        print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if winner is None:
        print("\n没有累计收益差>0 的版本。")
    else:
        print("\n选定版本:", winner["version"])
        print("选股:", winner["pick"], winner["n_hold_rule"])
        print("退出:", winner["exit"])
        print("权重:", winner["core"], winner["sat"])
        print("组合:", fmt_pct(winner["combo"]))
        print("沪深300:", fmt_pct(winner["hs300"]))
        print("累计收益差:", fmt_pct(winner["excess_add"]))
        print("最大回撤:", fmt_pct(winner["max_dd"]))
        print("平均持股:", f"{winner['avg_hold_active']:.1f}")
    _ = show


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "pass2":
        snapshot_baseline()
        uni = load_universe(use_cache=True)
        prev = pd.read_csv(OPT_DIR / "results.csv")
        df = run_env_followup(uni, prev)
        winner = write_outputs(df)
        if winner is not None:
            save_winner_equity(uni, winner)
            print("\n选定版本:", winner["version"])
            print("选股:", winner["pick"], winner["n_hold_rule"])
            print("退出:", winner["exit"])
            print("权重:", winner["core"], winner["sat"])
            print("组合:", fmt_pct(winner["combo"]))
            print("沪深300:", fmt_pct(winner["hs300"]))
            print("累计收益差:", fmt_pct(winner["excess_add"]))
            print("最大回撤:", fmt_pct(winner["max_dd"]))
            print("平均持股:", f"{winner['avg_hold_active']:.1f}")
    else:
        main()
