"""沪深300 指数增强：核心跟指数，增强仓让强势股多留一点弹性。"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from hs300_strategy.data import DATA_DIR, fetch_hs300
from hs300_strategy.formula import compute_signals
from hs300_strategy.moneyflow import read_cached_l2
from hs300_strategy.stock_data import STOCK_DIR, fetch_constituents, fetch_industries, fetch_many_klines

OUTPUT_DIR = DATA_DIR.parent / "output"
FEE = 0.0003
SATELLITE = 0.30
SATELLITE_TOP_N = 5  # 方案一：增强仓只等权持有当日环境评分最高的 N 只启动股
HEAT_BARS = 5
HEAT_TH = 0.04
HEAT_SCALE = 0.40
BT_START = "20240901"
# 方案二风格：下跌配老登，左拐右/上涨配科技。权重预先固定，不按样本内结果搜参。
STYLE_HI = 3.0
STYLE_LO = 0.35
DEFENSIVE_KEYS = (
    "货币金融",
    "资本市场",
    "保险",
    "其他金融",
    "房地产",
    "电力",
    "热力",
    "燃气",
    "煤炭",
    "石油",
    "天然气",
    "黑色金属",
    "土木工程",
    "酒、饮料",
    "食品制造",
    "农副",
    "铁路运输",
    "道路运输",
    "水上运输",
    "航空运输",
    "邮政",
    "非金属矿物",
)
TECH_KEYS = (
    "计算机",
    "通信和其他电子",
    "软件",
    "互联网",
    "电信",
    "电气机械",
    "通信设备",
    "航空航天",
    "广播",
    "电影",
    "研究和试验",
)
SCHEME_ENV_TOP5 = "env_top5"
SCHEME_CT_ALL = "ct_all"
ALL_SCHEMES = (SCHEME_ENV_TOP5, SCHEME_CT_ALL)
_WORKER: dict = {}


@dataclass
class EnhanceResult:
    equity: pd.DataFrame
    monthly: pd.DataFrame
    metrics: dict
    n_stocks: int
    selection: dict | None = None
    selection_text: str = ""
    scheme: str = SCHEME_ENV_TOP5


def run_enhance(
    start: str = "20100101",
    end: str | None = None,
    bt_start: str = BT_START,
    satellite: float = SATELLITE,
    use_cache: bool = True,
    with_flow: bool = True,
    limit: int | None = None,
    workers: int | None = None,
    schemes: tuple[str, ...] | list[str] | None = None,
) -> dict[str, EnhanceResult]:
    t0 = perf_counter()
    wanted = tuple(schemes) if schemes else ALL_SCHEMES
    for sid in wanted:
        if sid not in ALL_SCHEMES:
            raise ValueError(f"未知方案 {sid}，可选 {ALL_SCHEMES}")
    end = end or date.today().strftime("%Y%m%d")
    members = fetch_constituents(use_cache=use_cache)
    if limit:
        members = members.head(limit).copy()
    codes = members["ts_code"].tolist()
    names = dict(zip(members["ts_code"], members["name"]))
    print(f"成分股 {len(codes)} 只（当前名单，回测存在幸存者偏差）")

    print("沪深300 指数 …")
    hs300 = fetch_hs300(start=start, end=end, use_cache=True)
    hs300 = hs300.sort_values("date").reset_index(drop=True)
    hs300["date"] = pd.to_datetime(hs300["date"])
    hs300_sig = compute_signals(hs300.copy(), asset="index")
    market_env = hs300_sig.set_index("date")["env_level"]

    print("成分股日K …")
    klines = fetch_many_klines(codes, start, end, use_cache=use_cache)
    ready = [c for c in codes if c in klines and len(klines[c]) >= 160]
    print(f"可用K线 {len(ready)} 只  {perf_counter() - t0:.1f}s")

    n_workers = workers or max(1, min(8, (os.cpu_count() or 4) - 1))
    print(f"计算个股仓位  {n_workers} 进程 …")
    pos_map: dict[str, pd.Series] = {}
    close_map: dict[str, pd.Series] = {}
    open_map: dict[str, pd.Series] = {}
    score_map: dict[str, pd.Series] = {}
    launch_map: dict[str, pd.Series] = {}
    escape_map: dict[str, pd.Series] = {}
    done = 0
    jobs = [(code, names.get(code, code)) for code in ready]
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(market_env, start, end, with_flow),
    ) as pool:
        futs = [pool.submit(_one_position, job) for job in jobs]
        for fut in as_completed(futs):
            code, o, c, pos, score, launch, escape = fut.result()
            done += 1
            if c is not None:
                open_map[code] = o
                close_map[code] = c
                pos_map[code] = pos
                if score is not None:
                    score_map[code] = score
                if launch is not None:
                    launch_map[code] = launch
                if escape is not None:
                    escape_map[code] = escape
            if done % 30 == 0 or done == len(jobs):
                print(f"  仓位 {done}/{len(jobs)}  {perf_counter() - t0:.1f}s", flush=True)

    cal = pd.DatetimeIndex(hs300["date"].sort_values().unique())
    open_px = pd.DataFrame(open_map).reindex(cal)
    close_px = pd.DataFrame(close_map).reindex(cal)
    overlay = pd.DataFrame(pos_map).reindex(cal).fillna(0.0)
    launch_df = pd.DataFrame(launch_map).reindex(cal).fillna(0.0)
    escape_df = pd.DataFrame(escape_map).reindex(cal).fillna(0.0)
    env_score = pd.DataFrame(score_map).reindex(cal) if score_map else None

    from hs300_strategy.config import ENHANCE_HEAT_BARS, ENHANCE_HEAT_SCALE, ENHANCE_HEAT_TH

    idx_open = hs300.set_index("date")["open"].astype(float).reindex(cal)
    idx_close = hs300.set_index("date")["close"].astype(float).reindex(cal)
    idx_5 = idx_close.pct_change(ENHANCE_HEAT_BARS).shift(1)
    heat = (idx_5 > ENHANCE_HEAT_TH).fillna(False)

    books: dict[str, pd.DataFrame] = {}
    if SCHEME_ENV_TOP5 in wanted:
        pos = overlay.copy()
        if SATELLITE_TOP_N and env_score is not None:
            pos = _topn_equal(pos, env_score, SATELLITE_TOP_N)
            print(f"方案一：增强仓按个股环境评分取 Top{SATELLITE_TOP_N}，内部等权")
        books[SCHEME_ENV_TOP5] = pos
    if SCHEME_CT_ALL in wanted:
        pos = _hold_until_ct(launch_df, escape_df)
        ind = fetch_industries(list(pos.columns), use_cache=True)
        style = _style_of_names(pos.columns, ind)
        regime = _market_regime(market_env.reindex(cal))
        pos = _apply_style_tilt(pos, regime, style)
        print("方案二：启动后拿到逃顶；下跌偏老登，上涨/左拐右偏科技")
        books[SCHEME_CT_ALL] = pos

    results: dict[str, EnhanceResult] = {}
    for sid, pos in books.items():
        print(f"回测 {sid} …")
        results[sid] = _finalize_book(
            scheme=sid,
            pos=pos,
            open_px=open_px,
            close_px=close_px,
            idx_open=idx_open,
            idx_close=idx_close,
            cal=cal,
            heat=heat,
            satellite=satellite,
            bt_start=bt_start,
            n_stocks=len(ready),
        )
    _write_scheme_log(results, satellite)
    print(f"总耗时 {perf_counter() - t0:.1f}s")
    return results


def _hold_until_ct(launch: pd.DataFrame, escape: pd.DataFrame) -> pd.DataFrame:
    """Yellow-triangle launch stays 100% until CT. No JC / take-profit / env flatten."""
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


def _style_label(industry: str) -> str:
    text = str(industry)
    if any(k in text for k in DEFENSIVE_KEYS):
        return "defensive"
    if any(k in text for k in TECH_KEYS):
        return "tech"
    return "other"


def _style_of_names(codes, industry_df: pd.DataFrame) -> pd.Series:
    ind = industry_df.drop_duplicates("ts_code").set_index("ts_code")["industry"]
    return pd.Series({c: _style_label(ind.get(c, "")) for c in codes})


def _market_regime(market_env: pd.Series) -> pd.Series:
    """T-day index env_level: 1=下跌偏老登；>=3 上涨偏科技；震荡且近20日跌过=左拐右偏科技。"""
    e = pd.to_numeric(market_env, errors="coerce").fillna(2.0)
    recent_down = e.eq(1).rolling(20, min_periods=1).max().eq(1)
    regime = pd.Series("neutral", index=e.index, dtype=object)
    regime = regime.mask(e.eq(1), "defensive")
    regime = regime.mask(e.ge(3), "tech")
    regime = regime.mask(e.eq(2) & recent_down, "tech")
    return regime


def _apply_style_tilt(pos: pd.DataFrame, regime: pd.Series, style: pd.Series) -> pd.DataFrame:
    held = (pos.fillna(0.0) > 1e-12).astype(float)
    style = style.reindex(held.columns).fillna("other")
    def_m = (style.to_numpy() == "defensive")
    tech_m = (style.to_numpy() == "tech")
    reg = regime.reindex(held.index).fillna("neutral").to_numpy()
    raw = held.to_numpy(dtype=float)
    out = raw.copy()
    for i, name in enumerate(reg):
        row = raw[i]
        if name == "defensive":
            out[i] = np.where(def_m, row * STYLE_HI, np.where(tech_m, row * STYLE_LO, row))
        elif name == "tech":
            out[i] = np.where(tech_m, row * STYLE_HI, np.where(def_m, row * STYLE_LO, row))
    return pd.DataFrame(out, index=held.index, columns=held.columns)


def _finalize_book(
    *,
    scheme: str,
    pos: pd.DataFrame,
    open_px: pd.DataFrame,
    close_px: pd.DataFrame,
    idx_open: pd.Series,
    idx_close: pd.Series,
    cal: pd.DatetimeIndex,
    heat: pd.Series,
    satellite: float,
    bt_start: str,
    n_stocks: int,
) -> EnhanceResult:
    from hs300_strategy.config import ENHANCE_HEAT_SCALE
    from hs300_strategy.execution import EXECUTION_NOTE
    from hs300_strategy.selection import evaluate_selection, format_selection, save_selection
    from hs300_strategy.strategy_backtest import enhance_blend, portfolio_from_position, position_blotter

    sat_daily = portfolio_from_position(pos, open_px, close_px, idx_open, idx_close)
    blended = enhance_blend(sat_daily, sat_daily["hs300_ret"], satellite, heat, ENHANCE_HEAT_SCALE)
    bt0 = pd.Timestamp(bt_start)
    mask = cal >= bt0
    sample = blended.loc[mask].copy()
    sample = sample.reset_index(names="date") if "date" not in sample.columns else sample.reset_index(drop=True)
    if "date" not in sample.columns:
        sample.insert(0, "date", cal[mask].to_numpy())
    sample["date"] = pd.to_datetime(sample["date"])
    if len(sample) < 5:
        raise RuntimeError("回测窗口太短。")
    for col in ("gross_ret", "net_ret", "hs300_ret", "turnover", "cost"):
        if col in sample.columns:
            sample.loc[sample.index[0], col] = 0.0
    sample["strategy_ret"] = sample["net_ret"]
    sample["benchmark_ret"] = sample["hs300_ret"]
    sample["excess_ret"] = sample["net_ret"] - sample["hs300_ret"]
    if "n_hold" not in sample.columns:
        sample["n_hold"] = sat_daily["n_hold"].reindex(pd.DatetimeIndex(sample["date"])).fillna(0).to_numpy()
    if "satellite_w" not in sample.columns:
        sample["satellite_w"] = 0.0
    sample["nav"] = (1 + sample["strategy_ret"]).cumprod()
    sample["bench"] = (1 + sample["benchmark_ret"]).cumprod()
    sample["excess_nav"] = sample["nav"] / sample["bench"]
    sample["heat"] = heat.reindex(pd.DatetimeIndex(sample["date"])).fillna(False).astype(int).to_numpy()

    metrics = _metrics(sample, satellite, n_stocks, scheme)
    metrics["execution"] = EXECUTION_NOTE
    metrics["survivorship"] = "当前沪深300成分名单，基准和持仓都有幸存者偏差，不得称为无偏基准。"
    monthly = _monthly(sample)
    files = _scheme_files(scheme)
    selection = evaluate_selection(
        open_px, close_px, pos, idx_open, idx_close, cal, bt_start, plot_path=OUTPUT_DIR / files["selection_png"]
    )
    save_selection(selection, path=OUTPUT_DIR / files["selection_json"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUTPUT_DIR / files["equity"], index=False, encoding="utf-8-sig")
    monthly.to_csv(OUTPUT_DIR / files["monthly"], index=False, encoding="utf-8-sig")
    blotter = position_blotter(pos.loc[mask], open_px, close_px)
    if not blotter.empty:
        blotter.to_csv(OUTPUT_DIR / files["trades"], index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / files["metrics"]).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _plot(sample, monthly, metrics, OUTPUT_DIR / files["chart"])
    return EnhanceResult(
        equity=sample,
        monthly=monthly,
        metrics=metrics,
        n_stocks=n_stocks,
        selection=selection,
        selection_text=format_selection(selection),
        scheme=scheme,
    )


def _scheme_files(scheme: str) -> dict[str, str]:
    if scheme == SCHEME_CT_ALL:
        return {
            "metrics": "enhance_ct_metrics.json",
            "equity": "enhance_ct_equity.csv",
            "monthly": "enhance_ct_monthly.csv",
            "trades": "enhance_ct_trades.csv",
            "chart": "enhance_ct.png",
            "selection_json": "selection_ct.json",
            "selection_png": "selection_ct.png",
        }
    return {
        "metrics": "enhance_metrics.json",
        "equity": "enhance_equity.csv",
        "monthly": "enhance_monthly.csv",
        "trades": "enhance_trades.csv",
        "chart": "enhance.png",
        "selection_json": "selection.json",
        "selection_png": "selection.png",
    }


def _period_rows(sample: pd.DataFrame) -> list[dict]:
    d0 = pd.Timestamp(sample["date"].iloc[0])
    d1 = pd.Timestamp(sample["date"].iloc[-1])
    windows = [
        ("full", "全样本", d0, d1),
        ("y2425", "2024-09～2025-12", pd.Timestamp("2024-09-02"), pd.Timestamp("2025-12-31")),
        ("y24", "2024-09～2024-12", pd.Timestamp("2024-09-02"), pd.Timestamp("2024-12-31")),
        ("y25", "2025全年", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")),
        ("y26", "2026至今", pd.Timestamp("2026-01-01"), d1),
    ]
    rows = []
    for pid, label, a, b in windows:
        w = sample[(sample["date"] >= a) & (sample["date"] <= b)]
        if len(w) < 2:
            continue
        nav = (1 + w["strategy_ret"].astype(float)).cumprod()
        bench = (1 + w["benchmark_ret"].astype(float)).cumprod()
        tot = float(nav.iloc[-1] / nav.iloc[0] - 1)
        bh = float(bench.iloc[-1] / bench.iloc[0] - 1)
        rows.append(
            {
                "id": pid,
                "label": label,
                "start": pd.Timestamp(w["date"].iloc[0]).strftime("%Y-%m-%d"),
                "end": pd.Timestamp(w["date"].iloc[-1]).strftime("%Y-%m-%d"),
                "strategy": tot,
                "hs300": bh,
                "excess_additive": tot - bh,
                "max_drawdown": float((nav / nav.cummax() - 1).min()),
            }
        )
    return rows


def _write_scheme_log(results: dict[str, EnhanceResult], satellite: float) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated": date.today().isoformat(),
        "core": 1.0 - satellite,
        "satellite": satellite,
        "execution": "T日收盘信号，T+1开盘成交",
        "baseline": "output/enhance_opt/baseline/ 未覆盖",
        "decision": (
            "方案一保留环境评分Top5（现有产品）。"
            "方案二：启动拿到逃顶，并按T日大盘环境在老登/科技之间加权（下跌偏老登，上涨或左拐右偏科技）。"
            "风格倍率预先固定为 3.0 / 0.35，不按样本内结果搜参。"
        ),
        "schemes": {},
    }
    for sid, res in results.items():
        m = res.metrics
        payload["schemes"][sid] = {
            "product": m.get("product"),
            "files": _scheme_files(sid),
            "total_return": m.get("total_return"),
            "benchmark_return": m.get("benchmark_return"),
            "excess_additive": m.get("excess_additive"),
            "avg_holdings": m.get("avg_holdings"),
            "periods": m.get("periods"),
        }
    (OUTPUT_DIR / "enhance_schemes.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _init_worker(market_env: pd.Series, start: str, end: str, with_flow: bool) -> None:
    _WORKER["env"] = market_env
    _WORKER["start"] = start
    _WORKER["end"] = end
    _WORKER["with_flow"] = with_flow


def _topn_equal(pos: pd.DataFrame, score: pd.DataFrame, n: int) -> pd.DataFrame:
    """Keep the n highest T-day scores among names the overlay wants to hold."""
    cand = pos > 1e-12
    sc = score.reindex(index=pos.index, columns=pos.columns).where(cand)
    rnk = sc.rank(axis=1, ascending=False, method="first")
    return ((rnk <= float(n)) & cand).astype(float)


def _one_position(job: tuple[str, str]):
    code, _name = job
    empty = (code, None, None, None, None, None, None)
    path = STOCK_DIR / f"{code.replace('.', '_')}.csv"
    if not path.exists():
        return empty
    work = pd.read_csv(path, parse_dates=["date"])
    if len(work) < 160:
        return empty
    if _WORKER["with_flow"]:
        flow = read_cached_l2(code, _WORKER["start"], _WORKER["end"])
        if flow is not None and not flow.empty:
            work = work.merge(flow[["date", "l2jbl"]], on="date", how="left")
    work["market_env"] = work["date"].map(_WORKER["env"])
    try:
        sig = compute_signals(work, asset="stock", overlay=True)
    except Exception:
        return empty
    sig["date"] = pd.to_datetime(sig["date"])
    idx = sig.set_index("date")
    score = idx["env_score"].astype(float) if "env_score" in idx.columns else None
    launch = idx["launch_turn"].astype(float) if "launch_turn" in idx.columns else None
    escape = idx["escape_top"].astype(float) if "escape_top" in idx.columns else None
    return (
        code,
        idx["open"].astype(float),
        idx["close"].astype(float),
        idx["position"].astype(float),
        score,
        launch,
        escape,
    )


def _metrics(sample: pd.DataFrame, satellite: float, n_stocks: int, scheme: str = SCHEME_ENV_TOP5) -> dict:
    n = len(sample)
    years = n / 252 if n else 0
    nav = sample["nav"].astype(float)
    bench = sample["bench"].astype(float)
    sret = sample["strategy_ret"].astype(float)
    bret = sample["benchmark_ret"].astype(float)
    xret = sample["excess_ret"].astype(float)
    total = float(nav.iloc[-1] / nav.iloc[0] - 1)
    bh = float(bench.iloc[-1] / bench.iloc[0] - 1)
    rel = float(nav.iloc[-1] / bench.iloc[-1] - 1)
    ann = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else 0.0
    bh_ann = (1 + bh) ** (1 / years) - 1 if years > 0 and bh > -1 else 0.0
    excess = rel
    excess_additive = total - bh
    excess_ann = (1 + rel) ** (1 / years) - 1 if years > 0 and rel > -1 else 0.0
    te = float(xret.std() * (252 ** 0.5)) if n > 1 else 0.0
    ir = float(xret.mean() / xret.std() * (252 ** 0.5)) if xret.std() else 0.0
    sharpe = float(sret.mean() / sret.std() * (252 ** 0.5)) if sret.std() else 0.0
    dd = float((nav / nav.cummax() - 1).min())
    bh_dd = float((bench / bench.cummax() - 1).min())
    live = sample["n_hold"] > 0
    periods = _period_rows(sample)
    if scheme == SCHEME_CT_ALL:
        product = "沪深300指数增强（方案二 · 启动拿到逃顶）"
        scheme_name = "方案二 · 启动拿到逃顶"
        rules = (
            f"方案二（尝试分段超过指数）。核心仓 {1 - satellite:.0%} 跟踪沪深300，增强仓 {satellite:.0%}。"
            "买入仍是黄三角启动；离场只认逃顶 CT。不按环境评分截断，不用波段/趋势减仓。"
            "增强仓内部按大盘环境切换风格（T日环境等级，T+1开盘生效）："
            "下跌环境多配银行/公用/煤炭石油/白酒等老登；上涨或近20日跌过后的震荡（左拐右）多配电子/软件/电新等科技。"
            f"指数近{HEAT_BARS}日涨超{HEAT_TH:.0%}时，增强仓临时×{HEAT_SCALE:.0%}（与方案一相同）。"
            "方案一文件仍是 output/enhance_*；本方案写 output/enhance_ct_*。"
            "原全部启动 baseline 保留在 output/enhance_opt/baseline/。"
        )
        note = (
            "核心仓跟踪沪深300价格指数。Python Level-2 为日度大单净额近似，不是通达信 100% 复刻。"
            "成分股用当前名单（幸存者偏差）。个股前复权、指数不计分红。"
            "超额与相对净值同一口径：策略净值/指数净值−1。"
            f"累计收益差（组合区间收益−指数区间收益）为 {excess_additive:.2%}。"
            "目标是上涨时比指数强、回撤时比指数浅；风格权重预先固定，不是样本内搜参。"
            "不能据此声称选股 alpha。"
        )
    else:
        product = "沪深300指数增强（方案一 · 环境评分Top5）"
        scheme_name = "方案一 · 环境评分Top5"
        rules = (
            f"方案一（现有产品，保留对照）。核心仓 {1 - satellite:.0%} 跟踪沪深300，增强仓 {satellite:.0%}。"
            f"公式启动信号仍全部计算；增强仓只等权持有当日个股环境评分最高的 {SATELLITE_TOP_N} 只。"
            "增强仓不止盈清仓，只在逃顶/个股下跌环境离场；波段减到70%、趋势减到85%。"
            f"指数近{HEAT_BARS}日涨超{HEAT_TH:.0%}时，增强仓临时×{HEAT_SCALE:.0%}。"
            "JCTREND 是 JC_EVENT 且 QQS，只描述风险分类，不是导致上涨的趋势确认。"
            "分段超额不是设计目标：2024-09～2025-12 可能落后指数，全样本超额主要来自 2026。"
            "原全部启动 baseline 保留在 output/enhance_opt/baseline/。"
        )
        note = (
            "核心仓跟踪沪深300价格指数。Python Level-2 为日度大单净额近似，不是通达信 100% 复刻。"
            "成分股用当前名单（幸存者偏差）。个股前复权、指数不计分红。"
            "超额与相对净值同一口径：策略净值/指数净值−1。"
            f"累计收益差（组合区间收益−指数区间收益）为 {excess_additive:.2%}，牛市里会比相对超额更大，不是图上的跌幅。"
        )
    return {
        "product": product,
        "scheme": scheme,
        "scheme_name": scheme_name,
        "start": sample["date"].iloc[0].strftime("%Y-%m-%d"),
        "end": sample["date"].iloc[-1].strftime("%Y-%m-%d"),
        "days": n,
        "n_stocks": n_stocks,
        "satellite": satellite,
        "avg_satellite": float(sample["satellite_w"].mean()),
        "heat_days": int((sample["heat"] == 1).sum()) if "heat" in sample.columns else 0,
        "fee": FEE,
        "total_return": total,
        "annual_return": ann,
        "benchmark_return": bh,
        "benchmark_annual": bh_ann,
        "excess": excess,
        "excess_additive": excess_additive,
        "excess_annual": excess_ann,
        "tracking_error": te,
        "information_ratio": ir,
        "sharpe": sharpe,
        "max_drawdown": dd,
        "benchmark_drawdown": bh_dd,
        "excess_win_rate": float((xret.iloc[1:] > 0).mean()) if n > 1 else 0.0,
        "avg_holdings": float(sample["n_hold"].mean()),
        "avg_holdings_when_active": float(sample.loc[live, "n_hold"].mean()) if live.any() else 0.0,
        "days_with_satellite": int(live.sum()),
        "satellite_day_share": float(live.mean()),
        "avg_turnover": float(sample["turnover"].mean()) if "turnover" in sample.columns else 0.0,
        "execution": (
            "T日收盘生成信号，T+1日开盘成交。禁止使用T日收盘价作为成交价。"
        ),
        "survivorship": "当前沪深300成分名单，存在幸存者偏差，不得称为无偏基准。",
        "rules": rules,
        "note": note,
        "periods": periods,
    }


def _monthly(sample: pd.DataFrame) -> pd.DataFrame:
    eq = sample.set_index("date")[["nav", "bench"]]
    eq.index = pd.to_datetime(eq.index)
    try:
        last_s = eq["nav"].resample("ME").last()
        last_b = eq["bench"].resample("ME").last()
        first_s = eq["nav"].resample("ME").first()
        first_b = eq["bench"].resample("ME").first()
    except ValueError:
        last_s = eq["nav"].resample("M").last()
        last_b = eq["bench"].resample("M").last()
        first_s = eq["nav"].resample("M").first()
        first_b = eq["bench"].resample("M").first()
    # month return from previous month-end; first month from first NAV
    s = last_s.pct_change()
    b = last_b.pct_change()
    s.iloc[0] = last_s.iloc[0] / first_s.iloc[0] - 1
    b.iloc[0] = last_b.iloc[0] / first_b.iloc[0] - 1
    out = pd.DataFrame(
        {
            "month": last_s.index.strftime("%Y-%m"),
            "strategy": s.to_numpy(),
            "benchmark": b.to_numpy(),
            "excess": (s - b).to_numpy(),
        }
    )
    return out.reset_index(drop=True)


def _plot(sample: pd.DataFrame, monthly: pd.DataFrame, metrics: dict, path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12.8, 17.6),
        gridspec_kw={"height_ratios": [1.2, 2.15, 1.25]},
    )
    d = pd.to_datetime(sample["date"])
    nav = sample["nav"].astype(float)
    bench = sample["bench"].astype(float)
    rel_pct = (sample["excess_nav"].astype(float) - 1.0) * 100.0

    ax = axes[0]
    ax.plot(d, nav, color="#1f6feb", lw=2.0, label="指数增强")
    ax.plot(d, bench, color="#8b949e", lw=1.6, label="沪深300")
    ax.set_title(
        f"{metrics.get('scheme_name') or metrics.get('product') or '沪深300指数增强'}    "
        f"{metrics['start']} ~ {metrics['end']}    "
        f"相对超额 {metrics['excess']:.2%}    IR {metrics['information_ratio']:.2f}",
        fontsize=13,
        pad=10,
    )
    ax.set_ylabel("净值（期初=1）", fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.28)
    ax.yaxis.set_major_locator(MaxNLocator(8))
    ax.tick_params(labelsize=10)

    ax2 = axes[1]
    ax2.plot(d, rel_pct, color="#1e8449", lw=2.0, label="相对超额 = 策略净值/指数净值 − 1")
    ax2.axhline(0.0, color="#8b949e", lw=1.0)
    ax2.fill_between(d, rel_pct, 0.0, where=rel_pct.to_numpy() >= 0, color="#e74c3c", alpha=0.16, interpolate=True)
    ax2.fill_between(d, rel_pct, 0.0, where=rel_pct.to_numpy() < 0, color="#1abc9c", alpha=0.16, interpolate=True)
    last = float(rel_pct.iloc[-1])
    ax2.annotate(
        f"末值 {last:+.2f}%",
        xy=(d.iloc[-1], last),
        xytext=(-110, 12 if last < 0 else -18),
        textcoords="offset points",
        color="#1e8449",
        fontsize=11,
        fontweight="bold",
    )
    ax2.set_ylabel("相对超额（%）", fontsize=11)
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, which="major", alpha=0.32)
    lo = float(rel_pct.min())
    hi = float(rel_pct.max())
    span = max(hi - lo, 1.5)
    ax2.set_ylim(lo - 0.12 * span, hi + 0.12 * span)
    ax2.yaxis.set_major_locator(MaxNLocator(10))
    ax2.tick_params(labelsize=10)

    ax3 = axes[2]
    month_x = pd.to_datetime(monthly["month"] + "-01")
    month_y = monthly["excess"].astype(float) * 100.0
    colors = np.where(month_y.to_numpy() >= 0, "#e74c3c", "#1abc9c")
    ax3.bar(month_x, month_y, width=22, color=colors, alpha=0.9, linewidth=0)
    ax3.axhline(0, color="#8b949e", lw=1.0)
    mx = float(np.nanmax(np.abs(month_y.to_numpy())))
    ax3.set_ylim(-mx * 1.28, mx * 1.28)
    ax3.set_ylabel("月超额（%）", fontsize=11)
    ax3.set_xlabel("日期", fontsize=11)
    ax3.yaxis.set_major_locator(MaxNLocator(8))
    ax3.grid(True, axis="y", alpha=0.32)
    ax3.tick_params(labelsize=10)
    fig.tight_layout(h_pad=1.35)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def format_report(m: dict, monthly: pd.DataFrame) -> str:
    lines = [
        f"[{m['product']}]",
        f"规则：{m.get('rules', '')}",
        f"无信号日增强仓为 0，全部放指数。{m.get('execution', 'T日收盘信号，T+1开盘成交')}。"
        f"平均增强仓 {m.get('avg_satellite', m['satellite']):.1%}"
        + (f"  短线大涨日 {m.get('heat_days', 0)} 天" if m.get("heat_days") is not None else ""),
        m.get("survivorship", ""),
        f"区间 {m['start']} ~ {m['end']}  （{m['days']} 个交易日，{m['n_stocks']} 只成分股）",
        "",
        f"增强组合  {m['total_return']:.2%}  年化 {m['annual_return']:.2%}  最大回撤 {m['max_drawdown']:.2%}  夏普 {m['sharpe']:.2f}",
        f"沪深300    {m['benchmark_return']:.2%}  年化 {m['benchmark_annual']:.2%}  最大回撤 {m['benchmark_drawdown']:.2%}",
        f"相对超额   {m['excess']:.2%}  年化 {m['excess_annual']:.2%}  （相对净值末值 {1 + m['excess']:.3f}）",
        f"累计收益差 {m.get('excess_additive', m['total_return'] - m['benchmark_return']):.2%}  （组合区间收益 − 指数区间收益，与上项不同口径）",
        f"跟踪误差   {m['tracking_error']:.2%}  信息比率 {m['information_ratio']:.2f}  日胜率 {m['excess_win_rate']:.1%}",
        f"增强仓有票 {m['days_with_satellite']} 天（{m['satellite_day_share']:.1%}）  有票时平均持股 {m['avg_holdings_when_active']:.1f}  全日平均 {m['avg_holdings']:.1f}",
        f"日均换手   {m['avg_turnover']:.2%}",
        "",
        "分段累计收益差（组合区间收益 − 指数区间收益）",
    ]
    for p in m.get("periods") or []:
        lines.append(
            f"  {p['label']}  组合 {p['strategy']:+.2%}  沪深300 {p['hs300']:+.2%}  差 {p['excess_additive']:+.2%}"
        )
    lines.extend([
        "",
        "月度超额",
    ])
    for _, row in monthly.iterrows():
        lines.append(
            f"  {row['month']}  增强 {row['strategy']:+.2%}  沪深300 {row['benchmark']:+.2%}  超额 {row['excess']:+.2%}"
        )
    lines.append("")
    lines.append(m["note"])
    return "\n".join(lines)
