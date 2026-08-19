"""Research runner: event study + strategy backtest + tables. No parameter hunting."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from hs300_strategy.config import (
    COMMISSION,
    COST_SLIP_GRID,
    ENHANCE_HEAT_BARS,
    ENHANCE_HEAT_SCALE,
    ENHANCE_HEAT_TH,
    ENHANCE_WEIGHTS,
    ENHANCE_WINDOW_START,
    EVENT_PRIMARY_N,
    HOLD_PERIODS,
    L2_DISCLAIMER,
    OUTPUT_DIR,
    RATING_KIND,
    RESEARCH_DIR,
    SIGNAL_LAYERS,
    SLIPPAGE_BUY,
    SLIPPAGE_SELL,
    STAMP_TAX,
    as_frozen_dict,
)
from hs300_strategy.event_backtest import (
    build_launch_events,
    event_trades,
    summarize_events,
    walk_forward_events,
)
from hs300_strategy.research_check import format_leak_report, future_leak_check
from hs300_strategy.research_panels import load_research_universe
from hs300_strategy.strategy_backtest import (
    enhance_blend,
    metrics_from_daily,
    portfolio_from_position,
    position_blotter,
    reconstruct_positions,
)


def run_research(
    start: str = "20100101",
    end: str | None = None,
    use_cache: bool = True,
    with_flow: bool = True,
    limit: int | None = None,
    workers: int | None = None,
) -> dict:
    end = end or date.today().strftime("%Y%m%d")
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    uni = load_research_universe(
        start=start, end=end, use_cache=use_cache, with_flow=with_flow, limit=limit, workers=workers
    )
    leak = future_leak_check(uni.get("leak_sample"))
    (RESEARCH_DIR / "future_leak_check.txt").write_text(format_leak_report(leak), encoding="utf-8")
    print(format_leak_report(leak))

    events = build_launch_events(uni["launch"])
    print(f"启动事件（去抖后） {len(events)}")
    trades = event_trades(
        events,
        uni["open"],
        uni["high"],
        uni["low"],
        uni["close"],
        uni["idx_open"],
        uni["idx_high"],
        uni["idx_low"],
        uni["idx_close"],
        uni["launch"],
        HOLD_PERIODS,
    )
    trades.to_csv(RESEARCH_DIR / "event_trades.csv", index=False, encoding="utf-8-sig")

    table2 = pd.concat(
        [summarize_events(trades, s) for s in ("full", "train", "test")],
        ignore_index=True,
    )
    table2.to_csv(RESEARCH_DIR / "table2_hold_periods.csv", index=False, encoding="utf-8-sig")
    wf = walk_forward_events(trades, EVENT_PRIMARY_N)
    wf.to_csv(RESEARCH_DIR / "walk_forward_events.csv", index=False, encoding="utf-8-sig")

    # blotter for primary N
    blotter = trades[(trades["hold_n"] == EVENT_PRIMARY_N) & (trades.get("complete", 1) == 1)].copy()
    blotter = blotter[
        [
            "ts_code",
            "signal_date",
            "entry_date",
            "entry_price",
            "exit_date",
            "exit_price",
            "stock_ret",
            "hs300_ret",
            "excess_ret",
            "mfe",
            "mae",
            "hs300_mfe",
            "hs300_mae",
            "excess_mfe",
        ]
    ]
    blotter.to_csv(RESEARCH_DIR / "event_blotter_n20.csv", index=False, encoding="utf-8-sig")

    print("策略回测（状态机，T+1 开盘）…")
    machine_daily = portfolio_from_position(
        uni["position"], uni["open"], uni["close"], uni["idx_open"], uni["idx_close"]
    )
    machine_daily.to_csv(RESEARCH_DIR / "strategy_daily.csv", encoding="utf-8-sig")
    blotter_s = position_blotter(uni["position"], uni["open"], uni["close"])
    blotter_s.to_csv(RESEARCH_DIR / "strategy_blotter.csv", index=False, encoding="utf-8-sig")

    pos_100 = reconstruct_positions(
        uni["launch"], uni["reduce_band"], uni["reduce_trend"],
        uni["take_profit"], uni["escape"], uni["env"], "always_100",
    )
    pos_restore = reconstruct_positions(
        uni["launch"], uni["reduce_band"], uni["reduce_trend"],
        uni["take_profit"], uni["escape"], uni["env"], "restore_100",
    )
    d100 = portfolio_from_position(pos_100, uni["open"], uni["close"], uni["idx_open"], uni["idx_close"])
    drest = portfolio_from_position(pos_restore, uni["open"], uni["close"], uni["idx_open"], uni["idx_close"])

    table8_rows = []
    for sample in ("full", "train", "test"):
        for label, daily in (
            ("state_machine", machine_daily),
            ("always_100", d100),
            ("restore_100_on_non_risk_days", drest),
        ):
            table8_rows.append(metrics_from_daily(daily, sample, label, "net_ret"))
            table8_rows.append(metrics_from_daily(daily, sample, label + "_gross", "gross_ret"))
    table8 = pd.DataFrame(table8_rows)
    table8.to_csv(RESEARCH_DIR / "table8_position_modes.csv", index=False, encoding="utf-8-sig")

    table1 = pd.DataFrame(
        [
            metrics_from_daily(machine_daily, s, "state_machine_net", "net_ret")
            for s in ("full", "train", "test")
        ]
    )
    table1.to_csv(RESEARCH_DIR / "table1_strategy_full.csv", index=False, encoding="utf-8-sig")

    table3 = table1.copy()
    table3.to_csv(RESEARCH_DIR / "table3_vs_csi300.csv", index=False, encoding="utf-8-sig")
    table4 = table1.copy()
    table4.to_csv(RESEARCH_DIR / "table4_vs_unselected.csv", index=False, encoding="utf-8-sig")
    ev_primary = table2[table2["hold_n"] == EVENT_PRIMARY_N].copy()
    ev_primary.to_csv(RESEARCH_DIR / "table5_is_oos_events.csv", index=False, encoding="utf-8-sig")
    table1.to_csv(RESEARCH_DIR / "table5_is_oos_strategy.csv", index=False, encoding="utf-8-sig")

    # Table 6: hold-period sensitivity = table2 full sample (robustness, not max)
    table2[table2["sample"] == "full"].to_csv(
        RESEARCH_DIR / "table6_hold_sensitivity.csv", index=False, encoding="utf-8-sig"
    )

    print("交易成本敏感性…")
    cost_rows = []
    for slip in COST_SLIP_GRID:
        d = portfolio_from_position(
            uni["position"],
            uni["open"],
            uni["close"],
            uni["idx_open"],
            uni["idx_close"],
            slip_buy=slip,
            slip_sell=slip,
        )
        for sample in ("full", "train", "test"):
            m = metrics_from_daily(d, sample, f"slip_{slip:.4f}", "net_ret")
            m["slippage"] = slip
            m["commission"] = COMMISSION
            m["stamp_tax"] = STAMP_TAX
            cost_rows.append(m)
    # zero everything
    d0 = portfolio_from_position(
        uni["position"], uni["open"], uni["close"], uni["idx_open"], uni["idx_close"],
        commission=0.0, stamp_tax=0.0, slip_buy=0.0, slip_sell=0.0,
    )
    for sample in ("full", "train", "test"):
        m = metrics_from_daily(d0, sample, "zero_cost", "gross_ret")
        m["slippage"] = 0.0
        m["commission"] = 0.0
        m["stamp_tax"] = 0.0
        cost_rows.append(m)
    table7 = pd.DataFrame(cost_rows)
    table7.to_csv(RESEARCH_DIR / "table7_cost_sensitivity.csv", index=False, encoding="utf-8-sig")

    print("指数增强权重网格（不寻优）…")
    overlay_daily = portfolio_from_position(
        uni["position_overlay"], uni["open"], uni["close"], uni["idx_open"], uni["idx_close"]
    )
    idx_ret = overlay_daily["hs300_ret"]
    idx_5 = uni["idx_close"].pct_change(ENHANCE_HEAT_BARS).shift(1)
    heat = (idx_5 > ENHANCE_HEAT_TH).fillna(False)
    enh_rows = []
    for sat in ENHANCE_WEIGHTS:
        blended = enhance_blend(overlay_daily, idx_ret, sat, heat, ENHANCE_HEAT_SCALE)
        for sample, slabel in (("full", "full"), ("train", "train"), ("test", "test"), ("enhance", "from_202409")):
            m = metrics_from_daily(blended, slabel if sample != "enhance" else "enhance", f"core_{1-sat:.0%}_sat_{sat:.0%}", "net_ret")
            m["satellite"] = sat
            m["core"] = 1.0 - sat
            enh_rows.append(m)
    table_enh = pd.DataFrame(enh_rows)
    table_enh.to_csv(RESEARCH_DIR / "table_enhance_weights.csv", index=False, encoding="utf-8-sig")

    frozen = as_frozen_dict()
    (RESEARCH_DIR / "frozen_config.json").write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    audit = _write_reports(
        leak=leak,
        table1=table1,
        table2=table2,
        table7=table7,
        table8=table8,
        ev_primary=ev_primary,
        table_enh=table_enh,
        wf=wf,
        n_events=len(events),
        n_stocks=uni["n_stocks"],
    )
    print(f"输出目录 {RESEARCH_DIR}")
    return {
        "leak": leak,
        "n_events": len(events),
        "n_stocks": uni["n_stocks"],
        "audit_path": str(audit),
    }


def _write_reports(**kw) -> Path:
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    backtest_md = RESEARCH_DIR / "回测报告.md"
    audit_md = RESEARCH_DIR / "策略审查报告.md"
    backtest_md.write_text(_backtest_markdown(**kw), encoding="utf-8")
    audit_md.write_text(_audit_markdown(**kw), encoding="utf-8")
    return audit_md


def _fmt(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    if df is None or df.empty:
        return "（空）"
    work = df.copy()
    if cols:
        work = work[[c for c in cols if c in work.columns]]
    return work.to_string(index=False)


def _pct(x) -> str:
    try:
        if pd.isna(x):
            return "NA"
        return f"{float(x):.2%}"
    except (TypeError, ValueError):
        return str(x)


def _num(x, nd=2) -> str:
    try:
        if pd.isna(x):
            return "NA"
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _backtest_markdown(*, leak, table1, table2, table7, table8, ev_primary, table_enh, wf, n_events, n_stocks) -> str:
    t2f = table2[table2["sample"] == "full"] if not table2.empty else table2
    layers = "\n".join(f"- {k}: {v}" for k, v in SIGNAL_LAYERS.items())
    return f"""# 回测报告（研究设计，非收益优化）

{L2_DISCLAIMER}

评级标签类型：`{RATING_KIND}`。不得当作客观监督标签或实时过滤。

信号层次（不可混用）：
{layers}

执行口径：T 日收盘生成信号；T+1 日开盘买入/调仓；事件研究在持有期满以收盘价退出。禁止用 T 日收盘价作为成交价。

成分股为当前沪深300名单，存在幸存者偏差。个股前复权，沪深300为价格指数。行业中性基准：当前本地缓存无行业分类，未计算。

冻结参数见 `output/research/frozen_config.json`。本次未在 Test 调参。

future_leak_check: hard_leak={'NO' if leak.get('ok') else 'YES'}
股票数 {n_stocks}  去抖后启动事件 {n_events}

## 表1 完整样本统计（状态机策略，净收益）

{_fmt(table1, ['sample','start','end','n','gross_return','net_return','hs300_return','excess_hs300','max_drawdown','avg_turnover','t_vs_hs300','p_vs_hs300','t_vs_unselected','p_vs_unselected','verdict_style'])}

## 表2 不同持有期（事件研究，T+1开盘→N日收盘）

{_fmt(t2f, ['sample','hold_n','n_complete','mean_ret','median_ret','mean_mfe','mean_mae','win_rate','beat_hs300','mean_excess_hs300','t_excess_hs300','p_excess_hs300','mean_excess_unselected','t_excess_unselected','p_excess_unselected'])}

MFE/MAE 与持有期超额分列，不混用。20 日只是标准化窗口之一，不是因为结果最好才选。

## 表3 相对 CSI300

见上表 `excess_hs300` / `t_vs_hs300`（策略）与表2 `mean_excess_hs300`（事件）。

## 表4 相对未入选成分股等权

见 `t_vs_unselected` / `mean_excess_unselected`。若相对沪深300显著、相对未入选不显著：策略相对市场基准存在超额，但暂不能区分选股能力与风格暴露。

## 表5 样本内 / 样本外

事件（N={EVENT_PRIMARY_N}）：

{_fmt(ev_primary, ['sample','n_complete','mean_excess_hs300','t_excess_hs300','p_excess_hs300','mean_excess_unselected','t_excess_unselected','p_excess_unselected'])}

策略：

{_fmt(table1, ['sample','net_return','excess_hs300','t_vs_hs300','p_vs_hs300','t_vs_unselected','p_vs_unselected'])}

Walk-forward（事件 N={EVENT_PRIMARY_N}，参数冻结）：`walk_forward_events.csv`

{_fmt(wf.head(16) if wf is not None and not wf.empty else pd.DataFrame(), ['sample','n_complete','mean_excess_hs300','t_excess_hs300','p_excess_hs300'])}

## 表6 持有期参数敏感性

即表2全样本各 N。重点是附近窗口是否同号/同向，不是选最优 N。

## 表7 交易成本敏感性

默认佣金 {COMMISSION}、印花税(卖) {STAMP_TAX}、买卖滑点 {SLIPPAGE_BUY}/{SLIPPAGE_SELL}。

{_fmt(table7[table7['sample']=='full'] if not table7.empty else table7, ['label','sample','gross_return','net_return','excess_hs300','cost_drag','avg_turnover','slippage'])}

## 表8 状态机与固定仓位对比

{_fmt(table8[(table8['sample']=='full') & (~table8['label'].str.contains('_gross', na=False))] if not table8.empty else table8, ['label','sample','net_return','excess_hs300','max_drawdown','avg_turnover','t_vs_hs300','p_vs_hs300'])}

仓位规则未改：启动100%、JCBAND→50%、JCTREND→70%、止盈/逃顶/下跌环境清空。JCTREND 不是加仓确认。

## 指数增强权重网格（50/50 … 100/0）

近期窗口起点 {ENHANCE_WINDOW_START}。不按超额选权重。

{_fmt(table_enh[table_enh['sample']=='enhance'] if not table_enh.empty else table_enh, ['label','sample','net_return','hs300_return','excess_hs300','max_drawdown','avg_turnover','t_vs_hs300'])}

明细 CSV 均在 `output/research/`。
"""


def _audit_markdown(*, leak, table1, table2, table7, table8, ev_primary, table_enh, wf, n_events, n_stocks) -> str:
    # Fill A-H from actual numbers
    t2_20 = pd.DataFrame()
    if table2 is not None and not table2.empty:
        t2_20 = table2[(table2["sample"] == "full") & (table2["hold_n"] == EVENT_PRIMARY_N)]
    t1f = table1[table1["sample"] == "full"].iloc[0].to_dict() if table1 is not None and not table1.empty else {}
    evf = t2_20.iloc[0].to_dict() if not t2_20.empty else {}
    enh = table_enh[table_enh["sample"] == "enhance"] if table_enh is not None and not table_enh.empty else pd.DataFrame()
    p_hs = evf.get("p_excess_hs300")
    p_un = evf.get("p_excess_unselected")
    mu_hs = evf.get("mean_excess_hs300")
    style = (
        "是。相对沪深300的事件超额若显著，但相对未入选成分股等权不显著，则不能称为纯选股 alpha。"
        if (pd.notna(p_hs) and p_hs < 0.05 and (pd.isna(p_un) or p_un >= 0.05 or (mu_hs or 0) > 0))
        else "需看表2/表4：只有相对未入选等权也显著时，才更接近选股而非风格。"
    )
    if pd.notna(p_hs) and p_hs < 0.05 and (mu_hs or 0) > 0 and (pd.isna(p_un) or p_un >= 0.05):
        style = "是。事件/策略相对沪深300的显著超额，暂不能排除风格暴露：相对未入选成分股等权不显著（或不稳定）。正确表述：策略相对市场基准存在超额，但暂不能区分选股能力与风格暴露。"
    elif pd.notna(p_hs) and p_hs < 0.05 and pd.notna(p_un) and p_un < 0.05:
        style = "不完全是。相对沪深300与未入选等权均显著，选股方向证据更强，但仍可能混有行业/因子暴露；本地无行业中性基准。"
    else:
        style = "当前全样本下，相对沪深300的“显著超额”在 T+1 开盘口径下需要按下表重读；不能沿用旧的收盘成交结论。"

    leak_ok = leak.get("ok", False)
    enh_all_neg = False
    if not enh.empty and "excess_hs300" in enh.columns:
        enh_all_neg = bool((enh["excess_hs300"] <= 0).all())

    t8 = pd.DataFrame()
    if table8 is not None and not table8.empty:
        lab = table8["label"].astype(str)
        t8 = table8[(table8["sample"] == "full") & (~lab.str.contains("_gross"))]

    return f"""# 策略审查报告

范围：Python 策略与回测框架的研究方法。不优化收益。样本股票 {n_stocks} 只，去抖启动 {n_events} 次。评级为 `{RATING_KIND}`。

{L2_DISCLAIMER}

## A. 当前是否存在未来函数？

公式内部：`rolling` / `hhv` / `llv` / `ma` / `ref(+n)` 使用含当日的历史窗口，截断重放检查 hard_leak={'否' if leak_ok else '是'}。
旧 `events.excursion_row`、质量评级、`screener.rank_stocks` 使用信号日后的 N 日路径，这是事后标签，不是公式信号；若拿来做当日过滤则构成未来函数。
旧生产路径若仍按收盘成交则视为未整改；当前 `backtest.py` / `enhance.py` / `screen.py` / `execution.py` 已统一为 T+1 开盘。
Level-2 日度数据若实际要到 T+1 盘中才到，则即使用开盘成交也仍有信息延迟风险（见 mapping 假设）。

## B. 当前启动信号是否可以无未来数据实现？

可以，在以下前提下：启动只用 T 日及以前的 OHLC、成交量、日度大单净额近似；T 收盘出信号，T+1 开盘买入，不用 T 收盘价成交。黄三角条件含当日收盘价、当日量、当日 `l2jbl`，因此不能在 T 日盘中当成已成交。不能声称与通达信盘中 Level-2 同步。

## C. 当前20日事件回测是否定义严谨？

旧版不严谨：以信号日收盘为入场价，且常把 20 日当成“策略”。新定义：signal_date=T，entry_date=T+1 开盘，持有 N 个交易日以退出日收盘离场；沪深300用同一段 entry 开盘到 exit 收盘。5/10/15/20/30/40/60 全部报告，20 日只是标准化窗口。重叠事件使 t 值偏乐观。旧 20 日结果不可直接引用。

全样本 N=20：n={_num(evf.get('n_complete'),0)}  持有期收益均值 {_pct(evf.get('mean_ret'))}  超额沪深300 {_pct(evf.get('mean_excess_hs300'))}  t={_num(evf.get('t_excess_hs300'))}  p={_num(evf.get('p_excess_hs300'),3)}  相对未入选 {_pct(evf.get('mean_excess_unselected'))}  t={_num(evf.get('t_excess_unselected'))}  p={_num(evf.get('p_excess_unselected'),3)}

## D. 当前相对沪深300的显著超额是否可能来自风格暴露？

{style}

策略全样本净收益 {_pct(t1f.get('net_return'))}  vs 沪深300 {_pct(t1f.get('hs300_return'))}  超额 {_pct(t1f.get('excess_hs300'))}  t={_num(t1f.get('t_vs_hs300'))}  p={_num(t1f.get('p_vs_hs300'),3)}  vs 未入选 t={_num(t1f.get('t_vs_unselected'))}  p={_num(t1f.get('p_vs_unselected'),3)}。无行业中性基准。

## E. 当前是否存在明显过拟合风险？

存在。公式含大量未做 Train-only 冻结的历史阈值（20/45/60/120、0.9/0.85/0.95/1.2、持有20 日等）。此前全样本 2010-07～2026-08 直接检验，且有过按事件结果强调 20 日的风险。本次参数冻结并输出 Train 2010-07～2020-12 / Test 2021-01～2026-08 对照与 walk-forward；**没有**在 Test 调参。评级 5/4/3/2/1 是主观半定量，不能当标签去拟合。

## F. 当前Level-2数据是否与通达信原公式严格一致？

否。对照表见 `docs/l2_mapping.md`。Python 用 Tushare 日度大单净额近似 `L2JBL`，缺 `L2_AMO` 分档、`WINNER/COST/PPART`，无数据时用量价 CLV 代理。

## G. 当前70/30指数增强组合为什么没有跑赢？

增强仓是事件驱动启动后的少量股票，对核心 70% 指数的偏离小；选股腿相对未入选等权并不稳定；T+1 开盘、费用、热度降仓、overlay 减弱止盈，都不是稳定 alpha 引擎。权重网格（不寻优）若多数 satellite 仍无稳定正超额，则结论是：当前信号适合事件驱动选股检验，但尚未形成稳定指数增强组合。

近期窗口 {ENHANCE_WINDOW_START} 网格：

{_fmt(enh, ['label','net_return','hs300_return','excess_hs300','max_drawdown','avg_turnover'])}

网格是否全部非正超额：{enh_all_neg}

## H. 目前最值得优先修改的3个问题是什么？

1. 把生产路径（`backtest.py` / `enhance.py` / `screen.py` / UI）从收盘成交改到与研究模块相同的 T+1 开盘，并输出 signal_date/entry_date 账本，避免两套口径。
2. 质量评级与 20 日 MFE 标签彻底隔离出交易决策；事件研究与状态机策略分开展示（本次模块已分开，产品文案仍须改）。
3. 补点-in-time 成分股与（若数据允许）行业中性基准，并在 Train 冻结后再看 Test；不要用增强仓权重去“修”近期负超额。

## 状态机仓位审查（未改参数）

启动满仓 100%，随后 JCTREND 降到 70%、JCBAND 降到 50%，止盈/逃顶清空。经济含义上这是风险事件减仓，不是趋势确认加仓。启动 100% 而“趋势环境 JC”后 70%，与“趋势更好应加仓”的直觉冲突；是否合理取决于 JC_EVENT 是否真是风险。三种仓位对比见表8，本次不改公式仓位。

{_fmt(t8, ['label','net_return','excess_hs300','max_drawdown','avg_turnover'])}
"""
