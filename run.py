"""本地运行：下载沪深300，计算建仓/减仓/离场提示，并做指数回测。

用法：
    python run.py
    python run.py --start 20180101
    python run.py --cache-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="沪深300 A层状态机回测")
    parser.add_argument("--start", default="20100101", help="起始日期 YYYYMMDD")
    parser.add_argument("--end", default="", help="结束日期 YYYYMMDD，默认今天")
    parser.add_argument("--cache-only", action="store_true", help="只读本地 data/hs300.csv")
    parser.add_argument("--no-flow", action="store_true", help="不拉 Tushare 资金流，只用 K 线")
    args = parser.parse_args()

    from hs300_strategy.backtest import format_key_signals, format_metrics, run_backtest
    from hs300_strategy.data import CACHE_FILE, fetch_hs300
    from hs300_strategy.formula import compute_signals
    from hs300_strategy.moneyflow import fetch_moneyflow, merge_moneyflow
    import pandas as pd

    print("正在获取沪深300日K …")
    if args.cache_only:
        if not CACHE_FILE.exists():
            print(f"没有缓存文件：{CACHE_FILE}")
            return 1
        kline = pd.read_csv(CACHE_FILE, parse_dates=["date"])
    else:
        kline = fetch_hs300(start=args.start, end=args.end or None, use_cache=True)

    print(f"K线 {len(kline)} 根，{kline['date'].min().date()} ~ {kline['date'].max().date()}")
    if not args.no_flow:
        print("正在获取沪深300成分股资金流 …")
        try:
            flow = fetch_moneyflow(start=args.start, end=args.end or None, use_cache=True)
            kline = merge_moneyflow(kline, flow)
            n_flow = int(kline["l2jbl"].notna().sum())
            print(f"资金流对齐 {n_flow} 天，用作 L2 近似")
        except Exception as exc:
            print(f"资金流未接入，回退到纯 K 线：{exc}")
    print("正在计算状态机 …")
    signals = compute_signals(kline, asset="index")
    result = run_backtest(signals)

    latest = signals.iloc[-1]
    print()
    print(f"最新交易日 {latest['date'].date()}  收盘 {latest['close']:.2f}")
    print(
        f"今日提示：{latest['label']}  "
        f"（仓位 {float(latest.get('position', 0)):.0%}，环境 {int(latest['env_level'])}，出货评分 {latest['dist_score']:.0f}）"
    )
    if "l2_flow" in signals.columns and pd.notna(latest.get("l2_flow")):
        src = latest.get("l2_source", "")
        print(f"L2 {float(latest['l2_flow']):.3f}  来源 {src}")
    print()
    print(format_metrics("交易状态机：launch_turn 买入，止盈/CT/下跌环境退出", result.metrics_event))
    print()
    print(format_metrics("ui_state 覆盖状态（对照，不宜当仓位）", result.metrics_state))
    print()
    print("最近 10 个交易日：")
    cols = ["date", "close", "env_level", "label", "launch_turn", "take_profit", "escape_top"]
    if "l2_flow" in signals.columns:
        cols.insert(3, "l2_flow")
    tail = signals.tail(10)[cols]
    print(tail.to_string(index=False))
    print()
    print(format_key_signals(signals))
    print()
    print(f"信号表 {ROOT / 'output' / 'signals.csv'}")
    print(f"关键信号 {ROOT / 'output' / 'key_signals.csv'}")
    print(f"净值表 {result.equity_path}")
    print(f"净值图 {result.chart_path}")
    print(f"K线图 {result.kline_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
