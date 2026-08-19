"""事后档案：启动事件的主观质量评级（非实时选股）。

用法：
    python run_screener.py
实时交易决策请用 python run_screen.py（不用 MFE/质量过滤）。
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

    parser = argparse.ArgumentParser(description="启动事件事后质量档案（非实时过滤）")
    parser.add_argument("--start", default="20100101")
    parser.add_argument("--end", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-flow", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    from hs300_strategy.config import RATING_KIND
    from hs300_strategy.events import QUALITY_CN
    from hs300_strategy.screener import run_screener

    events, ranked = run_screener(
        start=args.start,
        end=args.end or None,
        use_cache=True,
        limit=args.limit or None,
        with_flow=not args.no_flow,
        plot_top=0 if args.no_plot else 12,
    )
    recent = ranked[ranked["recent_launch"] == 1].head(20)
    n_high = int((events["quality"] == "high_value").sum())
    n_low = int((events["quality"] == "low_value").sum())
    n_neu = int((events["quality"] == "neutral").sum())
    n_watch = int((events["quality"] == "watching").sum())
    print()
    print("======== 事后质量档案（非实时选股）========")
    print(f"评级类型：{RATING_KIND}")
    print("MFE/MAE/质量不得进入实时筛选与生产决策。")
    print("旧版「信号日收盘买入」口径已废弃；正式事件研究见 python run_research.py")
    print()
    print(f"启动点 {len(events)}  高价值 {n_high}  中性 {n_neu}  低价值 {n_low}  观察中 {n_watch}")
    print()
    print("近50日有启动的股票（档案排序，不是买入清单）")
    if recent.empty:
        print("（无）")
    else:
        show = recent.copy()
        show["last_quality"] = show["last_quality"].map(QUALITY_CN).fillna(show["last_quality"])
        cols = [c for c in (
            "ts_code", "name", "last_launch", "bars_ago", "last_quality",
            "last_excess_mfe_20", "last_mae_20", "hit_rate", "med_excess_mfe_20", "score",
        ) if c in show.columns]
        print(show[cols].to_string(index=False))
    print()
    print(f"启动明细 {ROOT / 'output' / 'stock_launches.csv'}")
    print(f"股票档案 {ROOT / 'output' / 'stock_rank.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
