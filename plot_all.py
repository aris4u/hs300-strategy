"""Draw 3-year K-line charts for all HS300 constituents.

Usage:
    python plot_all.py
    python plot_all.py --only-launch
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

    parser = argparse.ArgumentParser(description="Plot every HS300 stock K-line")
    parser.add_argument("--only-launch", action="store_true", help="only names that have launch_turn in stock_rank.csv")
    args = parser.parse_args()

    from hs300_strategy.charts import plot_universe
    from hs300_strategy.stock_data import fetch_constituents

    codes = None
    if args.only_launch:
        import pandas as pd

        rank = pd.read_csv(ROOT / "output" / "stock_rank.csv")
        codes = rank["ts_code"].tolist()
        print(f"只画有启动记录的 {len(codes)} 只")
    else:
        n = len(fetch_constituents(use_cache=True))
        print(f"画沪深300全部 {n} 只（不是全部深A）")

    ok, fail = plot_universe(codes)
    if fail:
        print(f"失败 {fail}")
        return 1
    print(f"完成 {ok} 张")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
