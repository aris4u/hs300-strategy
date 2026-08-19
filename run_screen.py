"""沪深300 启动筛选：L2 流入 + 时点质量，并做选股有效性检验。

用法：
    python run_screen.py
    python run_screen.py --no-live
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
    parser = argparse.ArgumentParser(description="沪深300 选股筛选")
    parser.add_argument("--start-bt", default="20230101")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-flow", action="store_true")
    parser.add_argument("--no-live", action="store_true", help="不拉通达信五档")
    args = parser.parse_args()

    from hs300_strategy.screen import run_screen

    result = run_screen(
        bt_start=args.start_bt,
        with_flow=not args.no_flow,
        with_live_tdx=not args.no_live,
        limit=args.limit or None,
    )
    print()
    today = result["today"]
    print(f"今日筛选  推荐 {result['n_pick']}  观察 {result['n_watch']}")
    if today.empty:
        print("（今日没有进入推荐或观察池的股票）")
    else:
        cols = [c for c in (
            "bucket", "ts_code", "name", "signal_date", "entry_date", "entry_price",
            "bars_ago", "l2jbl", "position", "tdx",
        ) if c in today.columns]
        print(today[cols].to_string(index=False))
    print()
    print(result["proof_text"])
    if result["proof"].get("screen_rule"):
        print()
        print(result["proof"]["screen_rule"])
    print()
    print(f"今日 {ROOT / 'output' / 'screen_today.csv'}")
    print(f"检验 {ROOT / 'output' / 'screen_selection.json'}")
    print(f"图   {ROOT / 'output' / 'screen_selection.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
