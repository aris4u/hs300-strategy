"""手动跑一遍收盘后自动更新（界面启动后也会自己跑）。

用法：
    python refresh_daily.py
    python refresh_daily.py --force
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
    parser = argparse.ArgumentParser(description="收盘后自动更新日K / 建议 / 筛选 / 增强")
    parser.add_argument("--force", action="store_true", help="即使今日已更新也重跑图和回测")
    args = parser.parse_args()

    from hs300_strategy.daily_update import run_daily_update

    result = run_daily_update(force=args.force)
    print()
    print(result.get("label") or "")
    if result.get("error"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
