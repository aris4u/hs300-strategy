"""沪深300指数增强回测。

默认同时跑两套：方案一环境评分Top5，方案二启动后拿到逃顶。

用法：
    python run_enhance.py
    python run_enhance.py --scheme both
    python run_enhance.py --scheme env_top5
    python run_enhance.py --scheme ct_all
    python run_enhance.py --start-bt 20240901 --satellite 0.30
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

    parser = argparse.ArgumentParser(description="沪深300指数增强回测")
    parser.add_argument("--start", default="20100101", help="信号计算起点")
    parser.add_argument("--end", default="")
    parser.add_argument("--start-bt", default="20240901", help="超额回测起点")
    parser.add_argument("--satellite", type=float, default=0.30, help="增强仓比例")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-flow", action="store_true")
    parser.add_argument(
        "--scheme",
        default="both",
        choices=["both", "env_top5", "ct_all"],
        help="both=两套都跑；env_top5=方案一；ct_all=方案二",
    )
    args = parser.parse_args()

    from hs300_strategy.enhance import ALL_SCHEMES, format_report, run_enhance

    wanted = ALL_SCHEMES if args.scheme == "both" else (args.scheme,)
    results = run_enhance(
        start=args.start,
        end=args.end or None,
        bt_start=args.start_bt,
        satellite=args.satellite,
        with_flow=not args.no_flow,
        limit=args.limit or None,
        schemes=wanted,
    )
    for sid, result in results.items():
        print()
        print("=" * 72)
        print(format_report(result.metrics, result.monthly))
        if result.selection_text:
            print()
            print(result.selection_text)
        files = {
            "env_top5": (
                ROOT / "output" / "enhance_equity.csv",
                ROOT / "output" / "enhance_monthly.csv",
                ROOT / "output" / "enhance_metrics.json",
                ROOT / "output" / "enhance.png",
                ROOT / "output" / "selection.json",
            ),
            "ct_all": (
                ROOT / "output" / "enhance_ct_equity.csv",
                ROOT / "output" / "enhance_ct_monthly.csv",
                ROOT / "output" / "enhance_ct_metrics.json",
                ROOT / "output" / "enhance_ct.png",
                ROOT / "output" / "selection_ct.json",
            ),
        }.get(sid)
        if files:
            print()
            print(f"净值 {files[0]}")
            print(f"月度 {files[1]}")
            print(f"指标 {files[2]}")
            print(f"图   {files[3]}")
            print(f"选股检验 {files[4]}")
    print()
    print(f"两套对照 {ROOT / 'output' / 'enhance_schemes.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
