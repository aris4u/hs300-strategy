"""Draw a stock candlestick chart with F / LAUNCH / JC_BAND / JC_TREND / CT.

Usage:
    python plot_stock.py 600039.SH
    python plot_stock.py 600039.SH --bars 1260
    python plot_stock.py 000333.SZ --no-flow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _normalize_code(raw: str) -> str:
    code = raw.strip().upper().replace("_", ".")
    if "." not in code and code.isdigit():
        if code.startswith("6") or code.startswith("9"):
            code = f"{code}.SH"
        else:
            code = f"{code}.SZ"
    return code


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Stock K-line with strategy markers")
    parser.add_argument("code", help="ts_code, e.g. 600039.SH or 600039")
    parser.add_argument("--bars", type=int, default=0, help="recent bars to draw, default 3 years (756)")
    parser.add_argument("--start", default="20100101")
    parser.add_argument("--end", default="")
    parser.add_argument("--no-flow", action="store_true")
    args = parser.parse_args()

    from hs300_strategy.charts import LOOKBACK_BARS, plot_stock

    code = _normalize_code(args.code)
    sig, path = plot_stock(
        code,
        bars=args.bars or LOOKBACK_BARS,
        start=args.start,
        end=args.end or None,
        with_flow=not args.no_flow,
    )
    last = sig.iloc[-1]
    n_launch = int((sig["launch_turn"] == 1).sum())
    print(f"{code}  {last['date'].date()}  close {last['close']:.2f}")
    print(f"label {last['label']}  position {float(last['position']):.0%}  env {int(last['env_level'])}  dist {last['dist_score']:.0f}")
    print(f"launch_turn {n_launch}  f_signal {(sig['f_signal']==1).sum()}  JC_BAND {(sig['reduce_band']==1).sum()}  JC_TREND {(sig['reduce_trend']==1).sum()}  CT {(sig['escape_top']==1).sum()}")
    print(f"K线图 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
