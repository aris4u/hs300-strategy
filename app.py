"""沪深300 策略桌面界面。

用法：
    python app.py
    python app.py --port 8765 --no-browser
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hs300_strategy.ui_server import main

if __name__ == "__main__":
    raise SystemExit(main())
