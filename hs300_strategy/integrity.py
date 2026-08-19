"""Refuse to start if core files were emptied by sync or a bad copy."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (relative path, minimum bytes)
REQUIRED = (
    ("app.py", 200),
    ("hs300_strategy/formula.py", 8000),
    ("hs300_strategy/ops.py", 500),
    ("hs300_strategy/ui_server.py", 2000),
    ("hs300_strategy/ui_static/index.html", 2000),
)


def problems() -> list[str]:
    out: list[str] = []
    for rel, min_size in REQUIRED:
        path = ROOT / rel
        if not path.exists():
            out.append(f"缺少 {rel}")
            continue
        size = path.stat().st_size
        if size < min_size:
            out.append(f"{rel} 只有 {size} 字节（应大于 {min_size}），多半被同步成空文件")
    return out


def main() -> int:
    bad = problems()
    if not bad:
        return 0
    print("核心文件损坏，不能启动。")
    for line in bad:
        print("  -", line)
    print("请重新解压软件包，或打开 C:\\Users\\%USERNAME%\\HS300 里的完整副本。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
