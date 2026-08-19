"""从本地 .env 读取密钥，不把 token 写进代码。"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
DEFAULT_HTTP_URL = "http://jiaoch.site"


def load_env(path: Path = ENV_FILE) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


def tushare_token() -> str:
    load_env()
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("未找到 TUSHARE_TOKEN。请写在项目根目录 .env 里。")
    return token


def tushare_http_url() -> str:
    load_env()
    url = os.environ.get("TUSHARE_HTTP_URL", DEFAULT_HTTP_URL).strip().rstrip("/")
    return url or DEFAULT_HTTP_URL


def tushare_pro(timeout: int = 60):
    """按第三方教程改写 DataApi 内部地址；官方 tushare.pro 不认这份 token。"""
    from hs300_strategy.data import disable_http_proxy

    disable_http_proxy()
    import tushare as ts

    token = tushare_token()
    pro = ts.pro_api(token, timeout=timeout)
    pro._DataApi__token = token
    pro._DataApi__http_url = tushare_http_url()
    return pro
