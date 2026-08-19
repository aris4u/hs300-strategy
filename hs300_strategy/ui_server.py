"""Local desktop UI for HS300 advice + charts. Bind 127.0.0.1 only."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime, time as dtime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd

from hs300_strategy.advise import ENV_CN, _grasp_note, _hist_text
from hs300_strategy.config import LIVE_POLL_CLOSED_SECONDS, LIVE_POLL_SECONDS, LIVE_PREVIEW_NOTE
from hs300_strategy.data import DATA_DIR, read_csv_cached
from hs300_strategy.events import QUALITY_CN

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
CHART_DIR = OUTPUT / "charts"
STATIC = Path(__file__).resolve().parent / "ui_static"
STOCK_DIR = DATA_DIR / "stocks"

ACTION_ORDER = {"试多": 0, "持有": 1, "减仓": 2, "清仓": 3, "观望": 4}


def _clean(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item") and not isinstance(v, (bytes, str, dict, list, tuple)):
        try:
            v = v.item()
        except (ValueError, AttributeError):
            pass
        try:
            if pd.isna(v):
                return None
        except (TypeError, ValueError):
            pass
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _json_safe(v):
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    return _clean(v)


def _pct(v, digits: int = 1) -> str | None:
    v = _clean(v)
    if v is None:
        return None
    return f"{float(v) * 100:.{digits}f}%"


def _asof() -> str | None:
    sample = STOCK_DIR / "600015_SH.csv"
    path = sample if sample.exists() else next(STOCK_DIR.glob("*.csv"), None)
    if path is None:
        return None
    dates = pd.read_csv(path, usecols=["date"])
    return str(pd.to_datetime(dates["date"]).max().date())


def _chart_name(code: str) -> str:
    return f"{code.replace('.', '_')}.png"


def _session_open_ui() -> bool:
    try:
        from zoneinfo import ZoneInfo

        now = datetime.now(ZoneInfo("Asia/Shanghai"))
    except Exception:
        now = datetime.utcnow() + timedelta(hours=8)
    if now.weekday() >= 5:
        return False
    t = now.time()
    return dtime(9, 15) <= t <= dtime(15, 5)


_DETAIL_CACHE: dict[str, tuple[float, dict]] = {}


def invalidate_ui_caches() -> None:
    _DETAIL_CACHE.clear()


def _load_scheme_pack(stem: str, chart_url: str) -> dict:
    metrics_path = OUTPUT / f"{stem}_metrics.json"
    if not metrics_path.exists():
        return {"ok": False, "error": f"还没有 {stem} 回测。请先运行 python run_enhance.py"}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    monthly = []
    mpath = OUTPUT / f"{stem}_monthly.csv"
    if mpath.exists():
        df = pd.read_csv(mpath)
        for _, row in df.iterrows():
            monthly.append(
                {
                    "month": str(row["month"]),
                    "strategy": _clean(row["strategy"]),
                    "benchmark": _clean(row["benchmark"]),
                    "excess": _clean(row["excess"]),
                }
            )
    chart_file = OUTPUT / f"{stem}.png"
    return {
        "ok": True,
        "id": metrics.get("scheme"),
        "name": metrics.get("scheme_name") or metrics.get("product"),
        "metrics": metrics,
        "monthly": monthly,
        "chart": chart_url,
        "has_chart": chart_file.exists(),
    }


def load_enhance() -> dict:
    env_pack = _load_scheme_pack("enhance", "/enhance.png")
    ct_pack = _load_scheme_pack("enhance_ct", "/enhance_ct.png")
    schemes = {}
    if env_pack.get("ok"):
        env_pack["id"] = env_pack.get("id") or "env_top5"
        schemes["env_top5"] = env_pack
    if ct_pack.get("ok"):
        ct_pack["id"] = ct_pack.get("id") or "ct_all"
        schemes["ct_all"] = ct_pack
    if not schemes:
        return {"ok": False, "error": "还没有增强回测。请先运行 python run_enhance.py"}
    primary = schemes.get("env_top5") or next(iter(schemes.values()))
    return {
        "ok": True,
        "default": "env_top5" if "env_top5" in schemes else next(iter(schemes)),
        "schemes": schemes,
        "metrics": primary["metrics"],
        "monthly": primary["monthly"],
        "chart": primary["chart"],
        "has_chart": primary["has_chart"],
    }


def load_screen() -> dict:
    today_path = OUTPUT / "screen_today.csv"
    proof_path = OUTPUT / "screen_selection.json"
    if not today_path.exists() and not proof_path.exists():
        return {"ok": False, "error": "还没有筛选结果。请先运行 python run_screen.py"}
    today = []
    if today_path.exists():
        df = pd.read_csv(today_path)
        for _, row in df.iterrows():
            today.append({c: _clean(row[c]) for c in df.columns})
    proof = {}
    if proof_path.exists():
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
    return {
        "ok": True,
        "today": today,
        "proof": proof,
        "chart": "/screen_selection.png",
        "has_chart": (OUTPUT / "screen_selection.png").exists(),
    }


def load_board() -> dict:
    advice_path = OUTPUT / "today_advice.csv"
    rank_path = OUTPUT / "stock_rank.csv"
    if not advice_path.exists():
        return {
            "ok": False,
            "error": "还没有当日建议表。请先运行 python plot_all.py",
            "asof": _asof(),
            "counts": {},
            "stocks": [],
            "live_poll_seconds": LIVE_POLL_SECONDS,
            "live_poll_closed_seconds": LIVE_POLL_CLOSED_SECONDS,
            "session_open": False,
            "live_note": LIVE_PREVIEW_NOTE,
        }
    advice = pd.read_csv(advice_path)
    rank = pd.read_csv(rank_path) if rank_path.exists() else pd.DataFrame()
    if not rank.empty:
        keep = [
            "ts_code",
            "last_launch",
            "bars_ago",
            "last_quality",
            "last_excess_mfe_20",
            "last_mae_20",
            "last_excess_ret_20",
            "n_hist",
            "n_high",
            "n_low",
            "hit_rate",
            "med_excess_mfe_20",
            "med_mae_20",
            "hist_ok",
            "last_is_low",
            "score",
            "env_level",
            "recommend_rank",
        ]
        keep = [c for c in keep if c in rank.columns]
        advice = advice.merge(rank[keep], on="ts_code", how="left")
    advice["ord"] = advice["action"].map(ACTION_ORDER).fillna(9)
    advice = advice.sort_values(["ord", "position", "ts_code"], ascending=[True, False, True])
    stocks = []
    for _, row in advice.iterrows():
        rec = {c: _clean(row[c]) for c in advice.columns if c != "ord"}
        code = str(rec["ts_code"])
        rec["chart"] = f"/charts/{_chart_name(code)}"
        png = CHART_DIR / _chart_name(code)
        rec["has_chart"] = png.exists()
        rec["chart_ts"] = int(png.stat().st_mtime) if png.exists() else None
        rec["last_quality_cn"] = QUALITY_CN.get(str(rec.get("last_quality") or ""), rec.get("last_quality"))
        rec["env_cn"] = ENV_CN.get(int(rec["env_level"]) if rec.get("env_level") is not None else 0, "")
        rec["hit_rate_s"] = _pct(rec.get("hit_rate"), 0)
        rec["med_mfe_s"] = _pct(rec.get("med_excess_mfe_20"), 1)
        rec["last_mfe_s"] = _pct(rec.get("last_excess_mfe_20"), 1)
        rec["last_mae_s"] = _pct(rec.get("last_mae_20"), 1)
        rank_row = rec if rec.get("n_hist") is not None else None
        rec["hist_text"] = _hist_text(rank_row)
        rec["grasp_note"] = _grasp_note(rec.get("confidence"))
        rec["archive_only"] = True
        rec["execution"] = (
            "T日收盘生成信号，T+1日开盘成交。禁止使用T日收盘价作为成交价。"
            "事后质量/MFE不参与今日动作。"
        )
        stocks.append(rec)
    counts = {k: 0 for k in ACTION_ORDER}
    for s in stocks:
        counts[s["action"]] = counts.get(s["action"], 0) + 1
    mtime = advice_path.stat().st_mtime
    return {
        "ok": True,
        "asof": _asof(),
        "generated_at": pd.Timestamp(mtime, unit="s").strftime("%Y-%m-%d %H:%M"),
        "n": len(stocks),
        "counts": counts,
        "stocks": stocks,
        "live_poll_seconds": LIVE_POLL_SECONDS,
        "live_poll_closed_seconds": LIVE_POLL_CLOSED_SECONDS,
        "session_open": _session_open_ui(),
        "live_note": LIVE_PREVIEW_NOTE,
    }


def load_stock_detail(code: str) -> dict:
    from hs300_strategy.advise import make_advice
    from hs300_strategy.charts import load_stock_signals

    now = time.time()
    hit = _DETAIL_CACHE.get(code)
    ttl = 5.0 if _session_open_ui() else 3600.0
    if hit and now - hit[0] < ttl:
        return hit[1]

    rank_path = OUTPUT / "stock_rank.csv"
    rank_row = None
    if rank_path.exists():
        rank = read_csv_cached(rank_path)
        rhit = rank[rank["ts_code"] == code]
        if not rhit.empty:
            rank_row = {c: _clean(rhit.iloc[0][c]) for c in rhit.columns}
    sig = load_stock_signals(code, use_cache=True, with_flow=True)
    last = sig.iloc[-1]
    adv = make_advice(sig, rank_row)
    launches_path = OUTPUT / "stock_launches.csv"
    launches = []
    if launches_path.exists():
        ev = pd.read_csv(launches_path)
        sub = ev[ev["ts_code"] == code].copy()
        if not sub.empty:
            sub["date"] = pd.to_datetime(sub["date"])
            sub = sub.sort_values("date", ascending=False).head(16)
            for _, row in sub.iterrows():
                launches.append(
                    {
                        "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                        "quality": QUALITY_CN.get(str(row.get("quality", "")), str(row.get("quality", ""))),
                        "excess_mfe_20": _pct(row.get("excess_mfe_20"), 1),
                        "mae_20": _pct(row.get("mae_20"), 1),
                        "excess_ret_20": _pct(row.get("excess_ret_20"), 1),
                        "efficiency_20": _clean(row.get("efficiency_20")),
                        "bars_20": _clean(row.get("bars_20")),
                    }
                )
    result = {
        "ok": True,
        "ts_code": code,
        "action": adv["action"],
        "position_hint": adv["position_hint"],
        "confidence": adv["confidence"],
        "headline": adv["headline"],
        "detail": adv["detail"],
        "flags": adv["flags"],
        "position": adv["position"],
        "color": adv["color"],
        "close": float(last["close"]),
        "date": pd.Timestamp(last["date"]).strftime("%Y-%m-%d"),
        "env": ENV_CN.get(int(last["env_level"]), str(last["env_level"])),
        "dist_score": float(last.get("dist_score", 0) or 0),
        "signal_date": adv.get("signal_date"),
        "entry_date": adv.get("entry_date"),
        "entry_price": adv.get("entry_price"),
        "exit_date": adv.get("exit_date"),
        "exit_price": adv.get("exit_price"),
        "execution": adv.get("execution"),
        "launches": launches,
        "launches_note": "事后20日路径档案，非实时过滤",
        "has_chart": (CHART_DIR / _chart_name(code)).exists(),
        "chart": f"/charts/{_chart_name(code)}",
    }
    _DETAIL_CACHE[code] = (now, result)
    return result


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        msg = fmt % args
        if "/api/tdx" in msg or "/api/live" in msg or "/api/daily" in msg:
            return
        sys.stderr.write("[ui] " + msg + "\n")

    def _send_png(self, file: Path) -> None:
        st = file.stat()
        etag = f'"{int(st.st_mtime)}-{st.st_size}"'
        if self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", "private, max-age=86400")
            self.end_headers()
            return
        data = file.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("ETag", etag)
        self.send_header("Cache-Control", "private, max-age=86400")
        self.end_headers()
        self.wfile.write(data)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(
            _json_safe(payload), ensure_ascii=False, default=str, allow_nan=False
        ).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/" or path == "/index.html":
            html = (STATIC / "index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
            return
        if path == "/api/board":
            self._json(load_board())
            return
        if path == "/api/daily":
            from hs300_strategy.daily_update import status

            self._json(status())
            return
        if path == "/api/tdx":
            qs = parse_qs(parsed.query)
            raw = (qs.get("codes") or [""])[0]
            codes = [c.strip() for c in raw.split(",") if c.strip()]
            if not codes:
                codes = ["000300.SH"]
            try:
                from datetime import datetime, timedelta

                from hs300_strategy.tdx_l2 import snap_quotes

                try:
                    from zoneinfo import ZoneInfo

                    now = datetime.now(ZoneInfo("Asia/Shanghai"))
                except Exception:
                    now = datetime.utcnow() + timedelta(hours=8)
                quotes = snap_quotes(codes)
                self._json(
                    {
                        "ok": True,
                        "quote_time": now.strftime("%H:%M:%S"),
                        "quotes": quotes,
                    }
                )
            except Exception as exc:
                self._json({"ok": False, "error": str(exc), "quotes": {}}, 500)
            return
        if path == "/api/enhance":
            self._json(load_enhance())
            return
        if path == "/api/screen":
            self._json(load_screen())
            return
        if path == "/screen_selection.png":
            file = OUTPUT / "screen_selection.png"
            if not file.exists():
                self._send(404, b"missing", "text/plain")
                return
            self._send_png(file)
            return
        if path in {"/enhance.png", "/enhance_ct.png", "/selection_ct.png"}:
            file = OUTPUT / path.lstrip("/")
            if not file.exists():
                self._send(404, b"missing", "text/plain")
                return
            self._send_png(file)
            return
        if path.startswith("/api/stock/"):
            code = path[len("/api/stock/") :].upper().replace("_", ".")
            if "." not in code and code.isdigit():
                code = f"{code}.SH" if code.startswith(("6", "9")) else f"{code}.SZ"
            try:
                self._json(load_stock_detail(code))
            except Exception as exc:
                self._json({"ok": False, "error": str(exc), "ts_code": code}, 500)
            return
        if path.startswith("/charts/"):
            name = Path(path).name
            if not name.endswith(".png") or ".." in name:
                self._json({"ok": False, "error": "bad chart"}, 400)
                return
            file = CHART_DIR / name
            if not file.exists():
                self._send(404, b"missing", "text/plain")
                return
            self._send_png(file)
            return
        if path.startswith("/api/live/"):
            code = path[len("/api/live/") :].upper().replace("_", ".")
            if "." not in code and code.isdigit():
                code = f"{code}.SH" if code.startswith(("6", "9")) else f"{code}.SZ"
            qs = parse_qs(parsed.query)
            force_plot = (qs.get("manual") or ["0"])[0] == "1"
            try:
                from hs300_strategy.live_refresh import refresh_stock

                self._json(refresh_stock(code, redraw=True, force_plot=force_plot))
            except Exception as exc:
                self._json({"ok": False, "error": str(exc), "ts_code": code}, 500)
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        if path == "/api/daily":
            from hs300_strategy.daily_update import run_daily_update, status

            threading.Thread(target=run_daily_update, kwargs={"force": True}, daemon=True).start()
            self._json(status())
            return
        self._send(404, b"not found", "text/plain")


def _free_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])


def open_app_window(url: str) -> None:
    local = os.environ.get("LOCALAPPDATA", "")
    candidates = [
        Path(local) / "Microsoft/Edge/Application/msedge.exe",
        Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
        Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
        Path(local) / "Google/Chrome/Application/chrome.exe",
        Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
    ]
    for exe in candidates:
        if exe.exists():
            subprocess.Popen(
                [
                    str(exe),
                    f"--app={url}",
                    "--window-size=1560,960",
                    "--window-position=60,30",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
    webbrowser.open(url)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="沪深300 策略桌面界面")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    port = _free_port(args.port)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"界面已启动  {url}")
    print("关掉这个窗口即退出。规则提示，不是投资建议。")
    from hs300_strategy.daily_update import start_background_loop

    start_background_loop()
    if not args.no_browser:
        threading.Timer(0.4, open_app_window, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")
    finally:
        httpd.server_close()
    return 0
