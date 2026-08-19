"""After-close daily refresh: klines, L2, advice charts, screen, enhance.

Opening the UI starts a background loop. CSV is only written for completed sessions
(T close). Forming intraday bars stay in live_refresh memory, never as official cache.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from datetime import date
from pathlib import Path
from time import sleep

from hs300_strategy.data import (
    cache_asof_date,
    clear_kline_stamp,
    last_closed_session,
    now_cn,
    official_end_key,
    vendor_likely_done,
    write_kline_stamp,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
STATE_FILE = OUTPUT / "daily_update.json"
MEMBERS_MAX_AGE = 7 * 86400

_lock = threading.Lock()
_state: dict = {
    "running": False,
    "step": "",
    "message": "",
    "error": None,
    "closed": None,
    "cache_asof": None,
    "generation": 0,
    "steps_done": [],
    "started_at": None,
    "finished_at": None,
    "ok": False,
    "no_new_bar": False,
}
_loop_started = False


def _mtime(path: Path) -> int | None:
    if not path.exists():
        return None
    return int(path.stat().st_mtime)


def _persist() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(_state, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_persisted() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _set(**kwargs) -> None:
    _state.update(kwargs)
    _persist()


def _label_cls() -> tuple[str, str]:
    closed = _state.get("closed") or last_closed_session().isoformat()
    asof = _state.get("cache_asof")
    if _state.get("running"):
        step = _state.get("message") or _state.get("step") or "更新中"
        return f"日K更新中：{step}", "warn"
    if _state.get("step") == "wait":
        have = asof or "—"
        return f"等待今日收盘K（已有 {have}）", "warn"
    if _state.get("error"):
        return f"日K更新失败：{_state['error'][:40]}", "err"
    if asof:
        extra = "　节假日无新K" if _state.get("no_new_bar") else ""
        return f"日K截至 {asof}{extra}", "on"
    return f"日K待更新（收盘日 {closed}）", "warn"


def status() -> dict:
    with _lock:
        closed = last_closed_session().isoformat()
        asof = cache_asof_date()
        payload = dict(_state)
    payload["closed"] = closed
    payload["cache_asof"] = asof.isoformat() if asof else payload.get("cache_asof")
    payload["session_closed"] = official_end_key()
    payload["advice_mtime"] = _mtime(OUTPUT / "today_advice.csv")
    payload["enhance_mtime"] = _mtime(OUTPUT / "enhance_metrics.json")
    payload["screen_mtime"] = _mtime(OUTPUT / "screen_today.csv")
    label, cls = _label_cls()
    if not payload.get("running") and payload.get("step") != "wait":
        asof_s = payload.get("cache_asof")
        if asof_s and asof_s < closed and not payload.get("error"):
            t = now_cn()
            if t.weekday() < 5 and t.strftime("%H:%M") < "15:20":
                label, cls = f"日K截至 {asof_s}　盘中不写未收盘K", "on"
            else:
                label, cls = f"日K截至 {asof_s}　待补 {closed}", "warn"
    payload["label"] = label
    payload["cls"] = cls
    payload["ok"] = True
    return payload


def _already_done(closed: date, force: bool) -> bool:
    if force:
        return False
    persisted = _load_persisted()
    if str(persisted.get("asof_closed") or "") != closed.isoformat():
        return False
    done = set(persisted.get("steps_done") or [])
    return "charts" in done and persisted.get("ok") is True


def _optional_pending(closed: date) -> list[str]:
    persisted = _load_persisted()
    if str(persisted.get("asof_closed") or "") != closed.isoformat():
        return ["moneyflow", "screener", "screen", "enhance"]
    done = set(persisted.get("steps_done") or [])
    return [s for s in ("moneyflow", "screener", "screen", "enhance") if s not in done]


def _refresh_members():
    from hs300_strategy.stock_data import CONSTITUENT_FILE, fetch_constituents, fetch_industries

    stale = (not CONSTITUENT_FILE.exists()) or (time.time() - CONSTITUENT_FILE.stat().st_mtime > MEMBERS_MAX_AGE)
    members = fetch_constituents(use_cache=not stale)
    if stale:
        print("成分名单已刷新", flush=True)
        try:
            fetch_industries(use_cache=False)
        except Exception as exc:
            print(f"行业分类跳过：{exc}", flush=True)
    return members


def _probe_index(end_key: str) -> date | None:
    from hs300_strategy.data import fetch_hs300

    fetch_hs300(start="20100101", end=end_key, use_cache=True)
    return cache_asof_date()


def _update_klines(end_key: str, members) -> None:
    from hs300_strategy.stock_data import fetch_many_klines

    codes = members["ts_code"].tolist()
    print(f"补成分股日K {len(codes)} 只 …", flush=True)
    fetch_many_klines(codes, "20100101", end_key, use_cache=True)


def _update_moneyflow(end_key: str, members) -> None:
    from hs300_strategy.moneyflow import fetch_moneyflow, fetch_stock_l2
    from hs300_strategy.secrets import tushare_pro

    fetch_moneyflow(start="20100101", end=end_key, use_cache=True)
    pro = tushare_pro()
    codes = members["ts_code"].tolist()
    for i, code in enumerate(codes, start=1):
        fetch_stock_l2(code, "20100101", end_key, pro=pro, use_cache=True)
        if i % 20 == 0 or i == len(codes):
            _set(message=f"资金流 {i}/{len(codes)}")
            print(f"  资金流 {i}/{len(codes)}", flush=True)
        sleep(0.12)


def _clear_runtime_caches() -> None:
    try:
        from hs300_strategy.charts import invalidate_caches

        invalidate_caches()
    except Exception:
        pass
    try:
        from hs300_strategy.ui_server import invalidate_ui_caches

        invalidate_ui_caches()
    except Exception:
        pass


def _bump() -> None:
    _state["generation"] = int(_state.get("generation") or 0) + 1
    _persist()


def run_daily_update(force: bool = False) -> dict:
    """Refresh stale official daily data and rebuild derived tables. Safe to call often."""
    closed = last_closed_session()
    end_key = closed.strftime("%Y%m%d")
    if not _lock.acquire(blocking=False):
        return status()
    try:
        if _state.get("running"):
            return status()
        persisted = _load_persisted()
        if persisted.get("generation"):
            _state["generation"] = int(persisted["generation"])
        if _already_done(closed, force) and not _optional_pending(closed):
            asof = cache_asof_date()
            _set(
                running=False,
                step="idle",
                message="",
                error=None,
                closed=closed.isoformat(),
                cache_asof=asof.isoformat() if asof else None,
                asof_closed=closed.isoformat(),
                steps_done=persisted.get("steps_done") or ["charts"],
                ok=True,
                no_new_bar=bool(persisted.get("no_new_bar")),
            )
            return status()

        if force:
            clear_kline_stamp()

        _set(
            running=True,
            step="klines",
            message="检查日K",
            error=None,
            closed=closed.isoformat(),
            started_at=now_cn().strftime("%Y-%m-%d %H:%M:%S"),
            finished_at=None,
            ok=False,
            no_new_bar=False,
            steps_done=list(persisted.get("steps_done") or []) if not force and str(persisted.get("closed") or persisted.get("asof_closed") or "") == closed.isoformat() else [],
        )
        print(f"自动更新  目标收盘日 {closed.isoformat()}", flush=True)

        members = _refresh_members()
        _set(message="沪深300指数")
        asof = _probe_index(end_key)
        done = list(_state.get("steps_done") or [])

        if asof is None or asof < closed:
            if not vendor_likely_done():
                _set(
                    running=False,
                    step="wait",
                    message="等待数据源今日收盘K",
                    cache_asof=asof.isoformat() if asof else None,
                    ok=False,
                )
                print("指数还没有今日K，稍后重试（盘中不把未收盘K写入 CSV）", flush=True)
                return status()
            write_kline_stamp(end_key)
            _set(no_new_bar=True, message="节假日或数据源无新K")
            print("今日无新K（假期或源未出），沿用现有缓存", flush=True)
        else:
            _set(message="成分股日K", cache_asof=asof.isoformat())
            _update_klines(end_key, members)
            write_kline_stamp(end_key)
            asof = cache_asof_date()
            _set(cache_asof=asof.isoformat() if asof else None)

        if "klines" not in done:
            done.append("klines")
            _set(steps_done=done)

        skip_rebuild = bool(_state.get("no_new_bar")) and not force
        if not skip_rebuild:
            if force or "moneyflow" not in done:
                _set(step="moneyflow", message="资金流")
                try:
                    _update_moneyflow(end_key, members)
                    if "moneyflow" not in done:
                        done.append("moneyflow")
                        _set(steps_done=done)
                except Exception as exc:
                    print(f"资金流跳过：{exc}", flush=True)
                    if "moneyflow" not in done:
                        done.append("moneyflow")
                        _set(steps_done=done)

            if force or "screener" not in done:
                _set(step="screener", message="事后档案")
                try:
                    from hs300_strategy.screener import run_screener

                    run_screener(end=end_key, use_cache=True, plot_top=0)
                    if "screener" not in done:
                        done.append("screener")
                        _set(steps_done=done)
                except Exception as exc:
                    print(f"事后档案跳过：{exc}", flush=True)
                    if "screener" not in done:
                        done.append("screener")
                        _set(steps_done=done)

            if force or "charts" not in done:
                _clear_runtime_caches()
                _set(step="charts", message="重画建议图")
                from hs300_strategy.charts import plot_universe

                ok, fail = plot_universe(None)
                print(f"建议图 成功 {ok}  失败 {fail}", flush=True)
                if "charts" not in done:
                    done.append("charts")
                _set(steps_done=done)
                _bump()
                _clear_runtime_caches()

            if force or "screen" not in done:
                _set(step="screen", message="选股筛选")
                try:
                    from hs300_strategy.screen import run_screen

                    run_screen(end=end_key, use_cache=True, with_live_tdx=True)
                    if "screen" not in done:
                        done.append("screen")
                        _set(steps_done=done)
                    _bump()
                except Exception as exc:
                    print(f"选股筛选失败：{exc}", flush=True)
                    if "screen" not in done:
                        done.append("screen")
                        _set(steps_done=done)

            if force or "enhance" not in done:
                _set(step="enhance", message="指数增强回测")
                try:
                    from hs300_strategy.enhance import run_enhance

                    run_enhance(end=end_key, use_cache=True)
                    if "enhance" not in done:
                        done.append("enhance")
                        _set(steps_done=done)
                    _bump()
                except Exception as exc:
                    print(f"指数增强失败：{exc}", flush=True)
                    if "enhance" not in done:
                        done.append("enhance")
                        _set(steps_done=done)
        elif "charts" not in done and (OUTPUT / "today_advice.csv").exists():
            done.append("charts")
            _set(steps_done=done)

        asof = cache_asof_date()
        _set(
            running=False,
            step="idle",
            message="",
            error=None,
            cache_asof=asof.isoformat() if asof else None,
            asof_closed=closed.isoformat(),
            finished_at=now_cn().strftime("%Y-%m-%d %H:%M:%S"),
            ok=True,
            steps_done=done,
        )
        print(f"自动更新完成  K截至 {asof}", flush=True)
        return status()
    except Exception as exc:
        traceback.print_exc()
        _set(
            running=False,
            step="error",
            message="",
            error=str(exc),
            finished_at=now_cn().strftime("%Y-%m-%d %H:%M:%S"),
            ok=False,
        )
        return status()
    finally:
        _lock.release()


def start_background_loop() -> None:
    global _loop_started
    if _loop_started:
        return
    _loop_started = True
    persisted = _load_persisted()
    if persisted:
        for key in ("generation", "steps_done", "cache_asof", "asof_closed", "ok", "no_new_bar", "error"):
            if key in persisted:
                _state[key] = persisted[key]

    def loop() -> None:
        while True:
            try:
                run_daily_update(force=False)
            except Exception:
                traceback.print_exc()
            time.sleep(60)

    threading.Thread(target=loop, name="daily-update", daemon=True).start()
    print("已启动收盘后自动更新（缺最新交易日才拉数，盘中不写未收盘K）", flush=True)
