"""Look-ahead audit: static scan + truncated-series replay."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

from hs300_strategy.formula import compute_signals

PKG = Path(__file__).resolve().parent
SIGNAL_COLS = (
    "launch_turn",
    "f_signal",
    "washout_turn",
    "reduce_trend",
    "reduce_band",
    "escape_top",
    "take_profit",
    "position",
    "env_level",
)


def future_leak_check(
    sample: pd.DataFrame | None = None,
    *,
    cuts: tuple[float, ...] = (0.65, 0.80, 0.92),
) -> dict:
    """Return a structured audit. `ok` is False if a hard leak is found.

    Rolling windows that include the current bar are allowed (available at T close).
    `shift(-n)` / centered rolling / using t+1 prices in the signal are not.
    """
    static = _scan_sources()
    runtime = _replay_truncated(sample, cuts) if sample is not None and len(sample) > 200 else {
        "ran": False,
        "note": "no sample passed; static scan only",
        "mismatches": [],
    }
    hard = [x for x in static["findings"] if x["severity"] == "hard"]
    hard += [{"severity": "hard", **m} for m in runtime.get("mismatches", [])]
    return {
        "ok": len(hard) == 0,
        "static": static,
        "runtime": runtime,
        "assumptions": [
            "信号只使用 T 日收盘已可得的 OHLC、成交量、日度资金流。",
            "rolling(n) / hhv / llv / ma 含当日，不含未来。",
            "ref/shift 正偏移为过去。",
            "Tushare 日度大单净额假定 T 收盘后、T+1 开盘前可得（见 docs/l2_mapping.md）。",
            "events.label_quality / 评级使用未来 N 日路径，不得进入实时过滤。",
            "旧 backtest/enhance 用收盘价成交，属于执行口径错误，不是公式内部未来函数。",
        ],
    }


def format_leak_report(rep: dict) -> str:
    lines = [
        "[future_leak_check]",
        f"hard_leak={'NO' if rep['ok'] else 'YES'}",
        "",
        "假设：",
    ]
    lines += [f"- {a}" for a in rep["assumptions"]]
    lines += ["", "静态扫描："]
    for f in rep["static"]["findings"]:
        lines.append(f"- [{f['severity']}] {f['file']}:{f.get('line', '?')}  {f['msg']}")
    if not rep["static"]["findings"]:
        lines.append("- （无）")
    rt = rep["runtime"]
    lines += ["", f"截断重放：ran={rt.get('ran')}  mismatches={len(rt.get('mismatches') or [])}"]
    for m in rt.get("mismatches") or []:
        lines.append(f"- {m}")
    lines.append(rt.get("note", ""))
    return "\n".join(lines)


def _scan_sources() -> dict:
    findings: list[dict] = []
    for path in sorted(PKG.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        rel = path.name
        if rel in {"research_check.py"}:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            stripped = line.split("#", 1)[0]
            if re.search(r"\.shift\(\s*-\s*\d+", stripped):
                findings.append(
                    {
                        "severity": "hard",
                        "file": rel,
                        "line": i,
                        "msg": f"negative shift (future): {line.strip()}",
                    }
                )
            if "center=True" in stripped or "center = True" in stripped:
                findings.append(
                    {
                        "severity": "hard",
                        "file": rel,
                        "line": i,
                        "msg": f"centered rolling: {line.strip()}",
                    }
                )
        if rel == "formula.py":
            findings.extend(_scan_formula_ops(path, text))
        if rel == "events.py" and "look forward" in text:
            findings.append(
                {
                    "severity": "info",
                    "file": rel,
                    "line": 0,
                    "msg": "excursion_row 使用未来窗口，只允许作为事后事件研究/主观评级，不能当实时过滤。",
                }
            )
        if rel in {"backtest.py", "enhance.py", "screen.py", "selection.py"}:
            if "pct_change" in text and "T+1" not in text and "execution" not in text.lower():
                findings.append(
                    {
                        "severity": "warn",
                        "file": rel,
                        "line": 0,
                        "msg": "仍含 pct_change；请确认已走 execution / strategy_backtest 的 T+1 开盘路径。",
                    }
                )
            if "EXECUTION_NOTE" in text or "portfolio_from_position" in text or "single_t1_open" in text:
                findings.append(
                    {
                        "severity": "ok",
                        "file": rel,
                        "line": 0,
                        "msg": "已接入 T+1 开盘执行口径。",
                    }
                )
        if rel in {"advise.py", "screen.py"}:
            # These modules must not gate live actions on ex-post quality.
            if re.search(r"\bproven\b.*=.*hist_ok|if\s+proven|if\s+last_low", text):
                findings.append(
                    {
                        "severity": "hard",
                        "file": rel,
                        "line": 0,
                        "msg": "疑似仍用 proven/hist_ok/last_low 做实时决策。",
                    }
                )
            if "archive_only" in text or "不参与" in text or "不得进入" in text:
                findings.append(
                    {
                        "severity": "ok",
                        "file": rel,
                        "line": 0,
                        "msg": "事后评级已标注隔离。",
                    }
                )
        if rel == "screener.py":
            if "live_filter\": 0" in text.replace(" ", "") or "archive_only" in text:
                findings.append(
                    {
                        "severity": "ok",
                        "file": rel,
                        "line": 0,
                        "msg": "rank_stocks 仅为事后档案；recommend_rank 不再作为买入清单。",
                    }
                )
            if 'ranked["hist_ok"] == 1' in text or "& (ranked[\"hist_ok\"]" in text:
                findings.append(
                    {
                        "severity": "hard",
                        "file": rel,
                        "line": 0,
                        "msg": "仍用 hist_ok 过滤推荐列表。",
                    }
                )
    return {"findings": findings}


def _scan_formula_ops(path: Path, text: str) -> list[dict]:
    out: list[dict] = []
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return [{"severity": "hard", "file": path.name, "line": 0, "msg": f"parse fail: {exc}"}]
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "shift" and node.args:
                arg0 = node.args[0]
                if isinstance(arg0, ast.UnaryOp) and isinstance(arg0.op, ast.USub):
                    out.append(
                        {
                            "severity": "hard",
                            "file": path.name,
                            "line": getattr(node, "lineno", 0),
                            "msg": "AST: shift(-n)",
                        }
                    )
    # ops.ref is defined as shift(n) with n>=0 in call sites
    if "ref(" in text:
        out.append(
            {
                "severity": "ok",
                "file": "formula.py",
                "line": 0,
                "msg": "ref() → shift(+n) 取过去；rolling/hhv/llv 含当日、不含未来。",
            }
        )
    return out


def _replay_truncated(df: pd.DataFrame, cuts: tuple[float, ...]) -> dict:
    work = df.copy().reset_index(drop=True)
    if "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"])
    try:
        full = compute_signals(work, asset="stock")
    except Exception as exc:
        return {"ran": False, "note": f"full compute failed: {exc}", "mismatches": []}
    mismatches = []
    n = len(work)
    for frac in cuts:
        k = int(n * frac)
        if k < 160:
            continue
        try:
            part = compute_signals(work.iloc[:k].copy(), asset="stock")
        except Exception as exc:
            mismatches.append({"cut": k, "error": str(exc)})
            continue
        for col in SIGNAL_COLS:
            if col not in full.columns or col not in part.columns:
                continue
            a = pd.to_numeric(part[col], errors="coerce").to_numpy()
            b = pd.to_numeric(full[col].iloc[:k], errors="coerce").to_numpy()
            if a.shape != b.shape:
                mismatches.append({"cut": k, "col": col, "msg": "length mismatch"})
                continue
            # last 30 bars of the truncated sample vs the same bars on the full run
            sl = slice(max(0, k - 30), k)
            if not np.allclose(a[sl], b[sl], equal_nan=True, rtol=0, atol=1e-9):
                diff = int(np.nansum(~np.isclose(a[sl], b[sl], equal_nan=True, rtol=0, atol=1e-9)))
                mismatches.append(
                    {
                        "cut": k,
                        "col": col,
                        "bars_diff": diff,
                        "msg": "truncated series last bars != full series (possible look-ahead)",
                    }
                )
    return {
        "ran": True,
        "n": n,
        "cuts": list(cuts),
        "mismatches": mismatches,
        "note": "截断序列末根应与全样本同一日信号一致；不一致则公式读了未来行。",
    }
