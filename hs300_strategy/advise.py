"""当日策略建议：只看当日信号 + 状态机仓位。事后质量/MFE 不进入决策。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from hs300_strategy.execution import EXECUTION_NOTE, last_bar_execution

ENV_CN = {0: "不明", 1: "下跌", 2: "震荡", 3: "上涨", 4: "强牛"}
LABEL_CN = {
    "watch": "观察",
    "bottom_watch": "底部观察",
    "entry": "开始建仓",
    "hold": "趋势持有",
    "reduce": "风险控制",
    "exit": "离场",
}

TODAY_FLAGS = (
    ("launch_turn", "启动"),
    ("f_signal", "底部预警F"),
    ("washout_turn", "洗盘转折"),
    ("caution", "注意YJ"),
    ("reduce_trend", "趋势减仓JC_TREND"),
    ("reduce_band", "波段减仓JC_BAND"),
    ("take_profit", "止盈"),
    ("escape_top", "逃顶"),
)

COLORS = {
    "试多": "#1e8449",
    "持有": "#1f6feb",
    "减仓": "#e67e22",
    "清仓": "#c0392b",
    "观望": "#5d6d7e",
}


def make_advice(sig: pd.DataFrame, rank_row: dict | None = None) -> dict[str, Any]:
    """rank_row 仅作事后档案展示，不参与 action 决策。"""
    work = sig.sort_values("date").reset_index(drop=True)
    last = work.iloc[-1]
    pos = float(last["position"]) if "position" in work.columns else 0.0
    env = int(last["env_level"]) if "env_level" in work.columns else 0
    flags = [name for col, name in TODAY_FLAGS if col in work.columns and int(last[col]) == 1]
    bars_since, last_launch_date = _bars_since_launch(work)

    action, pos_hint, conf, reasons = _decide(pos, env, flags, bars_since, last_launch_date)
    color = COLORS.get(action, "#5d6d7e")
    headline = f"{action}　仓位 {pos_hint}"
    today = "、".join(flags) if flags else "无新信号"
    exec_info = last_bar_execution(
        work["date"],
        work["position"] if "position" in work.columns else pd.Series(0.0, index=work.index),
        work["open"] if "open" in work.columns else work["close"],
        work["close"],
        work["launch_turn"] if "launch_turn" in work.columns else None,
    )
    env_cn = ENV_CN.get(env, str(env))
    why = "；".join(reasons) if reasons else "无"
    bits = [f"今日信号：{today}", f"环境{env_cn}"]
    if last_launch_date:
        bits.append(f"上次启动 {last_launch_date}（{bars_since}日）")
    detail = f"原因：{why}\n" + "　".join(bits)
    return {
        "action": action,
        "position_hint": pos_hint,
        "confidence": conf,
        "color": color,
        "headline": headline,
        "detail": detail,
        "flags": flags,
        "position": pos,
        "signal_date": _fmt(exec_info.get("signal_date")),
        "entry_date": _fmt(exec_info.get("entry_date")),
        "entry_price": _num(exec_info.get("entry_price")),
        "exit_date": _fmt(exec_info.get("exit_date")),
        "exit_price": _num(exec_info.get("exit_price")),
        "execution": EXECUTION_NOTE,
    }


def _num(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x != x or x in (float("inf"), float("-inf")):
        return None
    return x


def _fmt(v) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "strftime"):
        return v.strftime("%Y-%m-%d")
    return str(v) if v else None


def _bars_since_launch(work: pd.DataFrame) -> tuple[int | None, str | None]:
    if "launch_turn" not in work.columns:
        return None, None
    hits = work.index[work["launch_turn"].astype(int) == 1]
    if len(hits) == 0:
        return None, None
    i = int(hits[-1])
    d = pd.Timestamp(work.loc[i, "date"]).strftime("%Y-%m-%d")
    return len(work) - 1 - i, d


def _hist_archive_only(row: dict | None) -> str:
    if not row:
        return "无事后档案（或不读评级）"
    from hs300_strategy.events import QUALITY_CN

    n = int(row.get("n_hist") or 0)
    q = QUALITY_CN.get(str(row.get("last_quality", "")), str(row.get("last_quality", "")))
    return (
        f"历史启动 {n} 次（事后主观评级，非实时过滤）；"
        f"最近 {row.get('last_launch', '')}（{q}）。"
        "不得据此提高或降低今日仓位。"
    )


def _hist_text(row: dict | None) -> str:
    """UI 兼容：仅档案文案。"""
    return _hist_archive_only(row)


def _grasp_note(conf, proven=None, last_low=None, hit=None) -> str:
    return "把握不再由事后MFE/质量评级决定；今日动作只跟状态机信号。"


def _decide(pos, env, flags, bars_since, last_launch_date):
    launch = "启动" in flags
    band = any("波段减仓" in f for f in flags)
    trend = any("趋势减仓" in f for f in flags)
    esc = "逃顶" in flags
    tp = "止盈" in flags
    f_sig = "底部预警F" in flags
    wash = "洗盘转折" in flags
    yj = "注意YJ" in flags

    if esc:
        return "清仓", "0%", "信号", ["今日逃顶CT，按规则清仓，下一交易日开盘卖出"]
    if tp and pos > 0:
        return "清仓", "0%", "信号", ["今日止盈，下一交易日开盘卖出"]
    if pos > 0 and env == 1:
        return "清仓", "0%", "信号", ["持仓中环境转为下跌，清仓，下一交易日开盘卖出"]
    if band and pos > 0:
        return "减仓", "50%", "信号", ["今日波段减仓JC_BAND，仓位降到五成"]
    if trend and pos > 0:
        return "减仓", "70%", "信号", ["今日趋势减仓JC_TREND，仓位降到七成"]

    if pos >= 0.99:
        reasons = ["公式状态机已满仓，继续持有"]
        if launch:
            reasons = ["今日黄三角启动，下一交易日开盘买入"]
        if bars_since is not None and last_launch_date:
            reasons.append(f"距上次启动 {bars_since} 个交易日（{last_launch_date}）")
        if yj:
            reasons.append("今日有YJ注意")
        action = "试多" if launch else "持有"
        return action, "100%", "信号", reasons

    if pos >= 0.45:
        reasons = [f"已按减仓规则降到 {pos:.0%}，拿余仓"]
        if bars_since is not None and last_launch_date:
            reasons.append(f"距上次启动 {bars_since} 个交易日（{last_launch_date}）")
        return "持有", f"{pos:.0%}", "信号", reasons

    if launch:
        return "试多", "100%", "信号", ["今日黄三角启动，下一交易日开盘买入，今日收盘价不是成交价"]
    if f_sig:
        return "观望", "0%", "观察", ["今日底部预警F，只提示机会、不开仓，等洗盘后的启动"]
    if wash:
        return "观望", "0%", "观察", ["今日洗盘转折，还不是买入，等后面的黄三角"]

    reasons = ["状态机空仓，今日没有开平仓信号"]
    if bars_since is not None and bars_since <= 50 and last_launch_date:
        reasons.append(f"上次启动 {last_launch_date}，距今 {bars_since} 日，已经不在场")
    return "观望", "0%", "常规", reasons
