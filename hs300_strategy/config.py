"""Frozen research parameters. Do not retune on Test.

Values below are a snapshot of formula.py / costs as of 2026-08-18.
Any later change must record old/new, reason, and Train-only status.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
RESEARCH_DIR = OUTPUT_DIR / "research"

# ----- sample splits (signal_date) -----
FULL_START = "2010-07-01"
FULL_END = "2026-08-17"
TRAIN_START = "2010-07-01"
TRAIN_END = "2020-12-31"
TEST_START = "2021-01-01"
TEST_END = "2026-08-17"
ENHANCE_WINDOW_START = "2024-09-02"

WARMUP_BARS = 120
LAUNCH_DEBOUNCE = 10  # first-in-N bars; frozen with formula _first_in(..., 10)

# Event study holding periods. Report all. Do not pick the best after seeing results.
HOLD_PERIODS = (5, 10, 15, 20, 30, 40, 60)
EVENT_PRIMARY_N = 20  # documented default for the standardized event study, not an optimized choice

# Index-enhancement satellite weights. Report the grid; do not search for positive excess.
ENHANCE_WEIGHTS = (0.50, 0.30, 0.20, 0.10, 0.00)  # satellite; core = 1 - satellite
ENHANCE_HEAT_BARS = 5
ENHANCE_HEAT_TH = 0.04
ENHANCE_HEAT_SCALE = 0.40

# A-share cost defaults for strategy_backtest (not tuned).
COMMISSION = 0.0003          # one-way
STAMP_TAX = 0.001            # sell only
SLIPPAGE_BUY = 0.0005        # 5bp vs open
SLIPPAGE_SELL = 0.0005
COST_SLIP_GRID = (0.0, 0.0005, 0.001, 0.002)

# Subjective rating — never use as a supervised label or live filter.
RATING_KIND = "subjective/semi-quantitative"
QUALITY_HIGH_EXCESS_MFE = 0.05
QUALITY_HIGH_MAE = -0.12
QUALITY_HIGH_EFF = 1.5
QUALITY_LOW_EXCESS_MFE = 0.02
QUALITY_LOW_EFF = 1.0
QUALITY_LOW_MAE = -0.18

L2_DISCLAIMER = (
    "Python版本对Level-2资金行为采用日度大单净额近似映射，"
    "不是通达信 LARGEINTRDVOL / LARGEOUTTRDVOL / L2_AMO 的 100% 复刻。"
)

# UI live preview only. Does not change research execution (T close / T+1 open).
# Quotes in the footer stay 1s. Formula/chart auto-refresh is slower so clicking stays responsive.
LIVE_POLL_SECONDS = 5
LIVE_POLL_CLOSED_SECONDS = 30
LIVE_PLOT_DPI = 120
LIVE_PREVIEW_NOTE = (
    "盘中K线与信号是未完成日K预览，收盘后才确认。"
    "成交仍为 T 日收盘信号、T+1 开盘成交，禁止用盘中价当成交价。"
)

SIGNAL_LAYERS = {
    "f_signal": "F：机会发现/资金介入，不是买入指令。",
    "launch_turn": "黄三角：洗盘结束启动/启动事件。唯一买入触发。",
    "reduce_trend": (
        "JCTREND：JC_EVENT 在 QQS 强趋势环境下的分类。"
        "只描述统计关联，不是导致上涨的趋势确认。"
    ),
    "reduce_band": "JCBAND：JC_EVENT 在非强趋势环境下的波段风险事件。",
    "escape_top": "CT：顶部风险事件。",
    "take_profit": "止盈：K线幅度+吸货回落。不是启动因子。",
}


@dataclass(frozen=True)
class FormulaParams:
    """Named knobs from formula.py section 1. Frozen. Index vs stock."""

    market_filter: int = 1
    volume_confirm: int = 1
    accum_floor: float = 0.0
    wash_window: int = 10
    wash_min_pullback: float = 2.0
    wash_max_pullback: float = 10.0
    wash_vol_ratio: float = 0.85
    wash_stay: float = 0.5
    reverse_score: float = 50.0
    swing_gain: float = 1.15
    near_high_ratio: float = 0.88
    launch_lookback: int = 5
    take_profit_mult: float = 1.25
    pos_band: float = 0.50
    pos_trend: float = 0.70
    overlay_pos_band: float = 0.70
    overlay_pos_trend: float = 0.85


STOCK_FORMULA = FormulaParams()
INDEX_FORMULA = FormulaParams(
    wash_window=15,
    wash_min_pullback=1.2,
    launch_lookback=10,
    take_profit_mult=1.12,
)


def formula_params(is_index: bool) -> FormulaParams:
    return INDEX_FORMULA if is_index else STOCK_FORMULA


# Documented magic numbers still inlined in formula.py. Frozen; not a search grid.
FORMULA_MAGIC = {
    "ma": (5, 10, 13, 20, 50, 60, 120),
    "accum_ma": 45,
    "accum_ma_mult": 0.9,
    "accum_env_mult": {4: 0.85, 3: 0.95, 2: 1.00, 1: 1.20, 0: 1.05},
    "launch_price_mult": {">=3": 1.005, "else": 1.008},
    "launch_vol_mult": {">=3": 0.95, "else": 1.00},
    "launch_accum_stay": 0.6,
    "jc_lookback_launch": 40,
    "jc_debounce": 10,
    "vol_confirm_env": {4: 0.95, 3: 1.00, 2: 1.05, 1: 1.15, 0: 1.08},
    "l2_in": 0.0,
    "l2_strong": 0.20,
    "l2_out_big": -0.2,
}


def as_frozen_dict() -> dict:
    return {
        "full": [FULL_START, FULL_END],
        "train": [TRAIN_START, TRAIN_END],
        "test": [TEST_START, TEST_END],
        "hold_periods": list(HOLD_PERIODS),
        "event_primary_n": EVENT_PRIMARY_N,
        "debounce": LAUNCH_DEBOUNCE,
        "costs": {
            "commission": COMMISSION,
            "stamp_tax": STAMP_TAX,
            "slippage_buy": SLIPPAGE_BUY,
            "slippage_sell": SLIPPAGE_SELL,
        },
        "stock_formula": asdict(STOCK_FORMULA),
        "index_formula": asdict(INDEX_FORMULA),
        "formula_magic": FORMULA_MAGIC,
        "enhance_weights": list(ENHANCE_WEIGHTS),
        "l2_disclaimer": L2_DISCLAIMER,
        "rating_kind": RATING_KIND,
    }
