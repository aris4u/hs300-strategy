"""沪深300本地策略：通达信「主力行为 L2 VZZC」的 K 线子集。"""

from hs300_strategy.formula import STATE_LABELS, compute_signals
from hs300_strategy.backtest import run_backtest

__all__ = ["STATE_LABELS", "compute_signals", "run_backtest"]
