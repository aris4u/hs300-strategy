"""通达信常用运算符的 Pandas 实现。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _s(x: pd.Series | np.ndarray | float) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    raise TypeError("期望 pd.Series")


def ma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).mean()


def ema(x: pd.Series, n: int) -> pd.Series:
    """通达信 EMA：alpha = 2/(N+1)。"""
    return x.ewm(span=n, adjust=False, min_periods=1).mean()


def sma(x: pd.Series, n: int, m: int = 1) -> pd.Series:
    """通达信 SMA(X,N,M)：Y = (M*X + (N-M)*Y.prev) / N。"""
    return x.ewm(alpha=m / n, adjust=False, min_periods=1).mean()


def ref(x: pd.Series, n: int = 1) -> pd.Series:
    return x.shift(n)


def hhv(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).max()


def llv(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).min()


def abs_(x: pd.Series) -> pd.Series:
    return x.abs()


def maximum(a, b) -> pd.Series:
    idx = a.index if isinstance(a, pd.Series) else b.index
    return pd.Series(np.maximum(a, b), index=idx)


def minimum(a, b) -> pd.Series:
    idx = a.index if isinstance(a, pd.Series) else b.index
    return pd.Series(np.minimum(a, b), index=idx)


def safe_div(a, b) -> pd.Series:
    idx = a.index if isinstance(a, pd.Series) else b.index
    return pd.Series(np.where((b == 0) | pd.isna(b), 0.0, a / b), index=idx)


def iff(cond, a, b) -> pd.Series:
    if isinstance(cond, pd.Series):
        idx = cond.index
    elif isinstance(a, pd.Series):
        idx = a.index
    else:
        idx = b.index
    return pd.Series(np.where(cond, a, b), index=idx)


def cross(a: pd.Series, b: pd.Series) -> pd.Series:
    """A 上穿 B。"""
    return (a > b) & (ref(a) <= ref(b))


def count(cond: pd.Series, n: int) -> pd.Series:
    return cond.fillna(False).astype(int).rolling(n, min_periods=1).sum()


def exist(cond: pd.Series, n: int) -> pd.Series:
    return count(cond, n) > 0


def true_range_bool(x: pd.Series) -> pd.Series:
    return x.fillna(False).astype(bool)
