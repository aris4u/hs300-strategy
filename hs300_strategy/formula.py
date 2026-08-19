"""Tongdaxin MainForce L2 VZZC, K-line subset in pandas.

Python版本对Level-2资金行为采用日度大单净额近似映射，不是通达信
LARGEINTRDVOL / LARGEOUTTRDVOL / L2_AMO 的 100% 复刻。无资金流时用量价 CLV 代理。
对照表：docs/l2_mapping.md。阈值见 config.py（冻结，禁止用 Test 调参）。

Position follows launch_turn / reduce_band / reduce_trend / escape_top.
ui_state is display-only and is not used as size.
JCTREND is JC_EVENT under QQS — statistical association only, not trend causation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hs300_strategy.config import formula_params
from hs300_strategy.ops import (
    abs_,
    count,
    cross,
    ema,
    exist,
    hhv,
    iff,
    llv,
    ma,
    maximum,
    minimum,
    ref,
    safe_div,
    sma,
)

STATE_LABELS = {
    0: "watch",
    1: "bottom_watch",
    2: "entry",
    3: "hold",
    4: "reduce",
    5: "exit",
}


def compute_signals(df: pd.DataFrame, *, asset: str = "stock", overlay: bool = False) -> pd.DataFrame:
    """OHLC + volume required. Optional l2jbl / market_env.

    asset='stock': native L2 scale. asset='index': rescale aggregated L2.
    Buy marker is launch_turn only (F is alert, not an entry).
    overlay=True: index-enhancement sleeve — do not flatten on take_profit,
    and cut less on JC so winners keep more beta.
    """
    is_index = asset == "index"
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    v = df["volume"].astype(float)
    idx = df.index

    proxy_l2 = _volume_flow_proxy(h, l, c, v)
    if "l2jbl" in df.columns:
        raw_l2 = pd.to_numeric(df["l2jbl"], errors="coerce")
        l2_from_ts = raw_l2.notna()
        if is_index:
            l2jbl = _rescale_constituent_l2(raw_l2).combine_first(proxy_l2)
        else:
            l2jbl = raw_l2.combine_first(proxy_l2)
    else:
        raw_l2 = pd.Series(np.nan, index=idx)
        l2jbl = proxy_l2
        l2_from_ts = pd.Series(False, index=idx)
    has_l2 = l2jbl.notna()
    l2_in = has_l2 & (l2jbl > 0)
    l2_strong = has_l2 & (l2jbl > 0.20)
    l2_lx3 = count(l2_in, 3) >= 2
    l2_lx5 = count(l2_in, 5) >= 3
    l2_out_big = has_l2 & (l2jbl < -0.2)

    # ----- 1. 参数（冻结于 config.py，禁止按回测收益改）-----
    P = formula_params(is_index)
    大盘过滤 = P.market_filter
    放量确认 = P.volume_confirm
    吸筹门槛 = P.accum_floor
    洗盘窗口 = P.wash_window
    洗盘最小回调 = P.wash_min_pullback
    洗盘最大回调 = P.wash_max_pullback
    洗盘缩量比 = P.wash_vol_ratio
    洗盘主力留存 = P.wash_stay
    反转强度门槛 = P.reverse_score
    波段涨幅 = P.swing_gain
    近期高点比 = P.near_high_ratio
    启动回看 = P.launch_lookback
    止盈倍数 = P.take_profit_mult

    # ----- 2. 市场环境 -----
    ma20e = ma(c, 20)
    ma60e = ma(c, 60)
    ma120e = ma(c, 120)
    s20e = (safe_div(ma20e, ref(ma20e, 5)) - 1) * 100
    s60e = (safe_div(ma60e, ref(ma60e, 5)) - 1) * 100
    bias20e = safe_div(c - ma20e, ma20e) * 100
    bias60e = safe_div(c - ma60e, ma60e) * 100
    range20e = safe_div(hhv(c, 20) - llv(c, 20), llv(c, 20)) * 100
    vol20e = ma(v, 20)
    volar = safe_div(v, vol20e)

    trend_score = (
        iff((ma20e > ma60e) & (ma60e > ma120e), 20, 0)
        + iff(c > ma20e, 10, 0)
        + iff(s20e > 0, 10, 0)
        + iff(s60e > 0, 10, 0)
        + iff(c >= hhv(c, 20) * 0.98, 10, 0)
    )
    pose = (
        iff(bias20e > 0, 10, 0)
        + iff((bias20e > 3) & (bias20e < 12), 10, 0)
        + iff(bias60e > 0, 5, 0)
        + iff(bias20e < -8, 0, 5)
    )
    volate = (
        iff(range20e < 6, 20, 0)
        + iff((range20e >= 6) & (range20e < 12), 10, 0)
        + iff((range20e >= 12) & (range20e < 20), 5, 0)
    )
    volume_score = (
        iff(volar > 1.5, 20, 0)
        + iff((volar > 1.2) & (volar <= 1.5), 15, 0)
        + iff((volar > 0.8) & (volar <= 1.2), 10, 0)
        + iff((volar > 0.6) & (volar <= 0.8), 5, 0)
    )
    环境评分 = trend_score + pose + volate + volume_score

    强牛环境 = (ma20e > ma60e) & (ma60e > ma120e) & (c > ma20e) & (trend_score >= 45) & (pose >= 15) & (volume_score >= 10)
    上涨环境 = (ma20e > ma60e) & (ma60e >= ma120e) & (c > ma20e) & (trend_score >= 35) & (pose >= 10) & (~强牛环境)
    下跌环境 = (ma20e < ma60e) & (ma60e < ma120e) & (c < ma20e) & (s20e < 0) & (s60e < 0) & (trend_score <= 15)
    震荡环境 = (range20e < 10) & (abs_(bias20e) < 6) & (~强牛环境) & (~上涨环境) & (~下跌环境)

    环境等级 = iff(强牛环境, 4, iff(上涨环境, 3, iff(震荡环境, 2, iff(下跌环境, 1, 0))))
    if is_index:
        环境允许 = 环境等级 != 1
    else:
        环境允许 = 环境等级 >= 2
        mkt = None
        if "market_env" in df.columns:
            mkt = pd.to_numeric(df["market_env"], errors="coerce")
        elif "大盘环境等级" in df.columns:
            mkt = pd.to_numeric(df["大盘环境等级"], errors="coerce")
        if mkt is not None:
            环境允许 = 环境允许 & (mkt.fillna(2) != 1)
    大盘安全 = iff(大盘过滤 == 1, 环境允许, 1).astype(bool)
    环境强势 = 环境等级 >= 3

    # ----- 3. 主力吸筹（TJ 公式里两处 IF 恒真，按原意显式化）-----
    tj1 = ref(l, 1)
    tj2 = safe_div(sma(abs_(l - tj1), 3, 1), sma(maximum(l - tj1, 0), 3, 1)) * 100
    tj3 = ema(tj2 * 10, 3)
    tj4 = llv(l, 38)
    tj5 = hhv(tj3, 38)
    tj7 = ema(iff(l <= tj4, (tj3 + tj5 * 2) / 2, 0), 3) / 618
    主力吸货1 = tj7

    动态阈值 = ma(主力吸货1, 45) * 0.9
    吸筹倍率 = iff(环境等级 == 4, 0.85, iff(环境等级 == 3, 0.95, iff(环境等级 == 2, 1.00, iff(环境等级 == 1, 1.20, 1.05))))
    吸筹阈值 = maximum(动态阈值 * 吸筹倍率, 吸筹门槛)
    吸筹达标 = 主力吸货1 >= 吸筹阈值
    吸筹强度 = iff(吸筹阈值 > 0, safe_div(主力吸货1, 吸筹阈值) * 100, 0)

    # ----- 4. 神秘资金 -----
    a1 = sma(maximum(c - ref(c, 1), 0), 12, 1) * 4
    a2 = sma(abs_(c - ref(c, 1)), 12, 1)
    窄幅震荡 = count(abs_(safe_div(c, ref(c, 1)) - 1) < 0.03, 5) >= 2
    神秘原始 = ((a1 - a2 + 50 < 55) & (a2 - a1 + 50 > 45)) | 窄幅震荡
    神秘资金允许 = 环境等级 != 1
    神秘 = 神秘原始 & 神秘资金允许

    # ----- 5. 全资金 -----
    dsszy1 = iff(环境等级 == 4, 8, iff(环境等级 == 3, 9, iff(环境等级 == 2, 10, iff(环境等级 == 1, 11, 9))))
    dsszy2 = iff(环境等级 == 4, 18, iff(环境等级 == 3, 20, iff(环境等级 == 2, 20, iff(环境等级 == 1, 22, 20))))
    dsszy3 = iff(环境等级 == 4, 8, iff(环境等级 == 3, 10, iff(环境等级 == 2, 10, iff(环境等级 == 1, 12, 10))))
    dsszy4 = 34

    dsszy6 = (3 * c + l + o + h) / 6
    dsszy7 = _dsszy7(dsszy6)
    dsszy8 = ema(dsszy7, 13)
    dsszy9 = ema(c, 5)
    dsszy10 = ema(dsszy9, 8)
    dsszy11 = ema(dsszy10, 13)
    dsszy12 = ema(dsszy11, 50)
    dsszy13 = 100 * safe_div(c - llv(l, dsszy4), hhv(c, dsszy4) - llv(l, dsszy4))
    dsszy14 = (ref(dsszy13, 1) < 3) & (dsszy13 > 1)

    rsv27 = 100 * safe_div(c - llv(l, 27), hhv(h, 27) - llv(l, 27))
    dsszy15 = 3 * sma(rsv27, 5, 1) - 2 * sma(sma(rsv27, 5, 1), 3, 1)
    dsszy16 = llv(dsszy15, 3)
    dsszy17 = ma(dsszy15, 12)
    dsszy19 = llv(l, 10)
    dsszy20 = hhv(h, 25)
    dsszy21 = ema(safe_div(c - dsszy19, dsszy20 - dsszy19) * 4, 4) * 30
    dsszy22 = dsszy6
    dsszy23 = ema(dsszy22, 6)
    dsszy24 = ema(dsszy22, 5)
    dsszy28 = ema(c, 12) - ema(c, 26)
    dsszy29 = ema(dsszy28, 9)
    dsszy30 = (dsszy28 - dsszy29) * 2

    dsszy33 = 100 * safe_div(c - llv(l, 9), hhv(h, 9) - llv(l, 9))
    k = sma(dsszy33, 3, 1)
    d = sma(k, 3, 1)
    j = 3 * k - 2 * d

    dsszy34 = exist(dsszy14, 4)
    dsszy35 = safe_div(c, ref(c, 1)) > 1 + 0.005 * dsszy1
    dsszy36 = dsszy16 > dsszy17
    dsszy37 = cross(dsszy23, dsszy24)
    dsszy38 = _exist_dynamic(safe_div(c, ref(c, 1)) > 1 + 0.005 * dsszy1, dsszy3)
    dsszy40 = c > dsszy8
    dsszy41 = dsszy35
    dsszy42 = safe_div(dsszy8, dsszy12) < 1 + 0.03 * dsszy2

    超级资金1 = dsszy34 & dsszy35
    超级资金2 = dsszy36 & dsszy37 & dsszy38
    超级资金3 = dsszy40 & dsszy41 & dsszy42

    资金1 = dsszy14
    资金2 = exist(cross(dsszy23, dsszy24), 3)
    资金3 = dsszy36
    资金4 = (j < d) & (dsszy28 > dsszy29)
    资金6 = cross(dsszy17, dsszy21)
    # 原公式把资金5（MACD柱<0）也 OR 进去，空头条件会误当成「有资金」。这里去掉。
    满足资金 = (
        资金1.fillna(False)
        | 资金2.fillna(False)
        | 资金3.fillna(False)
        | 资金4.fillna(False)
        | 资金6.fillna(False)
        | 超级资金1.fillna(False)
        | 超级资金2.fillna(False)
        | 超级资金3.fillna(False)
    )

    # ----- 6. 风控：指数本身不过滤板块 -----
    风控 = pd.Series(True, index=idx)

    # ----- 7. 放量 -----
    基础放量倍数 = iff(环境等级 == 4, 0.95, iff(环境等级 == 3, 1.00, iff(环境等级 == 2, 1.05, iff(环境等级 == 1, 1.15, 1.08))))
    放量 = safe_div(v, ma(v, 5)) >= 基础放量倍数
    放量条件 = iff(放量确认 == 1, 放量, True).astype(bool)

    # ----- 9. 底部反转：有资金流时要求大单净流入，否则退回 K 线弱化版 -----
    near_low = c <= llv(l, 20) * 1.08
    fz1 = minimum(safe_div(主力吸货1, 吸筹阈值) * 25, 25)
    fz2 = iff(has_l2, minimum(l2jbl * 40, 25), 0)
    fz3 = iff(has_l2 & l2_lx3, 15, 0)
    fz4 = iff(has_l2 & l2_lx5, 10, 0)
    fzqd = fz1 + fz2 + fz3 + fz4
    base_bottom = 环境允许 & 吸筹达标 & (c > ref(l, 5)) & near_low
    # 有资金流时不应关掉 K 线底部路径，否则指数 L2 偏弱会让底部反转全年为 0。
    kline_fz = base_bottom & (吸筹强度 >= 80)
    l2_fz = base_bottom & l2_strong & (fzqd > 反转强度门槛)
    f_hold = kline_fz | l2_fz
    f_signal = _rising(f_hold)

    # ----- 10. accumulation -----
    神秘资金 = 神秘 & 满足资金 & 风控 & 大盘安全 & 放量条件 & 环境允许
    主力吸货优化值 = iff(神秘资金, 主力吸货1, 0)

    # ----- 11. 洗盘 -----
    if is_index:
        近几日有吸筹 = count((主力吸货优化值 > 0) | 吸筹达标, 洗盘窗口) >= 1
        今日无吸筹 = (主力吸货优化值 == 0) & (~吸筹达标)
    else:
        近几日有吸筹 = count(主力吸货优化值 > 0, 洗盘窗口) >= 1
        今日无吸筹 = 主力吸货优化值 == 0
    近5日高点 = hhv(h, 5)
    近20日低点 = llv(l, 20)
    回调深度 = safe_div(近5日高点 - c, 近5日高点) * 100
    合理回调 = (回调深度 >= 洗盘最小回调) & (回调深度 <= 洗盘最大回调)
    支撑有效 = l >= 近20日低点 * 0.97
    量能萎缩 = (v < ma(v, 5) * 洗盘缩量比) & (v < ref(v, 1) * 1.05)
    主力仍在 = 主力吸货1 > maximum(吸筹阈值, 吸筹门槛) * 洗盘主力留存

    实体长度 = abs_(c - o)
    下影线长度 = minimum(c, o) - l
    上影线长度 = h - maximum(c, o)
    长下影止跌 = (下影线长度 > 实体长度 * 1.2) & (下影线长度 > 上影线长度 * 0.5)
    缩量十字星 = (safe_div(实体长度, c) < 0.01) & 量能萎缩
    小阳线止跌 = (c > o) & (safe_div(c - o, o) < 0.02) & 量能萎缩
    止跌形态 = 长下影止跌 | 缩量十字星 | 小阳线止跌
    洗盘形态 = (止跌形态 | (回调深度 >= 2.0)) if is_index else 止跌形态

    洗盘信号 = (
        环境允许
        & 近几日有吸筹
        & 今日无吸筹
        & 合理回调
        & 支撑有效
        & 量能萎缩
        & 主力仍在
        & 洗盘形态
    )
    washout_turn = _rising(洗盘信号)

    # ----- 12. launch: F is NOT an entry. Yellow bar = launch_turn only. -----
    启动涨幅v2 = iff(环境等级 >= 3, 1.005, 1.008)
    启动放量v2 = iff(环境等级 >= 3, 0.95, 1.00)
    启动条件1 = c > ref(c, 1) * 启动涨幅v2
    启动条件2 = v >= ma(v, 5) * 启动放量v2
    启动条件3 = 主力吸货1 > maximum(吸筹阈值, 吸筹门槛) * 0.6
    启动条件4 = count(washout_turn, 启动回看) > 0
    launch_raw = 环境允许 & 启动条件1 & 启动条件2 & 启动条件3 & 启动条件4
    launch_turn = _first_in(launch_raw, 10)
    recent_launch = count(launch_turn, 40) > 0

    # ----- 14. 顶部（仅 K 线因子）-----
    ma5 = ma(c, 5)
    ma10 = ma(c, 10)
    ma20 = ma(c, 20)
    ma60 = ma(c, 60)
    dtpl = (ma5 > ma10) & (ma10 > ma20) & (ma20 > ma60)
    qsqd = (
        iff(dtpl, 30, 0)
        + iff(ma20 > ref(ma20, 5), 25, 0)
        + iff(c > ma5, 20, 0)
        + iff(c > ref(hhv(h, 20), 1), 15, 0)
        + iff((c > ma5) & (ma5 > ref(ma5, 3)), 10, 0)
    )
    趋势强度门槛 = iff(环境等级 == 4, 65, iff(环境等级 == 3, 70, iff(环境等级 == 2, 72, iff(环境等级 == 1, 75, 70))))
    qqs = qsqd >= 趋势强度门槛
    qszl = qsqd >= 60

    jgfw = safe_div(c - llv(l, 60), hhv(h, 60) - llv(l, 60))
    q10_th = iff(环境等级 == 4, 0.88, iff(环境等级 == 3, 0.90, iff(环境等级 == 2, 0.92, iff(环境等级 == 1, 0.94, 0.90))))
    q10 = jgfw > q10_th
    zd1 = c > llv(l, 20) * 波段涨幅
    zd2 = c > hhv(h, 10) * 近期高点比
    顶部区域阈值 = iff(环境等级 == 4, 0.95, iff(环境等级 == 3, 0.96, iff(环境等级 == 2, 0.97, iff(环境等级 == 1, 0.98, 0.96))))
    高位区域 = (c > hhv(h, 20) * 顶部区域阈值) & (jgfw > iff(环境等级 == 4, 0.82, 0.85))

    zz1 = count(c < ref(c, 1), 3) >= 2
    syx = h - maximum(c, o)
    zz2 = (syx > (maximum(c, o) - minimum(c, o)) * 1.5) & (v > ma(v, 5) * 1.1)
    zz3 = (c < ref(c, 1)) & (v > ma(v, 5) * 1.2)
    zz8 = 高位区域 & ((h - maximum(c, o)) > abs_(c - o) * 2) & (v > ma(v, 5) * 1.2) & (c < ref(c, 1))
    zz9 = 高位区域 & (safe_div(abs_(c - o), o) < 0.025) & (safe_div(h - l, o) > 0.04) & (v > ma(v, 5) * 1.3) & (c > ref(c, 1))
    zz12 = 高位区域 & (safe_div(abs_(c - o), o) < 0.015) & (v > ma(v, 5))
    zz13 = 高位区域 & (c < ref(c, 1)) & ((o - c) > (c - l) * 1.5) & (v > ma(v, 5))
    zz14 = 高位区域 & (h > ref(h, 1)) & (c < ref(c, 1)) & (v > ma(v, 5) * 1.1)
    zz15 = 高位区域 & (h > ref(h, 1)) & (c < ref(c, 1)) & (v > ma(v, 5) * 1.2)
    zz16 = 高位区域 & (safe_div(abs_(c - o), o) < 0.01) & (safe_div(h - l, o) > 0.03) & (v > ma(v, 5) * 1.2)
    zz4 = has_l2 & (c < ma5) & (l2jbl < -0.08)
    zz10 = has_l2 & (c > hhv(h, 15)) & (c > ref(c, 1)) & (l2jbl < -0.03) & (v > ma(v, 5))
    zz11 = (
        has_l2
        & (c > hhv(h, 15))
        & (count(c > ref(c, 1), 3) >= 2)
        & (l2jbl < ref(l2jbl, 1))
        & (l2jbl < ref(l2jbl, 2))
    )

    ztp = ref(c, 1) * 1.1
    db2 = (v > ma(v, 5) * 1.5) & (safe_div(abs_(c - o), o) < 0.02) & (c < ref(h, 1))
    db3 = (h >= ztp) & (c < ztp * 0.98) & (v > ma(v, 5) * 2)
    db7 = count((c < ref(c, 1)) & (o > ref(c, 1)), 3) == 3
    db8 = (c < ma(c, 20)) & (ref(c, 1) > ma(c, 20)) & (v > ma(v, 5) * 1.2)
    zz7 = db2 | db3 | db7 | db8

    bddb = (c > hhv(h, 20) * iff(环境等级 == 4, 0.92, 0.93)) & (jgfw > iff(环境等级 == 4, 0.58, 0.60)) | (
        jgfw > iff(环境等级 == 4, 0.72, 0.75)
    )
    fqqs = ~qqs
    h10t = hhv(h, 10)
    h10f = (count(h > h10t * 0.98, 3) >= 2) & (c < h10t)
    zjbl3 = has_l2 & (l2jbl < ref(l2jbl, 1)) & (ref(l2jbl, 1) < ref(l2jbl, 2))
    zz17 = qszl & 高位区域 & (c >= hhv(h, 20)) & zjbl3 & (l2jbl < 0.05)
    zz18 = (
        qszl
        & 高位区域
        & (count(abs_(safe_div(c - ref(c, 1), ref(c, 1))) < 0.02, 3) == 3)
        & (l2jbl < -0.08)
    )
    zz19 = fqqs & bddb & h10f & (v > ma(v, 5) * 1.05) & has_l2 & (l2jbl < -0.03)
    zz20 = fqqs & bddb & ((h - maximum(c, o)) > abs_(c - o) * 1.5) & (v > ma(v, 5) * 1.2) & (c < ref(c, 1))
    jddb = zd1 & zd2 & (
        zz1 | zz2 | zz3 | zz4 | zz7 | zz8 | zz9 | zz10 | zz11 | zz12 | zz13 | zz14 | zz15 | zz16 | zz17 | zz18 | zz19 | zz20
    )

    df1 = iff(
        has_l2 & (l2jbl < -0.4),
        30,
        iff(has_l2 & (l2jbl < -0.3), 25, iff(has_l2 & (l2jbl < -0.2), 20, iff(has_l2 & (l2jbl < -0.1), 10, 0))),
    )
    df4 = iff(has_l2 & (count(has_l2 & (l2jbl < 0), 3) >= 2), 10, 0)
    df6 = iff((c < ma(c, 5)) & (c > ma(c, 20)), 5, 0)
    df7 = iff(q10, 10, 0)
    df11 = iff(高位区域 & (count(c > ref(c, 1), 3) <= 1) & (v > ma(v, 5)), 8, 0)
    df12 = iff(zz8, 15, 0)
    df13 = iff(zz17, 15, 0)
    df14 = iff(zz15, 12, 0)
    df15 = iff(zz16, 10, 0)
    df16 = iff(zz18, 12, 0)
    df17 = iff(zz19, 15, 0)
    df18 = iff(zz20, 15, 0)
    chqd = df1 + df4 + df6 + df7 + df11 + df12 + df13 + df14 + df15 + df16 + df17 + df18

    ct_ns = q10 & jddb & (~qqs) & (chqd >= iff(环境等级 == 2, 48, 50))
    jc_ns = q10 & jddb & (~qqs) & (chqd >= iff(环境等级 == 2, 32, 30)) & (chqd < iff(环境等级 == 2, 48, 50))
    yj_ns = q10 & jddb & (~qqs) & (chqd >= 25) & (chqd < iff(环境等级 == 2, 32, 30))
    jc_qs = q10 & jddb & qqs & (chqd >= 55) & (chqd < 75)
    jc_qs_v9 = qszl & 高位区域 & (chqd >= 45) & (zz17 | zz18)
    jc_bc = q10 & 高位区域 & (chqd >= 55) & l2_out_big
    jc_bd = bddb & fqqs & (zz19 | zz20) & (chqd >= 25)
    yj_bd = bddb & fqqs & (zz19 | zz20) & (chqd >= 15) & (chqd < 25)

    ct_state = ct_ns
    jc_state = jc_ns | jc_qs | jc_qs_v9 | jc_bc | jc_bd
    yj_state = yj_ns | yj_bd

    jc_cross = jc_state & cross(chqd, pd.Series(50, index=idx)) & recent_launch
    ct_cross = ct_state & cross(chqd, pd.Series(70, index=idx)) & recent_launch
    yj_cross = yj_state & cross(chqd, pd.Series(25, index=idx))
    jc_event = jc_cross & (count(jc_cross, 10) == 1)
    escape_top = ct_cross & (count(ct_cross, 10) == 1)
    caution = yj_cross
    reduce_trend = jc_event & qqs
    reduce_band = jc_event & (~qqs)

    涨幅 = safe_div(hhv(h, 10), llv(l, 30)) > 止盈倍数
    take_profit = 涨幅 & (ref(主力吸货优化值, 2) > 0) & (主力吸货优化值 < ref(主力吸货优化值, 1))

    ui_state = (
        iff(count(escape_top, 60) > 0, 5,
        iff(count(reduce_band, 30) > 0, 4,
        iff(count(reduce_trend, 60) > 0, 3,
        iff(count(launch_turn, 20) > 0, 2,
        iff(count(f_signal, 60) > 0, 1, 0)))))
    ).fillna(0).astype(int)

    position, state = _sequential_position(
        launch=launch_turn.fillna(False),
        reduce_band=reduce_band.fillna(False),
        reduce_trend=reduce_trend.fillna(False),
        take_profit=(pd.Series(False, index=idx) if overlay else take_profit.fillna(False)),
        escape=escape_top.fillna(False),
        bottom=f_signal.fillna(False),
        env=环境等级,
        band_level=P.overlay_pos_band if overlay else P.pos_band,
        trend_level=P.overlay_pos_trend if overlay else P.pos_trend,
    )

    out = df.copy()
    out["env_level"] = 环境等级
    out["env_score"] = 环境评分
    out["env_ok"] = 环境允许.fillna(False).astype(int)
    out["accum"] = 主力吸货1
    out["accum_ok"] = 吸筹达标.fillna(False).astype(int)
    out["accum_score"] = 吸筹强度
    out["f_signal"] = f_signal.fillna(False).astype(int)
    out["washout"] = 洗盘信号.fillna(False).astype(int)
    out["washout_turn"] = washout_turn.fillna(False).astype(int)
    out["launch_raw"] = launch_raw.fillna(False).astype(int)
    out["launch_turn"] = launch_turn.fillna(False).astype(int)
    out["dist_score"] = chqd
    out["fzqd"] = fzqd
    out["trend_score"] = qsqd
    out["live_chip"] = safe_div(v, ma(v, 20))
    out["caution"] = caution.fillna(False).astype(int)
    out["reduce_trend"] = reduce_trend.fillna(False).astype(int)
    out["reduce_band"] = reduce_band.fillna(False).astype(int)
    out["escape_top"] = escape_top.fillna(False).astype(int)
    out["take_profit"] = take_profit.fillna(False).astype(int)
    out["l2_flow"] = l2jbl
    out["l2_flow_raw"] = raw_l2
    out["l2_strong"] = l2_strong.fillna(False).astype(int)
    out["l2_source"] = np.where(l2_from_ts, "moneyflow", "volume_proxy")
    out["position"] = position
    out["ui_state"] = ui_state
    out["state"] = state
    out["label"] = pd.Series(state, index=out.index).map(STATE_LABELS)
    prev_pos = out["position"].shift(1).fillna(0.0)
    out["entry_ok"] = ((out["position"] > 0) & (prev_pos <= 0)).astype(int)
    out["exit_ok"] = ((out["position"] <= 0) & (prev_pos > 0)).astype(int)
    return out


def _rising(x: pd.Series) -> pd.Series:
    return x.fillna(False) & (~ref(x).fillna(False))


def _first_in(x: pd.Series, n: int) -> pd.Series:
    """Keep the rising edge only if it is the first one in n bars."""
    edge = _rising(x)
    return edge & (count(edge, n) == 1)


def _volume_flow_proxy(h: pd.Series, l: pd.Series, c: pd.Series, v: pd.Series) -> pd.Series:
    """CLV * volume / MA(vol,20) as a stand-in for L2JBL (~ -1.5 to 1.5)."""
    clv = safe_div((c - l) - (h - c), h - l)
    return safe_div(clv * v, ma(v, 20))


def _rescale_constituent_l2(raw: pd.Series) -> pd.Series:
    """把成分股加总 L2JBL 缩放到原公式个股量纲：0.20 ≈ 0.8 倍 60 日波动。"""
    vol = raw.rolling(60, min_periods=20).std().replace(0, np.nan)
    scaled = raw / vol * 0.25
    return scaled.combine_first(raw * 5.0)


def _sequential_position(
    launch,
    reduce_band,
    reduce_trend,
    take_profit,
    escape,
    bottom,
    env,
    band_level: float = 0.5,
    trend_level: float = 0.7,
):
    """Buy on launch_turn. Band/trend cuts scale size. Never auto-restore to 100%."""
    n = len(launch)
    pos = 0.0
    hold = 0
    just_exit = False
    positions = np.zeros(n)
    states = np.zeros(n, dtype=int)
    launch_a = launch.astype(bool).to_numpy()
    band_a = reduce_band.astype(bool).to_numpy()
    trend_a = reduce_trend.astype(bool).to_numpy()
    tp_a = take_profit.astype(bool).to_numpy()
    es_a = escape.astype(bool).to_numpy()
    bt_a = bottom.astype(bool).to_numpy()
    env_a = pd.to_numeric(env, errors="coerce").fillna(0).to_numpy()

    for i in range(n):
        if launch_a[i]:
            pos = 1.0
            hold = 0
            just_exit = False
        if pos > 0:
            hold += 1
        if band_a[i] and pos > 0:
            pos = band_level
        elif trend_a[i] and pos > 0:
            pos = min(pos, trend_level)
        env_fail = pos > 0 and env_a[i] == 1
        if tp_a[i] or es_a[i] or env_fail:
            if pos > 0:
                just_exit = True
            pos = 0.0
            hold = 0

        if pos >= 0.99:
            states[i] = 2 if hold <= 5 else 3
        elif pos >= 0.65:
            states[i] = 3
        elif pos >= 0.4:
            states[i] = 4
        elif just_exit:
            states[i] = 5
            just_exit = False
        elif bt_a[i]:
            states[i] = 1
        else:
            states[i] = 0
        positions[i] = pos

    idx = launch.index
    return pd.Series(positions, index=idx), pd.Series(states, index=idx)


def _dsszy7(x: pd.Series) -> pd.Series:
    """Weights 20..2 on REF 0..18, plus REF(...,20); skip REF 19."""
    vals = x.to_numpy(dtype=float)
    n = len(vals)
    out = np.full(n, np.nan)
    if n <= 20:
        return pd.Series(out, index=x.index)
    acc = 20.0 * vals
    for lag, weight in enumerate(range(19, 1, -1), start=1):
        acc[lag:] += weight * vals[:-lag]
    acc[20:] += vals[:-20]
    out[20:] = acc[20:] / 210.0
    return pd.Series(out, index=x.index)


def _exist_dynamic(cond: pd.Series, windows: pd.Series) -> pd.Series:
    """EXIST(X, dynamic N) via cumsum; N clipped to 1..12."""
    n_max = 12
    cond_i = cond.fillna(False).astype(np.int8).to_numpy()
    n = windows.fillna(10).clip(1, n_max).round().astype(int).to_numpy()
    cs = np.concatenate(([0], np.cumsum(cond_i, dtype=np.int32)))
    idx = np.arange(len(cond_i))
    left = np.maximum(idx + 1 - n, 0)
    return pd.Series((cs[idx + 1] - cs[left]) > 0, index=cond.index)
