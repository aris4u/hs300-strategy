# Level-2 变量对照

Python版本对Level-2资金行为采用日度大单净额近似映射，**不是**通达信 `LARGEINTRDVOL` / `LARGEOUTTRDVOL` / `L2_AMO` 的 100% 复刻。

| 原通达信变量 | 含义（公式侧） | Python 替代 | 差异 |
|---|---|---|---|
| `LARGEINTRDVOL` | 逐笔/Level-2 大单买入量 | Tushare `moneyflow` 大单+超大单买入量（日度） | 不是盘口逐笔；无盘中时点，只有日终合计 |
| `LARGEOUTTRDVOL` | 逐笔/Level-2 大单卖出量 | Tushare 大单+超大单卖出量（日度） | 同上 |
| `L2JBL = (LARGEINTRDVOL-LARGEOUTTRDVOL)/CAPITAL*100` | 大单净量 / 流通股本 ×100 | `(buy_lg_vol+buy_elg_vol - sell_lg_vol-sell_elg_vol) / float_share * 100`，写入 `l2jbl` | 口径接近净量比，但成交分级规则、成交量单位、是否含盘后与通达信插件不同 |
| `L2_AMO(0/1/2/3, 0/1)` | Level-2 金额分档 | 未使用分档金额；部分路径用大单净额 | **未复刻** 四档主动买卖金额 |
| `L2JEL` | 大单净额类 | 未作为独立因子 | 公式侧金额行为被日度净额近似，不声称一致 |
| `WINNER` / `COST` / `PPART` | 筹码/成本类 | 未接入 | **缺失** |
| `CAPITAL` | 流通股本 | Tushare 流通市值或股本（见 `moneyflow.py`） | 复权、股本变动日与通达信可能错位 |
| 无 L2 时 | 通达信无数据则公式空值 | `CLV * volume / MA(volume,20)` 量价代理 | **完全不同**；`l2_source=volume_proxy` |

假设（研究执行）：Tushare 日度资金流在 **T 日收盘后、T+1 开盘前** 可获得。若实际要到 T+1 盘中才更新，则用 T 日 L2 在 T+1 开盘成交仍偏乐观，不能当作无延迟。
