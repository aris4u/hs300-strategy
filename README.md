# HS300 公式研究框架

Python版本对Level-2资金行为采用日度大单净额近似映射，**不是**通达信 Level-2 的 100% 复刻。对照表：[docs/l2_mapping.md](docs/l2_mapping.md)。

目标：**无未来函数、T+1 开盘成交、事件研究与完整策略分开、Train 冻结 / Test 不调参**。不优化回测收益。

## 执行口径（研究与生产统一）

- T 日收盘生成信号（可用当日收盘价、成交量、日度大单净额）。
- T+1 日开盘买入 / 调仓。
- **禁止**用 T 日收盘价作为成交价。
- 账本：`signal_date`, `entry_date`, `entry_price`, `exit_date`, `exit_price`。
- 实现：`hs300_strategy/execution.py`；`backtest.py` / `enhance.py` / `screen.py` / UI 均走此口径。
- UI 盘中刷新（`live_refresh`）只给当前查看的股票打上通达信未完成日K并重算信号，**不是**已确认收盘信号，不能当成交依据。

## 信号层次（不要混用）

| 层次 | 含义 | 交易 |
|---|---|---|
| F (`f_signal`) | 机会发现 / 资金介入 | 不买入 |
| 黄三角 (`launch_turn`) | 洗盘结束启动事件 | 唯一买入触发 |
| JCTREND (`reduce_trend`) | `JC_EVENT AND QQS` | 风险减仓到 70%。**不是**导致上涨的趋势确认 |
| JCBAND (`reduce_band`) | `JC_EVENT AND NOT(QQS)` | 减仓到 50% |
| CT (`escape_top`) | 顶部风险 | 清仓 |
| 评级 / MFE / MAE | **subjective/semi-quantitative**，事后事件研究用 | **禁止**进入实时筛选与生产决策 |

## 两个研究对象

- **A. Event Study**（`event_backtest`）：启动 → T+1 开盘 → 固定持有 N 日。N=5/10/15/20/30/40/60。其中 N=20 为**标准事件窗口**，不是因为结果最好才选。这不是完整策略。
- **B. Full Strategy**（`strategy_backtest`）：完整状态机动态持仓 → 净值、换手、回撤、gross/net。

三类基准：沪深300、全部成分股等权、未入选成分股等权。当前为**非 point-in-time 成分名单**，存在**幸存者偏差**，不得称为无偏基准。

## 从 GitHub 运行（Windows）

克隆下来**不能**像桌面 zip 那样只解压双击就稳。需要本机已装 Python 3.11+（建议 3.13）。

```text
git clone <仓库地址>
cd <仓库目录>
python -m pip install -r requirements.txt
copy .env.example .env
python app.py
```

或双击 `HS300.cmd`（会找本机已安装的 Python）。

- 第一次启动会按收盘日自动拉沪深300成分股日K（BaoStock，无需密钥），并重画建议图，可能要十几分钟。
- 资金流 / L2 近似需要在 `.env` 里自己填 `TUSHARE_TOKEN`。密钥不会也不应出现在仓库里。
- 盘中通达信五档只在本机已安装通达信时可用；没有也能看日K和建议。
- 规则提示，不是投资建议。

## 命令

```text
python run_research.py          # 研究结论与表1–8
python run_screen.py            # 实时筛选（无质量过滤）
python run_enhance.py           # 指数增强（T+1开盘）
python run.py                   # 指数状态机回测（T+1开盘）
python app.py                   # UI（选股建议页对当前股票约每 20 秒拉通达信行情，重画K线并重算信号；盘中为未收盘预览）
```

输出：`output/research/`。参数冻结于 `hs300_strategy/config.py`。任何改动必须记录旧值/新值、原因，且只能在 Train 讨论，Test 冻结。
