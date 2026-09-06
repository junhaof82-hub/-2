# Stock AI MAX ULTRA X v4.0

这是给你现有 Streamlit 股票 AI 的**直接覆盖升级版**。设计原则：尽量扩大实时/事件/财报/估值覆盖，同时保护 Streamlit Community Cloud，避免因为一次分析过重而黑屏。

## 你只需要覆盖 3 个文件

把本包中的以下文件上传到你现在的 GitHub 仓库，覆盖旧文件：

- `app.py`
- `streamlit_app.py`
- `requirements.txt`

Streamlit 的 Main file path 继续设为：

```text
app.py
```

`streamlit_app.py` 是保险入口：如果以后 Streamlit 又误选它，它会用 `runpy` 每次重新执行 `app.py`，不会再用 `from app import *` 导致空白/黑屏。

## v4 新增核心

### 1. 30 秒实时价格脉冲
- 页面打开后，只有轻量 Quote 卡片每 30 秒独立刷新。
- 不会每 30 秒重新训练模型。
- Massive/Polygon 有 Key 时优先 Unified Snapshot；否则 Finnhub，再 fallback Yahoo。
- 是否真正实时取决于你的数据套餐 / 交易所 entitlement。

### 2. 新闻 + SEC + AI Web
- Yahoo Finance news
- Alpha Vantage NEWS_SENTIMENT（有 Key）
- Finnhub company news（有 Key）
- Marketaux（有 Key）
- SEC EDGAR 官方 8-K / 10-Q / 10-K / Form 4 / 13D / 13G / S-3 / 424B5
- OpenAI Responses + Web Search（可选、按按钮启用）

OpenAI Web 只做**最新事件核验和解释**，不会直接把 LLM 的文字当成交易概率，避免“AI 说多就是多”的问题。

### 3. 财报/机构系统
- Alpha Vantage quarterly EPS history / surprise
- Finnhub earnings calendar
- Finnhub recommendation trends
- Finnhub analyst price target
- Finnhub upgrade/downgrade（取决于套餐）
- 财报临近会提高 Risk，而不会自动假定涨或跌

### 4. 新估值引擎
同时尝试：
- Growth-adjusted Forward P/E
- Free Cash Flow Yield
- 5-year FCF DCF（Bear / Base / Bull assumptions）
- Analyst target consensus

最后输出：
- Valuation Score
- Fair Value
- Fair Low / Fair High
- Upside / Downside
- Valuation Confidence
- WACC assumption
- Growth assumption

缺数据、负 FCF、亏损公司会自动减少可用估值方法，不硬算。

### 5. 概率融合升级
最终 1D / 3D / 5D 概率现在融合：
- ML ensemble
- Technical
- 1h / 15m / 5m Intraday
- News
- Macro
- Fundamentals
- Earnings history/event risk
- Analyst consensus
- Valuation

权重按预测周期变化：1D 更看盘中/新闻；5D 增加基本面/估值权重。

### 6. 黑屏保护
- 主程序单文件结构
- `streamlit_app.py` 安全 wrapper
- 顶层 Crash Shield
- 每个 API 独立 timeout
- retry 次数上限
- ML `n_jobs=1`
- 不在页面启动时训练模型
- Options 默认关闭
- OpenAI Web 默认关闭
- 历史数据、新闻条数、Monte Carlo 路径数有上限
- 重模块顺序执行
- 只有 Quote 使用 30 秒 `st.fragment` 独立刷新
- 模块失败只降低 Data Quality，不应该拖垮整页

## Streamlit Secrets

在：

`Manage app -> Settings -> Secrets`

按需填：

```toml
MASSIVE_API_KEY = ""
ALPHA_VANTAGE_API_KEY = ""
FINNHUB_API_KEY = ""
MARKETAUX_API_KEY = ""
FRED_API_KEY = ""
OPENAI_API_KEY = ""
OPENAI_MODEL = "gpt-5"
SEC_USER_AGENT = "Your Name your-email@example.com"
```

**不要把真实 API Key 放进 GitHub。**

## 推荐启用顺序

如果你暂时不想一次申请很多：

1. Massive/Polygon — 更好的实时/盘中价格
2. Finnhub — 公司新闻、财报日历、机构评级、目标价
3. Alpha Vantage — 新闻情绪、EPS surprise、财务 Overview
4. Marketaux — 扩大媒体新闻覆盖
5. FRED — 官方宏观数据
6. OpenAI — 最新网页事件搜索与解释（按需开启）

## 使用

部署后直接输入：

```text
NVDA
AVGO
SNDK
AMAT
AMZN
GOOGL
NBIS
```

点击 `开始 ULTRA X`。

如果已经填了 `OPENAI_API_KEY`，需要最新网页核验时再打开 `AI Web 最新情报`。

## 重要限制

没有任何系统能覆盖互联网上 100% 的消息、保证实时零延迟或保证未来价格预测准确。这个版本的目标是：**多来源交叉验证 + 时间顺序回测 + 概率校准 + 数据质量收缩 + 风险保护**，而不是制造虚假的 90%+ 确定性。
