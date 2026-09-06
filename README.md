# Stock AI MAX — INSTITUTIONAL FINAL (Cloud-Safe)

这是给现有 Streamlit Cloud 股票 AI 的直接覆盖升级版。目标是：尽量扩大实时信息覆盖、提高概率可信度，同时优先避免 Community Cloud 因 CPU / 内存 / 超时而黑屏。

## 你只需要覆盖 3 个文件

上传到你当前 GitHub 仓库根目录并覆盖：

- `app.py`
- `streamlit_app.py`
- `requirements.txt`

Streamlit 的 Main file path 保持：

`app.py`

提交后等待自动部署；若没有自动更新：Manage app -> Reboot app。

## 核心升级

- 实时行情 fallback：Massive/Polygon -> Finnhub -> FMP -> Alpha Vantage -> Yahoo Finance
- 盘中 1h / 15m / 5m：Massive/Polygon -> Alpha Vantage -> Yahoo Finance
- 新闻/事件：Yahoo + SEC EDGAR + Alpha Vantage + Finnhub + Marketaux + FMP
- 新闻事件分类：财报/指引、分析师动作、M&A、产品/AI需求、监管/法律、资本结构、内部人/持股
- 来源可靠度、新闻重要度、时间衰减、方向一致度
- 宏观：SPY / QQQ / SMH / VIX / 10Y / DXY + 11 个行业 ETF 市场广度 + 可选 FRED
- 基本面：Yahoo + Alpha + Finnhub + FMP 互相补充
- 财报：EPS surprise / beat rate / 下次财报 / Alpha earnings calendar
- 机构：Finnhub recommendation / target / upgrade-downgrade / forward EPS & revenue estimates
- 同行相对强弱：Finnhub peers + 20D percentile
- 估值：增长调整 Forward P/E + FCF Yield + 5Y FCF DCF + analyst target consensus
- ML：Logistic + Random Forest + Extra Trees + Histogram Gradient Boosting
- 1D / 3D / 5D 概率校准、时间顺序 holdout、模型质量加权
- 市场 regime 与同行/预期的小幅动态修正
- Monte Carlo empirical bootstrap（保留肥尾）
- Bull / Base / Bear 场景概率
- Confidence / Data Quality / Freshness / Reliability Grade
- 可选 OpenAI Web 最新事件核验（只在手动开启时运行）

## 黑屏保护

- BLAS / sklearn 单线程限制
- 所有外部 API 有 connect/read timeout
- retry 数量有限
- 新闻源约 22 秒总预算
- 补充基本面约 20 秒总预算
- 总分析软预算默认 72 秒
- Options 默认关闭
- AI Web 默认关闭
- Monte Carlo 路径数量受限
- Scanner 不对每一只股票训练全部模型
- 完整分析缓存 5 分钟；实时报价单独 20 秒刷新，不会反复训练模型
- 单个模块失败只降低 Data Quality，不让整个页面失败
- 顶层 Crash Shield 显示错误，而不是空白页
- `streamlit_app.py` 使用 `exec` 安全执行主文件，避免之前 `from app import *` 的 rerun 黑屏问题

## 推荐 Streamlit Secrets

复制 `SECRETS_TEMPLATE.toml` 的内容到 Streamlit -> Manage app -> Settings -> Secrets。

建议优先：

1. MASSIVE_API_KEY
2. FINNHUB_API_KEY
3. ALPHA_VANTAGE_API_KEY
4. MARKETAUX_API_KEY
5. FMP_API_KEY
6. FRED_API_KEY
7. OPENAI_API_KEY（可选，AI Web 核验）

不要把真实 API key 上传到 GitHub。

## 重要说明

这是研究/决策支持系统，不是保证盈利的软件。没有系统能够覆盖 100% 的市场信息或保证某个概率一定实现。系统在数据不完整、模型分歧大或实时源较旧时会主动降低 Confidence，而不是给出虚假的高确定性。
