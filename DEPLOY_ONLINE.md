# Stock AI MAX — 最简单的在线网页部署

这个版本的重点是：**部署到云端后，你以后只打开网址，不需要 Windows BAT、CMD 或本地 Python。**

## 最简单方案：Streamlit Community Cloud

1. 注册/登录 GitHub。
2. 新建一个 repository，例如 `stock-ai-max`。
3. 把本项目文件全部上传到 repository 根目录。
4. 打开 Streamlit Community Cloud，使用 GitHub 登录。
5. 点击 **Create app**。
6. 选择刚才的 repository。
7. Main file / Entrypoint 选择：`app.py`（也可使用 `streamlit_app.py`）。
8. Python 建议选择 3.12。
9. 点 **Deploy**。
10. 部署成功后会得到一个 `*.streamlit.app` 网址。以后只打开这个网址即可。

## API Key 放哪里？

不要把 `.env` 上传到公开 GitHub。云端部署时，在 Streamlit 的 **Advanced settings → Secrets** 中填写 key。

例如：

```toml
ALPHA_VANTAGE_API_KEY = ""
FMP_API_KEY = ""
POLYGON_API_KEY = ""
FINNHUB_API_KEY = ""
MARKETAUX_API_KEY = ""
FRED_API_KEY = ""
OPENAI_API_KEY = ""
OPENAI_MODEL = "gpt-5-mini"
SEC_USER_AGENT = "StockAIMAX your-email@example.com"
```

没有 key 也能先运行，默认使用 yfinance，并尝试 SEC/FRED 免费公开数据。

## 推荐的新闻覆盖配置

如果希望尽量扩大新闻覆盖，建议按优先级配置：

1. Marketaux — 大范围财经媒体聚合
2. Massive/Polygon — 股票新闻 + ticker 关联
3. Alpha Vantage — 市场新闻与情绪
4. Finnhub — 公司新闻
5. FMP — 财经新闻 / 基本面补充
6. SEC EDGAR — 官方 8-K / 10-Q / 10-K / Form 4 等申报
7. OpenAI — 对高重要度新闻进行可选 LLM 分类

注意：不存在可以保证抓到“互联网全部消息”的数据源。MAX 会展示 **新闻覆盖评分、活跃源数量和数据质量**，缺数据时主动降低置信度。
