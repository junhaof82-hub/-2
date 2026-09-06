# 从你现在的版本升级：最简单步骤

1. 下载并解压 `Stock_AI_MAX_ULTRA_X_V4_DROPIN.zip`。
2. 打开你的 GitHub 股票 AI 仓库。
3. 点击 `Add file -> Upload files`。
4. 只上传这 3 个文件并覆盖旧版：
   - `app.py`
   - `streamlit_app.py`
   - `requirements.txt`
5. 点击 `Commit changes`。
6. 回到 Streamlit，确认 Main file path = `app.py`。
7. 等 Streamlit 自动重新部署；如果没有，`Manage app -> Reboot app`。
8. 页面成功打开后先用 `NVDA` 测试；Options 和 AI Web 先保持关闭。
9. 正常后再去 `Manage app -> Settings -> Secrets` 添加 API keys。

如果旧仓库里还有很多以前的 `.py` 文件，不用删除。v4 主程序是单文件 `app.py`，不会依赖那些旧文件。
