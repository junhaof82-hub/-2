# Safe Streamlit Cloud entrypoint
# Prefer app.py as Main file. This wrapper exists only if Streamlit selects this file.
from pathlib import Path
code = Path(__file__).with_name("app.py").read_text(encoding="utf-8")
exec(compile(code, "app.py", "exec"), {"__name__": "__main__", "__file__": str(Path(__file__).with_name("app.py"))})
