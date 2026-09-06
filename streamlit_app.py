from pathlib import Path
import runpy

# Safe compatibility entry point: always execute app.py on every Streamlit rerun.
runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
