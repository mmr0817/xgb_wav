"""无需安装即可启动 GUI：python run_gui.py

（`python -m xgb_wav.gui` 需要先 `pip install -e .` 或设置 PYTHONPATH=src；
 这个入口自己把 src/ 加进 sys.path，双击/直接运行都可以。）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from xgb_wav.gui import main  # noqa: E402

if __name__ == "__main__":
    main()
