"""PyInstaller 入口（避免相对导入问题）。"""

from xgb_wav.gui import main

if __name__ == "__main__":
    main()
