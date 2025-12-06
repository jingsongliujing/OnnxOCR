# main.py
# 仅用于启动UI
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from onnxocr.onnxocr_ui import OCRApp

if __name__ == "__main__":
    app = OCRApp()
    app.run()
