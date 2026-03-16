# tests/conftest.py
"""
Pytest 全域配置 - 確保 src 目錄在 Python 路徑中
"""

import sys
import os

# 確保 src 目錄在路徑中，使所有測試都能正確 import
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
