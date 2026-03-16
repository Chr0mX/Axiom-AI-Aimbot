
import json
import urllib.request
import webbrowser
from PyQt6.QtCore import QThread, pyqtSignal

from version import __version__

# GitHub Repository Info
REPO_OWNER = "iishong0w0"
REPO_NAME = "Axiom-AI-Aimbot"
API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases/latest"

def parse_version(v_str):
    """解析版本號字符串為元組，例如 'v1.0.2' -> (1, 0, 2)"""
    v_str = v_str.lower().strip()
    if v_str.startswith('v'):
        v_str = v_str[1:]
    
    parts = []
    for part in v_str.split('.'):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(0)
    
    # 補齊至少三位
    while len(parts) < 3:
        parts.append(0)
        
    return tuple(parts)

class UpdateChecker(QThread):
    """
    後台檢查更新線程
    
    Signals:
        update_available (str, str, str): 發現新版本時發送 (version, url, body)
        up_to_date: 已是最新版本時發送
        check_failed (str): 檢查失敗時發送錯誤訊息
    """
    update_available = pyqtSignal(str, str, str)
    up_to_date = pyqtSignal()
    check_failed = pyqtSignal(str)

    def run(self):
        try:
            # 設定 User-Agent 避免被 GitHub 拒絕
            req = urllib.request.Request(
                API_URL, 
                headers={'User-Agent': f'Axiom-AI-Aimbot/{__version__}'}
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status != 200:
                    self.check_failed.emit(f"HTTP {response.status}")
                    return
                
                data = json.loads(response.read().decode())
                
                latest_tag = data.get('tag_name', '')
                html_url = data.get('html_url', '')
                body = data.get('body', '')  # 發布說明
                
                if not latest_tag:
                    self.check_failed.emit("找不到版本標籤")
                    return
                
                current_ver = parse_version(__version__)
                latest_ver = parse_version(latest_tag)
                
                if latest_ver > current_ver:
                    self.update_available.emit(latest_tag, html_url, body)
                else:
                    self.up_to_date.emit()
                    
        except Exception as e:
            self.check_failed.emit(str(e))

def open_update_url(url):
    """打開預設瀏覽器下載頁面"""
    webbrowser.open(url)
