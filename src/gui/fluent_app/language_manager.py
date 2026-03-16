"""
Language Manager for Axiom GUI
Handles loading and switching of language translations using the core language manager.
"""
from PyQt6.QtCore import QObject, pyqtSignal

# Import the core language manager
try:
    from core.language_manager import language_manager as global_manager
except ImportError:
    # Fallback for IDEs/standalone running where core might not be resolved directly
    import sys
    import os
    # Try to find src
    current = os.path.dirname(os.path.abspath(__file__))
    src = os.path.dirname(os.path.dirname(os.path.dirname(current)))
    if src not in sys.path:
        sys.path.insert(0, src)
    from core.language_manager import language_manager as global_manager

# Language code to file prefix mapping (GUI Code -> Core Filename)
LANGUAGE_FILE_MAP = {
    "English": "English_English",
    "Chinese": "Chinese_中文",
    "Japanese": "Japanese_日本語",
    "Korean": "Korean_한국어",
    "German": "German_Deutsch",
    "French": "French_Français",
    "Spanish": "Spanish_Español",
    "Portuguese": "Portuguese_Português",
    "Russian": "Russian_Русский",
    "Hindi": "Hindi_हिन्दी",
}


class LanguageManager(QObject):
    """Bridge between PyQt signals and Core LanguageManager."""
    
    languageChanged = pyqtSignal(str)  # Emits when language changes
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        super().__init__()
        self._initialized = True
        
        # Initialize from global manager
        global_lang_full = global_manager.get_current_language() # e.g. "Chinese_中文"
        
        # Reverse map to get short code (e.g. "Chinese")
        self._currentLanguage = "English" # Default
        for code, filename in LANGUAGE_FILE_MAP.items():
            if filename == global_lang_full:
                self._currentLanguage = code
                break
    
    @property
    def currentLanguage(self) -> str:
        return self._currentLanguage
    
    def setLanguage(self, languageCode: str) -> bool:
        """Set the current language and emit signal."""
        # Map short code to full filename
        full_name = LANGUAGE_FILE_MAP.get(languageCode, languageCode)
        
        # Call core manager
        if global_manager.set_language(full_name):
            self._currentLanguage = languageCode
            self.languageChanged.emit(languageCode)
            return True
        return False
    
    def get(self, key: str, default: str = None) -> str:
        """Get translation for key. Returns key if not found."""
        return global_manager.get_text(key, default)
    
    def t(self, key: str, default: str = None) -> str:
        """Alias for get()."""
        return self.get(key, default)


# Global instance
_languageManager = None


def getLanguageManager() -> LanguageManager:
    """Get the global language manager instance."""
    global _languageManager
    if _languageManager is None:
        _languageManager = LanguageManager()
    return _languageManager


def t(key: str, default: str = None) -> str:
    """Convenience function to get translation."""
    return getLanguageManager().get(key, default)
