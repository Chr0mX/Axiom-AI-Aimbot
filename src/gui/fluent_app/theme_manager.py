# theme_manager.py
"""
主題管理器 - 負責將 ThemeColors 應用到所有 GUI 元素
統一管理亮色和暗色主題的樣式表
"""

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication

try:
    from qfluentwidgets import isDarkTheme, setThemeColor, qconfig, Theme
    HAS_FLUENT_WIDGETS = True
except ImportError:
    HAS_FLUENT_WIDGETS = False
    def isDarkTheme():
        return False

from .theme_colors import ThemeColors, to_css_rgba, ColorPairWithAlpha


class ThemeManager:
    """
    主題管理器
    負責生成和應用全局樣式表
    """
    
    _instance = None
    
    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self._currentTheme = "light"
        self._windowAlpha = 204  # 預設約 80% (CC)
        self._elementAlpha = 25   # 預設約 10% (19)
    
    def setAcrylicAlphas(self, window_alpha: int, element_alpha: int):
        """設定 Acrylic 效果的透明度"""
        self._windowAlpha = window_alpha
        self._elementAlpha = element_alpha
    
    def setTheme(self, theme: str):
        """設定主題 ('light' 或 'dark')"""
        self._currentTheme = theme
    
    def isDark(self) -> bool:
        """返回當前是否為暗色主題"""
        if HAS_FLUENT_WIDGETS:
            return isDarkTheme()
        return self._currentTheme == "dark"
    
    def getGlobalStyleSheet(self) -> str:
        """獲取全局樣式表"""
        return self._buildStyleSheet()
    
    def applyToApplication(self, app: QApplication = None):
        """將樣式應用到整個應用程序"""
        if app is None:
            app = QApplication.instance()
        if app:
            app.setStyleSheet(self.getGlobalStyleSheet())
    
    def _buildStyleSheet(self) -> str:
        """構建完整的樣式表"""
        # 獲取當前主題的顏色
        is_dark = self.isDark()
        
        # 基礎顏色
        bg_primary = ThemeColors.BACKGROUND_PRIMARY.dark if is_dark else ThemeColors.BACKGROUND_PRIMARY.light
        bg_secondary = ThemeColors.BACKGROUND_SECONDARY.dark if is_dark else ThemeColors.BACKGROUND_SECONDARY.light
        bg_tertiary = ThemeColors.BACKGROUND_TERTIARY.dark if is_dark else ThemeColors.BACKGROUND_TERTIARY.light
        bg_card = ThemeColors.CARD_BACKGROUND.dark if is_dark else ThemeColors.CARD_BACKGROUND.light
        bg_hover = ThemeColors.BACKGROUND_HOVER.dark if is_dark else ThemeColors.BACKGROUND_HOVER.light
        bg_pressed = ThemeColors.BACKGROUND_PRESSED.dark if is_dark else ThemeColors.BACKGROUND_PRESSED.light
        
        # 文字顏色
        text_primary = ThemeColors.TEXT_PRIMARY.dark if is_dark else ThemeColors.TEXT_PRIMARY.light
        text_secondary = ThemeColors.TEXT_SECONDARY.dark if is_dark else ThemeColors.TEXT_SECONDARY.light
        text_tertiary = ThemeColors.TEXT_TERTIARY.dark if is_dark else ThemeColors.TEXT_TERTIARY.light
        text_disabled = ThemeColors.TEXT_DISABLED.dark if is_dark else ThemeColors.TEXT_DISABLED.light
        
        # 邊框顏色
        border_default = ThemeColors.BORDER_DEFAULT.dark if is_dark else ThemeColors.BORDER_DEFAULT.light
        border_subtle = ThemeColors.BORDER_SUBTLE.dark if is_dark else ThemeColors.BORDER_SUBTLE.light
        
        # 強調色
        accent = ThemeColors.ACCENT.dark if is_dark else ThemeColors.ACCENT.light
        accent_hover = ThemeColors.BUTTON_PRIMARY_HOVER.dark if is_dark else ThemeColors.BUTTON_PRIMARY_HOVER.light
        accent_pressed = ThemeColors.BUTTON_PRIMARY_PRESSED.dark if is_dark else ThemeColors.BUTTON_PRIMARY_PRESSED.light
        
        # 按鈕顏色
        btn_primary_bg = ThemeColors.BUTTON_PRIMARY_BG.dark if is_dark else ThemeColors.BUTTON_PRIMARY_BG.light
        btn_primary_text = ThemeColors.BUTTON_PRIMARY_TEXT.dark if is_dark else ThemeColors.BUTTON_PRIMARY_TEXT.light
        btn_secondary_bg = ThemeColors.BUTTON_SECONDARY_BG.dark if is_dark else ThemeColors.BUTTON_SECONDARY_BG.light
        btn_secondary_text = ThemeColors.BUTTON_SECONDARY_TEXT.dark if is_dark else ThemeColors.BUTTON_SECONDARY_TEXT.light
        
        # 輸入框顏色
        input_bg = ThemeColors.INPUT_BG.dark if is_dark else ThemeColors.INPUT_BG.light
        input_border = ThemeColors.INPUT_BORDER.dark if is_dark else ThemeColors.INPUT_BORDER.light
        input_border_focus = ThemeColors.INPUT_BORDER_FOCUS.dark if is_dark else ThemeColors.INPUT_BORDER_FOCUS.light
        
        # 滑桿顏色
        slider_track = ThemeColors.SLIDER_TRACK.dark if is_dark else ThemeColors.SLIDER_TRACK.light
        slider_fill = ThemeColors.SLIDER_FILL.dark if is_dark else ThemeColors.SLIDER_FILL.light
        slider_thumb = ThemeColors.SLIDER_THUMB.dark if is_dark else ThemeColors.SLIDER_THUMB.light
        slider_thumb_border = ThemeColors.SLIDER_THUMB_BORDER.dark if is_dark else ThemeColors.SLIDER_THUMB_BORDER.light
        
        # 開關顏色
        switch_off_bg = ThemeColors.SWITCH_OFF_BG.dark if is_dark else ThemeColors.SWITCH_OFF_BG.light
        switch_on_bg = ThemeColors.SWITCH_ON_BG.dark if is_dark else ThemeColors.SWITCH_ON_BG.light
        
        # 下拉框顏色
        combo_bg = ThemeColors.COMBOBOX_BG.dark if is_dark else ThemeColors.COMBOBOX_BG.light
        combo_border = ThemeColors.COMBOBOX_BORDER.dark if is_dark else ThemeColors.COMBOBOX_BORDER.light
        combo_dropdown = ThemeColors.COMBOBOX_DROPDOWN_BG.dark if is_dark else ThemeColors.COMBOBOX_DROPDOWN_BG.light
        combo_item_hover = ThemeColors.COMBOBOX_ITEM_HOVER.dark if is_dark else ThemeColors.COMBOBOX_ITEM_HOVER.light
        
        # 滾動條顏色
        scroll_bg = ThemeColors.SCROLLBAR_BG.dark if is_dark else ThemeColors.SCROLLBAR_BG.light
        scroll_thumb = ThemeColors.SCROLLBAR_THUMB.dark if is_dark else ThemeColors.SCROLLBAR_THUMB.light
        scroll_thumb_hover = ThemeColors.SCROLLBAR_THUMB_HOVER.dark if is_dark else ThemeColors.SCROLLBAR_THUMB_HOVER.light
        
        # 導航顏色
        nav_bg = ThemeColors.NAV_BACKGROUND.dark if is_dark else ThemeColors.NAV_BACKGROUND.light
        nav_item_hover = ThemeColors.NAV_ITEM_HOVER.dark if is_dark else ThemeColors.NAV_ITEM_HOVER.light
        nav_item_selected = ThemeColors.NAV_ITEM_SELECTED.dark if is_dark else ThemeColors.NAV_ITEM_SELECTED.light
        nav_item_text = ThemeColors.NAV_ITEM_TEXT.dark if is_dark else ThemeColors.NAV_ITEM_TEXT.light
        
        # 狀態顏色
        success = ThemeColors.SUCCESS.dark if is_dark else ThemeColors.SUCCESS.light
        warning = ThemeColors.WARNING.dark if is_dark else ThemeColors.WARNING.light
        error = ThemeColors.ERROR.dark if is_dark else ThemeColors.ERROR.light
        
        # 標題顏色
        title_color = ThemeColors.CARD_TITLE.dark if is_dark else ThemeColors.CARD_TITLE.light
        desc_color = ThemeColors.CARD_DESCRIPTION.dark if is_dark else ThemeColors.CARD_DESCRIPTION.light
        
        # ============================================
        # Acrylic 半透明背景色（讓模糊效果透出）
        # ============================================
        w_a = self._windowAlpha
        e_a = self._elementAlpha
        
        # 輔助函數：將 HEX 顏色與 alpha 結合生成 rgba
        def get_rgba(hex_color, alpha):
            # 確保 alpha 在 0-255 範圍內
            safe_alpha = max(0, min(255, int(alpha)))
            c = QColor(hex_color)
            return f"rgba({c.red()}, {c.green()}, {c.blue()}, {safe_alpha})"

        # 計算內部元件的不透明度（比外層更實）
        inner_a = min(255, e_a + 160)
        inner_hover_a = min(255, e_a + 180)

        if is_dark:
            # 暗色主題 - 外層較透，內層較實
            acrylic_nav_bg = get_rgba("#202020", e_a)        # 導航列 (較透)
            acrylic_card_bg = get_rgba("#2D2D2D", inner_a)   # 卡片 (較實)
            acrylic_card_hover = get_rgba("#3D3D3D", inner_hover_a)
            acrylic_input_bg = get_rgba("#2D2D2D", inner_hover_a)
            acrylic_combo_bg = get_rgba("#2D2D2D", inner_hover_a)
            acrylic_combo_dropdown = "rgba(45, 45, 45, 230)"
            acrylic_scroll_bg = get_rgba("#2D2D2D", 50)
            acrylic_tooltip_bg = "rgba(45, 45, 45, 230)"
            acrylic_btn_secondary = get_rgba("#404040", inner_hover_a)
            acrylic_btn_secondary_hover = get_rgba("#505050", min(255, e_a + 200))
            acrylic_msg_bg = "rgba(45, 45, 45, 200)"
            acrylic_expand_bg = get_rgba("#2D2D2D", inner_a)
            acrylic_nav_hover = get_rgba("#353535", min(255, e_a + 40))
            acrylic_nav_selected = get_rgba("#404040", min(255, e_a + 60))
            acrylic_border = "rgba(255, 255, 255, 0.1)"
        else:
            # 亮色主題 - 外層較透，內層較實
            acrylic_nav_bg = get_rgba("#F3F3F3", e_a)        # 導航列 (較透)
            acrylic_card_bg = get_rgba("#FFFFFF", inner_a)   # 卡片 (較實)
            acrylic_card_hover = get_rgba("#E9E9E9", inner_hover_a)
            acrylic_input_bg = get_rgba("#FFFFFF", inner_hover_a)
            acrylic_combo_bg = get_rgba("#FFFFFF", inner_hover_a)
            acrylic_combo_dropdown = "rgba(255, 255, 255, 230)"
            acrylic_scroll_bg = get_rgba("#F5F5F5", 50)
            acrylic_tooltip_bg = "rgba(255, 255, 255, 230)"
            acrylic_btn_secondary = get_rgba("#F0F0F0", inner_hover_a)
            acrylic_btn_secondary_hover = get_rgba("#E5E5E5", min(255, e_a + 200))
            acrylic_msg_bg = "rgba(255, 255, 255, 200)"
            acrylic_expand_bg = get_rgba("#FFFFFF", inner_a)
            acrylic_nav_hover = get_rgba("#E5E5E5", min(255, e_a + 40))
            acrylic_nav_selected = get_rgba("#DCDCDC", min(255, e_a + 60))
            acrylic_border = "rgba(0, 0, 0, 0.1)"
        
        stylesheet = f"""
        /* ============================================ */
        /* 全局基礎樣式                                   */
        /* ============================================ */
        QWidget {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei UI", Roboto, Helvetica, Arial, sans-serif;
            color: {text_primary};
        }}
        
        /* ============================================ */
        /* 主窗口和頁面背景                               */
        /* ============================================ */
        FluentWindow, QMainWindow {{
            background-color: transparent; /* 啟用 Mica 特效需要透明背景 */
        }}

        /* 內容區域背景 */
        #stackedWidget {{
            background-color: transparent; /* 啟用 Mica 特效需要透明背景 */
        }}
        
        /* ============================================ */
        /* 導航欄樣式                                    */
        /* ============================================ */
        NavigationInterface {{
            background-color: {acrylic_nav_bg} !important;
            border: none;
        }}
        
        NavigationPanel {{
            background-color: {acrylic_nav_bg} !important;
        }}
        
        NavigationTreeWidget {{
            background-color: {acrylic_nav_bg} !important;
        }}
        
        NavigationPushButton {{
            color: {nav_item_text};
            background-color: transparent;
            border: none;
            border-radius: 14px;
            padding: 8px 12px;
        }}
        
        NavigationPushButton:hover {{
            background-color: {acrylic_nav_hover};
        }}
        
        NavigationPushButton:pressed {{
            background-color: {acrylic_nav_selected};
        }}
        
        NavigationPushButton[isSelected="true"] {{
            background-color: {acrylic_nav_selected};
        }}
        
        /* 導航指示器（選中時的左側小條） */
        NavigationWidget {{
            border-radius: 4px;
        }}

        /* ============================================ */
        /* 開關按鈕 (SwitchButton) - 更圓潤的膠囊形       */
        /* ============================================ */
        SwitchButton {{
            border-radius: 14px;
        }}
        
        SwitchButton > QWidget {{
            border-radius: 12px;
        }}

        /* ============================================ */
        /* 核取方塊 (CheckBox) - 更圓潤                   */
        /* ============================================ */
        CheckBox::indicator {{
            border-radius: 6px;
            width: 20px;
            height: 20px;
        }}

        /* ============================================ */
        /* 單選按鈕 (RadioButton) - 更圓潤                */
        /* ============================================ */
        RadioButton::indicator {{
            border-radius: 11px;
            width: 22px;
            height: 22px;
        }}

        /* ============================================ */
        /* 資訊提示條 (InfoBar) - 圓潤通知                 */
        /* ============================================ */
        InfoBar {{
            border-radius: 14px;
        }}
        
        InfoBarCloseButton {{
            border-radius: 10px;
        }}

        /* ============================================ */
        /* 分段控制器和 Pivot 標籤 - 圓潤選中效果          */
        /* ============================================ */
        SegmentedWidget {{
            border-radius: 14px;
        }}
        
        SegmentedWidget > SegmentedToolWidget,
        SegmentedWidget > SegmentedItem {{
            border-radius: 12px;
        }}
        
        Pivot {{
            border-radius: 14px;
        }}
        
        PivotItem {{
            border-radius: 12px;
        }}

        /* ============================================ */
        /* 標籤列 (TabBar) - 圓潤標籤                     */
        /* ============================================ */
        TabBar > QToolButton {{
            border-radius: 12px;
        }}
        
        TabItem {{
            border-radius: 12px;
        }}

        /* ============================================ */
        /* 進度條 (ProgressBar) - 更圓潤的軌道             */
        /* ============================================ */
        ProgressBar {{
            border-radius: 4px;
        }}
        
        ProgressBar::chunk {{
            border-radius: 4px;
        }}
        
        IndeterminateProgressBar {{
            border-radius: 4px;
        }}

        /* ============================================ */
        /* 選單項目 (MenuItem) - 圓潤懸停                  */
        /* ============================================ */
        QMenu {{
            border-radius: 14px;
            padding: 6px;
        }}
        
        QMenu::item {{
            border-radius: 10px;
            padding: 6px 16px;
        }}
        
        RoundMenu {{
            border-radius: 14px;
        }}
        
        MenuItemWidget {{
            border-radius: 10px;
        }}

        /* ============================================ */
        /* 焦點邊框 - 所有可聚焦控件的圓潤焦點環           */
        /* ============================================ */
        QWidget:focus {{
            outline: none;
        }}
        
        /* ============================================ */
        /* 設定卡片樣式                                  */
        /* ============================================ */
        SettingCard, SwitchSettingCard, PrimaryPushSettingCard, 
        ExpandSettingCard, PushSettingCard, HyperlinkCard, 
        CheckableSettingCard, RadioButtonSettingCard, 
        RangeSettingCard, CustomSettingCard, ColorSettingCard,
        SliderSpinCard, SliderDoubleSpinCard, SliderLabelCard {{
            background-color: {acrylic_card_bg} !important;
            border: 1px solid {acrylic_border} !important;
            border-radius: 18px;
        }}

        SettingCard:hover, SwitchSettingCard:hover, PrimaryPushSettingCard:hover,
        ExpandSettingCard:hover, PushSettingCard:hover, HyperlinkCard:hover,
        CheckableSettingCard:hover, RadioButtonSettingCard:hover,
        RangeSettingCard:hover, CustomSettingCard:hover, ColorSettingCard:hover,
        SliderSpinCard:hover, SliderDoubleSpinCard:hover, SliderLabelCard:hover {{
            background-color: {acrylic_card_hover} !important;
            border: 1px solid {acrylic_border} !important;
        }}

        SettingCardGroup {{
            background-color: transparent;
        }}

        /* 卡片內的標題文字 */
        SettingCard > QLabel {{
            color: {text_primary};
        }}

        /* 卡片組標題 */
        SettingCardGroup > QLabel {{
            color: {title_color};
            font-size: 14px;
            font-weight: 600;
        }}
        
        /* ============================================ */
        /* 標題標籤                                      */
        /* ============================================ */
        TitleLabel {{
            color: {title_color};
            font-size: 28px;
            font-weight: bold;
        }}

        SubtitleLabel {{
            color: {title_color};
            font-size: 18px;
            font-weight: 600;
        }}

        StrongBodyLabel {{
            color: {text_primary};
            font-weight: 600;
        }}

        BodyLabel {{
            color: {text_primary};
        }}

        CaptionLabel {{
            color: {text_secondary};
            font-size: 12px;
        }}

        /* ============================================ */
        /* 按鈕樣式                                      */
        /* ============================================ */
        PrimaryPushButton {{
            background-color: {btn_primary_bg};
            color: {btn_primary_text};
            border: none;
            border-radius: 14px;
            padding: 8px 20px;
            font-weight: 500;
        }}

        PrimaryPushButton:hover {{
            background-color: {accent_hover};
        }}

        PrimaryPushButton:pressed {{
            background-color: {accent_pressed};
        }}

        PrimaryPushButton:disabled {{
            background-color: {bg_tertiary};
            color: {text_disabled};
        }}

        PushButton {{
            background-color: {acrylic_btn_secondary};
            color: {btn_secondary_text};
            border: 1px solid {acrylic_border};
            border-radius: 14px;
            padding: 8px 16px;
        }}

        PushButton:hover {{
            background-color: {acrylic_btn_secondary_hover};
        }}

        PushButton:pressed {{
            background-color: {bg_pressed};
        }}

        TransparentPushButton {{
            background-color: transparent;
            color: {text_primary};
            border: none;
            border-radius: 14px;
            padding: 8px 12px;
        }}

        TransparentPushButton:hover {{
            background-color: {bg_hover};
        }}
        
        /* ============================================ */
        /* 開關樣式 - SwitchButton 使用內建主題顏色     */
        /* ============================================ */
        
        /* ============================================ */
        /* 滑桿樣式                                      */
        /* ============================================ */
        Slider::groove:horizontal {{
            background-color: {slider_track};
            height: 6px;
            border-radius: 4px;
        }}

        Slider::handle:horizontal {{
            background-color: {slider_thumb};
            border: 2px solid {slider_thumb_border};
            width: 20px;
            height: 20px;
            margin: -8px 0;
            border-radius: 12px;
        }}

        Slider::handle:horizontal:hover {{
            background-color: {slider_thumb};
            border: 2px solid {accent_hover};
        }}

        Slider::sub-page:horizontal {{
            background-color: {slider_fill};
            border-radius: 4px;
        }}

        /* ============================================ */
        /* 下拉框樣式                                    */
        /* ============================================ */
        ComboBox {{
            background-color: {acrylic_combo_bg} !important;
            color: {text_primary};
            border: 1px solid {acrylic_border} !important;
            border-radius: 14px;
            padding: 6px 12px;
        }}

        ComboBox:hover {{
            border-color: {accent};
        }}

        ComboBox:focus {{
            border-color: {accent};
        }}

        ComboBox::drop-down {{
            border: none;
        }}

        ComboBoxMenu {{
            background-color: {acrylic_combo_dropdown};
            border: 1px solid {acrylic_border};
            border-radius: 18px;
        }}

        ComboBoxMenu > QListWidget {{
            background-color: {acrylic_combo_dropdown};
            border: none;
        }}

        ComboBoxMenu > QListWidget::item {{
            color: {text_primary};
            padding: 8px 12px;
            border-radius: 12px;
        }}

        ComboBoxMenu > QListWidget::item:hover {{
            background-color: {combo_item_hover};
        }}

        ComboBoxMenu > QListWidget::item:selected {{
            background-color: {nav_item_selected};
        }}

        /* ============================================ */
        /* 輸入框樣式                                    */
        /* ============================================ */
        LineEdit {{
            background-color: {acrylic_input_bg} !important;
            color: {text_primary};
            border: 1px solid {acrylic_border} !important;
            border-radius: 14px;
            padding: 6px 12px;
        }}

        LineEdit:hover {{
            border-color: {accent};
        }}

        LineEdit:focus {{
            border-color: {input_border_focus};
        }}

        SpinBox {{
            background-color: {acrylic_input_bg} !important;
            color: {text_primary};
            border: 1px solid {acrylic_border} !important;
            border-radius: 14px;
            padding: 4px 8px;
        }}

        SpinBox:hover {{
            border-color: {accent};
        }}

        SpinBox:focus {{
            border-color: {input_border_focus};
        }}

        DoubleSpinBox {{
            background-color: {acrylic_input_bg} !important;
            color: {text_primary};
            border: 1px solid {acrylic_border} !important;
            border-radius: 14px;
            padding: 4px 8px;
        }}
        
        /* ============================================ */
        /* 滾動條樣式                                    */
        /* ============================================ */
        QScrollBar:vertical {{
            background: {acrylic_scroll_bg};
            width: 8px;
            border-radius: 6px;
            margin: 4px 0;
        }}
        
        QScrollBar::handle:vertical {{
            background: {scroll_thumb};
            border-radius: 6px;
            min-height: 32px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background: {scroll_thumb_hover};
        }}
        
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {{
            height: 0;
            background: none;
        }}
        
        QScrollBar::add-page:vertical,
        QScrollBar::sub-page:vertical {{
            background: none;
        }}
        
        QScrollBar:horizontal {{
            background: {acrylic_scroll_bg};
            height: 8px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:horizontal {{
            background: {scroll_thumb};
            border-radius: 6px;
            min-width: 32px;
        }}
        
        QScrollBar::handle:horizontal:hover {{
            background: {scroll_thumb_hover};
        }}
        
        /* SmoothScrollArea */
        SmoothScrollArea {{
            background-color: transparent;
            border: none;
        }}
        
        /* ============================================ */
        /* 提示框樣式                                    */
        /* ============================================ */
        QToolTip {{
            background-color: {acrylic_tooltip_bg};
            color: {text_primary};
            border: 1px solid {acrylic_border};
            border-radius: 14px;
            padding: 8px 12px;
        }}
        
        /* ============================================ */
        /* 展開設定卡片                                  */
        /* ============================================ */
        ExpandGroupSettingCard {{
            background-color: {acrylic_expand_bg} !important;
            border: 1px solid {acrylic_border} !important;
            border-radius: 18px;
        }}
        
        ExpandGroupSettingCard > QWidget {{
            background-color: transparent;
        }}
        
        /* ============================================ */
        /* 分隔線                                        */
        /* ============================================ */
        QFrame[frameShape="4"] {{
            background-color: {border_subtle};
            max-height: 1px;
        }}
        
        QFrame[frameShape="5"] {{
            background-color: {border_subtle};
            max-width: 1px;
        }}
        
        /* ============================================ */
        /* 訊息框                                        */
        /* ============================================ */
        MessageBox {{
            background-color: {acrylic_msg_bg};
            border-radius: 18px;
        }}
        
        MessageBox > QLabel {{
            color: {text_primary};
        }}
        
        /* ============================================ */
        /* 特定頁面樣式覆蓋                               */
        /* ============================================ */
        
        /* 狀態指示器顏色 */
        QLabel[status="success"] {{
            color: {success};
        }}
        
        QLabel[status="warning"] {{
            color: {warning};
        }}
        
        QLabel[status="error"] {{
            color: {error};
        }}
        """
        
        return stylesheet


def get_theme_manager() -> ThemeManager:
    """獲取主題管理器單例"""
    return ThemeManager.instance()


def apply_theme_to_app(app: QApplication = None):
    """將當前主題樣式應用到應用程序"""
    get_theme_manager().applyToApplication(app)


def get_current_stylesheet() -> str:
    """獲取當前主題的樣式表"""
    return get_theme_manager().getGlobalStyleSheet()
