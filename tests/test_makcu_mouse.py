# tests/test_makcu_mouse.py
"""
MAKCU KM Host 滑鼠控制模組測試套件

測試範圍：
1. MakcuMouse 類 - connect, disconnect, is_connected, move, click
2. 模組級函式 - send_mouse_move_makcu, send_mouse_click_makcu, connect_makcu, etc.
3. ASCII 命令格式驗證
4. send_mouse_move / send_mouse_click 調度包含 makcu
5. config _validate_mouse_method 包含 makcu
"""

import sys
import os
from unittest import mock
from unittest.mock import patch, MagicMock, PropertyMock, call

import pytest

# 確保 src 目錄在路徑中
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


# ============================================================
# 1. MakcuMouse 類測試
# ============================================================

class TestMakcuMouseConnect:
    """測試 MAKCU 連線/斷線"""

    @patch("win_utils.makcu_mouse.serial.Serial")
    def test_connect_success(self, mock_serial_cls):
        from win_utils.makcu_mouse import MakcuMouse
        mock_ser = MagicMock()
        mock_ser.is_open = True
        mock_serial_cls.return_value = mock_ser

        m = MakcuMouse()
        result = m.connect("COM3")

        assert result is True
        assert m.is_connected() is True
        assert m.com_port == "COM3"
        mock_serial_cls.assert_called_once_with("COM3", 115200, timeout=0.1, write_timeout=0.1)
        # Should send version then echo-off command on connect
        mock_ser.write.assert_any_call(b"km.version()\r\n")
        mock_ser.write.assert_any_call(b"km.echo(0)\r\n")

    @patch("win_utils.makcu_mouse.serial.Serial", side_effect=Exception("port busy"))
    def test_connect_failure(self, mock_serial_cls):
        from win_utils.makcu_mouse import MakcuMouse
        m = MakcuMouse()
        result = m.connect("COM99")

        assert result is False
        assert m.is_connected() is False

    @patch("win_utils.makcu_mouse.serial.Serial")
    def test_disconnect(self, mock_serial_cls):
        from win_utils.makcu_mouse import MakcuMouse
        mock_ser = MagicMock()
        mock_ser.is_open = True
        mock_serial_cls.return_value = mock_ser

        m = MakcuMouse()
        m.connect("COM3")
        m.disconnect()

        assert m.is_connected() is False
        mock_ser.close.assert_called()

    @patch("win_utils.makcu_mouse.serial.Serial")
    def test_reconnect_closes_old(self, mock_serial_cls):
        """再次連線應先關閉舊連線"""
        from win_utils.makcu_mouse import MakcuMouse
        mock_ser1 = MagicMock()
        mock_ser1.is_open = True
        mock_ser2 = MagicMock()
        mock_ser2.is_open = True
        mock_serial_cls.side_effect = [mock_ser1, mock_ser2]

        m = MakcuMouse()
        m.connect("COM3")
        m.connect("COM4")

        mock_ser1.close.assert_called()
        assert m.com_port == "COM4"


class _FakeVersionProbeSerial:
    """Minimal serial stand-in for _try_open()'s km.version() probe: feeds a
    fixed reply once (all available immediately, like a real single UART
    write), then reports no further data. Unlike a bare MagicMock, this
    avoids a tight/unbounded spin in _try_open()'s deadline-polling loop
    when the reply never matches (the "unknown identity" rejection case)."""

    def __init__(self, reply: bytes):
        self._reply = bytearray(reply)
        self.is_open = True

    @property
    def in_waiting(self):
        return len(self._reply)

    def read(self, n):
        chunk = bytes(self._reply[:n])
        del self._reply[:n]
        return chunk

    def reset_input_buffer(self):
        pass

    def write(self, data):
        pass

    def flush(self):
        pass


class TestMakcuTryOpenIdentity:
    """測試 _try_open() 對裝置身分字串的辨識 (km.MAKCU / km.MAKXD)"""

    @patch("win_utils.makcu_mouse.serial.Serial")
    def test_accepts_legacy_makcu_identity(self, mock_serial_cls):
        from win_utils.makcu_mouse import MakcuMouse
        mock_serial_cls.return_value = _FakeVersionProbeSerial(b"km.MAKCU\r\n>>> ")

        m = MakcuMouse()
        assert m._try_open("COM3", 4_000_000) is True

    @patch("win_utils.makcu_mouse.serial.Serial")
    def test_accepts_makxd_identity(self, mock_serial_cls):
        """新的 MAKXD 裝置身分字串 (per KM_API.md) 也應被接受。"""
        from win_utils.makcu_mouse import MakcuMouse
        mock_serial_cls.return_value = _FakeVersionProbeSerial(b"km.MAKXD\r\n>>> ")

        m = MakcuMouse()
        assert m._try_open("COM3", 4_000_000) is True

    @patch("win_utils.makcu_mouse.serial.Serial")
    def test_rejects_unknown_identity(self, mock_serial_cls):
        from win_utils.makcu_mouse import MakcuMouse
        mock_serial_cls.return_value = _FakeVersionProbeSerial(b"km.SOMETHINGELSE\r\n>>> ")

        m = MakcuMouse()
        assert m._try_open("COM3", 4_000_000) is False


class TestMakcuQueryInfoStreamRace:
    """測試 _query_info() 與按鍵事件流讀取執行緒之間的競態條件修復。

    Real bug #1, confirmed against a hardware capture: other_page.py's
    periodic Hardware-status refresh calls query_info() every ~3s while
    MAKCU is connected — concurrently with the always-running button-event
    stream reader thread, which polls the same serial port with no lock
    (by design, per _stream_reader()'s own docstring: it assumes every
    incoming byte while the stream is active IS a button-stream event).
    km.info()'s own write + its "...\\r\\n>>> " reply landing mid-stream
    broke that assumption — on a MAKXD device (whose real button frames
    have no verifiable suffix at all) this got misread as a spurious
    button-mask byte, i.e. "aim activates for no reason".

    Real bug #2, introduced by that first fix and fixed here: an earlier
    version stopped and restarted the device-side stream itself
    (km.buttons(0)/(1)) around the query, which reset _btn_mask to 0 —
    but MAKCU's button stream is edge-triggered ("streams only emit on
    new frames" per the protocol docs), so an already-held button never
    got re-reported after the stream came back up, and holding the aim
    key appeared to silently "stop working" partway through every hold
    (timed by this method's own ~3s call cadence). _query_info() now only
    pauses _stream_reader()'s own reading (via _stream_read_pause) around
    its request/reply — it never stops/restarts the stream or touches
    _btn_mask, so an in-progress hold survives the query untouched.
    """

    def test_pauses_stream_reader_while_querying(self):
        from win_utils.makcu_mouse import MakcuMouse
        m = MakcuMouse()
        m._serial = MagicMock()
        m._serial.is_open = True
        m._serial.in_waiting = 0

        fake_thread = MagicMock()
        fake_thread.is_alive.return_value = True
        m._stream_thread = fake_thread

        observed = {}

        def fake_sleep(_seconds):
            observed["paused_during_sleep"] = m._stream_read_pause.is_set()

        with patch("win_utils.makcu_mouse.time.sleep", side_effect=fake_sleep), \
             patch.object(m, "_stop_stream") as mock_stop, \
             patch.object(m, "_start_stream") as mock_start:
            m._query_info()

        assert observed.get("paused_during_sleep") is True
        assert m._stream_read_pause.is_set() is False  # cleared afterward
        mock_stop.assert_not_called()
        mock_start.assert_not_called()

    def test_does_not_pause_when_stream_not_running(self):
        """例如 connect() 期間第一次呼叫 _query_info() 時，串流尚未啟動。"""
        from win_utils.makcu_mouse import MakcuMouse
        m = MakcuMouse()
        m._serial = MagicMock()
        m._serial.is_open = True
        m._serial.in_waiting = 0
        m._stream_thread = None

        observed = {}

        def fake_sleep(_seconds):
            observed["paused_during_sleep"] = m._stream_read_pause.is_set()

        with patch("win_utils.makcu_mouse.time.sleep", side_effect=fake_sleep):
            m._query_info()

        assert observed.get("paused_during_sleep") is False
        assert m._stream_read_pause.is_set() is False

    def test_pause_flag_always_cleared_even_on_disconnect_mid_query(self):
        """查詢途中若序列埠消失（模擬斷線），暫停旗標仍應被清除。"""
        from win_utils.makcu_mouse import MakcuMouse
        m = MakcuMouse()
        m._serial = MagicMock()
        m._serial.is_open = True
        m._serial.in_waiting = 0

        fake_thread = MagicMock()
        fake_thread.is_alive.return_value = True
        m._stream_thread = fake_thread

        def fake_sleep(_seconds):
            m._serial = None  # simulate the device disappearing mid-query

        with patch("win_utils.makcu_mouse.time.sleep", side_effect=fake_sleep):
            result = m._query_info()

        assert result == {}
        assert m._stream_read_pause.is_set() is False

    def test_stream_reader_skips_reading_and_keeps_mask_while_paused(self):
        """_stream_reader() 迴圈在暫停旗標設定時應完全跳過讀取，且不動 _btn_mask。"""
        from win_utils.makcu_mouse import MakcuMouse
        m = MakcuMouse()
        m._serial = MagicMock()
        m._serial.is_open = True
        m._btn_mask = 0x01  # simulate an already-held left button
        m._stream_read_pause.set()

        def stop_after_one_pass(*_a, **_k):
            m._stream_stop.set()

        with patch("win_utils.makcu_mouse.time.sleep", side_effect=stop_after_one_pass):
            m._stream_reader()

        m._serial.read.assert_not_called()
        assert m._btn_mask == 0x01


class TestMakcuMouseMove:
    """測試 MAKCU 滑鼠移動指令格式"""

    @patch("win_utils.makcu_mouse.serial.Serial")
    def test_move_basic(self, mock_serial_cls):
        from win_utils.makcu_mouse import MakcuMouse
        mock_ser = MagicMock()
        mock_ser.is_open = True
        mock_serial_cls.return_value = mock_ser

        m = MakcuMouse()
        m.connect("COM3")
        mock_ser.write.reset_mock()

        m.move(10, -3)

        mock_ser.write.assert_called_once_with(b"km.move(10,-3)\r\n")

    @patch("win_utils.makcu_mouse.serial.Serial")
    def test_move_large_values(self, mock_serial_cls):
        """MAKCU 支援 int16 範圍（遠大於 Arduino 的 -128~127）"""
        from win_utils.makcu_mouse import MakcuMouse
        mock_ser = MagicMock()
        mock_ser.is_open = True
        mock_serial_cls.return_value = mock_ser

        m = MakcuMouse()
        m.connect("COM3")
        mock_ser.write.reset_mock()

        m.move(500, -300)
        mock_ser.write.assert_called_once_with(b"km.move(500,-300)\r\n")

    @patch("win_utils.makcu_mouse.serial.Serial")
    def test_move_clamps_to_int16(self, mock_serial_cls):
        """超過 int16 範圍的值應被 clamp"""
        from win_utils.makcu_mouse import MakcuMouse
        mock_ser = MagicMock()
        mock_ser.is_open = True
        mock_serial_cls.return_value = mock_ser

        m = MakcuMouse()
        m.connect("COM3")
        mock_ser.write.reset_mock()

        m.move(50000, -50000)
        mock_ser.write.assert_called_once_with(b"km.move(32767,-32768)\r\n")

    def test_move_not_connected(self):
        """未連線時移動不應報錯"""
        from win_utils.makcu_mouse import MakcuMouse
        m = MakcuMouse()
        m.move(10, 20)  # 不應拋出異常

    @patch("win_utils.makcu_mouse.serial.Serial")
    def test_move_zero(self, mock_serial_cls):
        from win_utils.makcu_mouse import MakcuMouse
        mock_ser = MagicMock()
        mock_ser.is_open = True
        mock_serial_cls.return_value = mock_ser

        m = MakcuMouse()
        m.connect("COM3")
        mock_ser.write.reset_mock()

        m.move(0, 0)
        mock_ser.write.assert_called_once_with(b"km.move(0,0)\r\n")


class TestMakcuMouseClick:
    """測試 MAKCU 滑鼠點擊指令格式"""

    @patch("win_utils.makcu_mouse.serial.Serial")
    def test_click_action1(self, mock_serial_cls):
        """action=1: 點擊（按下後放開）"""
        from win_utils.makcu_mouse import MakcuMouse
        mock_ser = MagicMock()
        mock_ser.is_open = True
        mock_serial_cls.return_value = mock_ser

        m = MakcuMouse()
        m.connect("COM3")
        mock_ser.write.reset_mock()

        m.click(1)
        assert mock_ser.write.call_args_list == [
            call(b"km.left(1)\r\n"),
            call(b"km.left(0)\r\n"),
        ]

    @patch("win_utils.makcu_mouse.serial.Serial")
    def test_click_action2_press(self, mock_serial_cls):
        """action=2: 按下"""
        from win_utils.makcu_mouse import MakcuMouse
        mock_ser = MagicMock()
        mock_ser.is_open = True
        mock_serial_cls.return_value = mock_ser

        m = MakcuMouse()
        m.connect("COM3")
        mock_ser.write.reset_mock()

        m.click(2)
        mock_ser.write.assert_called_once_with(b"km.left(1)\r\n")

    @patch("win_utils.makcu_mouse.serial.Serial")
    def test_click_action3_release(self, mock_serial_cls):
        """action=3: 放開"""
        from win_utils.makcu_mouse import MakcuMouse
        mock_ser = MagicMock()
        mock_ser.is_open = True
        mock_serial_cls.return_value = mock_ser

        m = MakcuMouse()
        m.connect("COM3")
        mock_ser.write.reset_mock()

        m.click(3)
        mock_ser.write.assert_called_once_with(b"km.left(0)\r\n")

    def test_click_not_connected(self):
        """未連線時點擊不應報錯"""
        from win_utils.makcu_mouse import MakcuMouse
        m = MakcuMouse()
        m.click(1)  # 不應拋出異常

    @patch("win_utils.makcu_mouse.serial.Serial")
    def test_click_invalid_action(self, mock_serial_cls):
        """無效 action 不應發送任何命令"""
        from win_utils.makcu_mouse import MakcuMouse
        mock_ser = MagicMock()
        mock_ser.is_open = True
        mock_serial_cls.return_value = mock_ser

        m = MakcuMouse()
        m.connect("COM3")
        mock_ser.write.reset_mock()

        m.click(99)
        mock_ser.write.assert_not_called()


# ============================================================
# 1b. 按鍵事件流解析測試 (_stream_reader)
# ============================================================

class _FakeStreamSerial:
    """Minimal serial stand-in that feeds a fixed byte payload once, then
    signals the reader thread to stop so _stream_reader() returns."""

    def __init__(self, payload, stop_event):
        self._data = bytearray(payload)
        self.is_open = True
        self._stop = stop_event

    @property
    def in_waiting(self):
        return len(self._data)

    def read(self, n):
        chunk = bytes(self._data[:n])
        del self._data[:n]
        if not self._data:
            self._stop.set()  # drained — let the reader loop exit
        return chunk


def _run_stream(payload):
    """Drive _stream_reader() over `payload` and return the final _btn_mask."""
    from win_utils.makcu_mouse import MakcuMouse
    m = MakcuMouse()
    m._serial = _FakeStreamSerial(payload, m._stream_stop)
    m._stream_reader()
    return m


_KM_PREFIX = b"km."
_SUFFIX = b"\r\n>>> "


def _frame(mask):
    """A single legacy-MAKCU button-stream frame, in the format confirmed
    against real MAKCU hardware (raw hex capture, 2026-08-04): "km." + a
    single mask byte + the same "\\r\\n>>> " suffix the docs describe for
    one-off ASCII command replies. E.g. a Left-button press is literally
    `6b 6d 2e 01 0d 0a 3e 3e 3e 20` on the wire.

    This is the fourth framing model this reader has used — three earlier
    guesses (a "km."-prefixed 5-byte frame with a 2-byte mask, a bare
    2-byte mask with no prefix, and the MAKCU V2 [0x50]-wrapped binary
    protocol) were each tried and reported as not detecting real clicks.
    See _stream_reader()'s docstring for the confirmed capture.
    """
    return _KM_PREFIX + bytes([mask]) + _SUFFIX


def _makxd_frame(mask):
    """A single MAKXD button-stream frame, per the official
    terrafirma2021/mak-suite KM_API.md spec: "km." + a single mask byte,
    unframed — no CR/LF/prompt at all. E.g. a Left-button press is
    literally `6b 6d 2e 01` on the wire, 4 bytes total.
    """
    return _KM_PREFIX + bytes([mask])


class TestMakcuStreamReader:
    """測試按鍵事件流 (buttons stream) 的 km.<mask>\\r\\n>>> 封包解析與按鍵位元遮罩"""

    def test_real_lmb_press_parses(self):
        """km. + 01 + \\r\\n>>>  → 左鍵按下"""
        m = _run_stream(_frame(0x01))
        assert m._btn_mask == 0x01
        assert m.lmb_held is True
        assert m.rmb_held is False

    def test_release_parses(self):
        """press then release → 全部放開"""
        m = _run_stream(_frame(0x01) + _frame(0x00))
        assert m._btn_mask == 0x00
        assert m.lmb_held is False

    def test_rmb_press_parses(self):
        """mask=0x02 → 右鍵按下，不誤報左鍵"""
        m = _run_stream(_frame(0x02))
        assert m._btn_mask == 0x02
        assert m.lmb_held is False
        assert m.rmb_held is True

    def test_side1_press_parses(self):
        """mask=0x08 → 側鍵一按下，不誤報其他按鍵"""
        m = _run_stream(_frame(0x08))
        assert m._btn_mask == 0x08
        assert m.side1_held is True
        assert m.side2_held is False
        assert m.lmb_held is False
        assert m.rmb_held is False

    def test_side2_press_parses(self):
        """mask=0x10 → 側鍵二按下，不誤報其他按鍵"""
        m = _run_stream(_frame(0x10))
        assert m._btn_mask == 0x10
        assert m.side2_held is True
        assert m.side1_held is False

    def test_high_bits_masked_off(self):
        """遮罩位元組的高位元 (bits 5-7) 應被 _BTN_BITS 遮罩"""
        # 0xE1 = 0b111_00001 -> only bits 0-4 (0x01, Left) should survive
        m = _run_stream(_frame(0xE1))
        assert m._btn_mask == 0x01
        assert m.lmb_held is True

    def test_last_frame_wins_in_batch(self):
        """同一次讀取含多幀時，最後一幀為最終狀態"""
        m = _run_stream(_frame(0x01) + _frame(0x03) + _frame(0x00))
        assert m._btn_mask == 0x00

    def test_isolated_km_mask_with_no_suffix_applies_as_makxd_short_frame(self):
        """"km." + mask 後若在偵測寬限期內始終沒有更多位元組送達，應視為完整的
        MAKXD 4-byte 無框封包並套用（而非永遠等待一個不會出現的合法後綴）。

        This exact byte sequence is genuinely ambiguous in isolation — it's
        either a legacy frame whose suffix hasn't arrived yet, or a
        complete, real MAKXD short frame. Since a real legacy device always
        writes its full 10 bytes in one shot, "nothing more arrives within
        the bounded grace window" is the correct signal to treat it as a
        complete MAKXD event rather than block forever waiting on a suffix
        that a MAKXD device will never send.
        """
        m = _run_stream(_KM_PREFIX + bytes([0x01]))
        assert m._btn_mask == 0x01
        assert m.lmb_held is True

    def test_makxd_short_frame_applies_immediately(self):
        """MAKXD 的 4-byte 無框封包應能單獨被正確解析（無需等待任何後綴）。"""
        m = _run_stream(_makxd_frame(0x02))
        assert m._btn_mask == 0x02
        assert m.rmb_held is True

    def test_makxd_consecutive_short_frames_all_apply_in_order(self):
        """連續多個 MAKXD 短封包應依序套用，以最後一幀為最終狀態。"""
        m = _run_stream(_makxd_frame(0x01) + _makxd_frame(0x02) + _makxd_frame(0x00))
        assert m._btn_mask == 0x00

    def test_makxd_side_buttons_parse(self):
        """MAKXD 短封包也應正確解析 side1/side2 (bit3/bit4)。"""
        m = _run_stream(_makxd_frame(0x08))
        assert m._btn_mask == 0x08
        assert m.side1_held is True
        assert m.side2_held is False

        m2 = _run_stream(_makxd_frame(0x10))
        assert m2._btn_mask == 0x10
        assert m2.side2_held is True
        assert m2.side1_held is False

    def test_isolated_single_frame_applies(self):
        """孤立的單一封包應直接套用。"""
        m = _run_stream(_frame(0x01))
        assert m._btn_mask == 0x01
        assert m.lmb_held is True

    def test_consecutive_frames_all_apply_in_order(self):
        """連續多幀應依序套用，以最後一幀為最終狀態。"""
        m = _run_stream(_frame(0x01) + _frame(0x02) + _frame(0x00))
        assert m._btn_mask == 0x00

    def test_mismatched_suffix_discarded_and_resynced(self):
        """後綴不符（失步/損毀）的候選幀應被捨棄，並在下一個真正的 km. 前綴重新同步。"""
        corrupt = _KM_PREFIX + bytes([0x01]) + b"\r\nXXXXX"  # wrong suffix
        m = _run_stream(corrupt + _frame(0x02))
        assert m._btn_mask == 0x02
        assert m.rmb_held is True

    def test_real_captured_sequence_from_hardware(self):
        """回歸測試：對真實硬體擷取的原始位元組序列逐幀套用，最終狀態應為左鍵按下。

        Verbatim raw capture (2026-08-04) of a user testing L, R, side-top,
        side-bottom, middle, middle, L — confirms the parser tracks the
        entire real sequence correctly end to end, not just synthetic frames.
        """
        raw_hex = (
            "6b6d2e010d0a3e3e3e20"
            "6b6d2e000d0a3e3e3e20"
            "6b6d2e020d0a3e3e3e20"
            "6b6d2e000d0a3e3e3e20"
            "6b6d2e100d0a3e3e3e20"
            "6b6d2e000d0a3e3e3e20"
            "6b6d2e080d0a3e3e3e20"
            "6b6d2e000d0a3e3e3e20"
            "6b6d2e040d0a3e3e3e20"
            "6b6d2e000d0a3e3e3e20"
            "6b6d2e040d0a3e3e3e20"
            "6b6d2e000d0a3e3e3e20"
            "6b6d2e010d0a3e3e3e20"
        )
        m = _run_stream(bytes.fromhex(raw_hex))
        assert m._btn_mask == 0x01
        assert m.lmb_held is True

    def test_resyncs_past_garbage_before_prefix(self):
        """"km." 之前的雜訊位元組應被丟棄，並在下一個有效封包重新同步。"""
        garbage = bytes([0x11, 0x22, 0x33])
        m = _run_stream(garbage + _frame(0x01))
        assert m._btn_mask == 0x01
        assert m.lmb_held is True

    def test_side1_press_parses(self):
        """mask=0x08 → side1 (S1) 按下"""
        m = _run_stream(_frame(0x08))
        assert m._btn_mask == 0x08
        assert m.side1_held is True
        assert m.side2_held is False

    def test_side2_press_parses(self):
        """mask=0x10 → side2 (S2) 按下"""
        m = _run_stream(_frame(0x10))
        assert m._btn_mask == 0x10
        assert m.side2_held is True
        assert m.side1_held is False


def _bin_frame(btn_id, state, tag=0x53, length=3, sub_type=0x01):
    """A single binary button-event frame, reverse-engineered from a raw
    capture of a newer MAKCU/MAKXD firmware revision that stopped
    speaking the ASCII buttons(1) stream entirely: 2-byte 0xDE 0xAD sync
    + little-endian 2-byte length field + 1-byte report-type tag (0x53
    for button events) + payload `[sub_type][btn_id][state]`. E.g. a
    Left-button (bit 0) press is literally `de ad 03 00 53 01 00 01` on
    the wire; release is the same with the trailing byte 0x00.
    """
    return bytes([0xDE, 0xAD, length & 0xFF, (length >> 8) & 0xFF, tag, sub_type, btn_id, state])


class TestMakcuBinaryStreamReader:
    """測試新韌體的二進位按鍵事件流封包解析 (0xDE 0xAD sync)。

    Real, reported bug: a newer MAKCU firmware revision stopped emitting
    either ASCII "km." button-stream form at all, so clicks silently
    stopped registering (the format-detection logic never found a "km."
    prefix anywhere and _btn_mask simply never updated). Confirmed
    against a raw capture of two button presses+releases (button-index 0
    and 1, i.e. Left and Right):

        de ad 03 00 53 01 00 01   de ad 03 00 53 01 00 00
        de ad 03 00 53 01 01 01   de ad 03 00 53 01 01 00
    """

    def test_left_press_release_parses(self):
        m = _run_stream(_bin_frame(0x00, 1) + _bin_frame(0x00, 0))
        assert m._btn_mask == 0x00
        assert m.lmb_held is False

    def test_left_press_isolated_applies(self):
        m = _run_stream(_bin_frame(0x00, 1))
        assert m._btn_mask == 0x01
        assert m.lmb_held is True

    def test_right_press_isolated_applies(self):
        """對應實際擷取到的第二組封包 (btn_id=0x01)。"""
        m = _run_stream(_bin_frame(0x01, 1))
        assert m._btn_mask == 0x02
        assert m.rmb_held is True

    def test_side1_and_side2_bit_indices_apply(self):
        """side1/side2 的位元索引對應目前沿用既有的 5-bit 慣例 (未經硬體獨立驗證)。"""
        m1 = _run_stream(_bin_frame(0x03, 1))
        assert m1._btn_mask == 0x08
        assert m1.side1_held is True

        m2 = _run_stream(_bin_frame(0x04, 1))
        assert m2._btn_mask == 0x10
        assert m2.side2_held is True

    def test_real_captured_sequence_from_hardware(self):
        """回歸測試：對真實硬體擷取的原始位元組序列逐幀套用。"""
        raw_hex = (
            "dead030053010001"
            "dead030053010000"
            "dead030053010101"
            "dead030053010100"
        )
        m = _run_stream(bytes.fromhex(raw_hex))
        assert m._btn_mask == 0x00  # last frame was a release

    def test_resyncs_past_garbage_before_sync(self):
        garbage = bytes([0x11, 0x22, 0x33])
        m = _run_stream(garbage + _bin_frame(0x00, 1))
        assert m._btn_mask == 0x01
        assert m.lmb_held is True

    def test_unrecognized_tag_is_skipped_not_misread(self):
        """未知的 report-type tag 應被跳過，不應誤判為按鍵事件。"""
        unknown = _bin_frame(0x00, 1, tag=0x99)
        m = _run_stream(unknown + _bin_frame(0x01, 1))
        assert m._btn_mask == 0x02  # only the recognized frame applied

    def test_implausible_length_resyncs_instead_of_hanging(self):
        """異常長度值應視為雜訊並重新同步，而非等待永遠不會到齊的位元組。"""
        corrupt = bytes([0xDE, 0xAD, 0xFF, 0xFF, 0x00])  # length = 65535
        m = _run_stream(corrupt + _bin_frame(0x00, 1))
        assert m._btn_mask == 0x01
        assert m.lmb_held is True

    def test_binary_marker_wins_detection_over_km_when_earlier(self):
        """若緩衝區中 0xDE 0xAD 早於任何 "km." 出現，應鎖定為二進位模式。"""
        # A coincidental "km." substring appearing only *after* the binary
        # sync marker must not derail detection.
        m = _run_stream(_bin_frame(0x00, 1) + b"km.stray")
        assert m._btn_mask == 0x01
        assert m._stream_binary_mode is True


# ============================================================
# 2. 模組級便利函式測試
# ============================================================

class TestModuleFunctions:
    """測試模組級便利函式"""

    @patch("win_utils.makcu_mouse.makcu_mouse")
    def test_send_mouse_move_makcu(self, mock_singleton):
        from win_utils.makcu_mouse import send_mouse_move_makcu
        send_mouse_move_makcu(10, -5)
        mock_singleton.move.assert_called_once_with(10, -5)

    @patch("win_utils.makcu_mouse.makcu_mouse")
    def test_send_mouse_click_makcu(self, mock_singleton):
        from win_utils.makcu_mouse import send_mouse_click_makcu
        result = send_mouse_click_makcu(1)
        mock_singleton.click.assert_called_once_with(1)
        assert result is True

    @patch("win_utils.makcu_mouse.makcu_mouse")
    def test_connect_makcu(self, mock_singleton):
        mock_singleton.connect.return_value = True
        from win_utils.makcu_mouse import connect_makcu
        result = connect_makcu("COM3", 115200)
        mock_singleton.connect.assert_called_once_with("COM3", 115200)
        assert result is True

    @patch("win_utils.makcu_mouse.makcu_mouse")
    def test_disconnect_makcu(self, mock_singleton):
        from win_utils.makcu_mouse import disconnect_makcu
        disconnect_makcu()
        mock_singleton.disconnect.assert_called_once()

    @patch("win_utils.makcu_mouse.makcu_mouse")
    def test_is_makcu_connected(self, mock_singleton):
        mock_singleton.is_connected.return_value = True
        from win_utils.makcu_mouse import is_makcu_connected
        assert is_makcu_connected() is True


# ============================================================
# 3. 調度層測試 (send_mouse_move / send_mouse_click)
# ============================================================

class TestDispatchMakcu:
    """測試 makcu 在 send_mouse_move 和 send_mouse_click 調度中的整合"""

    @patch("win_utils.makcu_mouse.makcu_mouse")
    def test_send_mouse_move_dispatch_makcu(self, mock_singleton):
        from win_utils import send_mouse_move
        send_mouse_move(10, 20, method="makcu")
        mock_singleton.move.assert_called_once_with(10, 20)

    @patch("win_utils.makcu_mouse.send_mouse_click_makcu", return_value=True)
    def test_send_mouse_click_dispatch_makcu(self, mock_click):
        from win_utils.mouse_click import send_mouse_click
        result = send_mouse_click(method="makcu")
        mock_click.assert_called_once()
        assert result is True


# ============================================================
# 4. Config 驗證測試
# ============================================================

class TestConfigValidation:
    """測試 config 中 makcu 作為有效的滑鼠方式"""

    def test_makcu_is_valid_click_method(self):
        from core.config import Config, _validate_mouse_method
        config = Config()
        config.mouse_click_method = "makcu"
        _validate_mouse_method(config)
        assert config.mouse_click_method == "makcu"

    def test_invalid_method_falls_back(self):
        from core.config import Config, _validate_mouse_method
        config = Config()
        config.mouse_click_method = "nonexistent"
        _validate_mouse_method(config)
        assert config.mouse_click_method == "mouse_event"

    def test_makcu_com_port_in_config(self):
        from core.config import Config
        config = Config()
        assert hasattr(config, "makcu_com_port")
        assert config.makcu_com_port == ""

    def test_makcu_com_port_in_dict(self):
        from core.config import Config
        config = Config()
        config.makcu_com_port = "COM5"
        d = config.to_dict()
        assert "makcu_com_port" in d
        assert d["makcu_com_port"] == "COM5"

    def test_makcu_com_port_from_dict(self):
        from core.config import Config
        config = Config()
        config.from_dict({"makcu_com_port": "COM7"})
        assert config.makcu_com_port == "COM7"


# ============================================================
# 5. Serial 錯誤處理測試
# ============================================================

class TestMakcuSerialErrors:
    """測試串列通訊錯誤處理"""

    @patch("win_utils.makcu_mouse.serial.Serial")
    def test_move_serial_exception_disconnects(self, mock_serial_cls):
        """串列異常應將連線狀態設為 False"""
        import serial as real_serial
        from win_utils.makcu_mouse import MakcuMouse

        mock_ser = MagicMock()
        mock_ser.is_open = True
        mock_ser.write.side_effect = real_serial.SerialException("port gone")
        mock_serial_cls.return_value = mock_ser

        m = MakcuMouse()
        m.connect("COM3")
        m.move(10, 20)

        assert m._connected is False

    @patch("win_utils.makcu_mouse.serial.Serial")
    def test_click_serial_exception_disconnects(self, mock_serial_cls):
        """點擊時串列異常應將連線狀態設為 False"""
        import serial as real_serial
        from win_utils.makcu_mouse import MakcuMouse

        mock_ser = MagicMock()
        mock_ser.is_open = True
        mock_ser.write.side_effect = [None, real_serial.SerialException("port gone")]
        mock_serial_cls.return_value = mock_ser

        m = MakcuMouse()
        m.connect("COM3")
        m.click(1)

        assert m._connected is False
