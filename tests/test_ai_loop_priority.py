"""Process priority is restored when the inference loop exits.

ai_loop needs cv2/win_utils, so it is imported inside each test: on
non-Windows the ImportError is reported as a skip (tests/conftest.py)
instead of breaking collection.
"""


class _FakeK32:
    def __init__(self, current=0x20, set_ok=True):
        self.current = current
        self.set_ok = set_ok
        self.calls = []

    def GetCurrentProcess(self):
        return -1

    def GetPriorityClass(self, proc):
        return self.current

    def SetPriorityClass(self, proc, cls):
        self.calls.append(cls)
        if self.set_ok:
            self.current = cls
        return 1 if self.set_ok else 0


def _patch(monkeypatch, mod, k32):
    class _Windll:
        kernel32 = k32
    monkeypatch.setattr(mod.os, "name", "nt")
    monkeypatch.setattr(mod.ctypes, "windll", _Windll(), raising=False)


def test_set_returns_previous_and_restore_puts_it_back(monkeypatch):
    from core import ai_loop
    k32 = _FakeK32(current=0x20)
    _patch(monkeypatch, ai_loop, k32)
    prev = ai_loop._set_process_priority("high")
    assert prev == 0x20 and k32.current == 0x80
    ai_loop._restore_process_priority(prev)
    assert k32.current == 0x20


def test_failed_set_returns_none_and_restore_is_noop(monkeypatch):
    from core import ai_loop
    k32 = _FakeK32(current=0x20, set_ok=False)
    _patch(monkeypatch, ai_loop, k32)
    assert ai_loop._set_process_priority("high") is None
    k32.calls.clear()
    ai_loop._restore_process_priority(None)
    assert k32.calls == []
