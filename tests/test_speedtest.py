"""Tests for core/speedtest.py (LAN speed-test sizing, payload and statistics)."""
import math

import pytest

from core import speedtest as st


class TestClampMb:
    @pytest.mark.parametrize("raw,want", [
        (8, 8), ("16", 16), (0, 1), (-5, 1), (9999, 64), (1.9, 1), ("2.7", 2),
    ])
    def test_bounds(self, raw, want):
        assert st.clamp_mb(raw) == want

    @pytest.mark.parametrize("raw", [None, "abc", "", float("nan"), float("inf"), [], {}])
    def test_junk_falls_back_to_default(self, raw):
        assert st.clamp_mb(raw) == st.DEFAULT_DOWNLOAD_MB


class TestPayload:
    @pytest.mark.parametrize("total", [0, 1, 1000, st.CHUNK_BYTES, st.CHUNK_BYTES + 7, 3 * st.CHUNK_BYTES])
    def test_exact_length_in_bounded_chunks(self, total):
        chunks = list(st.iter_payload(total))
        assert sum(len(c) for c in chunks) == total
        assert all(0 < len(c) <= st.CHUNK_BYTES for c in chunks)

    def test_negative_total_yields_nothing(self):
        assert list(st.iter_payload(-10)) == []

    def test_one_mib_matches_content_length_math(self):
        assert sum(len(c) for c in st.iter_payload(st.MIB)) == st.MIB

    def test_payload_is_not_trivially_compressible(self):
        import zlib
        data = b"".join(st.iter_payload(st.CHUNK_BYTES))
        assert len(zlib.compress(data)) > 0.95 * len(data)


class TestMbps:
    def test_basic(self):
        assert st.mbps(1_000_000, 1.0) == pytest.approx(8.0)
        assert st.mbps(125_000_000, 2.0) == pytest.approx(500.0)

    @pytest.mark.parametrize("secs", [0, -1, float("nan"), float("inf")])
    def test_bad_duration_is_zero(self, secs):
        assert st.mbps(1000, secs) == 0.0

    def test_bad_bytes(self):
        assert st.mbps(None, 1.0) == 0.0
        assert st.mbps(-5, 1.0) == 0.0


class TestSummarizePings:
    def test_empty(self):
        assert st.summarize_pings([]) == {"count": 0, "min": 0.0, "median": 0.0, "max": 0.0, "jitter": 0.0}

    def test_single_sample_has_zero_jitter(self):
        s = st.summarize_pings([4.0])
        assert s["count"] == 1 and s["min"] == s["median"] == s["max"] == 4.0 and s["jitter"] == 0.0

    def test_stats_and_jitter(self):
        s = st.summarize_pings([2.0, 4.0, 3.0, 9.0])
        assert (s["min"], s["median"], s["max"]) == (2.0, 3.5, 9.0)
        assert s["jitter"] == pytest.approx((2 + 1 + 6) / 3)

    def test_ignores_bad_samples(self):
        s = st.summarize_pings([1.0, float("nan"), -3.0, float("inf"), "x", None, 3.0])
        assert s["count"] == 2 and s["jitter"] == pytest.approx(2.0)
        assert not math.isnan(s["median"])
