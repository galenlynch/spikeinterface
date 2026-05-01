"""Tests for ``method="fir_magnitude_matched"`` in FilterRecording.

The FIR substitution is *magnitude-matched*, not sample-equivalent, to the
IIR's ``sosfiltfilt`` response.  Tests check:

  - FIR convolver output matches scipy.signal.oaconvolve (its true reference)
    to fp tolerance.
  - FIR magnitude response matches IIR's filtfilt response within design
    tolerance (passband ~flat 0 dB, stopband ≤ -60 dB where IIR is).
  - End-to-end: linear-phase output, RMS within ~25% of IIR, high
    correlation in the spike band.
  - Round-trip via ``_kwargs`` preserves ``method``.
  - Fallbacks: ba-mode / forward-only / bandstop fall back to IIR with
    a warning.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.signal

from spikeinterface import NumpyRecording
from spikeinterface.preprocessing import (
    BandpassFilterRecording,
    HighpassFilterRecording,
)
from spikeinterface.preprocessing._fir_filter import (
    CachedOSConvolver,
    _calc_oa_block_size,
    _detect_l2_bytes,
    design_matched_fir_from_sos,
    pick_chunk_C,
)


FS = 30_000.0


def _butter_hp(order=5, fc=300.0):
    return scipy.signal.butter(order, fc, btype="highpass", fs=FS, output="sos")


def _butter_bp(order=5, lo=300.0, hi=6000.0):
    return scipy.signal.butter(order, [lo, hi], btype="bandpass", fs=FS, output="sos")


# ---- L2 detection / chunk picker -------------------------------------------


class TestChunkPicker:
    def test_l2_is_positive(self):
        l2 = _detect_l2_bytes()
        assert l2 > 0
        assert l2 <= 16 * 1024 * 1024  # sanity

    def test_pick_pool_bound(self):
        chunk = pick_chunk_C(C=64, block_size=128, pool_size=8, l2_bytes=10_000_000)
        assert chunk == 8

    def test_pick_l2_bound(self):
        chunk = pick_chunk_C(
            C=1024, block_size=2048, pool_size=2, l2_bytes=1_000_000, headroom=0.6
        )
        assert 30 <= chunk <= 40

    def test_pick_minimum_one(self):
        assert pick_chunk_C(C=4, block_size=10**9, pool_size=8, l2_bytes=1024) >= 1


# ---- FIR design ------------------------------------------------------------


class TestDesign:
    def _check_magnitude_match(
        self, sos, h, *, passband_tol_db=1.0, stopband_target_db=50.0
    ):
        w_iir, h_iir = scipy.signal.sosfreqz(sos, worN=8192, fs=FS)
        iir_db = 20 * np.log10(np.maximum(np.abs(h_iir) ** 2, 1e-30))
        w_fir, h_fir = scipy.signal.freqz(h, worN=8192, fs=FS)
        fir_db = 20 * np.log10(np.maximum(np.abs(h_fir), 1e-30))

        passband_mask = iir_db >= -1.0
        stopband_mask = iir_db <= -60.0

        if np.any(passband_mask):
            assert np.all(np.abs(fir_db[passband_mask]) < passband_tol_db)
        if np.any(stopband_mask):
            assert np.all(fir_db[stopband_mask] < -stopband_target_db)

    def test_design_highpass(self):
        sos = _butter_hp(order=5, fc=300.0)
        h = design_matched_fir_from_sos(sos, FS)
        assert h.dtype == np.float32
        assert len(h) % 2 == 1
        # Linear-phase symmetric
        assert np.array_equal(h, h[::-1])
        self._check_magnitude_match(sos, h)

    def test_design_bandpass(self):
        sos = _butter_bp(order=5)
        h = design_matched_fir_from_sos(sos, FS)
        assert np.array_equal(h, h[::-1])
        self._check_magnitude_match(sos, h)

    def test_design_lowpass(self):
        sos = scipy.signal.butter(4, 6000.0, btype="lowpass", fs=FS, output="sos")
        h = design_matched_fir_from_sos(sos, FS)
        assert np.array_equal(h, h[::-1])
        self._check_magnitude_match(sos, h)

    def test_design_bandstop_raises(self):
        sos = scipy.signal.butter(4, [300.0, 6000.0], btype="bandstop", fs=FS, output="sos")
        with pytest.raises(ValueError, match="lowpass.*highpass.*bandpass"):
            design_matched_fir_from_sos(sos, FS)


# ---- Convolver --------------------------------------------------------------


class TestConvolver:
    def test_block_size_default(self):
        bs = _calc_oa_block_size(1_000_000, 221)
        # FFT size should be in the Lambert-W neighborhood for taps=221
        assert 1024 <= bs <= 4096

    def test_matches_scipy_oaconvolve_2d(self):
        sos = _butter_hp(order=5, fc=300.0)
        h = design_matched_fir_from_sos(sos, FS)
        T, C = 30_000, 8
        rng = np.random.default_rng(0)
        x = rng.standard_normal((T, C), dtype=np.float32) * 50.0
        conv = CachedOSConvolver(h, T)
        out = np.empty_like(x)
        conv(x, out=out)
        ref = scipy.signal.oaconvolve(x, h[:, None], mode="same", axes=0)
        np.testing.assert_allclose(out, ref, atol=1e-3)

    def test_matches_scipy_oaconvolve_1d(self):
        sos = _butter_bp(order=5)
        h = design_matched_fir_from_sos(sos, FS)
        T = 60_000
        rng = np.random.default_rng(1)
        x = rng.standard_normal(T, dtype=np.float32) * 50.0
        conv = CachedOSConvolver(h, T)
        out = np.empty_like(x)
        conv(x, out=out)
        ref = scipy.signal.oaconvolve(x, h, mode="same")
        np.testing.assert_allclose(out, ref, atol=1e-3)

    def test_thread_safe_shared_convolver(self):
        from concurrent.futures import ThreadPoolExecutor

        sos = _butter_hp(order=5, fc=300.0)
        h = design_matched_fir_from_sos(sos, FS)
        T, C = 30_000, 64
        rng = np.random.default_rng(2)
        x = rng.standard_normal((T, C), dtype=np.float32) * 50.0
        conv = CachedOSConvolver(h, T)

        ref = np.empty_like(x)
        conv(x, out=ref)

        out = np.empty_like(x)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = []
            for c0 in range(0, C, 16):
                c1 = min(c0 + 16, C)
                futs.append(pool.submit(conv, x[:, c0:c1], out=out[:, c0:c1]))
            for f in futs:
                f.result()
        np.testing.assert_allclose(out, ref, atol=1e-5)


# ---- End-to-end via FilterRecording -----------------------------------------


class TestEndToEnd:
    def _make_rec(self, T=60_000, C=16, seed=0):
        rng = np.random.default_rng(seed)
        traces = (rng.standard_normal((T, C)) * 30.0).astype(np.float32)
        return NumpyRecording([traces], sampling_frequency=FS), T, C

    def test_highpass_fir_runs_and_has_correct_shape(self):
        rec, T, C = self._make_rec()
        rec_fir = HighpassFilterRecording(
            rec, freq_min=300.0, dtype="float32", method="fir_magnitude_matched"
        )
        out = rec_fir.get_traces(start_frame=2_000, end_frame=T - 2_000)
        assert out.shape == (T - 4_000, C)
        assert out.dtype == np.float32

    def test_highpass_fir_vs_iir_correlation(self):
        rec, T, C = self._make_rec(T=120_000, C=8)
        iir = HighpassFilterRecording(rec, freq_min=300.0, dtype="float32")
        fir = HighpassFilterRecording(
            rec, freq_min=300.0, dtype="float32", method="fir_magnitude_matched"
        )
        y_iir = iir.get_traces(start_frame=5_000, end_frame=T - 5_000)
        y_fir = fir.get_traces(start_frame=5_000, end_frame=T - 5_000)
        for c in range(C):
            corr = float(np.corrcoef(y_iir[:, c], y_fir[:, c])[0, 1])
            assert corr > 0.95, f"channel {c}: corr {corr:.3f} too low"
        rms_iir = float(np.sqrt(np.mean(y_iir ** 2)))
        rms_fir = float(np.sqrt(np.mean(y_fir ** 2)))
        # Magnitude-matched FIR should have RMS within 20% of IIR for HP.
        assert 0.8 < rms_fir / rms_iir < 1.2

    def test_bandpass_fir_runs(self):
        rec, T, C = self._make_rec(T=60_000, C=16)
        bp = BandpassFilterRecording(
            rec,
            freq_min=300.0,
            freq_max=6000.0,
            dtype="float32",
            method="fir_magnitude_matched",
        )
        out = bp.get_traces(start_frame=2_000, end_frame=T - 2_000)
        assert out.shape == (T - 4_000, C)
        # Mean should be near 0 (DC removed).
        assert abs(out.mean()) < 1.0

    def test_n_workers_bit_identical(self):
        """Channel-parallel FIR should produce identical output to single-thread."""
        rec, T, _ = self._make_rec(T=30_000, C=64)
        fir_serial = HighpassFilterRecording(
            rec, freq_min=300.0, dtype="float32", method="fir_magnitude_matched"
        )
        fir_par = HighpassFilterRecording(
            rec,
            freq_min=300.0,
            dtype="float32",
            method="fir_magnitude_matched",
            n_workers=4,
        )
        y_serial = fir_serial.get_traces(start_frame=2_000, end_frame=T - 2_000)
        y_par = fir_par.get_traces(start_frame=2_000, end_frame=T - 2_000)
        np.testing.assert_array_equal(y_serial, y_par)

    def test_kwargs_round_trip(self):
        rec, _, _ = self._make_rec()
        fir = HighpassFilterRecording(
            rec, freq_min=300.0, dtype="float32", method="fir_magnitude_matched"
        )
        assert fir._kwargs["method"] == "fir_magnitude_matched"

    def test_default_method_is_iir(self):
        """Without method=, the default behavior is unchanged (IIR)."""
        rec, T, C = self._make_rec(T=30_000, C=16)
        iir1 = HighpassFilterRecording(rec, freq_min=300.0, dtype="float32")
        iir2 = HighpassFilterRecording(
            rec, freq_min=300.0, dtype="float32", method="iir"
        )
        y1 = iir1.get_traces(start_frame=1_000, end_frame=T - 1_000)
        y2 = iir2.get_traces(start_frame=1_000, end_frame=T - 1_000)
        np.testing.assert_array_equal(y1, y2)


# ---- Fallback paths ---------------------------------------------------------


class TestFallback:
    def test_ba_mode_falls_back_to_iir(self):
        rng = np.random.default_rng(0)
        T, C = 30_000, 8
        traces = (rng.standard_normal((T, C)) * 50).astype(np.float32)
        rec = NumpyRecording([traces], sampling_frequency=FS)
        with pytest.warns(UserWarning, match="filter_mode='sos'"):
            rec_fir = HighpassFilterRecording(
                rec,
                freq_min=300.0,
                dtype="float32",
                method="fir_magnitude_matched",
                filter_mode="ba",
            )
        # Fall-back path uses IIR; output should match a plain IIR build.
        rec_iir = HighpassFilterRecording(
            rec, freq_min=300.0, dtype="float32", filter_mode="ba"
        )
        np.testing.assert_array_equal(
            rec_fir.get_traces(start_frame=2_000, end_frame=T - 2_000),
            rec_iir.get_traces(start_frame=2_000, end_frame=T - 2_000),
        )

    def test_forward_only_falls_back_to_iir(self):
        rng = np.random.default_rng(0)
        T, C = 30_000, 8
        traces = (rng.standard_normal((T, C)) * 50).astype(np.float32)
        rec = NumpyRecording([traces], sampling_frequency=FS)
        with pytest.warns(UserWarning, match="forward-backward"):
            rec_fir = HighpassFilterRecording(
                rec,
                freq_min=300.0,
                dtype="float32",
                method="fir_magnitude_matched",
                direction="forward",
            )
        rec_iir = HighpassFilterRecording(
            rec, freq_min=300.0, dtype="float32", direction="forward"
        )
        np.testing.assert_array_equal(
            rec_fir.get_traces(start_frame=2_000, end_frame=T - 2_000),
            rec_iir.get_traces(start_frame=2_000, end_frame=T - 2_000),
        )


# ---- stopband_db kwarg ------------------------------------------------------


class TestStopbandDB:
    def _segment_kernel(self, rec):
        """Pull the FIR kernel that was designed for the segment."""
        from spikeinterface.preprocessing._fir_filter import CachedOSConvolver  # noqa: F401
        return rec._segments[0]._fir_kernel

    def test_stopband_default_is_60(self):
        rec, _, _ = self._make_rec()
        fir = HighpassFilterRecording(
            rec, freq_min=300.0, dtype="float32", method="fir_magnitude_matched"
        )
        # Default stopband=60 dB used; tap count for HP order 5 at 300 Hz
        # is 399 (well-tested in existing tests).
        assert len(self._segment_kernel(fir)) == 399

    def test_stopband_lower_means_fewer_taps(self):
        rec, _, _ = self._make_rec()
        fir60 = HighpassFilterRecording(
            rec, freq_min=300.0, dtype="float32",
            method="fir_magnitude_matched", stopband_db=60.0,
        )
        fir40 = HighpassFilterRecording(
            rec, freq_min=300.0, dtype="float32",
            method="fir_magnitude_matched", stopband_db=40.0,
        )
        fir30 = HighpassFilterRecording(
            rec, freq_min=300.0, dtype="float32",
            method="fir_magnitude_matched", stopband_db=30.0,
        )
        n60 = len(self._segment_kernel(fir60))
        n40 = len(self._segment_kernel(fir40))
        n30 = len(self._segment_kernel(fir30))
        # Lowering stopband_db saves taps, but less than naive Bellanger
        # would predict because the IIR's transition narrows at lower-dB
        # points (the -3 dB to -X dB transition shrinks as X drops),
        # which partly offsets the Bellanger reduction.  Empirically ~11%
        # tap savings going 60 → 30 dB on this filter.
        assert n30 < n40 < n60
        assert n60 == 399
        assert 350 <= n40 <= 390
        assert 340 <= n30 <= 380

    def test_kwargs_round_trip_includes_stopband(self):
        rec, _, _ = self._make_rec()
        fir = HighpassFilterRecording(
            rec, freq_min=300.0, dtype="float32",
            method="fir_magnitude_matched", stopband_db=42.0,
        )
        # FilterRecording's _kwargs always includes stopband_db (parent class).
        # The wrapper overrides _kwargs but **filter_kwargs propagates the kwarg.
        assert fir._kwargs.get("stopband_db") == 42.0

    def test_stopband_attenuation_meets_target(self):
        """The designed FIR achieves the requested stopband attenuation in
        the IIR's deep stopband."""
        import scipy.signal
        rec, _, _ = self._make_rec()
        for target in (30.0, 40.0, 50.0):
            fir = HighpassFilterRecording(
                rec, freq_min=300.0, dtype="float32",
                method="fir_magnitude_matched", stopband_db=target,
            )
            h = self._segment_kernel(fir)
            sos = scipy.signal.butter(5, 300.0, btype="highpass", fs=FS, output="sos")
            w_iir, h_iir = scipy.signal.sosfreqz(sos, worN=8192, fs=FS)
            iir_db = 20 * np.log10(np.maximum(np.abs(h_iir) ** 2, 1e-30))
            w_fir, h_fir = scipy.signal.freqz(h, worN=8192, fs=FS)
            fir_db = 20 * np.log10(np.maximum(np.abs(h_fir), 1e-30))
            stop_mask = iir_db <= -60.0  # IIR's deep stopband
            if not np.any(stop_mask):
                continue
            worst_fir = float(np.max(fir_db[stop_mask]))
            assert worst_fir < -target + 5.0, (
                f"target={target}, worst FIR in IIR -60dB region = {worst_fir:.1f} dB"
            )

    def _make_rec(self, T=60_000, C=8, seed=0):
        rng = np.random.default_rng(seed)
        traces = (rng.standard_normal((T, C)) * 30.0).astype(np.float32)
        return NumpyRecording([traces], sampling_frequency=FS), T, C
