"""End-to-end preprocessing-chain benchmark.

Compares the full ``BandpassFilter → CommonReference → PhaseShift`` chain
in three configurations:

  - **Stock**: BP IIR sosfiltfilt, CMR serial median, PhaseShift FFT.
  - **Fast (already-landed PRs)**: BP IIR ``n_workers=8``, CMR
    ``n_workers=16``, PhaseShift ``method="fir"`` with int16 fast path.
    This is the configuration from PR #4562 / #4563 / #4564.
  - **Fast + FIR (this PR)**: BP ``method="fir_magnitude_matched"`` +
    ``n_workers=8``, CMR ``n_workers=16``, PhaseShift ``method="fir"``.

Per-stage and end-to-end timings are reported for both float32 and int16
input (the int16 path matters because PhaseShift's FIR variant has an
explicit int16 → float32 fast path, and BP+CMR+PS as a chain is the
bottleneck for the AIND spike-sorting prep workflow).
"""

from __future__ import annotations

import time

import numpy as np

from spikeinterface import NumpyRecording
from spikeinterface.preprocessing import (
    BandpassFilterRecording,
    CommonReferenceRecording,
    PhaseShiftRecording,
)


def _make_recording(*, dtype, T: int = 1_048_576, C: int = 384, fs: float = 30_000.0):
    rng = np.random.default_rng(0)
    if dtype == np.int16:
        traces = (rng.standard_normal((T, C)) * 100).astype(np.int16)
    else:
        traces = (rng.standard_normal((T, C)) * 50).astype(dtype)
    rec = NumpyRecording([traces], sampling_frequency=fs)
    # Phase-shift needs `inter_sample_shift` property (one-shift-per-channel).
    shifts = (np.arange(C) % 12) / 12.0  # NP-style 12-channel ADC group
    rec.set_property("inter_sample_shift", shifts)
    return rec


def _time_get_traces(rec, *, start_frame, end_frame, n_reps=3, warmup=1):
    for _ in range(warmup):
        rec.get_traces(start_frame=start_frame, end_frame=end_frame)
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        rec.get_traces(start_frame=start_frame, end_frame=end_frame)
        times.append(time.perf_counter() - t0)
    return float(min(times))


# ---- Per-stage timings -----------------------------------------------------


def bench_per_stage(*, dtype):
    print("=" * 80)
    print(f"Per-stage end-to-end (rec.get_traces()), dtype={np.dtype(dtype).name}")
    print("=" * 80)

    rec = _make_recording(dtype=dtype)
    T = rec.get_num_samples()
    s = T // 4
    e = s + T // 2  # half the recording, centered

    print(f"  Window: [{s}, {e}] = {e - s} samples × {rec.get_num_channels()} ch")
    print()
    print(f"  {'stage':<14} {'config':<48} {'time (s)':>9}")

    # Bandpass
    bp_stock = BandpassFilterRecording(
        rec, freq_min=300.0, freq_max=6000.0, dtype="float32"
    )
    bp_iir_par = BandpassFilterRecording(
        rec, freq_min=300.0, freq_max=6000.0, dtype="float32", n_workers=8
    )
    bp_fir_par = BandpassFilterRecording(
        rec,
        freq_min=300.0,
        freq_max=6000.0,
        dtype="float32",
        n_workers=8,
        method="fir_magnitude_matched",
    )
    print(f"  {'Bandpass':<14} {'stock (IIR n_workers=1)':<48} {_time_get_traces(bp_stock, start_frame=s, end_frame=e):>9.3f}")
    print(f"  {'Bandpass':<14} {'IIR n_workers=8':<48} {_time_get_traces(bp_iir_par, start_frame=s, end_frame=e):>9.3f}")
    print(f"  {'Bandpass':<14} {'FIR n_workers=8 (this PR)':<48} {_time_get_traces(bp_fir_par, start_frame=s, end_frame=e):>9.3f}")
    print()

    # CMR — feed the float32 BP output to make per-stage timings comparable
    cmr_stock = CommonReferenceRecording(
        bp_stock, operator="median", reference="global"
    )
    cmr_par = CommonReferenceRecording(
        bp_stock, operator="median", reference="global", n_workers=16
    )
    print(f"  {'CMR':<14} {'stock (n_workers=1) over BP-stock':<48} {_time_get_traces(cmr_stock, start_frame=s, end_frame=e):>9.3f}")
    print(f"  {'CMR':<14} {'n_workers=16 over BP-stock':<48} {_time_get_traces(cmr_par, start_frame=s, end_frame=e):>9.3f}")
    print()

    # PhaseShift — FFT vs FIR; for int16, FIR with output_dtype=float32 is the
    # fast path that skips int16→float64→int16 round-trip.
    ps_stock = PhaseShiftRecording(rec, method="fft")
    if dtype == np.int16:
        ps_fast = PhaseShiftRecording(rec, method="fir", output_dtype=np.float32)
    else:
        ps_fast = PhaseShiftRecording(rec, method="fir")
    print(f"  {'PhaseShift':<14} {'stock (method=fft)':<48} {_time_get_traces(ps_stock, start_frame=s, end_frame=e):>9.3f}")
    label = 'method=fir, output_dtype=f32' if dtype == np.int16 else 'method=fir'
    print(f"  {'PhaseShift':<14} {label:<48} {_time_get_traces(ps_fast, start_frame=s, end_frame=e):>9.3f}")
    print()


# ---- Full-chain timings ----------------------------------------------------


def _build_chain_stock(rec):
    """Stock chain: all defaults, no n_workers, IIR BP, FFT phase-shift."""
    ps = PhaseShiftRecording(rec, method="fft")
    bp = BandpassFilterRecording(ps, freq_min=300.0, freq_max=6000.0, dtype="float32")
    cmr = CommonReferenceRecording(bp, operator="median", reference="global")
    return cmr


def _build_chain_fast(rec, *, dtype):
    """Fast chain (PRs already in flight): IIR n=8 BP, CMR n=16, FIR PS."""
    if dtype == np.int16:
        ps = PhaseShiftRecording(rec, method="fir", output_dtype=np.float32)
    else:
        ps = PhaseShiftRecording(rec, method="fir")
    bp = BandpassFilterRecording(
        ps, freq_min=300.0, freq_max=6000.0, dtype="float32", n_workers=8
    )
    cmr = CommonReferenceRecording(
        bp, operator="median", reference="global", n_workers=16
    )
    return cmr


def _build_chain_fast_fir(rec, *, dtype):
    """Fast + FIR-bandpass (this PR's addition)."""
    if dtype == np.int16:
        ps = PhaseShiftRecording(rec, method="fir", output_dtype=np.float32)
    else:
        ps = PhaseShiftRecording(rec, method="fir")
    bp = BandpassFilterRecording(
        ps,
        freq_min=300.0,
        freq_max=6000.0,
        dtype="float32",
        n_workers=8,
        method="fir_magnitude_matched",
    )
    cmr = CommonReferenceRecording(
        bp, operator="median", reference="global", n_workers=16
    )
    return cmr


def bench_full_chain(*, dtype):
    print("=" * 80)
    print(f"Full chain (PhaseShift → Bandpass → CMR), dtype={np.dtype(dtype).name}")
    print("=" * 80)

    rec = _make_recording(dtype=dtype)
    T = rec.get_num_samples()
    s = T // 4
    e = s + T // 2

    chain_stock = _build_chain_stock(rec)
    chain_fast = _build_chain_fast(rec, dtype=dtype)
    chain_fir = _build_chain_fast_fir(rec, dtype=dtype)

    t_stock = _time_get_traces(chain_stock, start_frame=s, end_frame=e)
    t_fast = _time_get_traces(chain_fast, start_frame=s, end_frame=e)
    t_fir = _time_get_traces(chain_fir, start_frame=s, end_frame=e)

    print(f"  {'config':<60} {'time (s)':>9}  {'speedup':>8}")
    print(f"  {'stock (IIR / serial CMR / FFT phase-shift)':<60} {t_stock:>9.3f}  {'1.00x':>8}")
    print(f"  {'fast: IIR n_workers=8 / CMR n=16 / FIR phase-shift':<60} {t_fast:>9.3f}  {t_stock / t_fast:>7.2f}x")
    print(f"  {'fast + FIR-BP (this PR)':<60} {t_fir:>9.3f}  {t_stock / t_fir:>7.2f}x")
    print()


def main():
    for dtype in (np.float32, np.int16):
        bench_per_stage(dtype=dtype)
    print()
    for dtype in (np.float32, np.int16):
        bench_full_chain(dtype=dtype)


if __name__ == "__main__":
    main()
