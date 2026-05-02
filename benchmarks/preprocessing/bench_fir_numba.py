"""Compare numba-jitted OS inner-loop helpers vs pure-python.

Runs the FIR-magnitude-matched FilterRecording path on a 1 M × 384 float32
recording at n_workers ∈ {1, 4, 8} and times direct ``get_traces()``.

We run the numba-on case first (in-process), then re-import the modules
with ``SI_FIR_NO_NUMBA=1`` to switch to pure-python.  The recording is
recreated each pass so the FilterRecording picks up the freshly-imported
``CachedOSConvolver``.
"""

from __future__ import annotations

import importlib
import os
import sys
import time

import numpy as np


def _import_si_with_numba(use_numba: bool) -> None:
    if use_numba:
        os.environ.pop("SI_FIR_NO_NUMBA", None)
    else:
        os.environ["SI_FIR_NO_NUMBA"] = "1"
    # Drop preprocessing module so the SI_FIR_NO_NUMBA toggle takes effect.
    for name in list(sys.modules):
        if name.startswith("spikeinterface.preprocessing"):
            del sys.modules[name]


def _time_get_traces(rec, *, start_frame, end_frame, n_reps=3, warmup=2):
    for _ in range(warmup):
        rec.get_traces(start_frame=start_frame, end_frame=end_frame)
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        rec.get_traces(start_frame=start_frame, end_frame=end_frame)
        times.append(time.perf_counter() - t0)
    return float(min(times))


def bench_path(label: str, use_numba: bool, traces, fs):
    _import_si_with_numba(use_numba)
    from spikeinterface import NumpyRecording
    from spikeinterface.preprocessing import HighpassFilterRecording
    from spikeinterface.preprocessing._fir_filter import USE_NUMBA

    assert USE_NUMBA == use_numba, f"requested {use_numba} but module saw {USE_NUMBA}"
    print(f"\n--- {label}  (USE_NUMBA={USE_NUMBA}) ---")

    rec = NumpyRecording([traces], sampling_frequency=fs)
    T_full = rec.get_num_samples()
    Ts = [30_000, 300_000, 1_000_000]
    workers_set = [1, 4, 8]

    results = {}
    print(f"  {'T':>8}  {'n_workers':>9}  {'time (s)':>9}")
    for T in Ts:
        center = T_full // 2
        s = max(0, center - T // 2)
        e = s + T
        for nw in workers_set:
            fir = HighpassFilterRecording(
                rec, freq_min=300.0, dtype="float32",
                n_workers=nw, method="fir_magnitude_matched",
            )
            t = _time_get_traces(fir, start_frame=s, end_frame=e)
            results[(T, nw)] = t
            print(f"  {T:>8d}  {nw:>9d}  {t:>9.3f}")
    return results


def main():
    fs = 30_000.0
    rng = np.random.default_rng(0)
    traces = (rng.standard_normal((1_048_576, 384)) * 50).astype(np.float32)

    print("=" * 80)
    print("FIR-magnitude-matched: numba inner-loop vs pure-python")
    print("=" * 80)
    print("1 M × 384 fp32, highpass 300 Hz (5th-order Butterworth target)")

    # Numba first (so JIT compile cost is paid before warm-up).
    nb = bench_path("numba (njit nogil helpers)", True, traces, fs)
    py = bench_path("pure python", False, traces, fs)

    print()
    print("=" * 80)
    print("Speedup summary")
    print("=" * 80)
    print(f"  {'T':>8}  {'n_workers':>9}  {'python (s)':>11}  {'numba (s)':>10}  {'numba/py':>9}")
    for key in py:
        T, nw = key
        s = py[key] / nb[key]
        print(f"  {T:>8d}  {nw:>9d}  {py[key]:>11.3f}  {nb[key]:>10.3f}  {s:>8.2f}x")


if __name__ == "__main__":
    main()
