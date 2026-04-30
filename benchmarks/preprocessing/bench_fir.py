"""Benchmark FilterRecording's IIR vs ``method="fir_magnitude_matched"`` path.

Two regimes:

  1. Direct ``get_traces()`` at chunk T ∈ {30 k, 300 k, 1 M} samples,
     n_workers ∈ {1, 8}.  Captures the per-call relationship between IIR
     ``sosfiltfilt`` and the FIR overlap-save substitution at typical SI
     chunk sizes (the 30 k case is SI's default ``chunk_duration="1s"``
     at fs = 30 kHz).

  2. CRE (``TimeSeriesChunkExecutor``) outer × inner: outer ``n_jobs`` ∈
     {1, 8} × inner ``n_workers`` ∈ {1, 8}.  Captures the realistic
     spike-sorting prep workflow where chunks flow through the filter
     in batches.

Both regimes use the AIND default highpass (5th-order Butterworth at
300 Hz) on a 1 M × 384 float32 recording.
"""

from __future__ import annotations

import time

import numpy as np

from spikeinterface import NumpyRecording
from spikeinterface.preprocessing import HighpassFilterRecording


def _make_recording(T: int = 1_048_576, C: int = 384, fs: float = 30_000.0):
    rng = np.random.default_rng(0)
    traces = (rng.standard_normal((T, C)) * 50).astype(np.float32)
    return NumpyRecording([traces], sampling_frequency=fs)


def _time_get_traces(rec, *, start_frame, end_frame, n_reps=3, warmup=1):
    for _ in range(warmup):
        rec.get_traces(start_frame=start_frame, end_frame=end_frame)
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        rec.get_traces(start_frame=start_frame, end_frame=end_frame)
        times.append(time.perf_counter() - t0)
    return float(min(times))


def _time_cre(executor, *, n_reps=2, warmup=1):
    for _ in range(warmup):
        executor.run()
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        executor.run()
        times.append(time.perf_counter() - t0)
    return float(min(times))


def _cre_init(recording):
    return {"recording": recording}


def _cre_func(segment_index, start_frame, end_frame, worker_dict):
    rec = worker_dict["recording"]
    rec.get_traces(start_frame=start_frame, end_frame=end_frame, segment_index=segment_index)
    return None


# ---- Regime 1: direct get_traces, T × n_workers × method ----------------


def bench_get_traces_T_workers():
    print("=" * 80)
    print("Direct rec.get_traces():  IIR vs FIR  ×  n_workers={1,8}  ×  T={30k,300k,1M}")
    print("=" * 80)
    print("Highpass: 5th-order Butterworth at 300 Hz, fs=30 kHz, 384 channels")
    print()

    rec_full = _make_recording(T=1_048_576)
    Ts = [30_000, 300_000, 1_000_000]
    workers_set = [1, 8]

    print(
        f"  {'T':>8}  {'n_workers':>9}  {'IIR (s)':>8}  {'FIR (s)':>8}  "
        f"{'FIR/IIR':>8}  {'IIR speedup':>11}  {'FIR speedup':>11}"
    )
    iir_serial: dict[int, float] = {}
    fir_serial: dict[int, float] = {}
    for T in Ts:
        # Pick a centered window so we have margin headroom on both sides.
        center = rec_full.get_num_samples() // 2
        s = max(0, center - T // 2)
        e = s + T
        for nw in workers_set:
            iir = HighpassFilterRecording(rec_full, freq_min=300.0, dtype="float32", n_workers=nw)
            fir = HighpassFilterRecording(
                rec_full,
                freq_min=300.0,
                dtype="float32",
                n_workers=nw,
                method="fir_magnitude_matched",
            )
            t_iir = _time_get_traces(iir, start_frame=s, end_frame=e)
            t_fir = _time_get_traces(fir, start_frame=s, end_frame=e)
            if nw == 1:
                iir_serial[T] = t_iir
                fir_serial[T] = t_fir
                iir_speedup = "—"
                fir_speedup = "—"
            else:
                iir_speedup = f"{iir_serial[T] / t_iir:5.2f}x"
                fir_speedup = f"{fir_serial[T] / t_fir:5.2f}x"
            print(
                f"  {T:>8d}  {nw:>9d}  {t_iir:>8.3f}  {t_fir:>8.3f}  "
                f"{t_fir / t_iir:>7.2f}x  {iir_speedup:>11}  {fir_speedup:>11}"
            )
        print()


# ---- Regime 2: CRE outer × inner ---------------------------------------


def bench_cre_outer_x_inner():
    from spikeinterface.core.job_tools import TimeSeriesChunkExecutor

    print("=" * 80)
    print("CRE outer × inner:  IIR vs FIR  ×  n_jobs={1,8}  ×  n_workers={1,8}")
    print("=" * 80)
    print("1 M × 384 float32, chunk_duration='1s' (SI default), highpass 300 Hz")
    print()

    rec = _make_recording()

    def make_cre(filter_rec, n_jobs):
        return TimeSeriesChunkExecutor(
            time_series=filter_rec,
            func=_cre_func,
            init_func=_cre_init,
            init_args=(filter_rec,),
            pool_engine="thread",
            n_jobs=n_jobs,
            chunk_duration="1s",
            progress_bar=False,
        )

    print(f"  {'method':<5}  {'outer':>5}  {'inner':>5}  {'time (s)':>9}  {'vs baseline':>12}")
    baseline = None
    for method, label in [("iir", "IIR"), ("fir_magnitude_matched", "FIR")]:
        for n_jobs in (1, 8):
            for n_workers in (1, 8):
                rec_filt = HighpassFilterRecording(
                    rec, freq_min=300.0, dtype="float32",
                    n_workers=n_workers, method=method,
                )
                cre = make_cre(rec_filt, n_jobs=n_jobs)
                t = _time_cre(cre)
                if baseline is None:
                    baseline = t
                    speedup = "1.00x"
                else:
                    speedup = f"{baseline / t:5.2f}x"
                print(
                    f"  {label:<5}  {n_jobs:>5d}  {n_workers:>5d}  {t:>9.3f}  {speedup:>12}"
                )
        print()


def main():
    bench_get_traces_T_workers()
    bench_cre_outer_x_inner()


if __name__ == "__main__":
    main()
