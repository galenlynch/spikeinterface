"""Tight micro-bench: just CachedOSConvolver, no FilterRecording plumbing.

Measures the OS convolver's __call__ in isolation, with channel-block
parallelism via ThreadPoolExecutor.  Strips out get_chunk_with_margin,
dtype casts, segment dispatch.

Compares:
  - pure-python inner loop (SI_FIR_NO_NUMBA=1)
  - numba @njit nogil helpers (default)

at single-thread and 8-thread (8 channel chunks).
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.signal


def _import_with_numba(use_numba: bool):
    if use_numba:
        os.environ.pop("SI_FIR_NO_NUMBA", None)
    else:
        os.environ["SI_FIR_NO_NUMBA"] = "1"
    for name in list(sys.modules):
        if "_fir_filter" in name or name.startswith("spikeinterface.preprocessing"):
            del sys.modules[name]
    from spikeinterface.preprocessing import _fir_filter as ff
    return ff


def time_path(label: str, ff, x, h):
    T, C = x.shape
    conv = ff.CachedOSConvolver(h, T)
    print(f"\n--- {label}  USE_NUMBA={ff.USE_NUMBA}  block_size={conv.block_size} nblocks={conv.nblocks} ---")

    # Single-thread: full array
    out = np.empty_like(x)
    for _ in range(3):  # warmup (incl. JIT compile if numba)
        conv(x, out=out)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        conv(x, out=out)
        times.append(time.perf_counter() - t0)
    t_serial = min(times)
    print(f"  single-thread:  {t_serial * 1000:8.1f} ms")

    # 8 threads, 8 channel chunks
    n_workers = 8
    chunks = []
    chunk_size = (C + n_workers - 1) // n_workers
    for i in range(n_workers):
        lo = i * chunk_size
        hi = min(C, lo + chunk_size)
        if hi <= lo:
            break
        chunks.append((lo, hi))

    def _run_chunk(lo, hi, conv, x, out):
        conv(x[:, lo:hi], out=out[:, lo:hi])

    pool = ThreadPoolExecutor(max_workers=n_workers)

    # warmup
    for _ in range(2):
        list(pool.map(lambda c: _run_chunk(c[0], c[1], conv, x, out), chunks))
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        list(pool.map(lambda c: _run_chunk(c[0], c[1], conv, x, out), chunks))
        times.append(time.perf_counter() - t0)
    t_par = min(times)
    pool.shutdown(wait=True)
    print(f"  8-thread x8ch:  {t_par * 1000:8.1f} ms   (speedup vs serial: {t_serial / t_par:.2f}x)")
    return t_serial, t_par


def main():
    fs = 30_000.0
    sos = scipy.signal.butter(5, 300.0, btype="highpass", fs=fs, output="sos")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1_000_000, 384)).astype(np.float32)

    ff_nb = _import_with_numba(True)
    h = ff_nb.design_matched_fir_from_sos(sos, fs)
    print(f"taps={h.shape[0]}")

    s_nb, p_nb = time_path("numba", ff_nb, x, h)

    ff_py = _import_with_numba(False)
    s_py, p_py = time_path("pure python", ff_py, x, h)

    print("\n" + "=" * 80)
    print(f"single-thread:  py={s_py*1000:.1f}ms  nb={s_nb*1000:.1f}ms  numba/py={s_nb/s_py:.2f}x")
    print(f"8-thread x8ch:  py={p_py*1000:.1f}ms  nb={p_nb*1000:.1f}ms  numba/py={p_nb/p_py:.2f}x")
    print(f"  py serial speedup at 8 thr: {s_py/p_py:.2f}x")
    print(f"  nb serial speedup at 8 thr: {s_nb/p_nb:.2f}x")


if __name__ == "__main__":
    main()
