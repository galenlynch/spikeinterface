"""Cached overlap-save FIR convolver, magnitude-matched to a given IIR SOS spec.

Used by ``FilterRecording`` when ``method="fir_magnitude_matched"`` is set, to
substitute a linear-phase FIR (designed to match the IIR's ``sosfiltfilt``
magnitude response in dB) for the underlying ``sosfiltfilt`` call.  The FIR
output is *not* sample-equivalent to the IIR — only the magnitude response
matches within design tolerance — but it is much cheaper to compute on
long signals and has the bonus of constant group delay (no waveform-shape
distortion across frequency).

Design notes
------------

The FIR is designed via Parks–McClellan (``scipy.signal.remez``) to a target
stopband attenuation (default -60 dB) at transitions read off the IIR's
``sosfreqz`` magnitude.  For bandpass IIRs the lower- and upper-edge
transitions in absolute Hz are highly asymmetric (Butterworth rolls off at
fixed dB/octave, which is geometric in frequency); feeding both edges as
distinct bands to remez produces a numerically unstable design (typical
result: a kernel with hundreds of dB peak gain in the wider transition
gap).  The implementation picks the *narrower* of the two transitions and
uses it on both edges, producing an FIR that matches the IIR exactly on
the narrow-Δf edge and is *sharper* than the IIR on the wide-Δf edge.

Algorithm
---------

The convolver uses overlap-save with the kernel FFT precomputed once.  The
center-block loop is ported from JuliaDSP/DSP.jl's ``unsafe_conv_kern_os!``:
one small reusable time-domain buffer per block, FFT/multiply/IFFT in
place, then write valid (non-aliased) samples directly to the output.
Block size is chosen by Lambert-W minimization of FFT-ops/output-sample
(scipy's ``_calc_oa_lens`` uses the same formula in closed form).

The convolver is stateless during ``__call__`` (per-call buffers are
local), so multiple threads can share one instance, which the
``FilterRecording`` does — each segment has one convolver, and the
``n_workers`` channel-block thread pool calls it concurrently from
worker threads.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import scipy.fft as sp_fft
import scipy.signal


# ---- Stop-band / pass-band design targets ---------------------------------

_DESIGN_STOPBAND_DB = 60.0
_DESIGN_PASSBAND_DB = 3.0
_DESIGN_RIPPLE_DB = 0.5


# ---- L2 cache size detection ----------------------------------------------


def _detect_l2_bytes() -> int:
    """Best-effort per-core L2 cache size.

    On Linux, reads ``/sys/devices/system/cpu/cpu0/cache/indexN/size``.  POSIX
    ``sysconf`` lists L2 keys but glibc doesn't implement them, so we fall
    back to sysfs directly.  Returns 1 MB if detection fails (a
    conservative number that fits comfortably in any modern CPU's L2).
    """
    try:
        for idx in range(4):
            level_path = f"/sys/devices/system/cpu/cpu0/cache/index{idx}/level"
            type_path = f"/sys/devices/system/cpu/cpu0/cache/index{idx}/type"
            size_path = f"/sys/devices/system/cpu/cpu0/cache/index{idx}/size"
            try:
                with open(level_path) as f:
                    level = int(f.read().strip())
                with open(type_path) as f:
                    cache_type = f.read().strip()
                if level == 2 and cache_type in ("Unified", "Data"):
                    with open(size_path) as f:
                        s = f.read().strip()
                    if s.endswith("K"):
                        return int(s[:-1]) * 1024
                    if s.endswith("M"):
                        return int(s[:-1]) * 1024 * 1024
                    return int(s)
            except (FileNotFoundError, OSError, ValueError):
                continue
    except OSError:
        pass
    return 1_000_000


_L2_BYTES = _detect_l2_bytes()


def pick_chunk_C(
    C: int,
    block_size: int,
    pool_size: int,
    *,
    l2_bytes: int = _L2_BYTES,
    headroom: float = 0.6,
    itemsize: int = 4,
) -> int:
    """Pick channel-chunk size for the FIR overlap-save channel-parallel path.

    Two binding constraints:

    1. **L2 fit**: per-block working set ≈ ``2 × block_size × chunk_C × itemsize``
       (input ``tdbuf`` + ``rfft`` output ``sp1`` simultaneously alive during
       ``rfft``).  Must fit in L2 with some headroom (default 0.6) for the
       FFT to run cache-resident.

    2. **Pool utilization**: at least ``pool_size`` chunks so all workers
       run on the first round.

    The rule is ``min(target_l2, max(C/pool_size, 1))``: as large as possible
    subject to both ceilings.  At typical ephys shapes (C=384, pool=8, fp32,
    block≈2058) the L2 cap is ~46 channels, ``C/pool=48``, and the rule
    picks 46 — both constraints essentially tied.
    """
    ws_per_chunk_C = max(1, 2 * block_size * itemsize)
    target_l2 = max(1, int(headroom * l2_bytes / ws_per_chunk_C))
    target_pool = max(1, (C + pool_size - 1) // pool_size)
    return min(target_l2, max(target_pool, 1))


# ---- Magnitude-matched FIR design from IIR SOS ----------------------------


def design_matched_fir_from_sos(
    sos: np.ndarray,
    fs: float,
    *,
    stopband_db: float = _DESIGN_STOPBAND_DB,
) -> np.ndarray:
    """Design a Remez FIR whose magnitude response matches the IIR's
    ``sosfiltfilt`` response (``|H_iir|^2`` in dB).

    Detects highpass / lowpass / bandpass shape from the IIR's measured
    response.  Bandstop is unsupported (raises ValueError).

    Returns a ``float32`` symmetric (linear-phase) FIR kernel.
    """
    nyq = fs / 2

    w, h = scipy.signal.sosfreqz(sos, worN=16384, fs=fs)
    db = 20 * np.log10(np.maximum(np.abs(h) ** 2, 1e-30))

    pass_mask = db >= -_DESIGN_PASSBAND_DB
    if not np.any(pass_mask):
        raise ValueError(
            "could not detect passband in IIR magnitude response"
        )

    pass_indices = np.where(pass_mask)[0]
    pass_lo_idx = int(pass_indices[0])
    pass_hi_idx = int(pass_indices[-1])
    f_pass_lo = float(w[pass_lo_idx])
    f_pass_hi = float(w[pass_hi_idx])

    is_lowpass = pass_lo_idx == 0 and pass_hi_idx < len(w) - 1
    is_highpass = pass_lo_idx > 0 and pass_hi_idx == len(w) - 1
    is_bandpass = pass_lo_idx > 0 and pass_hi_idx < len(w) - 1

    delta_p = (10 ** (_DESIGN_RIPPLE_DB / 20) - 1) / (10 ** (_DESIGN_RIPPLE_DB / 20) + 1)
    delta_s = 10 ** (-stopband_db / 20)

    def _find_stop_below(pass_idx: int) -> int:
        below = db[:pass_idx] <= -stopband_db
        if not np.any(below):
            return 0
        return int(np.where(below)[0][-1])

    def _find_stop_above(pass_idx: int) -> int:
        offset = pass_idx + 1
        above = db[offset:] <= -stopband_db
        if not np.any(above):
            return len(w) - 1
        return int(np.where(above)[0][0]) + offset

    bands: list[float]
    desired: list[float]
    weights: list[float]
    if is_highpass:
        f_stop_lo = float(w[_find_stop_below(pass_lo_idx)])
        transition = max(f_pass_lo - f_stop_lo, 5.0)
        bands = [0.0, f_stop_lo, f_pass_lo, nyq]
        desired = [0.0, 1.0]
        weights = [1.0, delta_s / delta_p]
    elif is_lowpass:
        f_stop_hi = float(w[_find_stop_above(pass_hi_idx)])
        transition = max(f_stop_hi - f_pass_hi, 5.0)
        bands = [0.0, f_pass_hi, f_stop_hi, nyq]
        desired = [1.0, 0.0]
        weights = [delta_s / delta_p, 1.0]
    elif is_bandpass:
        # Narrower transition wins; mirror it onto both edges.  Asymmetric
        # bands break Remez (typical failure: 250+ dB ringing in the wide
        # transition gap).  Cost: FIR is sharper than IIR on the wider
        # edge, by design.
        f_stop_lo_iir = float(w[_find_stop_below(pass_lo_idx)])
        f_stop_hi_iir = float(w[_find_stop_above(pass_hi_idx)])
        transition = max(
            min(f_pass_lo - f_stop_lo_iir, f_stop_hi_iir - f_pass_hi), 5.0
        )
        f_stop_lo = max(0.0, f_pass_lo - transition)
        f_stop_hi = min(nyq - 1.0, f_pass_hi + transition)
        bands = [0.0, f_stop_lo, f_pass_lo, f_pass_hi, f_stop_hi, nyq]
        desired = [0.0, 1.0, 0.0]
        weights = [1.0, delta_s / delta_p, 1.0]
    else:
        raise ValueError(
            "FIR magnitude-matched design only supports lowpass, highpass, "
            "and bandpass IIRs"
        )

    # Bellanger tap-count rule
    n_taps = int(np.ceil(
        (2 / 3) * np.log10(1 / (10 * delta_p * delta_s)) * fs / transition
    ))
    if n_taps % 2 == 0:
        n_taps += 1
    n_taps = max(n_taps, 21)

    h_fir = scipy.signal.remez(n_taps, bands, desired, weight=weights, fs=fs)
    return np.asarray(h_fir, dtype=np.float32)


# ---- Vendored Lambert-W block-size selection ------------------------------


def _calc_oa_block_size(s1: int, s2: int) -> int:
    """Pick OS block size minimizing FFT-ops/output-sample.

    Vendored from scipy.signal._signaltools._calc_oa_lens (which is private)
    so this module doesn't depend on scipy internals.
    """
    from scipy.special import lambertw

    if s1 == s2 or s1 == 1 or s2 == 1 or s2 >= s1 / 2:
        return int(sp_fft.next_fast_len(s1 + s2 - 1))
    overlap = s2 - 1
    opt_size = -overlap * lambertw(-1 / (2 * math.e * overlap), k=-1).real
    block_size = int(sp_fft.next_fast_len(math.ceil(opt_size)))
    if block_size >= s1:
        return int(sp_fft.next_fast_len(s1 + s2 - 1))
    return block_size


# ---- Cached overlap-save convolver ----------------------------------------


class CachedOSConvolver:
    """Overlap-save FIR convolver with kernel FFT precomputed.

    Specialised to: 1-D linear-phase symmetric kernel applied along axis 0
    of a 2-D ``(T, C)`` input, ``mode='same'`` — i.e., zero-phase output via
    centered-slice on the linear-phase response.

    The center-block loop is a port of ``JuliaDSP/DSP.jl``'s
    ``unsafe_conv_kern_os!``: one small reusable ``tdbuf`` per call, FFT /
    multiply / IFFT in place, then write valid (non-aliased) samples
    directly to ``out``.  Per-call ``tdbuf`` allocation makes the ``__call__``
    method thread-safe — multiple workers can share one convolver instance.
    """

    def __init__(
        self,
        h: np.ndarray,
        T: int,
        block_size: int | None = None,
    ) -> None:
        h = np.asarray(h, dtype=np.float32)
        if h.ndim != 1:
            raise ValueError("kernel must be 1-D")
        self.h = h
        self.taps = int(h.shape[0])
        self.T = int(T)

        if block_size is None:
            block_size = _calc_oa_block_size(self.T, self.taps)
        if block_size < self.taps:
            raise ValueError(f"block_size {block_size} < taps {self.taps}")
        self.block_size = int(block_size)

        self.sv = self.taps - 1
        self.save_blocksize = self.block_size - self.sv
        self.start = (self.taps - 1) // 2
        full_out = self.T + self.sv
        self.nblocks = math.ceil(full_out / self.save_blocksize)

        # Pre-FFT the kernel padded to block_size, in both fp32 and fp64
        # paths so the convolver matches the input dtype.
        h_padded_f32 = np.zeros(self.block_size, dtype=np.float32)
        h_padded_f32[: self.taps] = h
        kfft = sp_fft.rfft(h_padded_f32)
        if kfft.dtype != np.complex64:
            kfft = kfft.astype(np.complex64)
        self.kernel_fd_f32: Any = kfft

        h_padded_f64 = h_padded_f32.astype(np.float64)
        self.kernel_fd_f64: Any = sp_fft.rfft(h_padded_f64)

    def __call__(
        self,
        x: np.ndarray,
        *,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convolve ``x`` with cached kernel via OS, return mode='same' result.

        Writes directly into ``out`` if provided; allocates a fresh array
        of the same shape and dtype as ``x`` otherwise.
        """
        if x.shape[0] != self.T:
            raise ValueError(f"x.shape[0]={x.shape[0]} but configured T={self.T}")
        if out is None:
            out = np.empty(x.shape, dtype=x.dtype)

        is_2d = x.ndim == 2
        if x.dtype == np.float32:
            kfft = self.kernel_fd_f32
        elif x.dtype == np.float64:
            kfft = self.kernel_fd_f64
        else:
            raise ValueError(f"unsupported dtype {x.dtype}")

        sv = self.sv
        bs = self.block_size
        sb = self.save_blocksize
        T = self.T
        gd = self.start

        if is_2d:
            tdbuf = np.empty((bs, x.shape[1]), dtype=x.dtype)
        else:
            tdbuf = np.empty(bs, dtype=x.dtype)

        for b in range(self.nblocks):
            save_start_full = b * sb
            in_start = save_start_full - sv
            in_stop = in_start + bs

            src_lo = max(0, in_start)
            src_hi = min(T, in_stop)
            tdbuf_lo = src_lo - in_start
            tdbuf_hi = tdbuf_lo + (src_hi - src_lo)

            if tdbuf_lo > 0:
                tdbuf[:tdbuf_lo] = 0
            if tdbuf_hi < bs:
                tdbuf[tdbuf_hi:] = 0
            tdbuf[tdbuf_lo:tdbuf_hi] = x[src_lo:src_hi]

            sp1 = sp_fft.rfft(tdbuf, axis=0)
            if is_2d:
                sp1 *= kfft[:, None]
            else:
                sp1 *= kfft
            ret_block = sp_fft.irfft(sp1, n=bs, axis=0, overwrite_x=True)

            out_lo = save_start_full - gd
            out_hi = save_start_full + sb - gd
            valid_lo = max(0, -out_lo)
            valid_hi = sb - max(0, out_hi - T)
            if valid_hi <= valid_lo:
                continue
            dst_lo = max(0, out_lo)
            dst_hi = min(T, out_hi)
            out[dst_lo:dst_hi] = ret_block[sv + valid_lo : sv + valid_hi]

        return out
