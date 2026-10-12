"""Fused FIR phase shift, FIR filter and global common reference.

``fuse`` replaces a ``[phase_shift(method="fir")] -> FIR filter -> [common_reference(reference="global")]`` chain,
where the filter is ``fir_filter`` or ``highpass_filter``/``bandpass_filter`` with ``method="fir"``, with one
``FusedFIRReferenceRecording`` that reads the raw chunk once and does all the work in L2-sized tiles:

- The phase-shift and filter kernels are both linear and time-invariant, so each channel gets one composite kernel
  ``filter * phase_shift`` whose FFT is cached per FFT length.
- Overlap-save runs on (time block x channel slab) tiles dispatched in time-block-major order, so the workers stay
  in the same time region and the raw rows are read from L3 once.
- When every channel slab of a time block is done, the block's rows are split into L2-sized pieces for the common
  reference, and threads take those ahead of new tiles while the block is in cache.

Intermediates stay float32 and the output is cast to the leaf's dtype once, so the result differs from an unfused
integer chain by the roundings that chain does after each step. At the recording's edges the unfused chain
zero-pads before each step and the fused one only before the first, so the outputs also differ within a kernel
length of the edges.
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np

from spikeinterface.core.job_tools import get_inner_pool, get_thread_budget

from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from ._row_median import row_median
from .common_reference import CommonReferenceRecording
from .fir_filter import _L2_BYTES, FIRFilterRecordingSegment, _channels_per_block, _overlap_save_fft_length
from .phase_shift import PhaseShiftRecording, _build_fir_kernels_kc


def _mean_reference(block):
    return block.mean(axis=1, keepdims=True)


def build_composite_kernels(taps, channel_groups, center, sample_shifts=None, n_taps=None):
    """Convolve each channel's filter kernel with its phase-shift kernel.

    `taps` is ``(num_kernels, num_taps)`` with channel ``c`` using row ``channel_groups[c]``, applied as
    ``y[n] = sum_j taps[j] * x[n + center - j]``. Returns ``(kernels_lc, kernel_index, center)``: float64 kernels as
    ``(L, K)`` columns, the column for each channel, and the tap aligned with the output sample in the same
    convention.
    """
    taps = np.atleast_2d(np.asarray(taps, dtype=np.float64))
    channel_groups = np.asarray(channel_groups, dtype=np.intp)
    if sample_shifts is None:
        return np.ascontiguousarray(taps.T), channel_groups, center

    # The phase shift correlates, y[n] = sum_k h[k] x[n - half + k]; reversed it is a convolution centred on
    # n_taps - 1 - half, and two convolutions compose with their centres added.
    half = n_taps // 2
    shift_kernels_kc = _build_fir_kernels_kc(sample_shifts, n_taps).astype(np.float64)[::-1]
    num_channels = shift_kernels_kc.shape[1]
    kernels_lc = np.empty((taps.shape[1] + n_taps - 1, num_channels))
    for c in range(num_channels):
        kernels_lc[:, c] = np.convolve(taps[channel_groups[c]], shift_kernels_kc[:, c])
    return kernels_lc, np.arange(num_channels), center + n_taps - 1 - half


class FusedFIRReferenceRecording(BasePreprocessor):
    """FIR filter with an optional per-channel FIR phase shift and global common reference, in one pass.

    Usually built by ``fuse`` from an unfused chain rather than directly.

    Parameters
    ----------
    recording : Recording
        The raw recording, before any of the fused steps.
    taps : array-like
        Filter kernel, ``(num_taps,)`` or ``(num_kernels, num_taps)``, as ``fir_filter`` takes it.
    channel_groups : array-like | None, default: None
        For a 2D `taps`, the row of `taps` for each channel, as ``fir_filter`` takes it.
    center : int | None, default: None
        Index of the filter tap aligned with the output sample. None means ``(num_taps - 1) // 2``.
    sample_shifts : array-like | None, default: None
        Per-channel fractional delay in samples, as ``phase_shift`` uses. None skips the phase shift.
    n_taps : int, default: 16
        Phase-shift kernel length, as ``phase_shift(method="fir")`` uses.
    operator : "median" | "average" | None, default: None
        Global common reference across all channels. None skips it.
    dtype : dtype | None, default: None
        Output dtype. None keeps the parent's.
    """

    def __init__(
        self,
        recording,
        taps,
        channel_groups=None,
        center=None,
        sample_shifts=None,
        n_taps=16,
        operator=None,
        dtype=None,
    ):
        if operator not in (None, "median", "average"):
            raise ValueError(f"operator must be None, 'median' or 'average', got {operator!r}")
        dtype = np.dtype(recording.get_dtype() if dtype is None else dtype)
        BasePreprocessor.__init__(self, recording, dtype=dtype)

        taps = np.asarray(taps, dtype=np.float64)
        num_channels = recording.get_num_channels()
        if channel_groups is None:
            groups = (
                np.zeros(num_channels, dtype=np.intp) if taps.ndim == 1 or len(taps) == 1 else np.arange(num_channels)
            )
        else:
            groups = np.asarray(channel_groups, dtype=np.intp)
        filter_center = (taps.shape[-1] - 1) // 2 if center is None else int(center)
        kernels_lc, kernel_index, composite_center = build_composite_kernels(
            taps, groups, filter_center, None if sample_shifts is None else np.asarray(sample_shifts), n_taps
        )
        for parent_segment in recording.segments:
            self.add_recording_segment(
                FusedFIRReferenceRecordingSegment(
                    parent_segment, kernels_lc, kernel_index, composite_center, operator, dtype
                )
            )

        self._kwargs = dict(
            recording=recording,
            taps=taps.tolist(),
            channel_groups=None if channel_groups is None else groups.tolist(),
            center=filter_center,
            sample_shifts=None if sample_shifts is None else np.asarray(sample_shifts, dtype=np.float64).tolist(),
            n_taps=int(n_taps),
            operator=operator,
            dtype=dtype.str,
        )


class FusedFIRReferenceRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, kernels_lc, kernel_index, center, operator, dtype):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.kernels_lc = kernels_lc
        self.kernel_index = kernel_index
        self.num_taps = kernels_lc.shape[0]
        self.center = center
        self.reference_func = {None: None, "median": row_median, "average": _mean_reference}[operator]
        self.dtype = dtype
        # Kernel spectra depend on the FFT length, which depends on the chunk length.
        self._kernel_spectra = {}
        self._kernel_spectra_lock = threading.Lock()

    def _get_kernel_spectra(self, fft_length):
        import scipy.fft

        with self._kernel_spectra_lock:
            spectra = self._kernel_spectra.get(fft_length)
            if spectra is None:
                spectra = scipy.fft.rfft(self.kernels_lc, n=fft_length, axis=0).astype(np.complex64)
                self._kernel_spectra[fft_length] = spectra
        return spectra

    def _get_input(self, start_frame, end_frame, channel_indices):
        """Raw traces covering every input sample the kernel reaches, zero beyond the recording."""
        left = self.num_taps - 1 - self.center
        right = self.center
        fetch_start = max(0, start_frame - left)
        fetch_end = min(self.parent_recording_segment.get_num_samples(), end_frame + right)
        traces = self.parent_recording_segment.get_traces(fetch_start, fetch_end, channel_indices)
        pad = ((fetch_start - (start_frame - left), end_frame + right - fetch_end), (0, 0))
        if pad[0] != (0, 0):
            traces = np.pad(traces, pad)
        return traces

    def get_traces(self, start_frame, end_frame, channel_indices):
        import scipy.fft

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()
        if channel_indices is None:
            channel_indices = slice(None)

        # The reference needs every channel, so with one the selection happens at the end.
        fetch_channels = slice(None) if self.reference_func is not None else channel_indices
        padded = self._get_input(start_frame, end_frame, fetch_channels)
        num_channels = padded.shape[1]
        kernel_index = self.kernel_index[fetch_channels]

        num_out = end_frame - start_frame
        overlap = self.num_taps - 1
        # Per-channel kernel spectra make long tiles spill out of L2 sooner than the FFT-work model allows for.
        fft_length = min(
            _overlap_save_fft_length(num_out + overlap, self.num_taps),
            scipy.fft.next_fast_len(4 * self.num_taps, real=True),
        )
        valid_per_block = fft_length - overlap
        num_blocks = -(-num_out // valid_per_block)

        spectra = self._get_kernel_spectra(fft_length)
        one_kernel = spectra.shape[1] == 1
        # The float32 block and its complex64 spectrum take 8 bytes per sample, and per-channel kernel spectra 4 more.
        slab_channels = _channels_per_block(num_channels, fft_length, bytes_per_sample=8 if one_kernel else 12)
        slabs = [(c0, min(c0 + slab_channels, num_channels)) for c0 in range(0, num_channels, slab_channels)]
        out = np.empty((num_out, num_channels), dtype=np.float32)
        # Reference rows are split into pieces that fit in L2, so a finished block is shared across threads.
        reference_rows = max(64, int(0.5 * _L2_BYTES / (4 * num_channels)))

        def run_tile(b, c0, c1):
            t0 = b * valid_per_block
            t1 = min(t0 + valid_per_block, num_out)
            # Output sample n reads padded[n : n + overlap + 1], so block b reads padded[t0 : t0 + fft_length].
            source = padded[t0 : t0 + fft_length, c0:c1]
            tile = np.empty((fft_length, c1 - c0), dtype=np.float32)
            tile[: source.shape[0]] = source
            tile[source.shape[0] :] = 0  # only the last block runs past the fetched rows
            spectrum = scipy.fft.rfft(tile, axis=0, overwrite_x=True)
            spectrum *= spectra if one_kernel else spectra[:, kernel_index[c0:c1]]
            result = scipy.fft.irfft(spectrum, n=fft_length, axis=0, overwrite_x=True)
            out[t0:t1, c0:c1] = result[overlap : overlap + (t1 - t0)]

        def run_reference(r0, r1):
            out[r0:r1] -= self.reference_func(out[r0:r1])

        tiles = [(b, c0, c1) for b in range(num_blocks) for c0, c1 in slabs]
        slabs_left = [len(slabs)] * num_blocks
        pending_references = deque()
        next_tile = 0
        tiles_running = 0
        state = threading.Condition()

        def work():
            # Each thread prefers a finished block's reference rows over a new tile, while the block is in cache.
            nonlocal next_tile, tiles_running
            while True:
                with state:
                    while True:
                        if pending_references:
                            task = ("reference", pending_references.popleft())
                            break
                        if next_tile < len(tiles):
                            task = ("tile", tiles[next_tile])
                            next_tile += 1
                            tiles_running += 1
                            break
                        if tiles_running == 0:
                            return
                        state.wait()
                kind, args = task
                if kind == "reference":
                    run_reference(*args)
                    continue
                run_tile(*args)
                with state:
                    tiles_running -= 1
                    b = args[0]
                    slabs_left[b] -= 1
                    if slabs_left[b] == 0 and self.reference_func is not None:
                        t0 = b * valid_per_block
                        t1 = min(t0 + valid_per_block, num_out)
                        pending_references.extend(
                            (r0, min(r0 + reference_rows, t1)) for r0 in range(t0, t1, reference_rows)
                        )
                    state.notify_all()

        pool = get_inner_pool()
        if pool is None or len(tiles) < 2:
            work()
        else:
            for future in [pool.submit(work) for _ in range(get_thread_budget())]:
                future.result()

        if self.reference_func is not None:
            out = out[:, channel_indices]
        if np.issubdtype(self.dtype, np.integer):
            info = np.iinfo(self.dtype)
            out = np.clip(np.rint(out), info.min, info.max)
        return out.astype(self.dtype, copy=False)


def fuse(recording):
    """Replace a fusable preprocessing suffix of `recording` with one ``FusedFIRReferenceRecording``.

    Recognises ``[phase_shift(method="fir")] -> FIR filter -> [common_reference(reference="global")]`` ending at
    `recording`, where the FIR filter is ``fir_filter``, or ``highpass_filter`` or ``bandpass_filter`` with
    ``method="fir"``, without reflect padding. Anything else is returned unchanged. Only the leaf's dtype is kept; the
    steps' intermediate dtypes are not reproduced.
    """
    leaf = recording
    node = recording
    operator = None
    if isinstance(node, CommonReferenceRecording):
        segment = node.segments[0]
        if (
            segment.reference != "global"
            or segment.group_indices is not None
            or segment.ref_channel_indices is not None
        ):
            return recording
        operator = segment.operator
        node = node._parent

    segment = node.segments[0]
    if not isinstance(segment, FIRFilterRecordingSegment) or segment.add_reflect_padding:
        return recording
    taps, channel_groups, center = segment.kernels, segment.channel_groups, segment.center
    node = node._parent

    sample_shifts = None
    n_taps = 16
    if isinstance(node, PhaseShiftRecording) and node.segments[0].method == "fir":
        sample_shifts = node.segments[0].sample_shifts
        n_taps = node.segments[0].n_taps
        node = node._parent

    fused = FusedFIRReferenceRecording(
        node,
        taps,
        channel_groups=channel_groups,
        center=center,
        sample_shifts=sample_shifts,
        n_taps=n_taps,
        operator=operator,
        dtype=leaf.get_dtype(),
    )
    leaf.copy_metadata(fused, only_main=False)
    return fused
