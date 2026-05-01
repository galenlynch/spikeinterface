import os
import threading
import warnings
import weakref

import numpy as np

from spikeinterface.core.core_tools import define_function_handling_dict_from_class
from spikeinterface.core import (
    get_chunk_with_margin,
    ensure_chunk_size,
    get_global_job_kwargs,
    is_set_global_job_kwargs_set,
)

from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment

HIGHPASS_ERROR_THRESHOLD_HZ = 100
MARGIN_TO_CHUNK_PERCENT_WARNING = 0.2  # 20%


_common_filter_docs = """**filter_kwargs : dict
        Certain keyword arguments for `scipy.signal` filters:
            filter_order : order
                The order of the filter. Note as filtering is applied with scipy's
                `filtfilt` functions (i.e. acausal, zero-phase) the effective
                order will be double the `filter_order`.
            filter_mode :  "sos" | "ba", default: "sos"
                Filter form of the filter coefficients:
                - second-order sections ("sos")
                - numerator/denominator : ("ba")
            ftype : str, default: "butter"
                Filter type for `scipy.signal.iirfilter` e.g. "butter", "cheby1"."""


class FilterRecording(BasePreprocessor):
    """
    A generic filter class based on:
        For filter coefficient generation:
            * scipy.signal.iirfilter
        For filter application:
            * scipy.signal.filtfilt or scipy.signal.sosfiltfilt when direction = "forward-backward"
            * scipy.signal.lfilter or scipy.signal.sosfilt when direction = "forward" or "backward"

    BandpassFilterRecording is built on top of it.

    Parameters
    ----------
    recording : Recording
        The recording extractor to be re-referenced
    band : float or list, default: [300.0, 6000.0]
        If float, cutoff frequency in Hz for "highpass" filter type
        If list. band (low, high) in Hz for "bandpass" filter type
    btype : "bandpass" | "highpass", default: "bandpass"
        Type of the filter
    margin_ms : float, default: None
        Margin in ms on border to avoid border effect.
        Must be provided by sub-class.
    coeff : array | None, default: None
        Filter coefficients in the filter_mode form.
    dtype : dtype or None, default: None
        The dtype of the returned traces. If None, the dtype of the parent recording is used
    add_reflect_padding : Bool, default False
        If True, uses a left and right margin during calculation.
    filter_order : order
        The order of the filter for `scipy.signal.iirfilter`
    filter_mode :  "sos" | "ba", default: "sos"
        Filter form of the filter coefficients for `scipy.signal.iirfilter`:
        - second-order sections ("sos")
        - numerator/denominator : ("ba")
    ftype : str, default: "butter"
        Filter type for `scipy.signal.iirfilter` e.g. "butter", "cheby1".
    direction : "forward" | "backward" | "forward-backward", default: "forward-backward"
        Direction of filtering:
        - "forward" - filter is applied to the timeseries in one direction, creating phase shifts
        - "backward" - the timeseries is reversed, the filter is applied and filtered timeseries reversed again. Creates phase shifts in the opposite direction to "forward"
        - "forward-backward" - Applies the filter in the forward and backward direction, resulting in zero-phase filtering. Note this doubles the effective filter order.
    n_workers : int, default: 1
        Channel-parallel pool size for the SOS path. See ``_apply_sos`` for details.
    method : "iir" | "fir_magnitude_matched", default: "iir"
        Filtering method.

        - "iir" preserves the historical behavior — applies ``scipy.signal.sosfiltfilt``
          (or ``filtfilt``/``lfilter`` per ``direction``) to the IIR coefficients above.
        - "fir_magnitude_matched" designs a linear-phase Remez FIR matching the IIR's
          ``sosfiltfilt`` magnitude response (within ``stopband_db`` tolerance) and
          applies it via cached overlap-save with the same ``n_workers`` channel pool.
          The FIR output is *not* sample-equivalent to the IIR — only the magnitude
          spec matches — but it is much cheaper for long signals and has constant
          group delay (no waveform-shape distortion across frequency).
    stopband_db : float, default: 60.0
        Target stopband attenuation in dB for the FIR design when
        ``method="fir_magnitude_matched"``. Lower values produce somewhat shorter
        FIRs at the cost of less aggressive out-of-band suppression. The savings are
        modest because lowering the stopband target also narrows the IIR's
        transition band (the IIR's −X dB point sits closer to its −3 dB cutoff than
        its −60 dB point), and the two effects mostly cancel: e.g., HP order-5 at
        300 Hz, fs=30 kHz: 60 dB → 399 taps; 30 dB → 355 taps (~11% savings).
        For typical Neuropixels recordings where the analog hardware has already
        AC-coupled most LFP, 30 dB is empirically sufficient to put residual
        out-of-band content well below the per-channel noise floor; the default of
        60 dB is conservative and adds margin against unmeasured worst-case
        recordings. Ignored when ``method="iir"``.

    Returns
    -------
    filter_recording : FilterRecording
        The filtered recording extractor object
    """

    def __init__(
        self,
        recording,
        band=(300.0, 6000.0),
        btype="bandpass",
        filter_order=5,
        ftype="butter",
        filter_mode="sos",
        margin_ms=None,
        add_reflect_padding=False,
        coeff=None,
        dtype=None,
        direction="forward-backward",
        n_workers=1,
        method="iir",
        stopband_db=60.0,
    ):
        import scipy.signal

        assert filter_mode in ("sos", "ba"), "'filter' mode must be 'sos' or 'ba'"
        assert int(n_workers) >= 1, "n_workers must be >= 1"
        assert method in ("iir", "fir_magnitude_matched"), (
            "'method' must be 'iir' (default) or 'fir_magnitude_matched'"
        )
        assert stopband_db > 0, "stopband_db must be positive"
        fs = recording.get_sampling_frequency()
        if coeff is None:
            assert btype in ("bandpass", "highpass"), "'bytpe' must be 'bandpass' or 'highpass'"
            # coefficient
            # self.coeff is 'sos' or 'ab' style
            filter_coeff = scipy.signal.iirfilter(
                filter_order, band, fs=fs, analog=False, btype=btype, ftype=ftype, output=filter_mode
            )
        else:
            filter_coeff = coeff
            if not isinstance(coeff, list):
                if filter_mode == "ba":
                    coeff = [c.tolist() for c in coeff]
                else:
                    coeff = coeff.tolist()
        dtype = fix_dtype(recording, dtype)

        BasePreprocessor.__init__(self, recording, dtype=dtype)
        self.annotate(is_filtered=True)

        if "offset_to_uV" in self.get_property_keys():
            self.set_channel_offsets(0)

        assert margin_ms is not None, "margin_ms must be provided!"
        margin = int(margin_ms * fs / 1000.0)

        # If method='fir_magnitude_matched', design the FIR once at construction
        # time and pass it to each segment.  The FIR is intended to substitute
        # for the IIR's filtfilt response (single-pass FIR ≈ filtfilt IIR in
        # magnitude), so we require direction='forward-backward'.  For
        # unsupported specs (filter_mode='ba', non-Butterworth IIRs that don't
        # design cleanly as FIR, bandstop) we fall back to IIR with a warning.
        fir_kernel = None
        fir_block_size = None
        effective_method = method
        if method == "fir_magnitude_matched":
            if filter_mode != "sos":
                warnings.warn(
                    "method='fir_magnitude_matched' requires filter_mode='sos'; "
                    "falling back to IIR.",
                    stacklevel=2,
                )
                effective_method = "iir"
            elif direction != "forward-backward":
                warnings.warn(
                    "method='fir_magnitude_matched' requires "
                    "direction='forward-backward'; falling back to IIR.",
                    stacklevel=2,
                )
                effective_method = "iir"
            else:
                from ._fir_filter import design_matched_fir_from_sos
                try:
                    fir_kernel = design_matched_fir_from_sos(
                        np.asarray(filter_coeff), fs, stopband_db=float(stopband_db)
                    )
                except (ValueError, RuntimeError) as exc:
                    warnings.warn(
                        f"FIR design failed ({exc}); falling back to IIR.",
                        stacklevel=2,
                    )
                    effective_method = "iir"

        global_job_kwargs_chunk_size = ensure_chunk_size(recording, **get_global_job_kwargs())
        if is_set_global_job_kwargs_set() and margin > MARGIN_TO_CHUNK_PERCENT_WARNING * global_job_kwargs_chunk_size:
            warnings.warn(
                f"The margin size ({margin} samples) is more than {int(MARGIN_TO_CHUNK_PERCENT_WARNING * 100)}% "
                f"of the global chunk size {global_job_kwargs_chunk_size} samples. This may lead to performance bottlenecks when "
                f"chunking. Consider increasing the chunk_size or chunk_duration to minimize margin overhead."
            )
        self.margin_samples = margin
        for parent_segment in recording.segments:
            self.add_recording_segment(
                FilterRecordingSegment(
                    parent_segment,
                    filter_coeff,
                    filter_mode,
                    margin,
                    dtype,
                    add_reflect_padding=add_reflect_padding,
                    direction=direction,
                    n_workers=int(n_workers),
                    method=effective_method,
                    fir_kernel=fir_kernel,
                    fir_block_size=fir_block_size,
                )
            )

        self._kwargs = dict(
            recording=recording,
            band=band,
            btype=btype,
            filter_order=filter_order,
            ftype=ftype,
            filter_mode=filter_mode,
            coeff=coeff,
            margin_ms=margin_ms,
            add_reflect_padding=add_reflect_padding,
            dtype=dtype.str,
            direction=direction,
            n_workers=int(n_workers),
            method=method,
            stopband_db=float(stopband_db),
        )


class FilterRecordingSegment(BasePreprocessorSegment):
    def __init__(
        self,
        parent_recording_segment,
        coeff,
        filter_mode,
        margin,
        dtype,
        add_reflect_padding=False,
        direction="forward-backward",
        n_workers=1,
        method="iir",
        fir_kernel=None,
        fir_block_size=None,
    ):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.coeff = coeff
        self.filter_mode = filter_mode
        self.direction = direction
        self.margin = margin
        self.add_reflect_padding = add_reflect_padding
        self.dtype = dtype
        self.n_workers = int(n_workers)
        self.method = method
        # Cached FIR convolvers, keyed by chunk-T (reused across calls).
        self._fir_kernel = fir_kernel
        self._fir_block_size = fir_block_size
        self._fir_convolvers: dict = {}
        self._fir_convolvers_lock = threading.Lock()
        # Per-caller-thread lazy pool map.  Each outer thread that calls
        # get_traces() on this segment gets its own inner pool, avoiding the
        # shared-pool queueing pathology that would occur if multiple outer
        # workers (e.g., a TimeSeriesChunkExecutor with n_jobs > 1) all
        # dispatched into a single shared pool on the segment.
        #
        # WeakKeyDictionary + weakref.finalize: entries are keyed by the Thread
        # object itself (not by thread-id integer, which can be reused after a
        # thread dies).  When the calling thread is garbage-collected, its
        # inner pool is shut down (non-blocking) and the dict entry drops, so
        # long-running processes don't accumulate zombie pools.
        self._filter_pools = weakref.WeakKeyDictionary()
        self._filter_pools_lock = threading.Lock()
        self._filter_pools_pid = os.getpid()

    def _get_pool(self):
        """Lazy per-caller-thread thread pool for channel-parallel filtering."""
        if self.n_workers <= 1:
            return None
        # os.fork() copies memory but only the calling thread.  In a forked
        # child, ThreadPoolExecutors stored on this segment reference parent
        # Thread objects whose OS threads don't exist here, and the pool lock
        # may even be in a held state.  Detect by pid and reset.  Pickling
        # (mp_context="spawn"/"forkserver") goes through __reduce__ and
        # rebuilds via __init__, so it sees fresh state already; this guard
        # is specifically for the fork path.
        if self._filter_pools_pid != os.getpid():
            self._filter_pools = weakref.WeakKeyDictionary()
            self._filter_pools_lock = threading.Lock()
            self._filter_pools_pid = os.getpid()
        thread = threading.current_thread()
        pool = self._filter_pools.get(thread)
        if pool is None:
            with self._filter_pools_lock:
                pool = self._filter_pools.get(thread)
                if pool is None:
                    from concurrent.futures import ThreadPoolExecutor

                    pool = ThreadPoolExecutor(max_workers=self.n_workers)
                    self._filter_pools[thread] = pool
                    # When the calling thread is GC'd, shut down its pool
                    # without blocking the finalizer thread.  In-flight
                    # tasks would be cancelled, but the owning thread
                    # submits + joins synchronously, so no such tasks
                    # exist when the thread actually exits.
                    weakref.finalize(thread, pool.shutdown, wait=False)
        return pool

    def _apply_sos(self, fn, traces, axis=0):
        """Apply a scipy SOS function across channel blocks in parallel.

        Each channel is independent of every other channel, so splitting the
        channel axis across threads is a safe parallelization.  scipy's C
        implementations of ``sosfiltfilt``/``sosfilt`` release the GIL during
        per-column work, so Python-thread parallelism delivers real speedup
        (measured ~3× on 8 threads for a 1M × 384 float32 chunk).

        Workers write directly into a pre-allocated output array — eliminating
        the per-block tuple return + post-loop allocate-and-copy that adds
        ~15 ms of wall time per call on a (30k, 384) float32 chunk.  Each
        block writes into a non-overlapping channel slice, so concurrent
        writes are safe.
        """
        if self.n_workers == 1:
            return fn(self.coeff, traces, axis=axis)
        C = traces.shape[1]
        if C < 2 * self.n_workers:
            return fn(self.coeff, traces, axis=axis)
        pool = self._get_pool()
        block = (C + self.n_workers - 1) // self.n_workers
        bounds = [(c0, min(c0 + block, C)) for c0 in range(0, C, block)]

        # Probe the output dtype on a tiny slice (longer than scipy's internal
        # padlen of 6 * len(sos)) so we can pre-allocate.  Cost: microseconds.
        probe_len = max(64, 6 * self.coeff.shape[0] + 1)
        out_dtype = fn(self.coeff, traces[:probe_len, :1], axis=axis).dtype
        out = np.empty((traces.shape[0], C), dtype=out_dtype)

        def _work(c0, c1):
            out[:, c0:c1] = fn(self.coeff, traces[:, c0:c1], axis=axis)

        futures = [pool.submit(_work, c0, c1) for c0, c1 in bounds]
        for fut in futures:
            fut.result()
        return out

    def _get_fir_convolver(self, T):
        """Lazy-build (and cache) a CachedOSConvolver for chunk length T."""
        conv = self._fir_convolvers.get(T)
        if conv is not None:
            return conv
        with self._fir_convolvers_lock:
            conv = self._fir_convolvers.get(T)
            if conv is None:
                from ._fir_filter import CachedOSConvolver

                conv = CachedOSConvolver(
                    self._fir_kernel, T, block_size=self._fir_block_size
                )
                self._fir_convolvers[T] = conv
            return conv

    def _apply_fir(self, traces, axis=0):
        """Apply the cached overlap-save FIR to a (T, C) chunk.

        Channel-block parallelism follows the same per-caller-thread pool
        pattern as ``_apply_sos``, but channels are sized for L2 fit (the
        FIR's per-block working set is ~2 * block_size * chunk_C * 4 bytes,
        vs IIR which is purely streaming and L2-insensitive).
        """
        from ._fir_filter import pick_chunk_C

        assert axis == 0, "FIR path only supports axis=0"
        T = traces.shape[0]
        C = traces.shape[1] if traces.ndim == 2 else 1
        conv = self._get_fir_convolver(T)

        if self.n_workers <= 1 or C < 2 * self.n_workers or traces.ndim == 1:
            out = np.empty(traces.shape, dtype=traces.dtype)
            conv(traces, out=out)
            return out

        chunk_C = pick_chunk_C(C, conv.block_size, self.n_workers)
        bounds = [(c0, min(c0 + chunk_C, C)) for c0 in range(0, C, chunk_C)]
        out = np.empty(traces.shape, dtype=traces.dtype)
        pool = self._get_pool()

        def _work(c0, c1):
            conv(traces[:, c0:c1], out=out[:, c0:c1])

        futures = [pool.submit(_work, c0, c1) for c0, c1 in bounds]
        for fut in futures:
            fut.result()
        return out

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces_chunk, left_margin, right_margin = get_chunk_with_margin(
            self.parent_recording_segment,
            start_frame,
            end_frame,
            channel_indices,
            self.margin,
            add_reflect_padding=self.add_reflect_padding,
        )

        traces_dtype = traces_chunk.dtype
        # if uint --> force int
        if traces_dtype.kind == "u":
            traces_chunk = traces_chunk.astype("float32")

        import scipy.signal

        if self.method == "fir_magnitude_matched":
            # Linear-phase FIR matched to IIR's filtfilt magnitude.  No
            # forward/backward distinction (single-pass, zero-phase via
            # centered slice on the linear-phase response).  FIR design
            # was promoted to fp32 in __init__; promote int input here too.
            if not np.issubdtype(traces_chunk.dtype, np.floating):
                traces_chunk = traces_chunk.astype("float32")
            filtered_traces = self._apply_fir(traces_chunk, axis=0)
        elif self.direction == "forward-backward":
            if self.filter_mode == "sos":
                filtered_traces = self._apply_sos(scipy.signal.sosfiltfilt, traces_chunk, axis=0)
            elif self.filter_mode == "ba":
                b, a = self.coeff
                filtered_traces = scipy.signal.filtfilt(b, a, traces_chunk, axis=0)
        else:
            if self.direction == "backward":
                traces_chunk = np.flip(traces_chunk, axis=0)

            if self.filter_mode == "sos":
                filtered_traces = self._apply_sos(scipy.signal.sosfilt, traces_chunk, axis=0)
            elif self.filter_mode == "ba":
                b, a = self.coeff
                filtered_traces = scipy.signal.lfilter(b, a, traces_chunk, axis=0)

            if self.direction == "backward":
                filtered_traces = np.flip(filtered_traces, axis=0)

        if right_margin > 0:
            filtered_traces = filtered_traces[left_margin:-right_margin, :]
        else:
            filtered_traces = filtered_traces[left_margin:, :]

        if np.issubdtype(self.dtype, np.integer):
            filtered_traces = filtered_traces.round()

        return filtered_traces.astype(self.dtype)


class BandpassFilterRecording(FilterRecording):
    """
    Bandpass filter of a recording

    Parameters
    ----------
    recording : Recording
        The recording extractor to be re-referenced
    freq_min : float
        The highpass cutoff frequency in Hz
    freq_max : float
        The lowpass cutoff frequency in Hz
    margin_ms : float | str, default: "auto"
        Margin in ms on border to avoid border effect.
        If "auto", margin is computed as 3 times the filter highpass cutoff period.
    dtype : dtype or None
        The dtype of the returned traces. If None, the dtype of the parent recording is used
    ignore_low_freq_error : bool, default: False
        If True, does not raise an error if freq_min is too low for the sampling frequency.
    {}

    Returns
    -------
    filter_recording : BandpassFilterRecording
        The bandpass-filtered recording extractor object
    """

    def __init__(
        self,
        recording,
        freq_min=300.0,
        freq_max=6000.0,
        margin_ms="auto",
        dtype=None,
        ignore_low_freq_error=False,
        _skip_margin_warning_for_old_version=False,
        **filter_kwargs,
    ):
        if margin_ms == "auto":
            margin_ms = adjust_margin_ms_for_highpass(freq_min)
        highpass_check(
            freq_min,
            margin_ms,
            ignore_low_freq_error=ignore_low_freq_error,
            skip_warning=_skip_margin_warning_for_old_version,
        )
        FilterRecording.__init__(
            self, recording, band=[freq_min, freq_max], margin_ms=margin_ms, dtype=dtype, **filter_kwargs
        )
        dtype = fix_dtype(recording, dtype)
        self._kwargs = dict(
            recording=recording,
            freq_min=freq_min,
            freq_max=freq_max,
            margin_ms=margin_ms,
            dtype=dtype.str,
            ignore_low_freq_error=ignore_low_freq_error,
        )
        self._kwargs.update(filter_kwargs)

    @classmethod
    def _handle_backward_compatibility(cls, old_kwargs, full_dict):
        new_kwargs = old_kwargs.copy()
        is_lfp_case = old_kwargs["freq_min"] < HIGHPASS_ERROR_THRESHOLD_HZ
        if "ignore_low_freq_error" not in new_kwargs:
            new_kwargs["ignore_low_freq_error"] = True
            if is_lfp_case:
                new_kwargs["_skip_margin_warning_for_old_version"] = False
            else:
                new_kwargs["_skip_margin_warning_for_old_version"] = True
        return new_kwargs


class HighpassFilterRecording(FilterRecording):
    """
    Highpass filter of a recording

    Parameters
    ----------
    recording : Recording
        The recording extractor to be re-referenced
    freq_min : float
        The highpass cutoff frequency in Hz
    margin_ms : float | str, default: "auto"
        Margin in ms on border to avoid border effect.
        If "auto", margin is computed as 3 times the filter highpass cutoff period.
    dtype : dtype or None
        The dtype of the returned traces. If None, the dtype of the parent recording is used
    ignore_low_freq_error : bool, default: False
        If True, does not raise an error if freq_min is too low for the sampling frequency.
    {}

    Returns
    -------
    filter_recording : HighpassFilterRecording
        The highpass-filtered recording extractor object
    """

    def __init__(
        self,
        recording,
        freq_min=300.0,
        margin_ms="auto",
        dtype=None,
        ignore_low_freq_error=False,
        _skip_margin_warning_for_old_version=False,
        **filter_kwargs,
    ):
        if margin_ms == "auto":
            margin_ms = adjust_margin_ms_for_highpass(freq_min)
        highpass_check(
            freq_min,
            margin_ms,
            ignore_low_freq_error=ignore_low_freq_error,
            skip_warning=_skip_margin_warning_for_old_version,
        )
        FilterRecording.__init__(
            self, recording, band=freq_min, margin_ms=margin_ms, dtype=dtype, btype="highpass", **filter_kwargs
        )
        dtype = fix_dtype(recording, dtype)
        self._kwargs = dict(
            recording=recording,
            freq_min=freq_min,
            margin_ms=margin_ms,
            dtype=dtype.str,
            ignore_low_freq_error=ignore_low_freq_error,
        )
        self._kwargs.update(filter_kwargs)

    @classmethod
    def _handle_backward_compatibility(cls, old_kwargs, full_dict):
        new_kwargs = old_kwargs.copy()
        is_lfp_case = old_kwargs["freq_min"] < HIGHPASS_ERROR_THRESHOLD_HZ
        if "ignore_low_freq_error" not in new_kwargs:
            new_kwargs["ignore_low_freq_error"] = True
            if is_lfp_case:
                new_kwargs["_skip_margin_warning_for_old_version"] = False
            else:
                new_kwargs["_skip_margin_warning_for_old_version"] = True
        return new_kwargs


class NotchFilterRecording(FilterRecording):
    """
    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor to be notch-filtered
    freq : int or float
        The target frequency in Hz of the notch filter
    q : int
        The quality factor of the notch filter
    dtype : None | dtype, default: None
        dtype of recording. If None, will take from `recording`
    margin_ms : float | str, default: "auto"
        Margin in ms on border to avoid border effect

    Returns
    -------
    filter_recording : NotchFilterRecording
        The notch-filtered recording extractor object
    """

    def __init__(self, recording, freq=3000, q=30, margin_ms="auto", dtype=None, **filter_kwargs):
        import scipy.signal

        if margin_ms == "auto":
            margin_ms = adjust_margin_ms_for_notch(q, freq)

        fn = 0.5 * float(recording.get_sampling_frequency())
        coeff = scipy.signal.iirnotch(freq / fn, q)

        dtype = fix_dtype(recording, dtype)

        # if uint --> unsupported
        if dtype.kind == "u":
            raise TypeError(
                "The notch filter only supports signed types. Use the 'dtype' argument"
                "to specify a signed type (e.g. 'int16', 'float32')"
            )

        FilterRecording.__init__(
            self, recording, coeff=coeff, filter_mode="ba", margin_ms=margin_ms, dtype=dtype, **filter_kwargs
        )
        self.annotate(is_filtered=True)
        self._kwargs = dict(recording=recording, freq=freq, q=q, margin_ms=margin_ms, dtype=dtype.str)
        self._kwargs.update(filter_kwargs)


# functions for API
filter = define_function_handling_dict_from_class(source_class=FilterRecording, name="filter")
bandpass_filter = define_function_handling_dict_from_class(source_class=BandpassFilterRecording, name="bandpass_filter")
notch_filter = define_function_handling_dict_from_class(source_class=NotchFilterRecording, name="notch_filter")
highpass_filter = define_function_handling_dict_from_class(source_class=HighpassFilterRecording, name="highpass_filter")


def causal_filter(
    recording,
    direction="forward",
    band=(300.0, 6000.0),
    btype="bandpass",
    filter_order=5,
    ftype="butter",
    filter_mode="sos",
    margin_ms=5.0,
    add_reflect_padding=False,
    coeff=None,
    dtype=None,
):
    """
    Generic causal filter built on top of the filter function.

    Parameters
    ----------
    recording : Recording
        The recording extractor to be re-referenced
    direction : "forward" | "backward", default: "forward"
        Direction of causal filter. The "backward" option flips the traces in time before applying the filter
        and then flips them back.
    band : float or list, default: [300.0, 6000.0]
        If float, cutoff frequency in Hz for "highpass" filter type
        If list. band (low, high) in Hz for "bandpass" filter type
    btype : "bandpass" | "highpass", default: "bandpass"
        Type of the filter
    margin_ms : float, default: 5.0
        Margin in ms on border to avoid border effect
    coeff : array | None, default: None
        Filter coefficients in the filter_mode form.
    dtype : dtype or None, default: None
        The dtype of the returned traces. If None, the dtype of the parent recording is used
    add_reflect_padding : Bool, default False
        If True, uses a left and right margin during calculation.
    filter_order : order
        The order of the filter for `scipy.signal.iirfilter`
    filter_mode :  "sos" | "ba", default: "sos"
        Filter form of the filter coefficients for `scipy.signal.iirfilter`:
        - second-order sections ("sos")
        - numerator/denominator : ("ba")
    ftype : str, default: "butter"
        Filter type for `scipy.signal.iirfilter` e.g. "butter", "cheby1".

    Returns
    -------
    filter_recording : FilterRecording
        The causal-filtered recording extractor object
    """
    assert direction in ["forward", "backward"], "Direction must be either 'forward' or 'backward'"
    return filter(
        recording=recording,
        direction=direction,
        band=band,
        btype=btype,
        filter_order=filter_order,
        ftype=ftype,
        filter_mode=filter_mode,
        margin_ms=margin_ms,
        add_reflect_padding=add_reflect_padding,
        coeff=coeff,
        dtype=dtype,
    )


bandpass_filter.__doc__ = bandpass_filter.__doc__.format(_common_filter_docs)
highpass_filter.__doc__ = highpass_filter.__doc__.format(_common_filter_docs)


def adjust_margin_ms_for_highpass(freq_min, multiplier=5):
    margin_ms = multiplier * (1000.0 / freq_min)
    return margin_ms


def adjust_margin_ms_for_notch(q, f0, multiplier=5):
    margin_ms = (multiplier / np.pi) * (q / f0) * 1000.0
    return margin_ms


def highpass_check(freq_min, margin_ms, ignore_low_freq_error=False, skip_warning=False):
    if freq_min < HIGHPASS_ERROR_THRESHOLD_HZ:
        if not ignore_low_freq_error:
            raise ValueError(
                f"The freq_min ({freq_min} Hz) is too low and may cause artifacts during chunk processing. "
                f"You can set 'ignore_low_freq_error=True' to bypass this error, but make sure you understand the implications. "
                f"It is recommended to use large chunks when processing/saving your filtered recording to minimize IO overhead."
                f"Refer to this documentation on LFP filtering and chunking artifacts for more details: "
                f"https://spikeinterface.readthedocs.io/en/latest/forhowto/plot_extract_lfps.html. "
            )
    if margin_ms == "auto":
        margin_ms = adjust_margin_ms_for_highpass(freq_min)
    else:
        auto_margin_ms = adjust_margin_ms_for_highpass(freq_min)
        if margin_ms < auto_margin_ms and not skip_warning:
            warnings.warn(
                f"The provided margin_ms ({margin_ms} ms) is smaller than the recommended margin for the given freq_min ({freq_min} Hz). "
                f"This may lead to artifacts at the edges of chunks during processing. "
                f"Consider increasing the margin_ms to at least {auto_margin_ms} ms."
            )


def fix_dtype(recording, dtype):
    """
    Fix recording dtype for preprocessing, by always returning a numpy.dtype.
    If `dtype` is not provided, the recording dtype is returned.
    If the dtype is unsigned, it raises a ValueError.

    Parameters
    ----------
    recording : BaseRecording
        The recording to fix the dtype for
    dtype : str | numpy.dtype
        A specified dtype to return as numpy.dtype

    Returns
    -------
    fixed_dtype : numpy.dtype
        The fixed numpy.dtype
    """
    if dtype is None:
        dtype = recording.get_dtype()
    dtype = np.dtype(dtype)

    # if uint --> force int
    if dtype.kind == "u":
        raise ValueError(
            "Unsigned types are not supported, since they don't interact well with "
            "various preprocessing steps. You can use "
            "`spikeinterface.preprocessing.unsigned_to_signed` to convert the recording to a signed type."
            "For more information, please see "
            "https://spikeinterface.readthedocs.io/en/stable/how_to/unsigned_to_signed.html"
        )

    return dtype
