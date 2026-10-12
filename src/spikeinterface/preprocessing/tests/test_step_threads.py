import numpy as np
import pytest

from spikeinterface import NumpyRecording
from spikeinterface.core import thread_budget
from spikeinterface.preprocessing import bandpass_filter, common_reference, phase_shift

FS = 30_000.0


@pytest.fixture(scope="module")
def recording():
    rng = np.random.default_rng(0)
    traces = (rng.standard_normal((int(1.3 * FS), 96)) * 50).astype(np.int16)
    rec = NumpyRecording(traces, FS)
    rec.set_property("inter_sample_shift", (np.arange(96) // 2 % 13) / 13.0)
    return rec


STEPS = {
    "phase_shift int16": lambda r: phase_shift(r, method="fir", dtype="float32"),
    "phase_shift float32": lambda r: phase_shift(bandpass_filter(r, method="fir", dtype="float32"), method="fir"),
    "fir bandpass": lambda r: bandpass_filter(r, method="fir", dtype="float32"),
    "fir bandpass int16": lambda r: bandpass_filter(r, method="fir"),
    "median": lambda r: common_reference(r, operator="median"),
    "median float32": lambda r: common_reference(bandpass_filter(r, method="fir", dtype="float32")),
    "average": lambda r: common_reference(r, operator="average", dtype="float32"),
    "median subset reference": lambda r: common_reference(r, ref_channel_ids=list(r.channel_ids[::3])),
}


@pytest.mark.parametrize("step", STEPS)
def test_threaded_step_equals_single_threaded(recording, step):
    rec = STEPS[step](recording)
    channel_ids = recording.channel_ids[5:40]
    for kwargs in (dict(), dict(start_frame=1234, end_frame=31234, channel_ids=channel_ids)):
        expected = rec.get_traces(**kwargs)
        with thread_budget(4):
            traces = rec.get_traces(**kwargs)
        assert traces.dtype == expected.dtype
        np.testing.assert_array_equal(traces, expected)
