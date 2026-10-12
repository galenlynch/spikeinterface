import numpy as np
import pytest

from spikeinterface import NumpyRecording
from spikeinterface.core import thread_budget
from spikeinterface.preprocessing import (
    bandpass_filter,
    common_reference,
    design_fir_highpass,
    fir_filter,
    highpass_filter,
    phase_shift,
)
from spikeinterface.preprocessing.fused_fir_reference import FusedFIRReferenceRecording, fuse

FS = 30_000.0


@pytest.fixture(scope="module")
def recording():
    rng = np.random.default_rng(0)
    traces = (rng.standard_normal((int(2 * FS), 32)) * 50).astype(np.int16)
    rec = NumpyRecording(traces, FS)
    rec.set_property("inter_sample_shift", (np.arange(32) // 2 % 13) / 13.0)
    return rec


def unfused_chain(rec, with_phase_shift, operator, filter_step="bandpass"):
    if with_phase_shift:
        rec = phase_shift(rec, method="fir", output_dtype="float32")
    if filter_step == "bandpass":
        rec = bandpass_filter(rec, method="fir", dtype="float32")
    elif filter_step == "highpass":
        rec = highpass_filter(rec, method="fir", dtype="float32")
    else:
        rec = fir_filter(rec, design_fir_highpass(300.0, FS), dtype="float32")
    if operator is not None:
        rec = common_reference(rec, operator=operator, dtype="float32")
    return rec


def assert_matches_away_from_edges(traces, expected):
    # The unfused chain zero-pads before each step at the recording's edges; the fused one only before the first.
    edge = 1000
    np.testing.assert_allclose(traces[edge:-edge], expected[edge:-edge], atol=1e-5 * np.abs(expected).max())


@pytest.mark.parametrize("with_phase_shift", [False, True])
@pytest.mark.parametrize("operator", [None, "median", "average"])
def test_fused_matches_unfused_chain(recording, with_phase_shift, operator):
    chain = unfused_chain(recording, with_phase_shift, operator)
    fused = fuse(chain)
    assert isinstance(fused, FusedFIRReferenceRecording)
    assert_matches_away_from_edges(fused.get_traces(), chain.get_traces())


@pytest.mark.parametrize("filter_step", ["highpass", "fir_filter"])
def test_fused_matches_other_fir_steps(recording, filter_step):
    chain = unfused_chain(recording, True, "median", filter_step)
    fused = fuse(chain)
    assert isinstance(fused, FusedFIRReferenceRecording)
    assert_matches_away_from_edges(fused.get_traces(), chain.get_traces())


def test_fused_matches_per_group_kernels(recording):
    highpass_300, highpass_600 = design_fir_highpass(300.0, FS), design_fir_highpass(600.0, FS)
    num_taps = max(highpass_300.size, highpass_600.size)
    taps = np.stack([np.pad(k, (num_taps - k.size) // 2) for k in (highpass_300, highpass_600)])
    groups = np.arange(recording.get_num_channels()) % 2
    rec = phase_shift(recording, method="fir", output_dtype="float32")
    chain = common_reference(fir_filter(rec, taps, channel_groups=groups, dtype="float32"), dtype="float32")
    assert_matches_away_from_edges(fuse(chain).get_traces(), chain.get_traces())


def test_fused_chunks_and_threads_agree(recording):
    fused = fuse(unfused_chain(recording, True, "median"))
    whole = fused.get_traces()
    chunked = np.concatenate(
        [fused.get_traces(start_frame=s, end_frame=s + 7000) for s in range(0, whole.shape[0], 7000)]
    )
    np.testing.assert_allclose(chunked, whole, atol=1e-5 * np.abs(whole).max())
    with thread_budget(4):
        np.testing.assert_array_equal(fused.get_traces(), whole)
    subset = fused.get_traces(start_frame=1000, end_frame=3000, channel_ids=recording.channel_ids[4:9])
    np.testing.assert_allclose(subset, whole[1000:3000, 4:9], atol=1e-5 * np.abs(whole).max())


def test_fuse_leaves_unsupported_chains_unchanged(recording):
    iir_chain = common_reference(bandpass_filter(recording))
    assert fuse(iir_chain) is iir_chain
    single_reference = common_reference(
        bandpass_filter(recording, method="fir"), reference="single", ref_channel_ids=[recording.channel_ids[0]]
    )
    assert fuse(single_reference) is single_reference
    reflect = bandpass_filter(recording, method="fir", add_reflect_padding=True)
    assert fuse(reflect) is reflect


def test_fused_int16_output_rounds_once(recording):
    chain = common_reference(bandpass_filter(phase_shift(recording, method="fir"), method="fir"))
    fused = fuse(chain)
    assert fused.get_dtype() == np.int16
    float_chain = fuse(unfused_chain(recording, True, "median"))
    np.testing.assert_array_equal(fused.get_traces(), np.rint(float_chain.get_traces()).astype(np.int16))
