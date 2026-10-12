"""Throughput, peak memory and error of phase shift -> bandpass -> median CMR, as one get_traces call.

Throughput is in channel-samples per second (MS/s; one 384-channel probe at 30 kHz is 11.5 MS/s) and as
a percentage of a memcpy of the same int16 chunk on the same number of threads.

Variants:
  - memcpy: copy of the int16 chunk into a preallocated buffer, split by rows across threads
  - stock: phase_shift FFT, IIR bandpass, CMR, every step int16 (the defaults)
  - unfused FIR: phase_shift(method="fir"), bandpass_filter(method="fir"), CMR, every step float32
  - fused: fuse() of the unfused FIR chain

Threads are set with thread_budget, as max_threads_per_worker sets them for each chunk task.

Each configuration runs in its own subprocess so peak RSS is not polluted by earlier runs. Peak RSS is
reported above the loaded data, as a percentage of the int16 chunk. Error is max |x - reference| in
counts, where the reference is the unfused FIR chain computed over a window 1 s wider on each side and
cropped, so it has no chunk-seam error of its own.

    PYTHONPATH=src python benchmarks/preprocessing/bench_fused.py \
        --data /mnt/Data/ephys/raw_chunk_786867_50213-1_10min.dat --durations 1 10
"""

import argparse
import json
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

FS = 30_000.0
START = int(2 * FS)
REFERENCE_PAD = int(1 * FS)


def build(variant, raw):
    import spikeinterface.preprocessing as spre
    from spikeinterface import NumpyRecording
    from spikeinterface.preprocessing.fused_fir_reference import fuse

    rec = NumpyRecording(raw, FS)
    # Thirteen ADC delays in a repeating pattern, as on a Neuropixels 1.0 probe.
    rec.set_property("inter_sample_shift", (np.arange(raw.shape[1]) // 2 % 13) / 13.0)
    if variant == "stock":
        rec = spre.phase_shift(rec)
        rec = spre.bandpass_filter(rec)
        return spre.common_reference(rec, operator="median")
    rec = spre.phase_shift(rec, method="fir", dtype="float32")
    rec = spre.bandpass_filter(rec, method="fir", dtype="float32")
    rec = spre.common_reference(rec, operator="median", dtype="float32")
    return fuse(rec) if variant == "fused" else rec


def max_rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Linux reports KiB


def memcpy_worker(args, chunk):
    dst = np.empty_like(chunk)
    bounds = np.linspace(0, chunk.shape[0], args.threads + 1).astype(int)
    pool = ThreadPoolExecutor(args.threads)

    def copy_rows(i):
        np.copyto(dst[bounds[i] : bounds[i + 1]], chunk[bounds[i] : bounds[i + 1]])

    times = []
    for _ in range(args.reps + 2):  # the first runs fault in dst's pages
        t0 = time.perf_counter()
        list(pool.map(copy_rows, range(args.threads)))
        times.append(time.perf_counter() - t0)
    print(json.dumps(dict(seconds=min(times), peak_pct=None, error=None)))


def worker(args):
    from spikeinterface.core import thread_budget

    num_samples = int(args.duration * FS)
    count = (START + num_samples + REFERENCE_PAD) * args.channels
    # fromfile, not memmap: file-backed pages would count toward RSS
    raw = np.fromfile(args.data, dtype=np.int16, count=count).reshape(-1, args.channels)
    start, end = START, START + num_samples
    if args.variant == "memcpy":
        return memcpy_worker(args, raw[start:end])
    rec = build(args.variant, raw)

    def run():
        with thread_budget(args.threads):
            return rec.get_traces(start_frame=start, end_frame=end)

    baseline = max_rss_bytes()
    out = run()  # first run sets the memory high-water mark
    peak_above = max_rss_bytes() - baseline
    times = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        run()
        times.append(time.perf_counter() - t0)

    if args.variant == "stock":
        error = None  # IIR and FIR responses differ by design, so a sample-wise error is meaningless
    else:
        reference = build("unfused", raw).get_traces(start_frame=start - REFERENCE_PAD, end_frame=end + REFERENCE_PAD)
        reference = reference[REFERENCE_PAD : REFERENCE_PAD + num_samples]
        error = float(np.abs(out.astype(np.float64) - reference).max())
    print(
        json.dumps(
            dict(
                seconds=min(times),
                peak_pct=100.0 * peak_above / (num_samples * args.channels * 2),
                error=error,
            )
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="raw int16 binary, samples x channels")
    parser.add_argument("--channels", type=int, default=384)
    parser.add_argument(
        "--durations",
        type=float,
        nargs="+",
        default=[1.0, 10.0],
        help="chunk length, s",
    )
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--duration", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--variant", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        args.threads = args.threads[0]
        return worker(args)

    configs = [("memcpy", t) for t in sorted(set(args.threads) | {1})]
    configs += [("stock", 1)] + [(v, t) for v in ("unfused", "fused") for t in args.threads]
    labels = dict(
        memcpy="memcpy of the int16 chunk",
        stock="stock (int16, IIR, FFT shift)",
        unfused="unfused FIR (float32)",
        fused="fused",
    )
    print(
        "| chunk | variant | threads | time | MS/s | % of memcpy | vs stock | peak RAM above chunk | max error, counts |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for duration in args.durations:
        channel_samples = int(duration * FS) * args.channels
        memcpy_seconds = {}
        stock_seconds = None
        for variant, threads in configs:
            cmd = [sys.executable, __file__, "--worker", "--data", args.data, "--channels", str(args.channels),
                   "--duration", str(duration), "--variant", variant, "--threads", str(threads),
                   "--reps", str(args.reps)]  # fmt: skip
            run = subprocess.run(cmd, capture_output=True, text=True, check=True)
            result = json.loads(run.stdout.strip().splitlines()[-1])
            seconds = result["seconds"]
            if variant == "memcpy":
                memcpy_seconds[threads] = seconds
            if variant == "stock":
                stock_seconds = seconds
            vs_stock = f"{stock_seconds / seconds:.1f}x" if stock_seconds else "n/a"
            peak = "n/a" if result["peak_pct"] is None else f"{result['peak_pct']:.0f}%"
            error = "n/a" if result["error"] is None else f"{result['error']:.2g}"
            print(
                f"| {duration:g} s | {labels[variant]} | {threads} | {seconds * 1e3:.1f} ms | "
                f"{channel_samples / seconds / 1e6:,.0f} | {100 * memcpy_seconds[threads] / seconds:.1f}% | "
                f"{vs_stock} | {peak} | {error} |",
                flush=True,
            )


if __name__ == "__main__":
    main()
