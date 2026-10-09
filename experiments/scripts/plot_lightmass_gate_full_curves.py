#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from packet_counts import validate_packet_count, window_packet_count


SPLICE_POINT = 100
LIGHT_SIZES = np.arange(1, SPLICE_POINT + 1, dtype=float)
TRACE_TITLES = {
    "caida_2016": "CAIDA 2016",
    "caida_2018": "CAIDA 2018",
    "caida_2018_new": "CAIDA 2018 new",
}


def format_float_for_path(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute full original-vs-light-mass-gate MRD/WMRD curves from existing "
            "pretrain_continuous predictions."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-full-root", type=Path, default=Path("data_full"))
    parser.add_argument("--packet-size", type=int, default=0)
    parser.add_argument("--mode", default="pretrain_continuous")
    parser.add_argument("--res", nargs="+", default=["64_64", "128_128"])
    parser.add_argument("--traces", nargs="+", default=["caida_2016", "caida_2018", "caida_2018_new"])
    parser.add_argument("--threshold", type=float, default=1.02)
    parser.add_argument("--cap", type=float, default=1.20)
    parser.add_argument(
        "--heavy-count-min",
        type=int,
        default=100,
        help=(
            "Minimum heavy-flow count subtracted from the packet residual. "
            "The default keeps the original sweep behavior (count >= 100); "
            "use 101 for the semantic splice where neural bins cover 1..100."
        ),
    )
    parser.add_argument(
        "--original-source",
        choices=["summary", "recompute"],
        default="summary",
        help="Use existing pipeline_eval minute_metrics.csv for original curves, or recompute raw curves from predictions.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def grouped_tasks(final_dir: Path) -> dict[str, str]:
    with (final_dir / "train_test_name_key.json").open() as f:
        items = list(json.load(f).items())

    mapping: dict[str, str] = {}
    i = 0
    while i < len(items):
        train_names: list[str] = []
        while i < len(items) and items[i][1]:
            train_names.append(Path(items[i][0]).stem)
            i += 1
        while i < len(items) and not items[i][1]:
            if train_names:
                mapping[Path(items[i][0]).stem] = train_names[0]
            i += 1
    return mapping


def read_original_metrics(final_dir: Path) -> dict[str, tuple[float, float]]:
    path = final_dir / "plots" / "pipeline_eval" / "minute_metrics.csv"
    if not path.exists():
        return {}

    metrics: dict[str, tuple[float, float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            metrics[row["dataset_id"]] = (float(row["mrd_avg"]), float(row["wmrd_avg"]))
    return metrics


def load_true(final_dir: Path, dataset_id: str) -> tuple[np.ndarray, np.ndarray]:
    real_1 = np.load(final_dir / "tr_ts" / "1_10_real" / f"{dataset_id}.npy")
    real_2 = np.load(final_dir / "tr_ts" / "10_1e4_real" / f"{dataset_id}.npy")
    real = np.vstack((real_1.reshape(-1, 2), real_2.reshape(-1, 2)))
    return real[:, 0].astype(int), real[:, 1].astype(float)


def load_heavy(final_dir: Path, dataset_id: str, heavy_count_min: int) -> tuple[np.ndarray, np.ndarray, float]:
    el_folder_idx = str(int(dataset_id.split("_")[-1]))
    path = final_dir / "EL" / el_folder_idx / "heavy_0.csv"
    freq: dict[int, int] = {}
    heavy_packet_mass = 0.0
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            count = int(row["count"])
            if count < heavy_count_min:
                continue
            freq[count] = freq.get(count, 0) + 1
            heavy_packet_mass += count

    if not freq:
        return np.empty(0, dtype=float), np.empty(0, dtype=float), 0.0
    values = np.array(sorted(freq), dtype=float)
    counts = np.array([freq[int(v)] for v in values], dtype=float)
    return values, counts, heavy_packet_mass


def linear_interp_extrap(xp: np.ndarray, fp: np.ndarray, x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.empty(0, dtype=float)
    if xp.size == 0:
        return np.zeros(x.size, dtype=float)
    if xp.size == 1:
        return np.full(x.size, fp[0], dtype=float)

    y = np.interp(x, xp, fp)
    lo = x < xp[0]
    if np.any(lo):
        slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
        y[lo] = fp[0] + (x[lo] - xp[0]) * slope
    hi = x > xp[-1]
    if np.any(hi):
        slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        y[hi] = fp[-1] + (x[hi] - xp[-1]) * slope
    return y


def small_predictions(pred_1: np.ndarray, pred_2: np.ndarray, small_sizes: np.ndarray) -> np.ndarray:
    out = np.empty((pred_1.shape[0], small_sizes.size), dtype=float)
    le10 = small_sizes <= 10
    if np.any(le10):
        out[:, le10] = pred_1[:, small_sizes[le10] - 1]
    if np.any(~le10):
        out[:, ~le10] = pred_2[:, small_sizes[~le10] - 11]
    return out


def light_mass_predictions(pred_1: np.ndarray, pred_2: np.ndarray) -> np.ndarray:
    light_preds = np.concatenate((pred_1[:, :10], pred_2[:, :90]), axis=1)
    return np.maximum(light_preds, 0).dot(LIGHT_SIZES)


def metrics(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    pred = np.maximum(pred, 0)
    denom = (pred + true) / 2
    valid = denom > 0
    mrd = float(np.mean(np.abs(pred[valid] - true[valid]) / denom[valid]))
    wmrd = float(np.mean(np.abs(pred - true)) / np.mean(denom))
    return mrd, wmrd


def compute_gate_dataset(
    final_dir: Path,
    dataset_id: str,
    block_name: str,
    threshold: float,
    cap: float,
    heavy_count_min: int,
    *,
    packet_count: int,
) -> dict[str, float]:
    root = final_dir / "0_finetuned_results"
    pred_1 = np.load(
        root / "ViT_1_10_results_1e-2" / f"finetuned_block_{block_name}" / "test_results" / dataset_id / "preds.npy"
    )
    pred_2 = np.load(
        root
        / "ViT_10_1e4_results_1e-2"
        / f"finetuned_block_{block_name}"
        / "test_results"
        / dataset_id
        / "preds.npy"
    )

    flow_sizes, true_counts = load_true(final_dir, dataset_id)
    small_mask = flow_sizes <= SPLICE_POINT
    large_mask = ~small_mask
    small_sizes = flow_sizes[small_mask]

    heavy_values, heavy_freqs, heavy_packet_mass = load_heavy(final_dir, dataset_id, heavy_count_min)
    base_large = np.zeros(flow_sizes.shape, dtype=float)
    base_large[large_mask] = linear_interp_extrap(heavy_values, heavy_freqs, flow_sizes[large_mask].astype(float))

    packet_count = validate_packet_count(packet_count)
    residual_light_mass = max(packet_count - heavy_packet_mass, 0.0)
    light_mass = light_mass_predictions(pred_1, pred_2)
    raw_scale = np.divide(
        residual_light_mass,
        light_mass,
        out=np.ones_like(light_mass, dtype=float),
        where=light_mass > 0,
    )
    applied_scale = np.where(raw_scale > threshold, np.minimum(raw_scale, cap), 1.0)

    small_pred = np.maximum(small_predictions(pred_1, pred_2, small_sizes), 0)
    raw_mrd: list[float] = []
    raw_wmrd: list[float] = []
    gated_mrd: list[float] = []
    gated_wmrd: list[float] = []
    for seed_idx in range(pred_1.shape[0]):
        pred = base_large.copy()
        if small_sizes.size:
            # Match the focused optimization script: round the base prediction,
            # then scale only the neural light bins when the residual gate fires.
            pred[small_mask] = small_pred[seed_idx]
        pred = np.around(pred, 0)
        pred[pred < 0] = 0
        mrd, wmrd = metrics(pred, true_counts)
        raw_mrd.append(mrd)
        raw_wmrd.append(wmrd)

        gated = pred.copy()
        if small_sizes.size and applied_scale[seed_idx] > 1.0:
            gated[small_mask] = np.around(gated[small_mask] * applied_scale[seed_idx], 0)
            gated[gated < 0] = 0
        mrd, wmrd = metrics(gated, true_counts)
        gated_mrd.append(mrd)
        gated_wmrd.append(wmrd)

    return {
        "recomputed_original_mrd": float(np.mean(raw_mrd)),
        "recomputed_original_wmrd": float(np.mean(raw_wmrd)),
        "gate_mrd": float(np.mean(gated_mrd)),
        "gate_wmrd": float(np.mean(gated_wmrd)),
        "raw_scale_mean": float(np.mean(raw_scale)),
        "raw_scale_p95": float(np.percentile(raw_scale, 95)),
        "raw_scale_max": float(np.max(raw_scale)),
        "applied_scale_mean": float(np.mean(applied_scale)),
        "applied_scale_p95": float(np.percentile(applied_scale, 95)),
        "applied_scale_max": float(np.max(applied_scale)),
        "gated_seed_fraction": float(np.mean(applied_scale > 1.0)),
        "residual_light_mass": residual_light_mass,
        "heavy_packet_mass": heavy_packet_mass,
        "n_seeds": float(pred_1.shape[0]),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"no rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_trace(path: Path, res: str, trace: str, rows: list[dict[str, object]]) -> None:
    minutes = np.array([int(r["minute"]) for r in rows], dtype=int)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True)
    for ax, metric, ylabel in [
        (axes[0], "mrd", "MRD error"),
        (axes[1], "wmrd", "WMRD error"),
    ]:
        ax.plot(minutes, [float(r[f"original_{metric}"]) for r in rows], label="Original", linewidth=1.35)
        ax.plot(minutes, [float(r[f"gate_{metric}"]) for r in rows], label="Light residual gate", linewidth=1.35)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left")
    axes[0].set_title(f"{res} {TRACE_TITLES.get(trace, trace)}: original vs light residual gate")
    axes[1].set_xlabel("Time (minute)")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_res_grid(out_dir: Path, res: str, traces: list[str], curves: dict[tuple[str, str], list[dict[str, object]]]) -> None:
    fig, axes = plt.subplots(len(traces), 2, figsize=(14, 3.3 * len(traces)), sharex=False)
    if len(traces) == 1:
        axes = np.array([axes])

    for row_idx, trace in enumerate(traces):
        rows = curves[(res, trace)]
        minutes = np.array([int(r["minute"]) for r in rows], dtype=int)
        for col_idx, metric in enumerate(["mrd", "wmrd"]):
            ax = axes[row_idx, col_idx]
            ax.plot(minutes, [float(r[f"original_{metric}"]) for r in rows], label="Original", linewidth=1.15)
            ax.plot(minutes, [float(r[f"gate_{metric}"]) for r in rows], label="Light residual gate", linewidth=1.15)
            ax.set_title(f"{TRACE_TITLES.get(trace, trace)} {metric.upper()}")
            ax.set_xlabel("Time (minute)")
            ax.set_ylabel("Error")
            ax.grid(True, alpha=0.25)
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper left")

    fig.suptitle(f"{res}: full original vs light residual gate curves", y=0.995)
    fig.tight_layout()
    path = out_dir / f"{res}_three_trace_original_vs_gate"
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def summarize(res: str, trace: str, rows: list[dict[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {"res": res, "trace": trace, "n": len(rows)}
    for metric in ["mrd", "wmrd"]:
        original = np.array([float(r[f"original_{metric}"]) for r in rows], dtype=float)
        gated = np.array([float(r[f"gate_{metric}"]) for r in rows], dtype=float)
        out[f"original_{metric}"] = float(np.mean(original))
        out[f"gate_{metric}"] = float(np.mean(gated))
        out[f"delta_{metric}"] = float(np.mean(gated) - np.mean(original))
        out[f"max_original_{metric}"] = float(np.max(original))
        out[f"max_gate_{metric}"] = float(np.max(gated))
    out["applied_scale_mean"] = float(np.mean([float(r["applied_scale_mean"]) for r in rows]))
    out["gated_seed_fraction"] = float(np.mean([float(r["gated_seed_fraction"]) for r in rows]))
    return out


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    suffix = f"t{format_float_for_path(args.threshold)}_c{format_float_for_path(args.cap)}"
    out_dir = args.out_dir or (run_root / "plots" / f"gate_full_curves_{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    curves: dict[tuple[str, str], list[dict[str, object]]] = {}
    summary_rows: list[dict[str, object]] = []
    for res in args.res:
        for trace in args.traces:
            final_dir = run_root / "online" / args.mode / f"{res}_{trace}_final"
            original = read_original_metrics(final_dir)
            rows: list[dict[str, object]] = []
            tasks = grouped_tasks(final_dir)
            print(f"compute {res} {trace}: {len(tasks)} test windows", flush=True)
            for idx, (dataset_id, block_name) in enumerate(tasks.items(), start=1):
                if idx == 1 or idx % 25 == 0 or idx == len(tasks):
                    print(f"  {res} {trace} {idx}/{len(tasks)} {dataset_id}", flush=True)
                gate = compute_gate_dataset(
                    final_dir,
                    dataset_id,
                    block_name,
                    args.threshold,
                    args.cap,
                    args.heavy_count_min,
                    packet_count=window_packet_count(
                        args.data_full_root / trace / "caida_1min_split",
                        int(dataset_id.split("_")[-1]), trace, args.packet_size,
                    ),
                )
                if args.original_source == "summary" and dataset_id in original:
                    original_mrd, original_wmrd = original[dataset_id]
                else:
                    original_mrd = gate["recomputed_original_mrd"]
                    original_wmrd = gate["recomputed_original_wmrd"]
                row: dict[str, object] = {
                    "minute": int(dataset_id.split("_")[-1]),
                    "dataset_id": dataset_id,
                    "train_block": block_name,
                    "original_mrd": original_mrd,
                    "original_wmrd": original_wmrd,
                    **gate,
                }
                rows.append(row)

            rows.sort(key=lambda r: int(r["minute"]))
            curves[(res, trace)] = rows
            write_csv(out_dir / f"{res}_{trace}_gate_full_curves.csv", rows)
            plot_trace(out_dir / f"{res}_{trace}_original_vs_gate_mrd_wmrd", res, trace, rows)
            summary_rows.append(summarize(res, trace, rows))

    for res in args.res:
        plot_res_grid(out_dir, res, args.traces, curves)

    write_csv(out_dir / "gate_full_summary.csv", summary_rows)
    print(f"wrote full gate curves to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
