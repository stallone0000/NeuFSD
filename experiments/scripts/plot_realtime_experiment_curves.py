#!/usr/bin/env python3
"""Plot realtime window_metrics.csv MRD/WMRD experiment curves."""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METRICS = ("mrd", "wmrd")
METHODS = (
    ("raw", "Raw / no gate"),
    ("gate", "Gate"),
    ("adaptive", "Adaptive gate"),
)
METHOD_COLORS = {
    "raw": "#334155",
    "gate": "#2563EB",
    "adaptive": "#D97706",
}
METHOD_STYLES = {
    "raw": "-",
    "gate": "--",
    "adaptive": "-.",
}
RUN_STYLES = {
    "fixed": "-",
    "continuous": "--",
}


@dataclass
class RunData:
    directory: Path
    label: str
    slug: str
    df: pd.DataFrame
    time_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot MRD/WMRD curves from realtime run directories containing "
            "window_metrics.csv."
        )
    )
    parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="*",
        default=[],
        help="Realtime run directories to plot individually; each should contain window_metrics.csv.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels for --run-dirs, in the same order.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: common input parent / realtime_experiment_curves.",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=None,
        help="Optional fixed/baseline realtime run directory for fixed vs continuous plots.",
    )
    parser.add_argument(
        "--continuous-dir",
        type=Path,
        default=None,
        help="Optional continuous realtime run directory for fixed vs continuous plots.",
    )
    parser.add_argument(
        "--baseline-label",
        default="fixed",
        help="Legend label for --baseline-dir in fixed vs continuous plots.",
    )
    parser.add_argument(
        "--continuous-label",
        default="continuous",
        help="Legend label for --continuous-dir in fixed vs continuous plots.",
    )
    parser.add_argument("--title-prefix", default="", help="Optional prefix prepended to figure titles.")
    parser.add_argument("--dpi", type=int, default=220, help="PNG output DPI.")
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.run_dirs):
        parser.error("--labels must have the same number of entries as --run-dirs")
    if bool(args.baseline_dir) != bool(args.continuous_dir):
        parser.error("pass both --baseline-dir and --continuous-dir, or neither")
    if not args.run_dirs and not (args.baseline_dir and args.continuous_dir):
        parser.error("provide --run-dirs and/or both --baseline-dir and --continuous-dir")
    return args


def metric_column(method: str, metric: str) -> str:
    if method == "raw":
        return metric
    return f"{method}_{metric}"


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    return slug or "run"


def unique_slug(base: str, used: set[str]) -> str:
    slug = base
    index = 2
    while slug in used:
        slug = f"{base}_{index}"
        index += 1
    used.add(slug)
    return slug


def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def normalize_dir(path: Path) -> Path:
    return path.expanduser().resolve()


def load_run(run_dir: Path, label: str, used_slugs: set[str]) -> RunData | None:
    run_dir = normalize_dir(run_dir)
    csv_path = run_dir / "window_metrics.csv"
    if not csv_path.exists():
        warn(f"missing window_metrics.csv under {run_dir}; skipping")
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        warn(f"empty metrics CSV: {csv_path}; skipping")
        return None

    df = df.copy()
    if "minute" in df.columns:
        df["_time"] = pd.to_numeric(df["minute"], errors="coerce")
        before = len(df)
        df = df.dropna(subset=["_time"]).sort_values("_time").reset_index(drop=True)
        if df.empty:
            warn(f"minute column has no numeric values in {csv_path}; using row index")
            df = pd.read_csv(csv_path)
            df["_time"] = range(len(df))
            time_label = "Window index"
        else:
            dropped = before - len(df)
            if dropped:
                warn(f"dropped {dropped} rows with non-numeric minute values in {csv_path}")
            time_label = "Time (minute)"
    else:
        warn(f"missing minute column in {csv_path}; using row index")
        df["_time"] = range(len(df))
        time_label = "Window index"

    for metric in METRICS:
        for method, _ in METHODS:
            col = metric_column(method, metric)
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    slug = unique_slug(slugify(label or run_dir.name), used_slugs)
    return RunData(directory=run_dir, label=label or run_dir.name, slug=slug, df=df, time_label=time_label)


def resolve_out_dir(args: argparse.Namespace, input_dirs: list[Path]) -> Path:
    if args.out_dir is not None:
        return args.out_dir.expanduser().resolve()
    parents = [normalize_dir(path).parent for path in input_dirs]
    if not parents:
        return (Path.cwd() / "realtime_experiment_curves").resolve()
    common = Path(os.path.commonpath([str(path) for path in parents]))
    return common / "realtime_experiment_curves"


def title_text(args: argparse.Namespace, text: str) -> str:
    if args.title_prefix:
        return f"{args.title_prefix}: {text}"
    return text


def save_figure(fig: plt.Figure, stem: Path, dpi: int) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    fig.tight_layout()
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]


def available_methods(run: RunData, metric: str) -> list[tuple[str, str]]:
    available = []
    for method, label in METHODS:
        col = metric_column(method, metric)
        if col in run.df.columns:
            available.append((method, label))
        else:
            warn(f"{run.label}: missing {col}; skipping that curve")
    return available


def plot_metric_vs_time(run: RunData, metric: str, out_dir: Path, args: argparse.Namespace) -> list[Path]:
    methods = available_methods(run, metric)
    if not methods:
        warn(f"{run.label}: no {metric.upper()} columns available; skipping time plot")
        return []

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    x = run.df["_time"]
    for method, label in methods:
        col = metric_column(method, metric)
        ax.plot(
            x,
            run.df[col],
            label=label,
            color=METHOD_COLORS[method],
            linestyle=METHOD_STYLES[method],
            linewidth=1.8,
        )

    ax.set_title(title_text(args, f"{run.label}: {metric.upper()} over time"))
    ax.set_xlabel(run.time_label)
    ax.set_ylabel(metric.upper())
    ax.grid(True, color="#E5E7EB", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")
    ax.margins(x=0.01)
    stem = out_dir / f"{run.slug}_raw_gate_adaptive_{metric}_vs_time"
    return save_figure(fig, stem, args.dpi)


def plot_delta_vs_nogate(run: RunData, metric: str, out_dir: Path, args: argparse.Namespace) -> list[Path]:
    raw_col = metric_column("raw", metric)
    if raw_col not in run.df.columns:
        warn(f"{run.label}: missing {raw_col}; skipping {metric.upper()} vs no-gate comparison")
        return []

    compare_methods = []
    for method, label in METHODS[1:]:
        col = metric_column(method, metric)
        if col in run.df.columns:
            compare_methods.append((method, label))
        else:
            warn(f"{run.label}: missing {col}; skipping that no-gate comparison")
    if not compare_methods:
        warn(f"{run.label}: no gate/adaptive {metric.upper()} columns available; skipping no-gate comparison")
        return []

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    x = run.df["_time"]
    for method, label in compare_methods:
        col = metric_column(method, metric)
        ax.plot(
            x,
            run.df[col] - run.df[raw_col],
            label=f"{label} - no gate",
            color=METHOD_COLORS[method],
            linestyle=METHOD_STYLES[method],
            linewidth=1.8,
        )

    ax.axhline(0.0, color="#111827", linewidth=1.0, alpha=0.75)
    ax.set_title(title_text(args, f"{run.label}: {metric.upper()} gate/adaptive vs no gate"))
    ax.set_xlabel(run.time_label)
    ax.set_ylabel(f"Delta {metric.upper()} (method - no gate)")
    ax.grid(True, color="#E5E7EB", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")
    ax.margins(x=0.01)
    stem = out_dir / f"{run.slug}_gate_adaptive_vs_nogate_{metric}_delta"
    return save_figure(fig, stem, args.dpi)


def plot_fixed_vs_continuous(
    fixed: RunData,
    continuous: RunData,
    metric: str,
    out_dir: Path,
    args: argparse.Namespace,
) -> list[Path]:
    plotted = 0
    fig, ax = plt.subplots(figsize=(11.5, 5.8))

    for run_key, run in [("fixed", fixed), ("continuous", continuous)]:
        for method, method_label in METHODS:
            col = metric_column(method, metric)
            if col not in run.df.columns:
                warn(f"{run.label}: missing {col}; skipping in fixed vs continuous plot")
                continue
            ax.plot(
                run.df["_time"],
                run.df[col],
                label=f"{run.label} - {method_label}",
                color=METHOD_COLORS[method],
                linestyle=RUN_STYLES[run_key],
                linewidth=1.8,
            )
            plotted += 1

    if plotted == 0:
        warn(f"no {metric.upper()} columns available for fixed vs continuous plot; skipping")
        plt.close(fig)
        return []

    ax.set_title(
        title_text(
            args,
            f"{fixed.label} vs {continuous.label}: {metric.upper()} over time",
        )
    )
    ax.set_xlabel("Time (minute)" if fixed.time_label == continuous.time_label == "Time (minute)" else "Time")
    ax.set_ylabel(metric.upper())
    ax.grid(True, color="#E5E7EB", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8.7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    ax.margins(x=0.01)
    stem = out_dir / f"{fixed.slug}_vs_{continuous.slug}_{metric}_vs_time"
    return save_figure(fig, stem, args.dpi)


def build_individual_specs(args: argparse.Namespace) -> list[tuple[Path, str]]:
    labels = args.labels or [path.name for path in args.run_dirs]
    specs = list(zip(args.run_dirs, labels))

    existing = {normalize_dir(path) for path, _ in specs}
    if args.baseline_dir and args.continuous_dir:
        for path, label in [
            (args.baseline_dir, args.baseline_label),
            (args.continuous_dir, args.continuous_label),
        ]:
            normalized = normalize_dir(path)
            if normalized not in existing:
                specs.append((path, label))
                existing.add(normalized)
    return specs


def main() -> None:
    args = parse_args()
    input_dirs = list(args.run_dirs)
    if args.baseline_dir and args.continuous_dir:
        input_dirs.extend([args.baseline_dir, args.continuous_dir])
    out_dir = resolve_out_dir(args, input_dirs)

    used_slugs: set[str] = set()
    outputs: list[Path] = []
    individual_runs: list[RunData] = []
    loaded_by_key: dict[tuple[Path, str], RunData] = {}
    for run_dir, label in build_individual_specs(args):
        run = load_run(run_dir, label, used_slugs)
        if run is not None:
            individual_runs.append(run)
            loaded_by_key[(run.directory, run.label)] = run

    for run in individual_runs:
        for metric in METRICS:
            outputs.extend(plot_metric_vs_time(run, metric, out_dir, args))
            outputs.extend(plot_delta_vs_nogate(run, metric, out_dir, args))

    if args.baseline_dir and args.continuous_dir:
        fixed_dir = normalize_dir(args.baseline_dir)
        continuous_dir = normalize_dir(args.continuous_dir)
        fixed = loaded_by_key.get((fixed_dir, args.baseline_label))
        if fixed is None:
            fixed = load_run(args.baseline_dir, args.baseline_label, used_slugs)
            if fixed is not None:
                loaded_by_key[(fixed.directory, fixed.label)] = fixed
        continuous = loaded_by_key.get((continuous_dir, args.continuous_label))
        if continuous is None:
            continuous = load_run(args.continuous_dir, args.continuous_label, used_slugs)
            if continuous is not None:
                loaded_by_key[(continuous.directory, continuous.label)] = continuous
        if fixed is not None and continuous is not None:
            for metric in METRICS:
                outputs.extend(plot_fixed_vs_continuous(fixed, continuous, metric, out_dir, args))
        else:
            warn("fixed vs continuous plots skipped because one comparison run could not be loaded")

    if not outputs:
        raise SystemExit("no figures were written")

    print(f"wrote {len(outputs)} files to {out_dir}")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
