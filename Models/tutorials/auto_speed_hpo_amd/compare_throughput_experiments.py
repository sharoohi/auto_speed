#!/usr/bin/env python3
"""Plot fps/node comparison across SPX, DPX, QPX, CPX experiments.

Reads job_throughput_*.log files from each experiment folder and creates two plots:
  1. Compute FPS/node vs batch size
  2. Full workload FPS/node vs batch size
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings

import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent


# Pattern to parse: "[RESULT] batch_size=8 sum_compute_fps=1569.92 sum_full_workload_fps=1473.53"
SUMMARY_RE = re.compile(
    r"\[RESULT\]\s+batch_size=(\d+)\s+sum_compute_fps=([\d.]+)\s+sum_full_workload_fps=([\d.]+)"
)
COMBINED_SUMMARY_RE = re.compile(
    r"\[RESULT\]\s+num_gpus=(\d+)\s+batch_size=(\d+)\s+sum_compute_fps=([\d.]+)\s+sum_full_workload_fps=([\d.]+)"
)

def extract_num_gpus_from_filename(log_path: Path) -> Optional[int]:
    """Extract GPU count from job_throughput_*.log filename variants."""
    match = re.search(r"job_throughput_(?:[a-z0-9]+_)?(\d+)\.log", log_path.name)
    if match:
        return int(match.group(1))
    return None


def parse_log_file(log_path: Path) -> Tuple[Optional[int], Dict[int, Dict[str, float]]]:
    """Parse job_throughput_*.log file and return (num_gpus, {batch_size: {compute_fps, full_fps}})."""
    num_gpus = extract_num_gpus_from_filename(log_path)
    if num_gpus is None:
        return None, {}

    data: Dict[int, Dict[str, float]] = {}

    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                match = SUMMARY_RE.search(line)
                if match:
                    batch_size = int(match.group(1))
                    compute_fps = float(match.group(2))
                    full_fps = float(match.group(3))

                    data[batch_size] = {
                        "compute_fps": compute_fps,
                        "full_fps": full_fps,
                    }
    except Exception as exc:
        print(f"Warning: failed to parse {log_path}: {exc}")
        return num_gpus, {}

    return num_gpus, data


def parse_combined_log_file(log_path: Path) -> Dict[str, Tuple[int, Dict[int, Dict[str, float]]]]:
    """Version with everything in one file."""
    data: Dict[int, Dict[int, Dict[str, float]]] = {}

    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                match = COMBINED_SUMMARY_RE.search(line)
                if match:
                    num_gpus = int(match.group(1))
                    batch_size = int(match.group(2))
                    compute_fps = float(match.group(3))
                    full_fps = float(match.group(4))

                    exp_data = data.setdefault(num_gpus,  {})

                    exp_data[batch_size] = {
                        "compute_fps": compute_fps,
                        "full_fps": full_fps,
                    }
    except Exception as exc:
        raise IOError(f"failed to parse {log_path}: {exc}") from exc
    
    ngpus_to_name = {
        8: "SPX (8 GPUs)",
        16: "DPX (16 GPUs)",
        32: "QPX (32 GPUs)",
        63: "CPX (63 GPUs)"
    }
    
    # Reformat to match expected output of find_all_logs + parse_log_file
    reformatted_data: Dict[str, Tuple[int, Dict[int, Dict[str, float]]]] = {}
    for num_gpus, exp_data in data.items():
        exp_name = ngpus_to_name.get(num_gpus, f"{num_gpus} GPUs")
        new_exp_data = {}
        for batch_size, fps_data in exp_data.items():
            new_exp_data[batch_size] = fps_data
        reformatted_data[exp_name] = (num_gpus, new_exp_data)

    return reformatted_data


def find_all_logs(root: Path) -> Dict[str, Path]:
    """Find job_throughput_*.log files in SPX, DPX, QPX, CPX subdirectories."""
    # Define expected locations for each experiment
    experiment_paths = {
        "SPX (8 GPUs)": root / "job_throughput_8.log",
        "DPX (16 GPUs)": root / "job_throughput_16.log",
        "QPX (32 GPUs)": root / "job_throughput_32.log",
        "CPX (63 GPUs)": root / "job_throughput_63.log",
    }

    # Check which ones exist
    return dict((exp_name, exp_path) for exp_name, exp_path in experiment_paths.items() if exp_path.exists())


def collect_run_logs(root: Path) -> Dict[str, Tuple[int, Dict[int, Dict[str, float]]]]:
    # Define expected locations for each experiment
    experiment_paths = [
        ("SPX (8 GPUs)", root / "partition_8", 8),
        ("DPX (16 GPUs)", root / "partition_16", 16),
        ("QPX (32 GPUs)", root / "partition_32", 32),
        ("CPX (63 GPUs)", root / "partition_63", 63),
    ]
    data_by_exp: Dict[str, Tuple[int, Dict[int, Dict[str, float]]]] = {}

    for exp_name, exp_path, num_gpus in experiment_paths:
        if not exp_path.exists() or not exp_path.is_dir():
            print(f"Warning: no logs found for {exp_name}: {exp_path}")
            continue
        print(f"Processing logs for {exp_name} in {exp_path}")

        # Parse all logs and aggregate by batch size
        aggregated_data: Dict[int, Dict[str, float]] = {}

        # Find the latest run by looking in an individual GPU's logs
        for bs_dir in sorted(exp_path.glob("bs_*"), key=lambda d: int(d.name.split("_", 1)[-1])):
            if bs_dir.is_dir():
                bs = int(bs_dir.name.split("_", 1)[-1])
                print(f"  BS {bs}")
                # There should always be at least GPU 0, so check that to find the run name
                runs = list(sorted(d.name for d in (bs_dir / "gpu_0").glob("run-*")))
                if not runs:
                    print(f"  Warning: no runs found for {exp_name} BS {bs} in {bs_dir / 'gpu_0'}")
                    continue
                latest_run = runs[-1]
                log_files = [(bs_dir / f"gpu_{i}" / latest_run / "training.log") for i in range(num_gpus)]
                missing_log_files = [log for log in log_files if not log.exists()]
                if missing_log_files:
                    print(f"  Warning: missing training.log files for {exp_name} BS {bs}: {', '.join(str(log) for log in missing_log_files)}")

                compute_throughput_sum = 0.0
                full_throughput_sum = 0.0
                oom = False
                for log_file in log_files:
                    if not log_file.exists():
                        print(f"  Warning: missing training.log file: {log_file}")
                        break
                    # Look for throughput lines
                    compute_throughput = None
                    full_throughput = None
                    with log_file.open("r") as f:
                        for line in reversed(f.readlines()):
                            if "Overall Average Training Throughput (full workload FPS)" in line:
                                full_throughput = float(line.split(":")[-1].strip("\n "))
                            elif "Overall Average Training Throughput (compute-only FPS)" in line:
                                compute_throughput = float(line.split(":")[-1].strip("\n "))
                    if compute_throughput is None or full_throughput is None:
                        # Check whether this was an OOM: this should be visible from the console.log file outside the run dir
                        console_log = log_file.parent.parent / "console.log"
                        if console_log.exists():
                            with console_log.open("r") as f:
                                for line in f:
                                    if line.startswith("torch.OutOfMemoryError"):
                                        # This is the stacktrace for an OOM
                                        print(f"Detected OOM for {exp_name} BS {bs}")
                                        oom = True
                        if not oom:
                            print(f"WARNING: no throughputs in {log_file}, skipping {exp_name} BS {bs}")
                        break
                    else:
                        compute_throughput_sum += compute_throughput
                        full_throughput_sum += full_throughput
                else:
                    # Only save if we successfully found throughput for all GPUs
                    aggregated_data[bs] = {
                        "compute_fps": compute_throughput_sum,
                        "full_fps": full_throughput_sum,
                    }
                if oom:
                    # Signal an OOM by -1 fps
                    aggregated_data[bs] = {"compute_fps": -1, "full_fps": -1}

        data_by_exp[exp_name] = (num_gpus, aggregated_data)
    return data_by_exp


def split_partition_and_precision(exp_name: str) -> Tuple[str, str]:
    """Return (partition, precision) from labels like DPX/fp32 or CPX/amp."""
    if "/" in exp_name:
        partition, precision = exp_name.split("/", 1)
        return partition, precision.lower()
    # Historical names without suffix are treated as AMP by default.
    return exp_name, "amp"


def _format_throughput_value(value: Optional[float]) -> str:
    if value is None:
        return ""
    elif value == -1:
        return "OOM"
    return f"{value:.2f}"


def _print_throughput_markdown_table(
    title: str,
    metric_name: str,
    data_by_exp: Dict[str, Tuple[int, Dict[int, Dict[str, float]]]],
    batch_sizes: list[int],
) -> None:
    print(f"**{title}**\n")
    header = ["**Mode**"] + [f"BS: {bs}" for bs in batch_sizes]
    header_row = "| " + " | ".join(header) + " |"
    separator_row = "|" + "|".join(["----------"] * len(header)) + "|"
    print(header_row)
    print(separator_row)

    for exp_name, (_, data) in sorted(data_by_exp.items(), key=lambda x: x[1][0]):
        display_name = exp_name.replace(" GPUs", "")
        row_values = [display_name]
        for bs in batch_sizes:
            value = data.get(bs, {}).get(metric_name)
            row_values.append(_format_throughput_value(value))
        print("| " + " | ".join(row_values) + " |")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate FPS/node comparison plots from throughput logs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=THIS_DIR / "output",
        help="Root directory containing SPX, DPX, etc. log files or path to combined log file",
    )
    parser.add_argument(
        "--from-root-log",
        action="store_true",
        help="Extract throughputs from the root log files (job_throughput_X.log) instead of collecting all training.log files from individual run logs in partition_X subdirectories",
    )
    parser.add_argument(
        "--out-compute",
        type=Path,
        default=THIS_DIR / "compare_compute_fps_per_node.png",
        help="Output path for compute FPS/node plot",
    )
    parser.add_argument(
        "--out-full",
        type=Path,
        default=THIS_DIR / "compare_full_workload_fps_per_node.png",
        help="Output path for full workload FPS/node plot",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        parser.error(f"Root directory not found: {root}")
    if root.is_file():
        print(f"Parsing single log file: {root}")
        try:
            data_by_exp = parse_combined_log_file(root)
        except OSError as exc:
            parser.error(str(exc))
    elif not args.from_root_log:
        print(f"Collecting run logs from {root}")
        data_by_exp = collect_run_logs(root)
    else:
        # Find all logs
        logs = find_all_logs(root)
        if not logs:
            print(f"No job_throughput_*.log files found under: {root}")
            return 1

        print(f"Found {len(logs)} log file(s): {', '.join(sorted(logs.keys()))}")
        print(" - {}".format("\n - ".join(str(log_path.resolve()) for log_path in sorted(logs.values()))))

        # Parse all logs
        data_by_exp: Dict[str, Tuple[int, Dict[int, Dict[str, float]]]] = {}
        print("Parsing log files:")
        for exp_name, log_path in sorted(logs.items()):
            num_gpus, data = parse_log_file(log_path)
            if num_gpus is None:
                warnings.warn(f"Could not extract num gpus from path: {log_path}")
            else:
                data_by_exp[exp_name] = (num_gpus, data)
                print(f"  {exp_name}: {log_path.name} (num_gpus={num_gpus}, batch_sizes={sorted(data.keys())})")

    if not data_by_exp:
        print("No valid data parsed from logs")
        return 1

    # Collect all batch sizes
    all_batch_sizes = set()
    for _, data in data_by_exp.values():
        all_batch_sizes.update(data.keys())
    all_batch_sizes_sorted = sorted(all_batch_sizes)

    # Print markdown throughput tables before plotting
    _print_throughput_markdown_table(
        "Compute throughput",
        "compute_fps",
        data_by_exp,
        all_batch_sizes_sorted,
    )
    _print_throughput_markdown_table(
        "Full throughput",
        "full_fps",
        data_by_exp,
        all_batch_sizes_sorted,
    )

    # Create plots
    fig_compute, ax_compute = plt.subplots(figsize=(10, 6))
    fig_full, ax_full = plt.subplots(figsize=(10, 6))

    # Color encodes partitioning family (SPX/DPX/QPX/CPX)
    partitions = list(sorted(data_by_exp.keys()))
    num_partitions = len(partitions)
    cmap = plt.cm.get_cmap("tab20" if num_partitions <= 20 else "hsv")
    partition_colors = {
        partition: cmap(i / max(num_partitions - 1, 1))
        for i, partition in enumerate(partitions)
    }
    x_ticks = [8, 16, 32, 64, 128, 256, 512, 1024]

    for exp_name, (num_gpus, data) in sorted(data_by_exp.items(), key=lambda x: x[1][0]):
        num_gpus, data = data_by_exp[exp_name]
        if num_gpus is None or not data:
            print(f"  Skipping {exp_name}: no valid data")
            continue

        batch_sizes = []
        compute_fps_per_node = []
        full_fps_per_node = []

        for bs in all_batch_sizes_sorted:
            if bs in data:
                if data[bs]["compute_fps"] > 0. and data[bs]["full_fps"] > 0.:
                    batch_sizes.append(bs)
                    compute_fps_per_node.append(data[bs]["compute_fps"])
                    full_fps_per_node.append(data[bs]["full_fps"])

        if batch_sizes:
            color = partition_colors.get(exp_name, "black")
            ax_compute.plot(
                batch_sizes,
                compute_fps_per_node,
                label=exp_name,
                color=color,
                linewidth=2,
                markersize=8,
                marker="x",
            )
            ax_full.plot(
                batch_sizes,
                full_fps_per_node,
                label=exp_name,
                color=color,
                linewidth=2,
                markersize=8,
                marker="x",
            )

    # Format compute plot
    ax_compute.set_xlabel("Batch Size", fontsize=12)
    ax_compute.set_ylabel("Compute FPS/node", fontsize=12)
    ax_compute.set_title("Compute FPS/node vs Batch Size", fontsize=14, fontweight="bold")
    ax_compute.legend(fontsize=11)
    ax_compute.grid(True, alpha=0.3)
    ax_compute.set_xscale("log")
    ax_compute.set_xticks(x_ticks)
    ax_compute.set_xticklabels([str(x) for x in x_ticks])

    # Format full workload plot
    ax_full.set_xlabel("Batch Size", fontsize=12)
    ax_full.set_ylabel("Full Workload FPS/node", fontsize=12)
    ax_full.set_title("Full Workload FPS/node vs Batch Size", fontsize=14, fontweight="bold")
    ax_full.legend(fontsize=11)
    ax_full.grid(True, alpha=0.3)
    ax_full.set_xscale("log")
    ax_full.set_xticks(x_ticks)
    ax_full.set_xticklabels([str(x) for x in x_ticks])

    # Save plots
    fig_compute.tight_layout()
    fig_compute.savefig(args.out_compute, dpi=150, bbox_inches="tight")
    print(f"Saved compute FPS/node plot: {args.out_compute}")

    fig_full.tight_layout()
    fig_full.savefig(args.out_full, dpi=150, bbox_inches="tight")
    print(f"Saved full workload FPS/node plot: {args.out_full}")

    plt.close(fig_compute)
    plt.close(fig_full)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
