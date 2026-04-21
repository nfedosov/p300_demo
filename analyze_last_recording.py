#!/usr/bin/env python3
"""Load the latest experiment recording and visualize TRCA-based decision metrics."""

from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np


VIS_EPOCH_TMIN = -0.2
VIS_EPOCH_TMAX = 1.0
METRIC_WINDOW_TMIN = 0.1
METRIC_WINDOW_TMAX = 0.7


def trca_fit(epochs_data: np.ndarray) -> np.ndarray:
    """Return leading TRCA spatial filter for epochs (n_trials, n_channels, n_times)."""
    if epochs_data.ndim != 3:
        raise ValueError("epochs_data must have shape (n_trials, n_channels, n_times)")

    n_trials, n_chan, _ = epochs_data.shape
    if n_trials < 2:
        raise ValueError("TRCA needs at least two trials per class")

    x_centered = epochs_data - epochs_data.mean(axis=2, keepdims=True)

    s_matrix = np.zeros((n_chan, n_chan), dtype=np.float64)
    for i in range(n_trials):
        xi = x_centered[i]
        for j in range(i + 1, n_trials):
            xj = x_centered[j]
            s_matrix += xi @ xj.T + xj @ xi.T

    q_matrix = np.zeros((n_chan, n_chan), dtype=np.float64)
    for k in range(n_trials):
        xk = x_centered[k]
        q_matrix += xk @ xk.T

    reg = 1e-6 * np.trace(q_matrix) / max(n_chan, 1)
    q_reg = q_matrix + reg * np.eye(n_chan)

    m_matrix = np.linalg.pinv(q_reg) @ s_matrix
    eigenvals, eigenvecs = np.linalg.eig(m_matrix)

    idx = np.argsort(np.real(eigenvals))[::-1]
    eigenvecs = np.real(eigenvecs[:, idx])
    return eigenvecs[:, 0]


def cross_cov_power(trials_a: np.ndarray, trials_b: np.ndarray | None = None) -> float:
    """Average absolute cross-covariance power across trial pairs."""
    if trials_a.size == 0:
        return 0.0

    a = trials_a - trials_a.mean(axis=1, keepdims=True)

    if trials_b is None:
        if len(a) < 2:
            return 0.0
        vals = [abs(float(np.dot(a[i], a[j]))) for i in range(len(a)) for j in range(i + 1, len(a))]
        return float(np.mean(vals)) if vals else 0.0

    b = trials_b - trials_b.mean(axis=1, keepdims=True)
    if b.size == 0:
        return 0.0

    vals = [abs(float(np.dot(a[i], b[j]))) for i in range(len(a)) for j in range(len(b))]
    return float(np.mean(vals)) if vals else 0.0


def find_latest_pickle(results_root: Path) -> Path:
    pattern = str(results_root / "baseline_experiment_*" / "data.pickle")
    matches = [Path(p) for p in glob.glob(pattern)]
    if not matches:
        raise FileNotFoundError(f"No data.pickle files found in {results_root}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def build_unique_events(sample_timestamps: np.ndarray, stim_events: Sequence[Dict], max_samples: int) -> np.ndarray:
    event_samples = []
    last_sample = -1
    for ev in stim_events:
        sample_idx = int(np.searchsorted(sample_timestamps, ev["onset_lsl"], side="left"))
        if sample_idx >= max_samples:
            sample_idx = max_samples - 1
        if sample_idx <= last_sample:
            sample_idx = last_sample + 1
        if sample_idx >= max_samples:
            continue
        event_samples.append([sample_idx, 0, int(ev["class_idx"]) + 1])
        last_sample = sample_idx

    if not event_samples:
        raise RuntimeError("No valid stimulus events after alignment to EEG samples")

    return np.asarray(event_samples, dtype=int)


def load_class_labels(data: Dict, n_classes: int) -> List[str]:
    labels = data.get("class_labels")
    if labels and len(labels) == n_classes:
        return list(labels)

    winner_label = data.get("winner_label")
    base = [f"class_{i + 1}" for i in range(n_classes)]
    if isinstance(winner_label, str) and 0 <= int(data.get("winner_idx", -1)) < n_classes:
        base[int(data["winner_idx"])] = winner_label
    return base


def prepare_epochs(data: Dict) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
    eeg = np.asarray(data["eeg"])
    timestamps = np.asarray(data["timestamps"])
    channel_names = data["channel_names"]
    eeg_indices = data["eeg_indices"]
    stim_events = data["stim_events"]

    eeg_names = [channel_names[i] for i in eeg_indices]
    eeg_data = eeg[:, eeg_indices]

    srate = int(data["exp_settings"]["srate"])
    info = mne.create_info(eeg_names, srate, ch_types="eeg")
    raw = mne.io.RawArray(eeg_data.T, info, verbose=False)
    raw.notch_filter([50.0, 100.0], verbose=False)
    raw.filter(0.3, 20.0, verbose=False)

    events = build_unique_events(timestamps, stim_events, len(timestamps))
    n_classes = int(events[:, 2].max())
    class_labels = load_class_labels(data, n_classes)
    event_id = {class_labels[i]: i + 1 for i in range(n_classes)}

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=VIS_EPOCH_TMIN,
        tmax=VIS_EPOCH_TMAX,
        baseline=None,
        preload=True,
        detrend=1,
        verbose=False,
    )

    epochs_by_class: Dict[str, np.ndarray] = {label: epochs[label].get_data(copy=True) for label in class_labels}
    return epochs.times.copy(), epochs_by_class, class_labels


def analyze_trca(
    times: np.ndarray,
    epochs_by_class: Dict[str, np.ndarray],
    class_labels: Sequence[str],
) -> Dict[str, np.ndarray]:
    p_target_vals, p_other_vals, ratio_vals = [], [], []
    target_templates, other_templates = [], []

    per_class = [epochs_by_class[label] for label in class_labels]
    metric_time_mask = (times >= METRIC_WINDOW_TMIN) & (times <= METRIC_WINDOW_TMAX)
    if not np.any(metric_time_mask):
        raise RuntimeError("Metric window mask is empty. Check epoch and metric window bounds.")

    for class_idx, cls_data in enumerate(per_class):
        cls_metric = cls_data[:, :, metric_time_mask]
        if cls_metric.shape[0] < 2:
            raise RuntimeError("TRCA needs at least two trials in the metric window.")

        w = trca_fit(cls_metric)
        cls_proj = np.einsum("c,tcn->tn", w, cls_metric)

        other_data = np.concatenate([per_class[i] for i in range(len(per_class)) if i != class_idx], axis=0)
        other_metric = other_data[:, :, metric_time_mask]
        other_proj = np.einsum("c,tcn->tn", w, other_metric)

        p_target = cross_cov_power(cls_proj)
        p_other = cross_cov_power(cls_proj, other_proj)
        ratio = p_target / (p_other + 1e-12)

        p_target_vals.append(p_target)
        p_other_vals.append(p_other)
        ratio_vals.append(ratio)
        target_templates.append(cls_proj.mean(axis=0))
        other_templates.append(other_proj.mean(axis=0))

    return {
        "p_target": np.asarray(p_target_vals),
        "p_other": np.asarray(p_other_vals),
        "ratio": np.asarray(ratio_vals),
        "target_templates": np.asarray(target_templates),
        "other_templates": np.asarray(other_templates),
    }


def plot_results(times: np.ndarray, labels: Sequence[str], stats: Dict[str, np.ndarray], output_dir: Path) -> Tuple[Path, Path]:
    n_classes = len(labels)

    fig_wave, axes = plt.subplots(n_classes, 1, figsize=(11, 2.6 * n_classes), sharex=True)
    if n_classes == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(times, stats["target_templates"][i], label=f"{labels[i]}: detected potential", linewidth=2)
        ax.plot(times, stats["other_templates"][i], label=f"{labels[i]}: other-classes potential", linewidth=2, alpha=0.85)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_ylabel("a.u.")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    fig_wave.suptitle("TRCA projected potentials: class vs other classes", fontsize=13)
    fig_wave.tight_layout(rect=(0, 0.02, 1, 0.98))

    wave_path = output_dir / "trca_overlapped_potentials.png"
    fig_wave.savefig(wave_path, dpi=170)
    plt.close(fig_wave)

    x = np.arange(n_classes)
    width = 0.35
    fig_bar, ax1 = plt.subplots(figsize=(11, 4.5))
    ax1.bar(x - width / 2, stats["p_target"], width, label="P_target")
    ax1.bar(x + width / 2, stats["p_other"], width, label="P_other")
    ax1.set_ylabel("Cross-covariance power")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, stats["ratio"], color="black", marker="o", linewidth=2, label="Decision metric ratio")
    ax2.set_ylabel("Ratio P_target / P_other")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    fig_bar.suptitle("TRCA decision metric per class", fontsize=13)
    fig_bar.tight_layout()

    bar_path = output_dir / "trca_decision_metric.png"
    fig_bar.savefig(bar_path, dpi=170)
    plt.close(fig_bar)

    return wave_path, bar_path


def save_numeric_summary(output_dir: Path, labels: Sequence[str], stats: Dict[str, np.ndarray]) -> Path:
    out = output_dir / "trca_metrics_summary.csv"
    with out.open("w", encoding="utf-8") as f:
        f.write("class,p_target,p_other,ratio\n")
        for i, label in enumerate(labels):
            f.write(f"{label},{stats['p_target'][i]:.8e},{stats['p_other'][i]:.8e},{stats['ratio'][i]:.8f}\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze latest recorded P300 data and produce plots.")
    parser.add_argument("--results-dir", default="results", help="Root folder with baseline_experiment_* runs.")
    parser.add_argument("--input", default=None, help="Optional explicit path to a data.pickle file.")
    parser.add_argument("--no-show", action="store_true", help="Do not open matplotlib windows.")
    args = parser.parse_args()

    data_path = Path(args.input) if args.input else find_latest_pickle(Path(args.results_dir))
    with data_path.open("rb") as f:
        data = pickle.load(f)

    times, epochs_by_class, labels = prepare_epochs(data)
    stats = analyze_trca(times, epochs_by_class, labels)

    output_dir = data_path.parent
    wave_path, bar_path = plot_results(times, labels, stats, output_dir)
    csv_path = save_numeric_summary(output_dir, labels, stats)

    winner_idx = int(np.argmax(stats["ratio"]))
    print(f"Loaded: {data_path}")
    print("\nTRCA numerical result:")
    for i, label in enumerate(labels):
        print(
            f"  {label:>12s} | P_target={stats['p_target'][i]:.5e} "
            f"P_other={stats['p_other'][i]:.5e} ratio={stats['ratio'][i]:.5f}"
        )
    print(f"\nPredicted class: {labels[winner_idx]} (index={winner_idx})")
    print(f"Saved: {wave_path}")
    print(f"Saved: {bar_path}")
    print(f"Saved: {csv_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
