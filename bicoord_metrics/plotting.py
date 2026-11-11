import glob
import os
import warnings

import h5py
import numpy as np
from absl import app, flags
from matplotlib import pyplot as plt

from core import (
    _load_task_config,
    _episode_masks,
    reduce_metrics,
    compute_windowed_sti,
    compute_windowed_coordination,
    reduce_coordination,
    role_asymmetry,
    compute_windowed_asym,
)

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", None, "Task root holding expert/okay/poor/mixture subfolders.", required=True)
flags.DEFINE_list("splits", ["expert", "okay", "poor", "mixture"], "Quality-tier subfolder names to look for under --data_dir.")
flags.DEFINE_bool("verbose", False, "Print per-episode metrics in addition to the per-split aggregate.")
flags.DEFINE_list("task", None, "Task name(s) from --config. With several tasks, --data_dir is the root holding one folder per task.", required=True)
flags.DEFINE_string("config", os.path.join(os.path.dirname(__file__), "tasks.yaml"), "Path to the per-task obs-layout / grasp-threshold config.")
flags.DEFINE_integer("window_size", 20, "Sliding-window length (timesteps) for the windowed curves.")
flags.DEFINE_integer("window_stride", 5, "Stride between consecutive windows.")
flags.DEFINE_integer("max_lag", 5, "Max lag (timesteps) for the windowed motion-coupling correlation.")
flags.DEFINE_string("plot_dir", None, "If set, save windowed plots here.")

KEYS = ["STI", "coord", "asym", "hold_frac", "length"]
# "ARD", "MRD", "SMP", "STI", "STI_obj", "coupling", "coord", "asym", "hold_frac", "length"
CURVES = {  # curve key -> y-axis label (one panel each)
    "sti": "Windowed STI (%)",
    "asym": "Windowed role asymmetry (%)",
    # "sti_obj": "Windowed object STI (%)",
    # "coupling": "Motion coupling (corr)",
    "coord": "Coord = STI_obj x coupling (%)",
}


def _plot_panel(ax, lines, key, grid):
    for label, group in lines.items():
        # NaN (e.g. asym windows where the hands don't both hold) stays NaN through interp
        y = np.array([np.interp(grid, c["t_frac"], c[key]) for c in group])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN grid columns
            mean, std = np.nanmean(y, axis=0), np.nanstd(y, axis=0)
        (line,) = ax.plot(grid, mean, linewidth=2.0, label=f"{label} (n={len(y)})")
        ax.fill_between(grid, mean - std, mean + std, color=line.get_color(), alpha=0.15, linewidth=0)
    ax.set_xlabel("Fraction of episode length")
    ax.set_ylabel(CURVES[key])


def plot_windowed(curves, plot_dir):
    """curves[task][split] = list of per-episode dicts {t_frac, sti, asym, sti_obj, coupling, coord}.
    One figure per task (lines = splits) and per split (lines = tasks); one panel per curve."""
    os.makedirs(plot_dir, exist_ok=True)
    grid = np.linspace(0, 1, 50)
    tasks = list(curves)
    splits = list(dict.fromkeys(split for task in tasks for split in curves[task]))

    figures = [(f"{task}_by_split", task, curves[task]) for task in tasks]
    if len(tasks) > 1:
        figures += [(f"{split}_by_task", split, {task: curves[task][split] for task in tasks if split in curves[task]}) for split in splits]

    for name, title, lines in figures:
        fig, axes = plt.subplots(1, len(CURVES), figsize=(4.5 * len(CURVES), 3.8))
        for ax, key in zip(axes, CURVES):
            _plot_panel(ax, lines, key, grid)
        axes[0].legend(frameon=False)
        fig.suptitle(f"{title} (mean +/- std)")
        fig.tight_layout()
        out_path = os.path.join(plot_dir, f"{name}_windowed.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


def _collect_splits(root):
    split_dirs = [(s, os.path.join(root, s)) for s in FLAGS.splits if os.path.isdir(os.path.join(root, s))]
    return split_dirs if split_dirs else [("all", root)]  # fall back: treat data_dir as one split


def main(_):
    curves = {}
    for task in FLAGS.task:
        task_cfg = _load_task_config(FLAGS.config, task)
        task_dir = os.path.join(FLAGS.data_dir, task)

        episode = []
        for split_name, split_dir in _collect_splits(task_dir):
            paths = sorted(glob.glob(os.path.join(split_dir, "*.h5")))
            if not paths:
                print(f"Warning: no .h5 files found in {split_dir}.")
                continue
            for path in paths:
                with h5py.File(path, "r") as f:
                    for ep_key in f.keys():
                        policy_obs = f[ep_key]["offline_data"]["observations"]["policy"][:-1]  # drop post-termination reset frame
                        dist, both_moving = _episode_masks(policy_obs, task_cfg)
                        coord = compute_windowed_coordination(
                            policy_obs, task_cfg, FLAGS.window_size, FLAGS.window_stride, FLAGS.max_lag)

                        metrics = reduce_metrics(dist, both_moving, len(policy_obs))
                        metrics.update(reduce_coordination(coord))
                        metrics.update(role_asymmetry(policy_obs, task_cfg))
                        metrics["file"] = os.path.basename(path)
                        metrics["split"] = split_name
                        episode.append(metrics)

                        _, sti = compute_windowed_sti(dist, both_moving, FLAGS.window_size, FLAGS.window_stride)
                        _, asym = compute_windowed_asym(policy_obs, task_cfg, FLAGS.window_size, FLAGS.window_stride)
                        curves.setdefault(task, {}).setdefault(split_name, []).append(dict(coord, sti=sti, asym=asym))

        assert len(episode) > 0, f"No .h5 files found under {task_dir} (looked in splits: {FLAGS.splits})"
        print(f"\n{task}: {len(episode)} episodes under {task_dir}")

        if FLAGS.verbose:
            for metrics in episode:
                print(f"[{metrics['split']}]", metrics["file"], {k: round(metrics[k], 2) for k in KEYS})
            print()

        print("Per-split aggregate (mean +/- std over episodes):")
        for split_name in curves[task]:
            split_metrics = [m for m in episode if m["split"] == split_name]
            print(f"  [{split_name}] n={len(split_metrics)}")
            for k in KEYS:
                values = np.array([m[k] for m in split_metrics], dtype=float)
                n_valid = int(np.isfinite(values).sum())
                if n_valid == 0:
                    print(f"{k}: n/a (no valid episodes)")
                    continue
                note = f"  (n_valid={n_valid})" if n_valid < len(values) else ""
                print(f"{k}: {np.nanmean(values):.2f} +/- {np.nanstd(values):.2f}{note}")

    if FLAGS.plot_dir:
        plot_windowed(curves, FLAGS.plot_dir)


if __name__ == "__main__":
    app.run(main)
