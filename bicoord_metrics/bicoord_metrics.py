import glob
import os

import h5py
import numpy as np
import yaml
from absl import app, flags

from bicoord_core import (_load_task_config, _episode_masks, reduce_metrics,
                          compute_windowed_sti)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "data_dir", None,
    "Task root folder containing quality-tier subfolders (e.g. .../boxLift, which holds "
    "expert/okay/poor). If none of --splits exist under it, --data_dir itself is treated "
    "as a single split.",
    required=True,
)
flags.DEFINE_list("splits", ["expert", "okay", "poor"], "Quality-tier subfolder names to look for under --data_dir.")
flags.DEFINE_bool("verbose", False, "Print per-episode metrics in addition to the per-split aggregate.")
flags.DEFINE_string("task", None, "Task name; must be a key in --config.", required=True)
flags.DEFINE_string(
    "config",
    os.path.join(os.path.dirname(__file__), "tasks.yaml"),
    "Path to the per-task obs-layout / grasp-threshold config.",
)
flags.DEFINE_integer("window_size", 20, "Sliding-window length (timesteps) for the windowed STI curve.")
flags.DEFINE_integer("window_stride", 5, "Stride between consecutive windows.")
flags.DEFINE_string(
    "plot_dir", None,
    "If set, save one PNG per split with per-episode windowed-STI curves (x-axis = fraction of "
    "episode completed) plus the split's mean curve -- useful for spotting a localized "
    "coordination dip (e.g. around a drop-and-recover event) that a single trajectory-level "
    "STI number averages away.",
)


def _plot_windowed_sti(curves_by_split: dict, plot_dir: str, task: str):
    from matplotlib import pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    grid = np.linspace(0, 1, 50)

    for split_name, curves in curves_by_split.items():
        plt.figure(figsize=(5, 4))
        interpolated = []
        for t_frac, sti_curve in curves:
            plt.plot(t_frac, sti_curve, color="tab:blue", alpha=0.15, linewidth=1)
            interpolated.append(np.interp(grid, t_frac, sti_curve))
        plt.plot(grid, np.mean(interpolated, axis=0), color="tab:blue", linewidth=2.5, label=f"mean (n={len(curves)})")
        plt.xlabel("Fraction of episode completed")
        plt.ylabel("Windowed STI (%)")
        plt.title(f"{task} - {split_name}")
        plt.legend(frameon=False)
        plt.tight_layout()
        out_path = os.path.join(plot_dir, f"{task}_{split_name}_windowed_sti.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")


def _collect_splits(root: str) -> list:
    split_dirs = [(s, os.path.join(root, s)) for s in FLAGS.splits if os.path.isdir(os.path.join(root, s))]
    return split_dirs if split_dirs else [("all", root)]  # fall back: treat data_dir as one split


def main(_):
    task_cfg = _load_task_config(FLAGS.config, FLAGS.task)

    per_episode = []
    curves_by_split = {}
    for split_name, split_dir in _collect_splits(FLAGS.data_dir):
        paths = sorted(glob.glob(os.path.join(split_dir, "*.h5")))
        if not paths:
            print(f"Warning: no .h5 files found in {split_dir}, skipping.")
            continue
        for path in paths:
            with h5py.File(path, "r") as f:
                for episode_key in f.keys():
                    policy_obs = f[episode_key]["offline_data"]["observations"]["policy"][:]
                    dist, both_moving = _episode_masks(policy_obs, task_cfg)

                    metrics = reduce_metrics(dist, both_moving, len(policy_obs))
                    metrics["file"] = os.path.basename(path)
                    metrics["split"] = split_name
                    per_episode.append(metrics)

                    if FLAGS.plot_dir:
                        curve = compute_windowed_sti(dist, both_moving, FLAGS.window_size, FLAGS.window_stride)
                        curves_by_split.setdefault(split_name, []).append(curve)

    assert len(per_episode) > 0, f"No .h5 files found under {FLAGS.data_dir} (looked in splits: {FLAGS.splits})"

    keys = ["ARD", "MRD", "SMP", "STI", "length"]
    print(f"Computed metrics for {len(per_episode)} episodes under {FLAGS.data_dir} (task={FLAGS.task})\n")

    if FLAGS.verbose:
        for metrics in per_episode:
            print(f"[{metrics['split']}]", metrics["file"], {k: round(metrics[k], 2) for k in keys})
        print()

    print("Per-split aggregate (mean +/- std over episodes):")
    seen_splits = dict.fromkeys(m["split"] for m in per_episode)  # preserve first-seen order
    for split_name in seen_splits:
        split_metrics = [m for m in per_episode if m["split"] == split_name]
        print(f"  [{split_name}] n={len(split_metrics)}")
        for k in keys:
            values = np.array([m[k] for m in split_metrics])
            print(f"    {k}: {values.mean():.2f} +/- {values.std():.2f}")

    if FLAGS.plot_dir:
        _plot_windowed_sti(curves_by_split, FLAGS.plot_dir, FLAGS.task)


if __name__ == "__main__":
    app.run(main)
