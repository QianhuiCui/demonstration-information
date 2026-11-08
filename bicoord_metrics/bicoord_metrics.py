import glob
import os

import h5py
import numpy as np
import yaml
from absl import app, flags
from matplotlib import pyplot as plt

from bicoord_core import _load_task_config, _episode_masks, reduce_metrics, compute_windowed_sti

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", None, "Task root holding expert/okay/poor/mixture subfolders.", required=True)
flags.DEFINE_list("splits", ["expert", "okay", "poor", "mixture"], "Quality-tier subfolder names to look for under --data_dir.")
flags.DEFINE_bool("verbose", False, "Print per-episode metrics in addition to the per-split aggregate.")
flags.DEFINE_list("task", None, "Task name(s) from --config. With several tasks, --data_dir is the root holding one folder per task.", required=True)
flags.DEFINE_string(
    "config",
    os.path.join(os.path.dirname(__file__), "tasks.yaml"),
    "Path to the per-task obs-layout / grasp-threshold config.",
)
flags.DEFINE_integer("window_size", 20, "Sliding-window length (timesteps) for the windowed STI curve.")
flags.DEFINE_integer("window_stride", 10, "Stride between consecutive windows.")
flags.DEFINE_string("plot_dir", None,  "If set, save per-split windowed-STI plots here.")


def _plot_windowed_sti(curves, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    grid = np.linspace(0, 1, 50)
    tasks = list(curves)
    splits = list(dict.fromkeys(split for task in tasks for split in curves[task]))

    # colors = {"expert": "tab:blue", "okay": "tab:orange", "poor": "tab:green", "mixture": "tab:red"}
    # task_palette = ["tab:red", "tab:orange", "tab:brown", "tab:green", "tab:blue", "tab:olive"]
    # colors.update({task: task_palette[i % len(task_palette)] for i, task in enumerate(tasks)})

    # one figure per task, one line per quality group
    figures = [(f"{task}_by_split", task, curves[task]) for task in tasks]
    # one figure per quality group, one line per task
    if len(tasks) > 1:
        figures += [(f"{split}_by_task", split, {task: curves[task][split] for task in tasks if split in curves[task]}) for split in splits]

    for name, title, lines in figures:
        plt.figure(figsize=(6, 4))
        for label, group in lines.items():
            y = np.array([np.interp(grid, t_frac, sti) for t_frac, sti in group])
            mean, std = y.mean(axis=0), y.std(axis=0)
            (line,) = plt.plot(grid, mean, linewidth=2.2, label=f"{label} (n={len(y)})")
            plt.fill_between(grid, mean - std, mean + std, color=line.get_color(), alpha=0.15, linewidth=0)
            # (line,) = plt.plot(grid, mean, color=colors.get(label), linewidth=2.2, label=f"{label} (n={len(y)})")
            # plt.fill_between(grid, mean - std, mean + std, color=line.get_color(), alpha=0.15, linewidth=0)        
        plt.xlabel("Fraction of episode length")
        plt.ylabel("Windowed STI (%)")
        plt.title(f"{title} (mean +/- std)")
        plt.legend(frameon=False)  
        plt.tight_layout()
        out_path = os.path.join(plot_dir, f"{name}_windowed_sti.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")


def _collect_splits(root):
    split_dirs = [(s, os.path.join(root, s)) for s in FLAGS.splits if os.path.isdir(os.path.join(root, s))]
    return split_dirs if split_dirs else [("all", root)]  # fall back: treat data_dir as one split


def main(_):
    keys = ["ARD", "MRD", "SMP", "STI", "length"]
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
                        policy_obs = f[ep_key]["offline_data"]["observations"]["policy"][:]
                        dist, both_moving = _episode_masks(policy_obs, task_cfg)
                        metrics = reduce_metrics(dist, both_moving, len(policy_obs))
                        metrics["file"] = os.path.basename(path)
                        metrics["split"] = split_name
                        episode.append(metrics)

                        curve = compute_windowed_sti(dist, both_moving, FLAGS.window_size, FLAGS.window_stride)
                        curves.setdefault(task, {}).setdefault(split_name, []).append(curve)

        assert len(episode) > 0, f"No .h5 files found under {task_dir} (looked in splits: {FLAGS.splits})"
        print(f"\n{task}: {len(episode)} episodes under {task_dir}")

        if FLAGS.verbose:
            for metrics in episode:
                print(f"[{metrics['split']}]", metrics["file"], {k: round(metrics[k], 2) for k in keys})
            print()

        print("Per-split aggregate (mean +/- std over episodes):")
        for split_name in curves[task]:
            split_metrics = [m for m in episode if m["split"] == split_name]
            print(f"  [{split_name}] n={len(split_metrics)}")
            for k in keys:
                values = np.array([m[k] for m in split_metrics])
                print(f"{k}: {values.mean():.2f} +/- {values.std():.2f}")

    if FLAGS.plot_dir:
        _plot_windowed_sti(curves, FLAGS.plot_dir)              


if __name__ == "__main__":
    app.run(main)
