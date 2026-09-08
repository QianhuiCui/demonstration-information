"""
Writes <path>/bicoord.pkl :
    {ep_idx -> {ARD, MRD, SMP, STI, length, quality, file,
               windowed_sti: {t_frac, sti}, dist, both_moving}}
"""
import os
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags
from scipy import stats

from bicoord_core import _load_task_config, _episode_masks, reduce_metrics, compute_windowed_sti

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", None, ".../symdex_rlds/sym_dex/<task>/1.0.0", required=True)
flags.DEFINE_string("task", None, "task name; must be a key in --config.", required=True)
flags.DEFINE_string("split", "train", "RLDS split to score (match S_coord: 'train').")
flags.DEFINE_string("config", os.path.join(os.path.dirname(__file__), "tasks.yaml"),
                    "per-task obs-layout / grasp-threshold config.")
flags.DEFINE_string("path", None, "directory to write bicoord.pkl into; if unset, only print.")
flags.DEFINE_bool("verbose", True, "print per-episode metrics as well as the per-tier aggregate.")
flags.DEFINE_integer("window", 20, "windowed-STI window length (frames); matches bicoord_metrics.py.")
flags.DEFINE_integer("stride", 5, "windowed-STI stride (frames); matches bicoord_metrics.py.")

_TIER = {3.0: "expert", 2.0: "okay", 1.0: "poor"}
_KEYS = ["ARD", "MRD", "SMP", "STI", "length"]


def _episode_obs(ep) -> np.ndarray:
    """ep['steps'] is a nested tf.data.Dataset -> (T, obs_dim) float array, in step order."""
    return np.stack([s["observation"]["state"].numpy() for s in ep["steps"]])


def main(_):
    tf.config.set_visible_devices([], "GPU")
    task_cfg = _load_task_config(FLAGS.config, FLAGS.task)

    ds = tfds.builder_from_directory(FLAGS.data_dir).as_dataset(split=FLAGS.split, shuffle_files=False)

    rows = {}
    for ep in ds:
        obs = _episode_obs(ep)
        meta = ep["episode_metadata"]
        ep_idx = int(meta["ep_idx"].numpy())
        dist, both_moving = _episode_masks(obs, task_cfg)

        m = reduce_metrics(dist, both_moving, len(obs))
        m["quality"] = float(meta["quality_score"].numpy())
        m["file"] = os.path.basename(meta["file_path"].numpy().decode())
        t_frac, sti_curve = compute_windowed_sti(dist, both_moving, FLAGS.window, FLAGS.stride)
        m["windowed_sti"] = {"t_frac": t_frac, "sti": sti_curve}
        m["dist"] = dist            # raw per-timestep arrays, so ds_viewer can re-window at any size
        m["both_moving"] = both_moving
        rows[ep_idx] = m

    assert rows, f"no episodes in split '{FLAGS.split}' of {FLAGS.data_dir}"
    order = sorted(rows)
    q = np.array([rows[e]["quality"] for e in order])

    print(f"\n{FLAGS.task}: {len(rows)} episodes ({FLAGS.split} split)\n")
    if FLAGS.verbose:
        for e in order:
            print(f"  ep {e:3d} [{_TIER.get(rows[e]['quality'], '?'):6s}] "
                  + "  ".join(f"{k}={rows[e][k]:7.2f}" for k in _KEYS))
        print()

    print("per-tier mean +/- std (scalars -- see ds_viewer §2c for the windowed profile):")
    for tier in (3.0, 2.0, 1.0):
        idx = [e for e in order if rows[e]["quality"] == tier]
        if not idx:
            continue
        print(f"  [{_TIER[tier]}] n={len(idx)}")
        for k in _KEYS:
            v = np.array([rows[e][k] for e in idx])
            print(f"    {k:6s} {v.mean():8.2f} +/- {v.std():6.2f}")

    print("\nfirst look -- per-episode spearman vs quality tier "
          "(the real test is the windowed profile vs S_coord, done in ds_viewer):")
    for k in ("ARD", "MRD", "SMP", "STI"):
        v = np.array([rows[e][k] for e in order])
        sp = stats.spearmanr(v, q)
        print(f"  {k:6s} {sp.statistic:+.3f}  (p={sp.pvalue:.3g})")

    if FLAGS.path is not None:
        tf.io.gfile.makedirs(FLAGS.path)
        out = os.path.join(FLAGS.path, "bicoord.pkl")
        with tf.io.gfile.GFile(out, "wb") as f:
            pickle.dump(rows, f)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    app.run(main)
