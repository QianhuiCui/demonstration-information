"""Fit a pooled PCA basis on the raw Allegro-hand joints of a symdex task.

  python tools/fit_hand_pca.py --data_dir /home/qianhui/deminf/symdex_rlds --task boxLift
  python tools/fit_hand_pca.py --data_dir /home/qianhui/deminf/symdex_rlds --task boxLift \
      --k 4 --out rlds/symdex/hand_pca_boxLift_k4.npz
"""
import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import yaml
from absl import app, flags

_TASKS_YAML = os.path.join(os.path.dirname(__file__), "..", "rlds", "symdex", "tasks.yaml")

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", None, "Root holding sym_dex/<task>/1.0.0.", required=True)
flags.DEFINE_string("task", "boxLift", "symdex task name.")
flags.DEFINE_string("split", "train", "Split to fit on.")
flags.DEFINE_integer("batch", 8192, "Streaming batch size for the moments pass.")
flags.DEFINE_integer("k", 0, "Components to keep and save. 0 => report variance only.")
flags.DEFINE_string("out", None, "Output .npz path.")


def stream_moments(ds, slices):
    """One streaming pass over every step. `slices` maps a name to a (start, end) column
    range, all of equal width d. Returns name -> [count, sum(d,), sum_of_outer(d, d)].
    Memory is O(d^2), independent of dataset size."""
    d = (lambda lo, hi: hi - lo)(*next(iter(slices.values())))
    acc = {name: [0, np.zeros(d), np.zeros((d, d))] for name in slices}
    batches = ds.flat_map(lambda ep: ep["steps"]).batch(FLAGS.batch).prefetch(tf.data.AUTOTUNE)
    for batch in tfds.as_numpy(batches):
        state = batch["observation"]["state"].astype(np.float64)      # (B, obs_dim)
        for name, (lo, hi) in slices.items():
            x = state[:, lo:hi]
            acc[name][0] += len(x)
            acc[name][1] += x.sum(axis=0)
            acc[name][2] += x.T @ x
    return acc


def within_group_cov(groups):
    """Covariance after removing each group's own mean, so a left/right mean offset
    doesn't masquerade as the top principal component. `groups`: list of [count, s1, s2]."""
    total = sum(n for n, _, _ in groups)
    scatter = sum(s2 - np.outer(s1, s1) / n for n, s1, s2 in groups)
    return scatter / total


def pca(cov):
    evals, evecs = np.linalg.eigh(cov)               # ascending, symmetric
    evals = np.clip(evals[::-1], 0.0, None)          # descending; drop roundoff negatives
    components = evecs[:, ::-1].T                    # (d, d), rows = PCs, descending
    return components, evals / evals.sum()



def main(_):
    with open(os.path.normpath(_TASKS_YAML)) as f:
        sl = yaml.safe_load(f)[FLAGS.task]["state_slices"]

    path = os.path.join(FLAGS.data_dir, "sym_dex", FLAGS.task, "1.0.0")
    ds = tfds.builder_from_directory(path).as_dataset(split=FLAGS.split)
    acc = stream_moments(ds, {"left": tuple(sl["left_hand"]), "right": tuple(sl["right_hand"])})

    groups = {"left": [acc["left"]], "right": [acc["right"]], "pooled": [acc["left"], acc["right"]]}
    n_pool = acc["left"][0] + acc["right"][0]
    print(f"\n{FLAGS.task}: {n_pool} hand-joint samples ({acc['left'][0]}/arm)")
    for name, g in groups.items():
        _, evr = pca(within_group_cov(g))
        print(f"  {name:6s} cumulative EVR: {np.round(np.cumsum(evr), 3)}")

    if FLAGS.k:
        components, evr = pca(within_group_cov(groups["pooled"]))
        mean_left = acc["left"][1] / acc["left"][0]
        mean_right = acc["right"][1] / acc["right"][0]
        out = FLAGS.out or f"rlds/symdex/hand_pca_{FLAGS.task}_k{FLAGS.k}.npz"
        np.savez(
            out,
            mean_left=mean_left.astype(np.float32),
            mean_right=mean_right.astype(np.float32),
            components=components[: FLAGS.k].astype(np.float32),
            evr=evr.astype(np.float32),
        )
        print(f"\nsaved k={FLAGS.k} -> {out}  ({np.cumsum(evr)[FLAGS.k - 1]:.1%} of within-hand variance)")



if __name__ == "__main__":
    app.run(main)
