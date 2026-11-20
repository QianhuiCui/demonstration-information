"""
S_coord scorer -- raw per-timestep and per-trajectory I(z_left; z_right).

Same KSG estimator as `estimate_quality.py --estimator ksg` (quality_estimators.ksg_from_dists),
but every frame is scored against one fixed reference set of frames instead of its random batch,
with the frame's own episode left out. No clip, no z-norm, and frames keep their time index.

Example command (same flags as estimate_quality.py):

python scripts/quality/estimate_quality_raw.py \
    --obs_ckpt path/to/left_vae/<step> \
    --action_ckpt path/to/right_vae/<step> \
    --batch_size 1024 \
    --estimator ksg \
    --path path/to/scores

Writes <path>/<dataset>.pkl with the same keys as estimate_quality.py (raw scores instead of
z-scores): ep_idx, quality_score, dataset_id, quality_by_ep_idx, plus
    per_frame  {ep_idx, t, quality, I_coord}   raw per-frame arrays, for windowed analysis
and <dataset>_hist.png / <dataset>_curve.png.
"""

import os
import pickle
import pprint

import jax
import numpy as np
import quality_estimators
import tensorflow as tf
from absl import app, flags
from jax.experimental import compilation_cache
from scipy import stats

from openx.utils.evaluate import load_checkpoint

FLAGS = flags.FLAGS
flags.DEFINE_string("path", None, "The path to save the results if desired.", required=False)
flags.DEFINE_string("obs_ckpt", None, "Path to the obs (left-arm) logs and checkpoints.", required=False)
flags.DEFINE_string("action_ckpt", None, "Path to the action (right-arm) logs and checkpoints.", required=False)
flags.DEFINE_string(
    "obs_action_ckpt", None, "Path to the obs action logs and checkpoints (not used by ksg).", required=False
)
flags.DEFINE_enum("estimator", "ksg", ["ksg"], "Type of quality estimator (only ksg is supported).")
flags.DEFINE_integer("batch_size", 1024, "The batch size for encoding and scoring.", required=False)
flags.DEFINE_integer("ref_size", 8192, "Number of frames in the fixed reference set.")
flags.DEFINE_bool("exclude_same_episode", True, "Leave the query frame's own episode out of the reference set.")
flags.DEFINE_integer("seed", 1, "Seed for the reference subsample and the VAE sampling.")


def _encode(ds, obs_alg, obs_state, action_alg, action_state, rng):
    """Encodes every frame in order. Returns z_obs (N, D), z_action (N, D), {key: (N,)} attributes."""

    @jax.jit
    def encode(batch, rng):
        obs_rng, action_rng = jax.random.split(rng)
        return obs_alg.predict(obs_state, batch, obs_rng), action_alg.predict(action_state, batch, action_rng)

    z_obs, z_action = [], []
    attrs = {k: [] for k in quality_estimators.AGGREGATION_KEYS}
    for i, batch in enumerate(ds):
        batch = jax.tree.map(lambda x: x._numpy(), batch)
        zo, za = encode(batch, jax.random.fold_in(rng, i))
        z_obs.append(np.asarray(zo).reshape(-1, zo.shape[-1]))
        z_action.append(np.asarray(za).reshape(-1, za.shape[-1]))
        for k in attrs:
            attrs[k].append(np.asarray(batch[k]).reshape(-1))
    attrs = {k: np.concatenate(v) for k, v in attrs.items()}
    attrs["ep_idx"] = attrs["ep_idx"].astype(np.int64)
    return np.concatenate(z_obs), np.concatenate(z_action), attrs


def _timesteps(ep_idx):
    """Within-episode time index; requires each episode's frames to be contiguous and in order."""
    t = np.empty(len(ep_idx), np.int64)
    bounds = np.r_[0, np.flatnonzero(np.diff(ep_idx)) + 1, len(ep_idx)]
    seen = set()
    for a, b in zip(bounds[:-1], bounds[1:]):
        e = int(ep_idx[a])
        assert e not in seen, "episode frames not contiguous -- the dataloader must be sequential"
        seen.add(e)
        t[a:b] = np.arange(b - a)
    return t


def main(_):
    assert FLAGS.obs_ckpt is not None and FLAGS.action_ckpt is not None, "ksg needs --obs_ckpt and --action_ckpt"

    # Initialize experimental jax compilation cache
    compilation_cache.compilation_cache.set_cache_dir(os.path.expanduser("~/.jax_compilation_cache"))

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    obs_alg, obs_state, _, _ = load_checkpoint(FLAGS.obs_ckpt)
    action_alg, action_state, _, _ = load_checkpoint(FLAGS.action_ckpt)
    # Same config lookup order as quality_estimators.get_dataset_and_score_fn.
    config = quality_estimators.load_config(FLAGS.obs_action_ckpt, FLAGS.obs_ckpt, FLAGS.action_ckpt)

    if FLAGS.path is not None:
        tf.io.gfile.makedirs(FLAGS.path)

    # One dataset at a time: each gets its own reference set, and ep_idx can't collide across datasets.
    for ds_name in config.dataloader.datasets.keys():
        # batch_size=1 then rebatch without drop_remainder, so the dataloader drops no frames.
        ds, _ = quality_estimators.get_dataset(
            config, batch_size=1, repeat=1, datasets=[ds_name], split_for_jax=False, sequential=True
        )
        ds = ds.unbatch().batch(FLAGS.batch_size)

        rng = jax.random.key(FLAGS.seed)
        z_obs, z_action, attrs = _encode(ds, obs_alg, obs_state, action_alg, action_state, rng)
        ep_idx = attrs["ep_idx"]
        t = _timesteps(ep_idx)
        n = len(ep_idx)

        ref_idx = np.random.RandomState(FLAGS.seed).choice(n, size=min(FLAGS.ref_size, n), replace=False)
        I_coord = quality_estimators.ksg_reference_estimator(
            z_obs,
            z_action,
            ep_idx,
            ref_idx,
            ks=quality_estimators.KSG_KS,
            exclude_same_episode=FLAGS.exclude_same_episode,
            block=FLAGS.batch_size,
        )

        ds_scores = quality_estimators._aggregate_stats(I_coord, attrs)
        ds_scores["per_frame"] = {"ep_idx": ep_idx, "t": t, "quality": attrs["quality_score"], "I_coord": I_coord}
        pprint.pprint({ds_name: {k: v for k, v in ds_scores.items() if k != "per_frame"}})

        if FLAGS.path is not None:
            with tf.io.gfile.GFile(os.path.join(FLAGS.path, ds_name + ".pkl"), "wb") as f:
                pickle.dump(ds_scores, f)

        r = quality_estimators.report_quality(ds_scores, ds_name, FLAGS.path)
        if r is not None:
            idxs = list(ds_scores["ep_idx"].keys())
            sp = stats.spearmanr(
                [ds_scores["ep_idx"][i] for i in idxs], [ds_scores["quality_by_ep_idx"][i] for i in idxs]
            )
            print(
                f"{ds_name}: {n} frames, {len(idxs)} episodes   "
                f"spearman(S_coord, quality) = {sp.statistic:+.3f} (p={sp.pvalue:.3f})   "
                f"pearson = {r.statistic:+.3f} (p={r.pvalue:.3f})"
            )


if __name__ == "__main__":
    app.run(main)
