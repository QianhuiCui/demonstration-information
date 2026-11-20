import functools
import json
import os
from typing import List

import jax
import numpy as np
import tensorflow as tf
import tqdm
from jax import numpy as jnp
from jax.scipy.special import digamma
from matplotlib import pyplot as plt
from ml_collections import ConfigDict
from scipy import stats as scipy_stats

from openx.data.dataloader import make_dataloader
from openx.utils.evaluate import load_checkpoint

AGGREGATION_KEYS = [
    "ep_idx",
    "quality_score",
    # "quality_score_continuous", # Enable only if using RoboCrowd.
    "dataset_id",
]
KSG_KS = np.arange(5, 8) 

"""
Quality Estimators
"""


def _l2_dists(z):
    return jnp.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)  # (B, B)


def _knn(z, ks):
    # Faster implementations exist, but this is OK for now since we are streaming.
    dist = _l2_dists(z)
    return jnp.sort(dist, axis=-1)[:, ks]


def ksg_estimator(batch, rng, ks, obs_alg, obs_state, action_alg, action_state):
    # Get the state and action encoding
    obs_rng, action_rng = jax.random.split(rng)
    z_obs = obs_alg.predict(obs_state, batch, obs_rng)
    z_action = action_alg.predict(action_state, batch, action_rng)

    obs_dist = _l2_dists(z_obs)
    action_dist = _l2_dists(z_action)

    # Use the InfNorm on Z
    joint_dist = jnp.maximum(obs_dist, action_dist)
    joint_knn_dists = jnp.sort(joint_dist, axis=-1)[:, ks]

    obs_count = jnp.sum(obs_dist[:, :, None] < joint_knn_dists[:, None, :], axis=1)
    action_count = jnp.sum(action_dist[:, :, None] < joint_knn_dists[:, None, :], axis=1)

    return -jnp.mean(digamma(obs_count) + digamma(action_count), axis=-1)


def biksg_estimator(batch, rng, ks, obs_alg, obs_state, action_alg, action_state):
    # Get the state and action encoding
    obs_rng, action_rng = jax.random.split(rng)
    z_obs = obs_alg.predict(obs_state, batch, obs_rng)
    z_action = action_alg.predict(action_state, batch, action_rng)
    z_joint = jnp.concatenate((z_obs, z_action), axis=-1)

    joint_knn_dists = _knn(z_joint, ks)  # (B, k)

    # Now we need counts for each of the marginals, omitting the self count.
    obs_count = jnp.sum(_l2_dists(z_obs)[:, :, None] <= joint_knn_dists[:, None, :], axis=1) - 1
    action_count = jnp.sum(_l2_dists(z_action)[:, :, None] <= joint_knn_dists[:, None, :], axis=1) - 1

    return -jnp.mean(jnp.log(obs_count) + jnp.log(action_count), axis=-1)


def kl_estimator(
    batch,
    rng,
    ks,
    obs_alg,
    obs_state,
    action_alg,
    action_state,
    obs_action_alg=None,
    obs_action_state=None,
    obs_weight: float = 1.0,
):
    obs_rng, action_rng, joint_rng = jax.random.split(rng, num=3)
    z_obs = obs_alg.predict(obs_state, batch, obs_rng)
    z_action = action_alg.predict(action_state, batch, action_rng)
    if obs_action_alg is not None:
        z_joint = obs_action_alg.predict(obs_action_state, batch, joint_rng)
    else:
        z_joint = jnp.concatenate((z_obs, z_action), axis=-1)

    # I(S;A) = H(S) + H(A) - H(S, A)
    h_obs = z_obs.shape[-1] * jnp.log(_knn(z_obs, ks))  # (B, k)
    h_action = z_action.shape[-1] * jnp.log(_knn(z_action, ks))  # (B, k)
    h_joint = z_joint.shape[-1] * jnp.log(_knn(z_joint, ks))  # (B, k)

    return jnp.mean(obs_weight * h_obs + h_action - h_joint, axis=-1)


def nce_estimator(batch, rng, obs_action_alg, obs_action_state):
    return obs_action_alg.predict(obs_action_state, batch, rng)


def vip_estimator(batch, rng, obs_alg, obs_state):
    return obs_alg.predict(obs_state, batch, rng)


def mine_estimator(batch, rng, alg, state):
    return alg.predict(state, batch, rng)


def l2_loss_estimator(batch, rng, alg, state):
    pred = alg.predict(state, batch, rng)
    loss = jnp.square(pred - batch["action"]).sum(axis=-1)  # (B, T, D) --> (B, T)
    return -jnp.sum(loss * batch["mask"], axis=-1) / (jnp.sum(batch["mask"], axis=-1))


def stddev_estimator(batch, rng, alg, state):
    # We need to get the prediction and the stddev
    _, stddev = alg.predict_with_confidence(state, batch, rng)
    return stddev  # For entropy estimators, higher stddev is supposedly good


def inverse_stddev_estimator(batch, rng, alg, state):
    _, stddev = alg.predict_with_confidence(state, batch, rng)
    return -stddev  # For entropy estimators, higher stddev is supposedly good


def compatibility_estimator(batch, rng, alg, state):
    # We need to get the prediction and the stddev
    pred, stddev = alg.predict_with_confidence(state, batch, rng)
    loss = jnp.square(pred - batch["action"]).sum(axis=-1)  # (B, T, D) --> (B, T)
    loss = jnp.sum(loss * batch["mask"], axis=-1) / (jnp.sum(batch["mask"], axis=-1))
    eta = 0.05
    lambd = 4
    return jnp.where(stddev < eta, 1 - jnp.minimum(loss / lambd, 1.0), 1.0)


def random_estimator(batch, rng):
    # determine the batch size somehow
    batch_size = jax.tree.leaves(batch["observation"])[0].shape[0]
    return jax.random.uniform(rng, shape=(batch_size,))


def get_dataset_and_score_fn(
    estimator: str,
    batch_size: int | None = None,
    obs_ckpt: str | None = None,
    action_ckpt: str | None = None,
    obs_action_ckpt: str | None = None,
    datasets: List[str] | None = None,
):
    # First get the prediction function, and set the correct dataset parameters for it.
    if estimator == "random":
        pred_fn = random_estimator
        repeat, discard_fraction = 1, 0.99  # Remove almost all the data, don't repeat
    elif estimator == "ksg":
        assert obs_ckpt is not None and action_ckpt is not None
        obs_alg, obs_state, _, _ = load_checkpoint(obs_ckpt)
        action_alg, action_state, _, _ = load_checkpoint(action_ckpt)
        pred_fn = functools.partial(
            ksg_estimator,
            ks=KSG_KS,
            obs_alg=obs_alg,
            obs_state=obs_state,
            action_alg=action_alg,
            action_state=action_state,
        )
        repeat, discard_fraction = 4, 0.5
    elif estimator == "biksg":
        assert obs_ckpt is not None and action_ckpt is not None
        obs_alg, obs_state, _, _ = load_checkpoint(obs_ckpt)
        action_alg, action_state, _, _ = load_checkpoint(action_ckpt)
        pred_fn = functools.partial(
            biksg_estimator,
            ks=np.arange(2, 5),
            obs_alg=obs_alg,
            obs_state=obs_state,
            action_alg=action_alg,
            action_state=action_state,
        )
        repeat, discard_fraction = 4, 0.5
    elif estimator == "kl":
        assert obs_ckpt is not None and action_ckpt is not None
        obs_alg, obs_state, _, _ = load_checkpoint(obs_ckpt)
        action_alg, action_state, _, _ = load_checkpoint(action_ckpt)
        if obs_action_ckpt is not None:
            obs_action_alg, obs_action_state, _, _ = load_checkpoint(obs_action_ckpt)
        else:
            obs_action_alg, obs_action_state = None, None
        pred_fn = functools.partial(
            kl_estimator,
            ks=np.arange(1, 4),
            obs_alg=obs_alg,
            obs_state=obs_state,
            action_alg=action_alg,
            action_state=action_state,
            obs_action_alg=obs_action_alg,
            obs_action_state=obs_action_state,
            obs_weight=1.0,
        )
        repeat, discard_fraction = 4, 0.5
    elif estimator == "nce":
        assert obs_action_ckpt is not None
        obs_action_alg, obs_action_state, _, _ = load_checkpoint(obs_action_ckpt)
        pred_fn = functools.partial(nce_estimator, obs_action_alg=obs_action_alg, obs_action_state=obs_action_state)
        repeat, discard_fraction = 1, 0.0
    elif estimator == "vip":
        assert obs_ckpt is not None
        obs_alg, obs_state, _, _ = load_checkpoint(obs_ckpt)
        pred_fn = functools.partial(vip_estimator, obs_alg=obs_alg, obs_state=obs_state)
        repeat, discard_fraction = 1, 0.0
    elif estimator == "l2":
        assert obs_ckpt is not None
        obs_alg, obs_state, _, _ = load_checkpoint(obs_ckpt)
        pred_fn = functools.partial(l2_loss_estimator, alg=obs_alg, state=obs_state)
        repeat, discard_fraction = 1, 0.0
    elif estimator == "stddev":
        assert obs_ckpt is not None
        obs_alg, obs_state, _, _ = load_checkpoint(obs_ckpt)
        pred_fn = functools.partial(stddev_estimator, alg=obs_alg, state=obs_state)
        repeat, discard_fraction = 1, 0.0
    elif estimator == "inv_stddev":
        assert obs_ckpt is not None
        obs_alg, obs_state, _, _ = load_checkpoint(obs_ckpt)
        pred_fn = functools.partial(inverse_stddev_estimator, alg=obs_alg, state=obs_state)
        repeat, discard_fraction = 1, 0.0
    elif estimator == "compatibility":
        assert obs_ckpt is not None
        obs_alg, obs_state, _, _ = load_checkpoint(obs_ckpt)
        pred_fn = functools.partial(compatibility_estimator, alg=obs_alg, state=obs_state)
        repeat, discard_fraction = 1, 0.0
    elif estimator == "mine":
        assert obs_action_ckpt is not None
        obs_action_alg, obs_action_state, _, _ = load_checkpoint(obs_action_ckpt)
        pred_fn = functools.partial(mine_estimator, alg=obs_action_alg, state=obs_action_state)
        repeat, discard_fraction = 1, 0.0
    else:
        raise ValueError("Invalid estimator type called: " + estimator)

    def wrapped_pred_fn(batch, rng):
        return pred_fn(batch, rng), {k: batch[k] for k in AGGREGATION_KEYS}

    # Try to load the config from one of the models
    path = next(iter(p for p in (obs_action_ckpt, obs_ckpt, action_ckpt) if p is not None))
    if os.path.basename(os.path.normpath(path)).isdigit():
        path = os.path.dirname(os.path.normpath(path))
    with tf.io.gfile.GFile(tf.io.gfile.join(path, "config.json"), "r") as f:
        config = json.load(f)
        config = ConfigDict(config)

    # Next get the dataloader from the config, set the parameters appropriately.
    dataloader_config = config.dataloader.to_dict()
    dataloader_config["repeat"] = repeat
    dataloader_config["discard_fraction"] = discard_fraction
    if batch_size is not None:
        dataloader_config["batch_size"] = batch_size
    assert dataloader_config["batch_size"] % jax.device_count() == 0

    # Subset the datasets if desired
    if datasets is not None:
        dataloader_config["datasets"] = {k: dataloader_config["datasets"] for k in datasets}
    for ds in dataloader_config["datasets"]:
        if "val_split" in dataloader_config["datasets"][ds]:
            del dataloader_config["datasets"][ds]["val_split"]

    if dataloader_config.get("goal_conditioning", None) is not None:
        dataloader_config["goal_conditioning"] = "last"  # Always set to last frame for preds.
    # Don't recompute statistics for quality estimation, they should already be there.
    dataloader_config["recompute_statistics"] = False

    # Add aggregation keys to structure.
    structure = config.structure.to_dict()
    for k in AGGREGATION_KEYS:
        if k == "dataset_id":
            continue
        structure[k] = None  # Add to the structure with empty field so we include it.

    ds, _, _, dataset_ids = make_dataloader(**dataloader_config, structure=structure, split_for_jax=True)

    return ds, wrapped_pred_fn, dataset_ids


def _aggregate_stats(stats, attrs):
    scores = dict()
    for attr_name, attr in attrs.items():
        scores[attr_name] = dict()
        for attr_val in np.unique(attr).tolist():
            scores[attr_name][attr_val] = np.mean(stats[attr == attr_val])

    if "ep_idx" in attrs and "quality_score" in attrs:
        scores["quality_by_ep_idx"] = dict()
        for ep_idx in np.unique(attrs["ep_idx"]):
            scores["quality_by_ep_idx"][ep_idx] = np.mean(attrs["quality_score"][attrs["ep_idx"] == ep_idx])

    if "ep_idx" in attrs and "quality_score_continuous" in attrs:
        scores["quality_continuous_by_ep_idx"] = dict()
        for ep_idx in np.unique(attrs["ep_idx"]):
            scores["quality_continuous_by_ep_idx"][ep_idx] = np.mean(
                attrs["quality_score_continuous"][attrs["ep_idx"] == ep_idx]
            )

    return scores


def estimate_quality(ds, pred_fn, dataset_ids, rng):
    """
    Note that this is a separate function to allow for jitting of pred_fn and sharding of the ds.
    """
    stats = []
    attributes = {attr: [] for attr in AGGREGATION_KEYS}

    for i, batch in tqdm.tqdm(enumerate(ds), dynamic_ncols=True):
        rng = jax.random.fold_in(rng, i)
        pred, attrs = pred_fn(batch, rng)
        no_nan_idx = ~jnp.isnan(pred)
        stats.append(pred[no_nan_idx])
        for k, v in attrs.items():
            attributes[k].append(v[no_nan_idx])

    # Concatenate everything
    stats = jnp.concatenate(stats, axis=0)
    attributes = {k: jnp.concatenate(v, axis=0) for k, v in attributes.items()}

    # Normalize
    stats = jnp.clip(stats, a_min=jnp.percentile(stats, 1), a_max=jnp.percentile(stats, 99))
    stats = (stats - jnp.mean(stats)) / jnp.std(stats)

    # Do only a single conversion to numpy
    stats = np.array(stats)
    attributes = {k: np.array(v) for k, v in attributes.items()}

    # Now aggregate per dataset
    scores = dict()
    for ds_name, ds_id in dataset_ids.items():
        mask = attributes["dataset_id"] == ds_id
        scores[ds_name] = _aggregate_stats(stats[mask], {k: v[mask] for k, v in attributes.items()})

    return scores


"""
S_coord helpers: per-frame KSG against a fixed reference set (used by estimate_quality_raw.py)
"""


def _l2_cross_dists(z, ref):
    return jnp.linalg.norm(z[:, None, :] - ref[None, :, :], axis=-1)  # (Q, R)


def ksg_from_dists(obs_dist, action_dist, ks):
    """Same KSG computation as ksg_estimator, but from precomputed (Q, N) distance matrices.

    The query must appear in its own row at distance 0 (as with in-batch dists), so sorted
    index k is the k-th neighbour and the counts include the query itself (= n + 1).
    """
    joint_dist = jnp.maximum(obs_dist, action_dist)  # InfNorm on Z
    joint_knn_dists = jnp.sort(joint_dist, axis=-1)[:, ks]
    obs_count = jnp.sum(obs_dist[:, :, None] < joint_knn_dists[:, None, :], axis=1)
    action_count = jnp.sum(action_dist[:, :, None] < joint_knn_dists[:, None, :], axis=1)
    return -jnp.mean(digamma(obs_count) + digamma(action_count), axis=-1)


def ksg_reference_estimator(z_obs, z_action, ep_idx, ref_idx, ks=KSG_KS, exclude_same_episode=True, block=1024):
    """KSG score of every frame against one fixed reference subset of frames (not its batch).

    All frames share the same reference cloud, so scores are comparable across frames and keep
    their time order. The query frame is never taken from the reference set (with
    exclude_same_episode, neither is any frame of its episode); it is added back as its own
    0-distance neighbour so the result follows ksg_estimator's conventions exactly.

    z_obs, z_action: (N, D) numpy encodings; ep_idx: (N,); ref_idx: (R,) indices into the N frames.
    Returns (N,) numpy scores.
    """
    zo_r, za_r = jnp.asarray(z_obs[ref_idx]), jnp.asarray(z_action[ref_idx])
    ep_r, idx_r = jnp.asarray(ep_idx[ref_idx]), jnp.asarray(ref_idx)

    @jax.jit
    def _block(q_obs, q_action, q_ep, q_idx):
        obs_dist = _l2_cross_dists(q_obs, zo_r)
        action_dist = _l2_cross_dists(q_action, za_r)
        drop = q_idx[:, None] == idx_r[None, :]
        if exclude_same_episode:
            drop = drop | (q_ep[:, None] == ep_r[None, :])
        obs_dist = jnp.where(drop, jnp.inf, obs_dist)
        action_dist = jnp.where(drop, jnp.inf, action_dist)
        self_dist = jnp.zeros((q_obs.shape[0], 1), obs_dist.dtype)
        obs_dist = jnp.concatenate((self_dist, obs_dist), axis=1)
        action_dist = jnp.concatenate((self_dist, action_dist), axis=1)
        return ksg_from_dists(obs_dist, action_dist, ks)

    idx = np.arange(len(ep_idx))
    out = [
        np.asarray(_block(z_obs[s : s + block], z_action[s : s + block], ep_idx[s : s + block], idx[s : s + block]))
        for s in range(0, len(ep_idx), block)
    ]
    return np.concatenate(out)


def load_config(*ckpts):
    """Loads config.json from the first non-None checkpoint path (step dir or its parent)."""
    path = next(iter(p for p in ckpts if p is not None))
    if os.path.basename(os.path.normpath(path)).isdigit():
        path = os.path.dirname(os.path.normpath(path))
    with tf.io.gfile.GFile(tf.io.gfile.join(path, "config.json"), "r") as f:
        return ConfigDict(json.load(f))


def get_dataset(
    config,
    batch_size: int | None = None,
    repeat: int = 1,
    discard_fraction: float = 0.0,
    datasets: List[str] | None = None,
    split_for_jax: bool = True,
    sequential: bool = False,
):
    """Same dataloader setup as get_dataset_and_score_fn, with the sampling made explicit.
    Returns (ds, dataset_ids).

    sequential: no shuffling and no parallel map/read, so the frames of an episode stay
        contiguous and in time order (needed for per-timestep scores).
    """
    dataloader_config = config.dataloader.to_dict()
    dataloader_config["repeat"] = repeat
    dataloader_config["discard_fraction"] = discard_fraction
    if batch_size is not None:
        dataloader_config["batch_size"] = batch_size
    if split_for_jax:
        assert dataloader_config["batch_size"] % jax.device_count() == 0

    # Subset the datasets if desired
    if datasets is not None:
        dataloader_config["datasets"] = {k: dataloader_config["datasets"][k] for k in datasets}
    for ds in dataloader_config["datasets"]:
        if "val_split" in dataloader_config["datasets"][ds]:
            del dataloader_config["datasets"][ds]["val_split"]

    if sequential:
        dataloader_config.update(shuffle_size=0, num_parallel_calls=1, num_parallel_reads=1)
        for ds_config in dataloader_config["datasets"].values():
            ds_config.update(num_parallel_calls=1, num_parallel_reads=1)

    if dataloader_config.get("goal_conditioning", None) is not None:
        dataloader_config["goal_conditioning"] = "last"  # Always set to last frame for preds.
    # Don't recompute statistics for quality estimation, they should already be there.
    dataloader_config["recompute_statistics"] = False

    # Add aggregation keys to structure.
    structure = config.structure.to_dict()
    for k in AGGREGATION_KEYS:
        if k == "dataset_id":
            continue
        structure[k] = None  # Add to the structure with empty field so we include it.

    ds, _, _, dataset_ids = make_dataloader(**dataloader_config, structure=structure, split_for_jax=split_for_jax)
    return ds, dataset_ids


def report_quality(ds_scores, ds_name, path=None):
    """Same R print and <ds_name>_hist.png / _curve.png plots as estimate_quality.py.
    Returns the pearsonr result (None if the scores have no quality labels)."""
    if "quality_by_ep_idx" not in ds_scores:
        return None

    # Compute statistics
    idxs = list(ds_scores["ep_idx"].keys())
    x = np.array([ds_scores["quality_by_ep_idx"][idx] for idx in idxs])
    y = np.array([ds_scores["ep_idx"][idx] for idx in idxs])
    r = scipy_stats.pearsonr(x, y)
    print("R:", r)
    if path is None:
        return r

    # Histogram of per-episode scores, one per quality label
    _, ax = plt.subplots(1, 1, figsize=(3, 3))
    for v in np.sort(np.unique(x)):
        ax.hist(y[x == v], bins=20, alpha=0.5, label=f"Quality {v}")
    ax.legend()
    title = f"r={r[0]:.2f}"
    ax.set_title(title)
    plt.tight_layout()
    with tf.io.gfile.GFile(os.path.join(path, ds_name + "_hist.png"), "wb") as f:
        plt.savefig(f)

    # Now plot the quality curve
    plt.clf()
    sort_idx = np.argsort(y)  # Sorts from low score to high score episodes.
    rev_sorted_quality_labels = x[sort_idx][::-1]  # The sorted quality labels
    total_quality_labels = np.cumsum(rev_sorted_quality_labels)
    num_data_points = 1 + np.arange(total_quality_labels.shape[0])
    avg_quality_label = total_quality_labels / num_data_points
    # Finally, re-reverse to set the axes back to num data points removed.
    plt.plot(np.arange(avg_quality_label.shape[0]), avg_quality_label[::-1], label="method")
    # Plot the oracle strategy
    oracle_labls = np.cumsum(np.sort(x)[::-1]) / num_data_points
    plt.plot(np.arange(oracle_labls.shape[0]), oracle_labls[::-1], color="gray", label="oracle")
    plt.gca().hlines(np.mean(x), xmin=0, xmax=oracle_labls.shape[0], color="red", linestyles="dashed")

    plt.xlabel("Episodes Removed")
    plt.ylabel("Average Quality Label")
    plt.legend(frameon=False)
    plt.ylim(np.min(x), np.max(x))

    plt.title(title)
    plt.tight_layout()
    with tf.io.gfile.GFile(os.path.join(path, ds_name + "_curve.png"), "wb") as f:
        plt.savefig(f)
    plt.close("all")
    return r
