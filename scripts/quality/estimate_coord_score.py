"""
Stage-3 S_coord scorer -- raw per-timestep and per-trajectory I(S_left; S_right).

Same KSG estimator as quality_estimators.ksg_estimator (max-norm joint, k in {5,6,7},
-[psi(n_left)+psi(n_right)] averaged over k), but: no clip, no z-norm, keeps the
within-episode time index, and scores each frame against a fixed leave-one-episode-out
reference cloud so episodes can be scored in time order.

  python scripts/quality/estimate_coord_score.py \
      --obs_ckpt=/home/qianhui/deminf/symdex_ckpts/boxLift/left_vae_20260902_161350/5000 \
      --action_ckpt=/home/qianhui/deminf/symdex_ckpts/boxLift/right_vae_20260902_161422/5000 \
      --path=/home/qianhui/deminf/symdex_scores/boxLift/scoord_raw

Writes <path>/<task>.pkl:
    ep_idx             {ep_idx -> S_coord(tau)}          raw, = mean_t I_coord(t)
    quality_by_ep_idx  {ep_idx -> quality tier}
    quality_score      {tier -> mean raw I_coord}
    per_frame          {ep_idx, t, quality, I_coord}     raw arrays, for windowed analysis

Also writes <path>/<task>_hist.png and <path>/<task>_curve.png -- same plots as
estimate_quality.py (per-tier score histogram; filter curve vs oracle / random baseline).
"""
import os
import pickle
import pprint

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from absl import app, flags
from jax.scipy.special import digamma
from matplotlib import pyplot as plt
from scipy import stats

from openx.data.dataloader import make_dataloader
from openx.utils.evaluate import load_checkpoint

FLAGS = flags.FLAGS
flags.DEFINE_string("obs_ckpt", None, "left-arm VAE checkpoint (.../left_vae_*/<step>).", required=True)
flags.DEFINE_string("action_ckpt", None, "right-arm VAE checkpoint (.../right_vae_*/<step>).", required=True)
flags.DEFINE_string("path", None, "directory to save <task>.pkl; if unset, only print.", required=False)

KS = (5, 6, 7)                # same as quality_estimators.ksg_estimator
REF_SIZE = 8192              # fixed reference-cloud size (frames)
BATCH_SIZE = 1024            # encode batch size
EXCLUDE_SAME_EPISODE = True  # drop reference frames from the query frame's own episode
SEED = 1                     # reference subsample (only RNG here)


def _build_loader(config):
    dl = config.dataloader.to_dict()
    dl.update(shuffle_size=0, discard_fraction=0.0, repeat=1, batch_size=BATCH_SIZE, recompute_statistics=False)
    for d in dl["datasets"].values():
        d.pop("val_split", None)
    structure = config.structure.to_dict()
    structure["ep_idx"] = None
    structure["quality_score"] = None
    ds, _, _, _ = make_dataloader(**dl, structure=structure, split_for_jax=False)
    return ds


def _encode_all(obs_alg, obs_state, act_alg, act_state, ds, rng):
    zL, zR, ep, ql = [], [], [], []

    @jax.jit
    def enc(batch, key):
        k1, k2 = jax.random.split(key)
        return obs_alg.predict(obs_state, batch, k1), act_alg.predict(act_state, batch, k2)

    for i, batch in enumerate(ds):
        batch = jax.tree.map(lambda x: x._numpy(), batch)
        zl, zr = enc(batch, jax.random.fold_in(rng, i))
        zL.append(np.asarray(zl).reshape(-1, zl.shape[-1]))
        zR.append(np.asarray(zr).reshape(-1, zr.shape[-1]))
        ep.append(np.asarray(batch["ep_idx"]).reshape(-1))
        ql.append(np.asarray(batch["quality_score"]).reshape(-1))
    return np.concatenate(zL), np.concatenate(zR), np.concatenate(ep).astype(np.int64), np.concatenate(ql)


def _timesteps(ep):
    t = np.empty(len(ep), np.int64)
    bounds = np.r_[0, np.flatnonzero(np.diff(ep)) + 1, len(ep)]
    seen = set()
    for a, b in zip(bounds[:-1], bounds[1:]):
        e = int(ep[a])
        assert e not in seen, "episode frames not contiguous -- shuffle_size must be 0"
        seen.add(e)
        t[a:b] = np.arange(b - a)
    return t


def _score(zL, zR, ep, ref, block=1024):
    ks = jnp.asarray(KS)
    zL_r, zR_r, ep_r = jnp.asarray(zL[ref]), jnp.asarray(zR[ref]), jnp.asarray(ep[ref])

    @jax.jit
    def blk(qL, qR, qe):
        dL = jnp.linalg.norm(qL[:, None, :] - zL_r[None, :, :], axis=-1)
        dR = jnp.linalg.norm(qR[:, None, :] - zR_r[None, :, :], axis=-1)
        if EXCLUDE_SAME_EPISODE:
            same = qe[:, None] == ep_r[None, :]
            dL = jnp.where(same, jnp.inf, dL)
            dR = jnp.where(same, jnp.inf, dR)
        rho = jnp.sort(jnp.maximum(dL, dR), axis=-1)[:, ks]
        nL = jnp.sum(dL[:, :, None] < rho[:, None, :], axis=1)
        nR = jnp.sum(dR[:, :, None] < rho[:, None, :], axis=1)
        return -jnp.mean(digamma(nL + 1) + digamma(nR + 1), axis=-1)

    out = [np.asarray(blk(jnp.asarray(zL[s:s + block]), jnp.asarray(zR[s:s + block]), jnp.asarray(ep[s:s + block])))
           for s in range(0, len(zL), block)]
    return np.concatenate(out)


def _save_plots(path, task, x, y, r):
    """<task>_hist.png + <task>_curve.png -- identical construction to estimate_quality.py.

    x = per-episode quality tier, y = per-episode S_coord.
    """
    title = f"r={r:.2f}"

    _, ax = plt.subplots(1, 1, figsize=(3, 3))
    for v in np.sort(np.unique(x)):
        ax.hist(y[x == v], bins=20, alpha=0.5, label=f"Quality {v}")
    ax.legend()
    ax.set_title(title)
    plt.tight_layout()
    with tf.io.gfile.GFile(os.path.join(path, task + "_hist.png"), "wb") as f:
        plt.savefig(f)
    plt.clf()

    sort_idx = np.argsort(y)                                  # low -> high S_coord
    total = np.cumsum(x[sort_idx][::-1])                       # best-scored episodes first
    count = 1 + np.arange(total.shape[0])
    avg_quality = total / count
    plt.plot(np.arange(avg_quality.shape[0]), avg_quality[::-1], label="method")
    oracle = np.cumsum(np.sort(x)[::-1]) / count
    plt.plot(np.arange(oracle.shape[0]), oracle[::-1], color="gray", label="oracle")
    plt.gca().hlines(np.mean(x), xmin=0, xmax=oracle.shape[0], color="red", linestyles="dashed")
    plt.xlabel("Episodes Removed")
    plt.ylabel("Average Quality Label")
    plt.legend(frameon=False)
    plt.ylim(np.min(x), np.max(x))
    plt.title(title)
    plt.tight_layout()
    with tf.io.gfile.GFile(os.path.join(path, task + "_curve.png"), "wb") as f:
        plt.savefig(f)
    plt.close("all")


def main(_):
    tf.config.set_visible_devices([], "GPU")

    obs_alg, obs_state, _, config = load_checkpoint(FLAGS.obs_ckpt)
    act_alg, act_state, _, _ = load_checkpoint(FLAGS.action_ckpt)
    task = next(iter(config.dataloader.datasets.keys()))

    ds = _build_loader(config)
    zL, zR, ep, ql = _encode_all(obs_alg, obs_state, act_alg, act_state, ds, jax.random.key(SEED))
    t = _timesteps(ep)
    n = len(ep)

    ref = np.random.RandomState(SEED).choice(n, size=min(REF_SIZE, n), replace=False)
    I = _score(zL, zR, ep, ref)

    uniq = np.unique(ep)
    scores = {
        "ep_idx": {int(e): float(np.nanmean(I[ep == e])) for e in uniq},
        "quality_by_ep_idx": {int(e): float(ql[ep == e][0]) for e in uniq},
        "quality_score": {float(q): float(np.nanmean(I[ql == q])) for q in np.unique(ql)},
        "per_frame": {"ep_idx": ep, "t": t, "quality": ql, "I_coord": I},
    }

    pprint.pprint({task: {k: scores[k] for k in ("ep_idx", "quality_by_ep_idx", "quality_score")}})
    s_by_ep = np.array([scores["ep_idx"][int(e)] for e in uniq])       # y: S_coord per episode
    q_by_ep = np.array([scores["quality_by_ep_idx"][int(e)] for e in uniq])  # x: quality tier
    sp = stats.spearmanr(s_by_ep, q_by_ep)
    pr = stats.pearsonr(q_by_ep, s_by_ep)
    print("R:", pr)
    print(f"\n{task}: {n} frames, {len(uniq)} episodes   "
          f"spearman(S_coord, quality) = {sp.statistic:+.3f} (p={sp.pvalue:.3f})   "
          f"pearson = {pr.statistic:+.3f} (p={pr.pvalue:.3f})")

    if FLAGS.path is not None:
        tf.io.gfile.makedirs(FLAGS.path)
        with tf.io.gfile.GFile(os.path.join(FLAGS.path, task + ".pkl"), "wb") as f:
            pickle.dump(scores, f)
        _save_plots(FLAGS.path, task, q_by_ep, s_by_ep, pr[0])


if __name__ == "__main__":
    app.run(main)
