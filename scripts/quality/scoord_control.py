"""
S_coord controls -- per-episode shuffle gap + time-shift decay + idle diagnostic,
on the same KSG / LOO-reference plumbing as estimate_coord_score.py.

  python scripts/quality/scoord_controls.py \
      --obs_ckpt=/home/qianhui/deminf/symdex_ckpts/boxLift/left_vae_20260902_161350/5000 \
      --action_ckpt=/home/qianhui/deminf/symdex_ckpts/boxLift/right_vae_20260902_161422/5000 \
      --deltas=5,20,50,100 --n_perm=16 --idle_pct=20

Sections printed per task:
  [1] real S_coord per episode + spearman/pearson vs quality  (reproduces estimate_coord_score.py)
  [2] per-episode debiased gap  g(e) = mean_t I_real(e,t) - mean_t I_shuf(e,t),
      I_shuf = mean over --n_perm permutations of the reference cloud's RIGHT latents
      (true MI 0 by construction; the psi(k)+psi(N) offset cancels in the gap).
      spearman(g, quality) is the test: does the quality signal survive debiasing?
  [3] time-shift curve gap(Delta): within-episode circular roll of z_right vs z_left,
      reference cloud rolled the same way. gap(0)-gap(large) = temporal coupling;
      gap(large) asymptote = time-invariant shared structure / idle floor.
  [4] idle diagnostic: per-episode fraction of frames with BOTH arms ~static
      (EE-pos step below --idle_pct percentile); corr with quality and with S_coord,
      plus S_coord recomputed on non-idle frames only.
"""
import numpy as np
import jax
import jax.numpy as jnp
import tensorflow as tf
from absl import app, flags
from jax.scipy.special import digamma
from scipy import stats

from openx.data.dataloader import make_dataloader
from openx.utils.evaluate import load_checkpoint

FLAGS = flags.FLAGS
flags.DEFINE_string("obs_ckpt", None, "left-arm VAE checkpoint.", required=True)
flags.DEFINE_string("action_ckpt", None, "right-arm VAE checkpoint.", required=True)
flags.DEFINE_string("deltas", "5,20,50,100", "comma-sep |Delta| values for the time-shift sweep.")
flags.DEFINE_integer("n_perm", 16, "permutations averaged for the shuffled null.")
flags.DEFINE_float("idle_pct", 20.0, "per-arm percentile of EE-pos step below which an arm is 'static'.")

KS = (5, 6, 7)
REF_SIZE = 8192
BATCH_SIZE = 1024
EXCLUDE_SAME_EPISODE = True
SEED = 1


# --------------------------------------------------------------------------- data
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


def _maybe_pos(batch, path, dims=3):
    """Navigate to the concatenated state array and take the first `dims` cols (EE_POS,
    which the structure config lists before EE_ROT6D)."""
    d = batch
    try:
        for k in path:
            d = d[k]
        a = np.asarray(d)
    except (KeyError, TypeError, IndexError):
        return None
    a = a.reshape(-1, a.shape[-1])
    return a[:, :dims] if a.shape[-1] >= dims else None


def _encode_all(obs_alg, obs_state, act_alg, act_state, ds, rng):
    zL, zR, ep, ql, pL, pR = [], [], [], [], [], []
    have_pos = True

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
        l = _maybe_pos(batch, ("observation", "state"))   # left arm,  cols 0:3 = EE_POS
        r = _maybe_pos(batch, ("action",))                # right arm, cols 0:3 = EE_POS
        if l is None or r is None:
            have_pos = False
        else:
            pL.append(l)
            pR.append(r)

    zL, zR = np.concatenate(zL), np.concatenate(zR)
    ep = np.concatenate(ep).astype(np.int64)
    ql = np.concatenate(ql)
    if have_pos:
        pL, pR = np.concatenate(pL), np.concatenate(pR)
    else:
        print("  [!] raw EE_POS not in batch -- idle diagnostic falls back to latent speed")
        pL, pR = zL, zR
    return zL, zR, ep, ql, pL, pR


def _timesteps(ep):
    t = np.empty(len(ep), np.int64)
    bounds = np.r_[0, np.flatnonzero(np.diff(ep)) + 1, len(ep)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        t[a:b] = np.arange(b - a)
    return t


def _episode_bounds(ep):
    b = np.r_[0, np.flatnonzero(np.diff(ep)) + 1, len(ep)]
    return list(zip(b[:-1], b[1:]))


# ------------------------------------------------------------------------- scoring
@jax.jit
def _blk(qL, qR, qe, zL_r, zR_r, ep_r, ks):
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


def _score(zL, zR, ep, ref, ref_perm=None, block=1024):
    """Per-frame KSG contribution vs the LOO reference cloud, psi(k)+psi(N) dropped.
    ref_perm != None permutes the reference cloud's RIGHT latents -> shuffled null."""
    ks = jnp.asarray(KS)
    zL_r = jnp.asarray(zL[ref])
    zR_r = jnp.asarray(zR[ref] if ref_perm is None else zR[ref][ref_perm])
    ep_r = jnp.asarray(ep[ref])
    out = []
    for s in range(0, len(zL), block):
        out.append(np.asarray(_blk(
            jnp.asarray(zL[s:s + block]), jnp.asarray(zR[s:s + block]), jnp.asarray(ep[s:s + block]),
            zL_r, zR_r, ep_r, ks)))
    return np.concatenate(out)


def _roll_within_episodes(z, ep, delta):
    out = np.empty_like(z)
    for a, b in _episode_bounds(ep):
        T = b - a
        d = (T // 2) if delta == "half" else int(delta)
        out[a:b] = np.roll(z[a:b], d % T if T else 0, axis=0)
    return out


def _speed(X, ep):
    d = np.zeros(len(X))
    for a, b in _episode_bounds(ep):
        if b - a > 1:
            step = np.linalg.norm(np.diff(X[a:b], axis=0), axis=1)
            d[a + 1:b] = step
            d[a] = np.median(step)
        else:
            d[a] = np.inf
    return d


def _per_ep(vals, ep, uniq):
    return np.array([np.nanmean(vals[ep == e]) for e in uniq])


def _report(name, x, y):
    sp = stats.spearmanr(x, y)
    pr = stats.pearsonr(x, y)
    print(f"    {name:<34s} spearman {sp.statistic:+.3f} (p={sp.pvalue:.3g})   "
          f"pearson {pr.statistic:+.3f} (p={pr.pvalue:.3g})")


# ---------------------------------------------------------------------------- main
def main(_):
    tf.config.set_visible_devices([], "GPU")
    deltas = [int(x) for x in FLAGS.deltas.split(",") if x.strip()]

    obs_alg, obs_state, _, config = load_checkpoint(FLAGS.obs_ckpt)
    act_alg, act_state, _, _ = load_checkpoint(FLAGS.action_ckpt)
    task = next(iter(config.dataloader.datasets.keys()))

    ds = _build_loader(config)
    zL, zR, ep, ql, pL, pR = _encode_all(obs_alg, obs_state, act_alg, act_state, ds, jax.random.key(SEED))
    n = len(ep)
    uniq = np.unique(ep)
    q_by_ep = np.array([float(ql[ep == e][0]) for e in uniq])
    ref = np.random.RandomState(SEED).choice(n, size=min(REF_SIZE, n), replace=False)

    print(f"\n==================  {task}: {n} frames, {len(uniq)} episodes  ==================")

    # [1] real S_coord ------------------------------------------------------------
    I_real = _score(zL, zR, ep, ref)
    s_by_ep = _per_ep(I_real, ep, uniq)
    print("\n[1] real S_coord vs quality (per episode)")
    _report("S_coord vs quality", s_by_ep, q_by_ep)

    # [2] per-episode shuffle gap ----------------------------------------------------
    rs = np.random.RandomState(SEED + 1)
    acc = np.zeros(n)
    for _p in range(FLAGS.n_perm):
        acc += _score(zL, zR, ep, ref, ref_perm=rs.permutation(len(ref)))
    I_shuf = acc / FLAGS.n_perm
    gap = I_real - I_shuf
    gap_by_ep = _per_ep(gap, ep, uniq)
    print(f"\n[2] debiased gap  (n_perm={FLAGS.n_perm})")
    print(f"    dataset mean:  real {I_real.mean():+.4f}   shuf {I_shuf.mean():+.4f}   "
          f"gap {gap.mean():+.4f}")
    _report("gap vs quality", gap_by_ep, q_by_ep)
    _report("(compare) raw S_coord vs quality", s_by_ep, q_by_ep)

    # [3] time-shift sweep -------------------------------------------------------------
    null = I_shuf.mean()
    grid = [0] + [s * d for d in deltas for s in (1, -1)] + ["half"]
    print(f"\n[3] time-shift  gap(Delta) = mean_frame I(roll z_right by Delta) - shuf_null")
    curve = {}
    for delta in grid:
        Id = _score(zL, _roll_within_episodes(zR, ep, delta), ep, ref)
        curve[str(delta)] = Id.mean() - null
        print(f"    Delta={str(delta):>5s}   gap {curve[str(delta)]:+.4f}")
    g0, gh = curve["0"], curve["half"]
    if g0 != 0:
        print(f"    temporal fraction  (gap(0)-gap(half))/gap(0) = {(g0 - gh) / g0:.2f}")

    # [4] idle diagnostic -----------------------------------------------------------
    sL, sR = _speed(pL, ep), _speed(pR, ep)
    thrL = np.percentile(sL[np.isfinite(sL)], FLAGS.idle_pct)
    thrR = np.percentile(sR[np.isfinite(sR)], FLAGS.idle_pct)
    both_static = (sL <= thrL) & (sR <= thrR)
    idle_frac = np.array([both_static[ep == e].mean() for e in uniq])
    print(f"\n[4] idle diagnostic  (both arms below p{FLAGS.idle_pct:g} EE step; "
          f"mean both-static frac = {both_static.mean():.3f})")
    _report("idle_frac vs quality", idle_frac, q_by_ep)
    _report("idle_frac vs S_coord", idle_frac, s_by_ep)

    keep = ~both_static
    zLk, zRk, epk = zL[keep], zR[keep], ep[keep]
    uk = np.unique(epk)
    refk = np.random.RandomState(SEED).choice(len(epk), size=min(REF_SIZE, len(epk)), replace=False)
    I_trim = _score(zLk, zRk, epk, refk)
    s_trim = _per_ep(I_trim, epk, uk)
    q_trim = np.array([q_by_ep[list(uniq).index(e)] for e in uk])
    _report("S_coord(non-idle frames) vs quality", s_trim, q_trim)


if __name__ == "__main__":
    app.run(main)
