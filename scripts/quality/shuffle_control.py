"""KSG shuffle control: is I(S_left; S_right) real, or just estimator bias?

Recomputes the KSG MI per batch twice -- once on the true (z_left, z_right) pairing,
once with z_right permuted along the batch axis (so the true MI is 0 by construction).
The estimator drops the additive psi(k)+psi(N) constant, so only the GAP (real - shuffled)
is calibrated: it estimates the actual MI. gap >> 0 => real dependence; gap ~ 0 => the
latents / estimator aren't capturing the coupling and nothing downstream is trustworthy.

  python scripts/quality/shuffle_control.py \
      --obs_ckpt=/home/qianhui/deminf/symdex_ckpts/boxLift/left_vae_20260902_161350/5000 \
      --action_ckpt=/home/qianhui/deminf/symdex_ckpts/boxLift/right_vae_20260902_161422/5000
"""
import jax
import jax.numpy as jnp
import numpy as np
import quality_estimators as qe
import tensorflow as tf
from absl import app, flags
from jax.scipy.special import digamma

from openx.utils.evaluate import load_checkpoint

FLAGS = flags.FLAGS
flags.DEFINE_string("obs_ckpt", None, "left-arm VAE checkpoint (…/left_vae_*/<step>).", required=True)
flags.DEFINE_string("action_ckpt", None, "right-arm VAE checkpoint (…/right_vae_*/<step>).", required=True)
flags.DEFINE_integer("batch_size", 1024, "KSG batch size.")
flags.DEFINE_integer("n_batches", 40, "Max batches to average over (runs fewer if the ds is shorter).")


def _l2(z):
    return jnp.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)


def _ksg(z_obs, z_action, ks):
    obs_d, act_d = _l2(z_obs), _l2(z_action)
    jknn = jnp.sort(jnp.maximum(obs_d, act_d), axis=-1)[:, ks]          # (B, |ks|)
    n_obs = jnp.sum(obs_d[:, :, None] < jknn[:, None, :], axis=1)
    n_act = jnp.sum(act_d[:, :, None] < jknn[:, None, :], axis=1)
    return jnp.mean(-jnp.mean(digamma(n_obs) + digamma(n_act), axis=-1))  # scalar: dataset MI


def main(_):
    tf.config.set_visible_devices([], "GPU")
    ks = np.arange(5, 8)  # k in {5,6,7}, same as estimate_quality's ksg

    obs_alg, obs_state, _, _ = load_checkpoint(FLAGS.obs_ckpt)
    act_alg, act_state, _, _ = load_checkpoint(FLAGS.action_ckpt)
    ds, _, _ = qe.get_dataset_and_score_fn("ksg", FLAGS.batch_size, FLAGS.obs_ckpt, FLAGS.action_ckpt)

    @jax.jit
    def batch_stats(batch, rng):
        r_o, r_a, r_p = jax.random.split(rng, 3)
        z_obs = obs_alg.predict(obs_state, batch, r_o)
        z_act = act_alg.predict(act_state, batch, r_a)
        perm = jax.random.permutation(r_p, z_act.shape[0])
        return _ksg(z_obs, z_act, ks), _ksg(z_obs, z_act[perm], ks)

    rng = jax.random.key(0)
    real, shuf = [], []
    for i, batch in enumerate(ds):
        if i >= FLAGS.n_batches:
            break
        batch = jax.tree.map(lambda x: x._numpy(), batch)
        r, s = batch_stats(batch, jax.random.fold_in(rng, i))
        real.append(float(r))
        shuf.append(float(s))

    real, shuf = np.array(real), np.array(shuf)
    gap = real.mean() - shuf.mean()
    print(f"\nbatches: {len(real)}   batch_size: {FLAGS.batch_size}")
    print(f"real KSG : {real.mean():+.4f}  ± {real.std():.4f}")
    print(f"shuf KSG : {shuf.mean():+.4f}  ± {shuf.std():.4f}")
    print(f"gap (real - shuf) = {gap:+.4f}   ({gap / (shuf.std() + 1e-9):.1f} sigma of the shuffled noise)")


if __name__ == "__main__":
    app.run(main)
