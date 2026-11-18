"""
Config for training beta VAEs on symdex per-arm state, for computing I(S_left; S_right).

Left-arm state is routed into the `observation.state` slot and right-arm state into the
`action` slot by `symdex_lr_dataset_transform`, so the existing (obs, action) VAE + KSG
plumbing computes mutual information between the two arms with no other code changes.
"""

import optax
import tensorflow as tf
from ml_collections import ConfigDict

from openx.algs.beta_vae import BetaVAE
from openx.data.datasets.symdex import symdex_lr_dataset_transform
from openx.data.utils import NormalizationType, StateEncoding
from openx.networks.components.mlp import MLP
from openx.networks.core import Concatenate, MultiDecoder, MultiEncoder
from openx.utils.spec import ModuleSpec


def get_config(config_str="pickObject,s,1,train"):
    task, config_type, seed, split = config_str.split(",")
    assert config_type in {"s", "a", "sa"}
    assert split in {"train", "all"}

    # "s" -> S_left  (observation.state slot)
    # "a" -> S_right (action slot)
    vae_keys = {
        "s": {"observation->state": None},
        "a": {"action": None},
        "sa": {"observation->state": None, "action": None},
    }[config_type]
    z_dim = {  # per-arm state is 25-dim (EE_POS 3 + EE_ROT6D 6 + HAND_POS 16)
        "s": 13,
        "a": 13,
        "sa": 26,
    }[config_type]
    seed = int(seed)

    # Left arm -> observation.state, right arm -> action.desired_delta.
    structure = {
        "observation": {
            "state": {
                StateEncoding.EE_POS: NormalizationType.GAUSSIAN,
                StateEncoding.EE_ROT6D: NormalizationType.GAUSSIAN,
            },
        },
        "action": {
            "desired_delta": {
                StateEncoding.EE_POS: NormalizationType.GAUSSIAN,
                StateEncoding.EE_ROT6D: NormalizationType.GAUSSIAN,
            },
        },
    }

    dataloader = dict(
        datasets={
            task: dict(
                path="/home/qianhui/deminf/symdex_rlds/sym_dex/{task}/1.0.0".format(task=task),
                train_split="train" if split == "train" else "train+val",
                val_split="val",
                transform=ModuleSpec.create(symdex_lr_dataset_transform, task=task),
            ),
        },
        n_obs=1,
        n_action=1,
        shuffle_size=100000,
        batch_size=256,
        recompute_statistics=False,
        cache=True,
        prefetch=tf.data.AUTOTUNE,
    )

    alg = ModuleSpec.create(
        BetaVAE,
        encoder=ModuleSpec.create(
            MultiEncoder,
            encoders=vae_keys,
            trunk=ModuleSpec.create(
                Concatenate, model=ModuleSpec.create(MLP, [128, 128], activate_final=True), flatten_time=True
            ),
        ),
        decoder=ModuleSpec.create(
            MultiDecoder,
            trunk=ModuleSpec.create(MLP, [128, 128], activate_final=True),
            decoders=vae_keys,
        ),
        z_dim=z_dim,
        beta=0.05,
    )

    lr_schedule = ModuleSpec.create(optax.constant_schedule, 0.0001)
    optimizer = ModuleSpec.create(optax.adam)
    if split == "all":
        steps = {"s": 40000, "a": 40000, "sa": 40000}[config_type]
    else:
        steps = {"s": 100000, "a": 100000, "sa": 100000}[config_type]

    return ConfigDict(
        dict(
            structure=structure,
            alg=alg,
            dataloader=dataloader,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            steps=steps,
            log_freq=500,
            val_freq=2500,
            save_freq=2500,
            val_steps=25,
            seed=seed,
        )
    )