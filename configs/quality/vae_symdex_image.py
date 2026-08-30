import optax
import tensorflow as tf
from ml_collections import ConfigDict

from openx.algs.beta_vae import BetaVAE
from openx.data.datasets.symdex import symdex_dataset_transform
from openx.data.utils import NormalizationType, StateEncoding
from openx.networks.components.mlp import MLP
from openx.networks.components.resnet import ResNet18, ResNet18Decoder
from openx.networks.core import Concatenate, MultiDecoder, MultiEncoder
from openx.utils.spec import ModuleSpec


def get_config(config_str="boxLift,s,1"):
    task, config_type, seed = config_str.split(",")
    seed = int(seed)
    assert config_type in {"s", "a", "sa"}

    encoder_keys = {
        "s": {"observation->state": None, "observation->image->agent": ModuleSpec.create(ResNet18, num_kp=64)},
        "a": {"action": None},
        "sa": {
            "observation->state": None,
            "observation->image->agent": ModuleSpec.create(ResNet18, num_kp=64),
            "action": None,
        }
    }[config_type]

    decoder_keys = {
        "s": {"observation->state": None, "observation->image->agent": ModuleSpec.create(ResNet18Decoder)},
        "a": {"action": None},
        "sa": {
            "observation->state": None,
            "observation->image->agent": ModuleSpec.create(ResNet18Decoder),
            "action": None,
        },
    }[config_type]

    z_dim = {
        "s": 32,   # ~half of the corrected raw 63-dim state (6+12+32+13)
        "a": 22,   # ~half of the raw 44-dim action (unaffected by the MISC fix)
        "sa": 54,  # = 32 + 22
    }[config_type]

    weights = {
        "s": {"observation->state": 1.0, "observation->image->agent": 1 / 200},
        "a": {"action": 1.0},
        "sa": {"observation->state": 1.0, "observation->image->agent": 1 / 200, "action": 1.0},
    }[config_type]

    structure = {
        "observation": {
            "state": {
                StateEncoding.EE_POS: NormalizationType.GAUSSIAN,
                StateEncoding.EE_ROT6D: NormalizationType.GAUSSIAN,
                StateEncoding.HAND_POS: NormalizationType.GAUSSIAN,
                StateEncoding.MISC: NormalizationType.GAUSSIAN,
            },
            "image": {"agent": (128, 128)},
        },
        "action": {
            "desired_delta": {
                StateEncoding.JOINT_POS: NormalizationType.GAUSSIAN,
                StateEncoding.HAND_POS: NormalizationType.GAUSSIAN,
            },
        },
    }

    dataloader = dict(
        datasets={
            task: dict(
                path="path/to/symdex_rlds/{task}/1.0.0".format(task=task),  # fill in after tfds build
                train_split="train",
                val_split="val",
                transform=ModuleSpec.create(symdex_dataset_transform, task=task),
            ),
        },
        n_obs=1,
        n_action=1,
        augment_kwargs=dict(scale_range=(0.85, 0.95), aspect_ratio_range=None),
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
            encoders=encoder_keys,
            trunk=ModuleSpec.create(
                Concatenate, model=ModuleSpec.create(MLP, [512, 512], activate_final=True), flatten_time=True
            ),
        ),
        decoder=ModuleSpec.create(
            MultiDecoder,
            trunk=ModuleSpec.create(MLP, [512, 512], activate_final=True),
            decoders=decoder_keys,
        ),
        z_dim=z_dim,
        beta=0.01,
        weights=weights,
    )

    lr_schedule = ModuleSpec.create(optax.constant_schedule, 0.0001)
    optimizer = ModuleSpec.create(optax.adam)

    return ConfigDict(
        dict(
            structure=structure,
            alg=alg,
            dataloader=dataloader,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            steps=100000,
            log_freq=500,
            val_freq=2500,
            save_freq=10000,
            val_steps=25,
            seed=seed,
        )
    )