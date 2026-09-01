import os
import yaml
from typing import Any, Iterator, Tuple

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# Higher = better; folder name -> quality score. Matches the convention robomimic's builder
# uses for its "better"/"okay"/"worse" masks.
QUALITY_SCORES = {"expert": 3.0, "okay": 2.0, "poor": 1.0}
_TASK_YAML = os.path.join(os.path.dirname(__file__), "tasks.yaml")
VAL_FRACTION = 0.1
SPLIT_SEED = 0


def _load_builder_configs():
    with open(_TASK_YAML, "r") as f:
        tasks = yaml.safe_load(f)
    return [
        SymDexConfig(
            name=task_name,
            task=task_name,
            obs_dim=cfg["obs_dim"],
            action_dim=cfg["action_dim"],
            language_instruction=cfg["language_instruction"],
        )
        for task_name, cfg in tasks.items()
    ]


class SymDexConfig(tfds.core.BuilderConfig):
    def __init__(self, *, task: str, obs_dim: int, action_dim: int, language_instruction: str, **kwargs):
        super().__init__(**kwargs)
        self.task = task
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.language_instruction = language_instruction


class SymDex(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for symdex bimanual dexterous-hand trajectories."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}
    MANUAL_DOWNLOAD_INSTRUCTIONS = (
        "Point --manual_dir at a task's trajectory root (e.g. trajectories_symdex/boxLift), "
        "which must contain expert/okay/poor quality-tier subfolders, each holding *.h5 files."
    )


    BUILDER_CONFIGS = _load_builder_configs()

    def _info(self) -> tfds.core.DatasetInfo:
        cfg = self.builder_config
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {"state": tfds.features.Tensor(shape=(cfg.obs_dim,),
                                                               dtype=np.float32,
                                                               doc="Flat symdex `policy` observation vector",),
                                  "image": tfds.features.Image(shape=(128, 128, 3),
                                                               dtype=np.uint8,
                                                               encoding_format="jpeg",
                                                               doc="Agent camera RGB observation.",),}),
                            "action": tfds.features.Tensor(shape=(cfg.action_dim,),
                                                           dtype=np.float32,
                                                           doc="Flat joint-position action",),
                            "discount": tfds.features.Scalar(dtype=np.float32, doc="Always 1.0; not tracked"),
                            "reward": tfds.features.Scalar(dtype=np.float32, doc="1.0 on the terminal step of a success episode, else 0"),
                            "is_first": tfds.features.Scalar(dtype=np.bool_),
                            "is_last": tfds.features.Scalar(dtype=np.bool_),
                            "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                            "language_instruction": tfds.features.Text(),
                        }),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(doc="Path to the original .h5 file."),
                            "ep_idx": tfds.features.Scalar(dtype=np.int32),
                            "quality_score": tfds.features.Scalar(dtype=np.float32, doc="3=expert, 2=okay, 1=poor"),
                        }),
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        root = dl_manager.manual_dir
        train_paths, val_paths = [], []
        rng = np.random.RandomState(SPLIT_SEED)

        for quality_name, score in QUALITY_SCORES.items():
            split_dir = os.path.join(root, quality_name)
            if not tf.io.gfile.isdir(split_dir):
                continue
            files = sorted(tf.io.gfile.glob(os.path.join(split_dir, "*.h5")))
            is_val = rng.random_sample(len(files)) < VAL_FRACTION
            for file_path, val in zip(files, is_val):
                (val_paths if val else train_paths).append((file_path, score))

        assert train_paths or val_paths, f"No .h5 files found under {root}/{{{','.join(QUALITY_SCORES)}}}"
        return {"train": self._generate_examples(train_paths),
                "val": self._generate_examples(val_paths),}

    def _generate_examples(self, paths) -> Iterator[Tuple[str, Any]]:
        cfg = self.builder_config
        ep_idx = 0
        for file_path, quality_score in paths:
            with h5py.File(file_path, "r") as f:
                for episode_key in f.keys():
                    data = f[episode_key]["offline_data"]
                    policy_obs = data["observations"]["policy"][:].astype(np.float32)
                    vision_obs = data["observations"]["vision"][:]
                    actions = data["actions"][:].astype(np.float32)
                    terminals = data["terminals"][:]
                    length = policy_obs.shape[0]

                    assert policy_obs.shape[-1] == cfg.obs_dim, (f"{file_path}/{episode_key}: expected obs_dim {cfg.obs_dim}, got {policy_obs.shape[-1]}")
                    assert actions.shape[-1] == cfg.action_dim, (f"{file_path}/{episode_key}: expected action_dim {cfg.action_dim}, got {actions.shape[-1]}")

                    episode = []
                    for t in range(length):
                        episode.append(dict(observation=dict(state=policy_obs[t], image=vision_obs[t]),
                                            action=actions[t],
                                            discount=1.0,
                                            reward=1.0 if bool(terminals[t]) else 0.0,
                                            is_first=t == 0,
                                            is_last=t == length - 1,
                                            is_terminal=bool(terminals[t]),
                                            language_instruction=cfg.language_instruction,))
                    metadata = dict(file_path=file_path,
                                    ep_idx=ep_idx,
                                    quality_score=quality_score,)
                    ep_idx += 1
                    yield f"{os.path.basename(file_path)}_{episode_key}", dict(steps=episode, episode_metadata=metadata)
