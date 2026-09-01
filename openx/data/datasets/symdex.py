import os
import yaml
from typing  import Dict

import tensorflow as tf 

from openx.data.utils import RobotType, StateEncoding, rmat_to_rot6d

_TASKS_YAML = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "rlds", "symdex", "tasks.yaml"))


def _load_tasks(task: str):
    with open(_TASKS_YAML, "r") as f:
        tasks = yaml.safe_load(f)
    assert task in tasks, f"Unknown symdex task '{task}'. Options: {list(tasks.keys())}"
    return tasks[task]


def _slice(bounds):
    return slice(bounds[0], bounds[1])


def _slices(bounds):
    """Normalizes a slice spec into a list of slice() objects. Accepts either a single
    [start, end] pair (most fields) or a list of such pairs (for a field assembled from
    multiple disjoint ranges, e.g. boxLift's MISC after dropping redundant fields)."""
    if isinstance(bounds[0], int):
        bounds = [bounds]
    return [slice(b[0], b[1]) for b in bounds]


def _rot6d(flat_rotmat):
    return rmat_to_rot6d(tf.reshape(flat_rotmat, (-1, 3, 3)))


def symdex_dataset_transform(ep: Dict, task: str):
    s, a = (lambda cfg: (cfg["state_slices"], cfg["action_slices"]))(_load_tasks(task))
    obs = ep["observation"]["state"]
    raw_action = ep["action"]

    observation = {
        "image": {"agent": ep["observation"]["image"]},
        "state": {
            StateEncoding.EE_POS: tf.concat((obs[..., _slice(s["left_ee_pos"])], obs[..., _slice(s["right_ee_pos"])]), axis=-1),
            StateEncoding.EE_ROT6D: tf.concat((_rot6d(obs[..., _slice(s["left_ee_rot"])]), _rot6d(obs[..., _slice(s["right_ee_rot"])])), axis=-1),
            StateEncoding.HAND_POS: tf.concat((obs[..., _slice(s["left_hand"])], obs[..., _slice(s["right_hand"])]), axis=-1),
            StateEncoding.MISC: tf.concat([obs[..., sl] for sl in _slices(s["misc"])], axis=-1),
        },
    }
    action = {
        "desired_delta": {
            StateEncoding.JOINT_POS: tf.concat((raw_action[..., _slice(a["left_arm"])], raw_action[..., _slice(a["right_arm"])]), axis=-1),
            StateEncoding.HAND_POS: tf.concat((raw_action[..., _slice(a["left_hand"])], raw_action[..., _slice(a["right_hand"])]), axis=-1),
        },
    }

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.UF_ALLEGRO
    ep["ep_idx"] = ep["episode_metadata"]["ep_idx"]
    ep["quality_score"] = ep["episode_metadata"]["quality_score"]
    return ep


def symdex_lr_dataset_transform(ep: Dict, task: str):
    s = _load_tasks(task)["state_slices"]
    obs = ep["observation"]["state"]

    state_l = {
        "state": {
            StateEncoding.EE_POS:   obs[..., _slice(s["left_ee_pos"])],
            StateEncoding.EE_ROT6D: _rot6d(obs[..., _slice(s["left_ee_rot"])]),
            StateEncoding.HAND_POS: obs[..., _slice(s["left_hand"])],
        },
    }
    state_r = {
        "desired_delta": {
            StateEncoding.EE_POS:   obs[..., _slice(s["right_ee_pos"])],
            StateEncoding.EE_ROT6D: _rot6d(obs[..., _slice(s["right_ee_rot"])]),
            StateEncoding.HAND_POS: obs[..., _slice(s["right_hand"])],
        },
    }

    ep["observation"] = state_l
    ep["action"] = state_r
    ep["robot"] = RobotType.UF_ALLEGRO
    ep["ep_idx"] = ep["episode_metadata"]["ep_idx"]
    ep["quality_score"] = ep["episode_metadata"]["quality_score"]
    return ep