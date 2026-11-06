import numpy as np
import yaml


def _slice(bounds):
    return slice(bounds[0], bounds[1])


def _load_task_config(path: str, task: str):
    with open(path, "r") as f:
        all_tasks = yaml.safe_load(f)
    assert task in all_tasks, f"Unknown task '{task}'. Options: {list(all_tasks.keys())}"
    cfg = all_tasks[task]
    cfg["pos_eps"] = float(cfg["pos_eps"])
    for side in ("left", "right"):
        cfg[side]["grasp_threshold"] = float(cfg[side]["grasp_threshold"])
    return cfg


def _episode_masks(policy_obs, task_cfg, per_arm=False):
    assert policy_obs.shape[-1] == task_cfg["obs_dim"], (
        f"Expected obs_dim {task_cfg['obs_dim']} from tasks.yaml, got  {policy_obs.shape[-1]}."
    )
    ee, active = {}, {}
    for side in ("left", "right"):
        hand = task_cfg[side]
        ee[side] = policy_obs[:, _slice(hand["ee_pos"])]
        moving = np.linalg.norm(np.diff(ee[side], axis=0), axis=-1) > task_cfg["pos_eps"]
        moving= np.concatenate([[False], moving])  # first timestep has no diff
        obj_dist = np.min([np.linalg.norm(ee[side] - policy_obs[:, _slice(obj)], axis=-1) for obj in hand["obj_pos"]], axis=0)
        active[side] = moving | (obj_dist < hand["grasp_threshold"])

    rel_dist = np.linalg.norm(ee["left"] - ee["right"], axis=-1)
    dist = rel_dist / rel_dist[0]
    if per_arm:
        return dist, active["left"], active["right"]
    return dist, active["left"] & active["right"]


def reduce_metrics(dist: np.ndarray, both_moving: np.ndarray, length: int):
    return {
        "ARD": 100 * dist.mean(),
        "MRD": 100 * dist.min(),
        "SMP": 100 * both_moving.mean(),
        "STI": 100 * (both_moving * np.clip(1 - dist, 0, None)).mean(),
        "length": length,
    }


def compute_episode_metrics(policy_obs: np.ndarray, task_cfg: dict):
    dist, both_moving = _episode_masks(policy_obs, task_cfg)
    return reduce_metrics(dist, both_moving, len(policy_obs))


def compute_windowed_sti(dist: np.ndarray, both_moving: np.ndarray, window: int, stride: int):
    """Returns (t_frac, sti_curve): STI computed over sliding windows instead of over the whole
    episode, so a localized coordination dip isn't averaged away by the rest of the trajectory.
    t_frac is each window's center as a fraction of episode length, so curves from episodes of
    different lengths can be overlaid and averaged.
    """
    length = len(dist)
    weighted = both_moving * np.clip(1 - dist, 0, None)
    if length < window:
        return np.array([0.5]), np.array([100 * weighted.mean()])

    starts = np.arange(0, length - window + 1, stride)
    sti_curve = np.array([100 * weighted[s : s + window].mean() for s in starts])
    t_frac = (starts + window / 2) / length
    return t_frac, sti_curve