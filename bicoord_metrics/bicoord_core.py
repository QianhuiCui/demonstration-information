import numpy as np
import yaml


def _slice(bounds) -> slice:
    return slice(bounds[0], bounds[1])


def _load_task_config(path: str, task: str) -> dict:
    with open(path, "r") as f:
        all_tasks = yaml.safe_load(f)
    assert task in all_tasks, f"Unknown task '{task}'. Options: {list(all_tasks.keys())}"
    cfg = all_tasks[task]
    # PyYAML's safe_load only parses e.g. "5.0e-3" as a float, not "5e-3" (no decimal point) --
    # cast explicitly so a YAML typo fails loudly here instead of deep inside numpy.
    for key in ("pos_eps", "grasp_threshold", "grasp_threshold_scale"):
        if key in cfg:
            cfg[key] = float(cfg[key])
    return cfg


def _movement_mask(pos: np.ndarray, pos_eps: float) -> np.ndarray:
    moving = np.linalg.norm(np.diff(pos, axis=0), axis=-1) > pos_eps
    return np.concatenate([[False], moving])  # no previous timestep to compare the first frame to.


def _holding_mask(palm_pos: np.ndarray, grasp_ref_slices: list, threshold) -> np.ndarray:
    # threshold may be a scalar or a per-timestep array (e.g. scaled by box size).
    dists = [np.linalg.norm(palm_pos - obs_slice, axis=-1) for obs_slice in grasp_ref_slices]
    min_dist = np.min(np.stack(dists, axis=0), axis=0)
    return min_dist < threshold


def _episode_masks(policy_obs: np.ndarray, task_cfg: dict):
    """Computes the per-timestep `dist` and `both_moving` arrays shared by the trajectory-level
    metrics and the windowed STI curve, so the two stay consistent."""
    assert policy_obs.shape[-1] == task_cfg["obs_dim"], (
        f"Expected obs_dim {task_cfg['obs_dim']} from tasks.yaml, got {policy_obs.shape[-1]}. "
        "The obs layout for this task is likely wrong -- check tasks.yaml against env_cfg.py."
    )

    right_pos = policy_obs[:, _slice(task_cfg["right"]["ee_pos"])]
    left_pos = policy_obs[:, _slice(task_cfg["left"]["ee_pos"])]

    rel_dist = np.linalg.norm(left_pos - right_pos, axis=-1)
    dist = rel_dist / rel_dist[0]

    if "grasp_threshold_obs_idx" in task_cfg:
        threshold = policy_obs[0, task_cfg["grasp_threshold_obs_idx"]] * task_cfg["grasp_threshold_scale"]
    else:
        threshold = task_cfg["grasp_threshold"]

    right_refs = [policy_obs[:, _slice(b)] for b in task_cfg["right"]["grasp_ref"]]
    left_refs = [policy_obs[:, _slice(b)] for b in task_cfg["left"]["grasp_ref"]]

    m_right = _movement_mask(right_pos, task_cfg["pos_eps"]) | _holding_mask(right_pos, right_refs, threshold)
    m_left = _movement_mask(left_pos, task_cfg["pos_eps"]) | _holding_mask(left_pos, left_refs, threshold)
    return dist, m_left & m_right


def reduce_metrics(dist: np.ndarray, both_moving: np.ndarray, length: int) -> dict:
    return {
        "ARD": 100 * dist.mean(),
        "MRD": 100 * dist.min(),
        "SMP": 100 * both_moving.mean(),
        "STI": 100 * (both_moving * np.clip(1 - dist, 0, None)).mean(),
        "length": length,
    }


def compute_episode_metrics(policy_obs: np.ndarray, task_cfg: dict) -> dict:
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