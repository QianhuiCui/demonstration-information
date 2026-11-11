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


# ---------------------------------------------------------------------------
# Object-centric proximity + motion coupling
# ---------------------------------------------------------------------------

def obj_dist(policy_obs, task_cfg):
    """Distance between the two manipulated objects (obj_pos[0] of each hand), normalized by its t=0 value."""
    pos = {side: policy_obs[:, _slice(task_cfg[side]["obj_pos"][0])] for side in ("left", "right")}
    dist = np.linalg.norm(pos["left"] - pos["right"], axis=-1)
    return dist / max(dist[0], 1e-6)


def hand_speeds(policy_obs, task_cfg):
    """Per-step end-effector speed of each hand -> (v_left, v_right); first step is 0."""
    speeds = []
    for side in ("left", "right"):
        ee = policy_obs[:, _slice(task_cfg[side]["ee_pos"])]
        speeds.append(np.concatenate([[0.0], np.linalg.norm(np.diff(ee, axis=0), axis=-1)]))
    return speeds[0], speeds[1]


def lagged_corr(a, b, max_lag):
    """Max positive Pearson corr between speed profiles a, b over lags in [-max_lag, max_lag].
    Returns (corr, lag); lag > 0 means a[t + lag] pairs with b[t], i.e. b (right) leads.
    Windows where either hand is static have no defined corr and count as 0.
    """
    best, best_lag = 0.0, 0
    for lag in range(-max_lag, max_lag + 1):
        x, y = (a[lag:], b[: len(b) - lag]) if lag >= 0 else (a[:lag], b[-lag:])
        if len(x) < 3 or x.std() < 1e-8 or y.std() < 1e-8:
            continue
        r = np.corrcoef(x, y)[0, 1]
        if r > best:
            best, best_lag = r, lag
    return best, best_lag


def compute_windowed_coordination(policy_obs, task_cfg, window: int, stride: int, max_lag: int):
    """Windowed object-centric STI, motion coupling and their product, on the same window grid
    as compute_windowed_sti. Returns dict of arrays: t_frac, sti_obj, coupling, lag, coord.
      sti_obj  = 100 * mean(both_active * clip(1 - obj_dist, 0))   (%)
      coupling = max lagged corr of the two hands' speed profiles   (0..1)
      coord    = sti_obj * coupling                                  (%)
    """
    length = len(policy_obs)
    window = min(window, length)
    _, both_active = _episode_masks(policy_obs, task_cfg)
    weighted = both_active * np.clip(1 - obj_dist(policy_obs, task_cfg), 0, None)
    v_left, v_right = hand_speeds(policy_obs, task_cfg)

    starts = np.arange(0, length - window + 1, stride)
    sti_obj, coupling, lag = [], [], []
    for s in starts:
        sl = slice(s, s + window)
        sti_obj.append(100 * weighted[sl].mean())
        c, l = lagged_corr(v_left[sl], v_right[sl], max_lag)
        coupling.append(c)
        lag.append(l)

    sti_obj, coupling = np.array(sti_obj), np.array(coupling)
    return {
        "t_frac": (starts + window / 2) / length,
        "sti_obj": sti_obj,
        "coupling": coupling,
        "lag": np.array(lag),
        "coord": sti_obj * coupling,
    }


def _hand_state(policy_obs, task_cfg):
    """Per hand: ee pos, its object pos, speed, and 'near its object' mask."""
    ee, obj, v, near = {}, {}, {}, {}
    for s in ("left", "right"):
        ee[s] = policy_obs[:, _slice(task_cfg[s]["ee_pos"])]
        obj[s] = policy_obs[:, _slice(task_cfg[s]["obj_pos"][0])]
        v[s] = np.concatenate([[0.0], np.linalg.norm(np.diff(ee[s], axis=0), axis=-1)])
        near[s] = np.linalg.norm(ee[s] - obj[s], axis=-1) < task_cfg[s]["grasp_threshold"]
    return ee, obj, v, near


def _asym_frames(policy_obs, task_cfg):
    """Per-frame asymmetry |v_L - v_R| / (v_L + v_R), the frames where it is defined
    (both hands near their object AND at least one moving), and the both-holding mask."""
    eps = task_cfg["pos_eps"]
    _, _, v, near = _hand_state(policy_obs, task_cfg)
    both_hold = near["left"] & near["right"]
    valid = both_hold & ((v["left"] > eps) | (v["right"] > eps))
    a = np.abs(v["left"] - v["right"]) / (v["left"] + v["right"] + 1e-9)
    return a, valid, both_hold


def role_asymmetry(policy_obs, task_cfg):
    """Episode-level role asymmetry.
      asym      = 100 * mean asymmetry over valid frames; 0 = hands move identically (parallel),
                  100 = one hand still while the other works (stabilize + manipulate).
                  NaN if the two hands never hold their objects at the same time.
      hold_frac = % of frames where both hands are near their objects (asym's denominator).
    Separates task coordination TYPE (matches volunteer ratings across tasks); within a task it
    is not a quality measure -- sequential/hesitant demos also look asymmetric.
    """
    a, valid, both_hold = _asym_frames(policy_obs, task_cfg)
    return {
        "asym": 100 * a[valid].mean() if valid.any() else np.nan,
        "hold_frac": 100 * both_hold.mean(),
    }


def compute_windowed_asym(policy_obs, task_cfg, window: int, stride: int):
    """(t_frac, asym_curve) on the same window grid as compute_windowed_sti.
    Windows without any valid frame are NaN (not 0), so they don't pull the average down."""
    a, valid, _ = _asym_frames(policy_obs, task_cfg)
    length = len(policy_obs)
    window = min(window, length)
    starts = np.arange(0, length - window + 1, stride)
    curve = np.array([
        100 * a[s : s + window][valid[s : s + window]].mean() if valid[s : s + window].any() else np.nan
        for s in starts
    ])
    return (starts + window / 2) / length, curve


def reduce_coordination(curves):
    """Episode-level scalars = mean over windows."""
    return {
        "STI_obj": curves["sti_obj"].mean(),
        "coupling": curves["coupling"].mean(),
        "coord": curves["coord"].mean(),
    }
