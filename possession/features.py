"""Per-frame evidence from the tracking: who controls the ball, and whether it is in play.

Three feature matrices, one for each question the models answer (models.py):

  team_ball   who has it, on frames where the ball is seen
  team_shape  who has it, from how the two teams stand and move; available on every frame, and the
              only evidence on the long ball gaps that make up nearly all of the missing frames
  dead        is the ball in play at all

The team matrices are antisymmetric by construction: every column is "home minus away", so swapping
the teams flips its sign. A logistic model with no intercept on such columns cannot learn a home
bias, which is what lets a model trained on other clubs apply to this one.

CHALLENGE.md's central rule is built in directly: "possession stays with the last team that had it".
last_sign and next_sign below say which team last controlled the ball and which team controls it
next. Frames of a ball in flight have no evidence of their own, and those two columns are what stop
a pass being cut in half at the moment the ball is nearest an opponent.
"""

import numpy as np

from possession.tracking import HOME, MAX_BALL_SPEED, MAX_PLAYER_SPEED, derived_speed

# Params
FPS = 25
# Control of the ball: a player close to it and moving with it.
CONTROL_RADIUS = 2.0
CONTROL_RELATIVE_SPEED = 3.0
CONTROL_THRESHOLD = 0.25
CONTROL_SMOOTH = 2
# Pairwise distances are the one expensive feature and change slowly: every 5th frame is plenty.
PAIRWISE_STRIDE = 5
PAIRWISE_CHUNK = 4000
DISTANCE_CAP = 30.0
VELOCITY_HALF_WIDTH = 2
# Decay constants, in frames, of evidence carried from the last and next sighting of a controller.
CARRY_FRAMES = (25, 125, 500)
SHAPE_WINDOWS = (12, 50, 125)
# Half-widths of the windows the dead-ball features average over.
DEAD_WINDOWS = (12, 50, 125)


def window_mean(values, low, high, blocks):
    """Mean of values[t+low .. t+high], NaN-aware, clipped to the block, NaN if nothing is finite."""
    out = np.full(len(values), np.nan, dtype=np.float64)
    for start, stop in blocks:
        window = values[start:stop].astype(np.float64)
        n = len(window)
        finite = np.isfinite(window)
        total = np.concatenate(([0.0], np.cumsum(np.where(finite, window, 0.0))))
        count = np.concatenate(([0], np.cumsum(finite)))
        index = np.arange(n)
        lo = np.clip(index + low, 0, n)
        hi = np.clip(index + high + 1, 0, n)
        members = count[hi] - count[lo]
        out[start:stop] = np.where(members > 0, (total[hi] - total[lo]) / np.maximum(members, 1),
                                   np.nan)
    return out


def _centred(values, half_width, blocks):
    return window_mean(values, -half_width, half_width, blocks)


def _zero_nan(values):
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _velocity(xy, blocks):
    """Per-axis velocity in m/s, from a central difference, NaN where an end is missing."""
    velocity = np.full(xy.shape, np.nan, dtype=np.float32)
    width = VELOCITY_HALF_WIDTH
    for start, stop in blocks:
        if stop - start <= 2 * width:
            continue
        velocity[start + width:stop - width] = (
            (xy[start + 2 * width:stop] - xy[start:stop - 2 * width]) * FPS / float(2 * width))
    return velocity


def _masked_stats(values):
    """(mean, std) along axis 1, ignoring NaN, NaN where a frame has nothing."""
    finite = np.isfinite(values)
    count = finite.sum(axis=1)
    safe = np.maximum(count, 1)
    filled = np.where(finite, values, 0.0).astype(np.float64)
    mean = filled.sum(axis=1) / safe
    variance = (filled ** 2).sum(axis=1) / safe - mean ** 2
    mean[count == 0] = np.nan
    variance[count == 0] = np.nan
    return mean, np.sqrt(np.maximum(variance, 0.0))


def _attack_direction(home_x, away_x, blocks, n_frames):
    """+1 where the home side attacks towards larger x, one value per block. 0 outside blocks."""
    direction = np.zeros(n_frames, dtype=np.float64)
    for start, stop in blocks:
        direction[start:stop] = 1.0 if np.nanmean(home_x[start:stop]) < np.nanmean(away_x[start:stop]) else -1.0
    return direction


def _nearest_opponent(xy, home_slots, away_slots, blocks):
    """(mean distance from home players to their nearest opponent, the same for away), per frame."""
    n_frames = xy.shape[0]
    home_out = np.full(n_frames, np.nan, dtype=np.float64)
    away_out = np.full(n_frames, np.nan, dtype=np.float64)
    for start, stop in blocks:
        frames = np.arange(start, stop, PAIRWISE_STRIDE)
        for chunk_start in range(0, len(frames), PAIRWISE_CHUNK):
            chunk = frames[chunk_start:chunk_start + PAIRWISE_CHUNK]
            home_xy = xy[chunk][:, home_slots]
            away_xy = xy[chunk][:, away_slots]
            home_xy = home_xy[:, np.isfinite(home_xy[:, :, 0]).any(axis=0)]
            away_xy = away_xy[:, np.isfinite(away_xy[:, :, 0]).any(axis=0)]
            if not home_xy.shape[1] or not away_xy.shape[1]:
                continue
            delta = home_xy[:, :, None, :] - away_xy[:, None, :, :]
            distance = np.sqrt((delta ** 2).sum(axis=3))
            distance = np.where(np.isfinite(distance), distance, np.inf)
            to_away = distance.min(axis=2)
            to_home = distance.min(axis=1)
            to_away = np.where(np.isfinite(to_away), to_away, np.nan)
            to_home = np.where(np.isfinite(to_home), to_home, np.nan)
            home_out[chunk] = _masked_stats(to_away)[0]
            away_out[chunk] = _masked_stats(to_home)[0]
        # Fill the frames between samples.
        index = np.arange(start, stop)
        for out in (home_out, away_out):
            known = np.isfinite(out[frames])
            if known.any():
                out[start:stop] = np.interp(index, frames[known], out[frames][known])
    return home_out, away_out


def _fill_sign(signs, blocks, forward):
    """(sign of the nearest nonzero entry looking back or ahead, its distance in frames)."""
    sign_out = np.zeros(len(signs), dtype=np.float64)
    age_out = np.full(len(signs), 1e6, dtype=np.float64)
    for start, stop in blocks:
        window = signs[start:stop]
        index = np.arange(len(window))
        if forward:
            source = np.maximum.accumulate(np.where(window != 0, index, -1))
            valid = source >= 0
            age = index - source
        else:
            source = np.minimum.accumulate(
                np.where(window != 0, index, len(window))[::-1])[::-1]
            valid = source < len(window)
            age = source - index
        sign_out[start:stop][valid] = window[source[valid]]
        age_out[start:stop][valid] = age[valid]
    return sign_out, age_out


def _gap_position(visible, blocks):
    """(frames since the ball was last seen, frames until it is next seen, gap length). 0 if seen."""
    since = np.zeros(len(visible), dtype=np.float64)
    until = np.zeros(len(visible), dtype=np.float64)
    length = np.zeros(len(visible), dtype=np.float64)
    for start, stop in blocks:
        window = visible[start:stop]
        edges = np.diff(np.concatenate(([0], (~window).astype(np.int8), [0])))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            since[start + a:start + b] = np.arange(1, b - a + 1)
            until[start + a:start + b] = np.arange(b - a, 0, -1)
            length[start + a:start + b] = b - a
    return since, until, length


class Features(object):
    """The three matrices and the columns the decoder wants directly."""

    def __init__(self, team_ball, team_shape, dead, visible, names):
        self.team_ball = team_ball
        self.team_shape = team_shape
        self.dead = dead
        self.visible = visible
        self.names = names


def build(match):
    """Features for one Match. Every matrix has one row per frame; rows outside blocks are unused."""
    xy, ball, team, blocks = match.xy, match.ball_xy, match.team, match.blocks
    length, width = match.pitch
    n_frames = xy.shape[0]
    home_slots = np.flatnonzero(team == HOME)
    away_slots = np.flatnonzero(team != HOME)
    visible = np.isfinite(ball[:, 0])

    speed = derived_speed(xy, blocks, MAX_PLAYER_SPEED)
    ball_speed = derived_speed(ball, blocks, MAX_BALL_SPEED).astype(np.float64)
    velocity = _velocity(xy, blocks)
    ball_velocity = _velocity(ball, blocks)

    # -- distances to the ball -----------------------------------------------------------------
    delta = xy - ball[:, None, :]
    distance = np.sqrt((delta ** 2).sum(axis=2))
    frames = np.arange(n_frames)

    def nearest(slots):
        table = np.where(np.isfinite(distance[:, slots]), distance[:, slots], np.inf)
        slot = table.argmin(axis=1)
        best = table[frames, slot].astype(np.float64)
        best[~np.isfinite(best)] = np.nan
        relative = velocity[frames, slots[slot]] - ball_velocity
        relative = np.sqrt((relative.astype(np.float64) ** 2).sum(axis=1))
        within = (table < 3.0).sum(axis=1).astype(np.float64)
        return best, relative, within

    home_d, home_rel, home_near = nearest(home_slots)
    away_d, away_rel, away_near = nearest(away_slots)

    def control(d, relative):
        relative = np.where(np.isfinite(relative), relative, CONTROL_RELATIVE_SPEED)
        return _zero_nan(np.exp(-d / CONTROL_RADIUS) * np.exp(-relative / CONTROL_RELATIVE_SPEED))

    home_control = _zero_nan(_centred(control(home_d, home_rel), CONTROL_SMOOTH, blocks))
    away_control = _zero_nan(_centred(control(away_d, away_rel), CONTROL_SMOOTH, blocks))

    signs = np.zeros(n_frames, dtype=np.int8)
    signs[(home_control > CONTROL_THRESHOLD) & (home_control >= away_control)] = 1
    signs[(away_control > CONTROL_THRESHOLD) & (away_control > home_control)] = -1
    last_sign, last_age = _fill_sign(signs, blocks, forward=True)
    next_sign, next_age = _fill_sign(signs, blocks, forward=False)

    # -- the shape of each team ----------------------------------------------------------------
    home_x, home_spread_x = _masked_stats(xy[:, home_slots, 0])
    away_x, away_spread_x = _masked_stats(xy[:, away_slots, 0])
    home_y, home_spread_y = _masked_stats(xy[:, home_slots, 1])
    away_y, away_spread_y = _masked_stats(xy[:, away_slots, 1])
    home_speed = _masked_stats(speed[:, home_slots])[0]
    away_speed = _masked_stats(speed[:, away_slots])[0]
    home_vx = _masked_stats(velocity[:, home_slots, 0])[0]
    away_vx = _masked_stats(velocity[:, away_slots, 0])[0]
    direction = _attack_direction(home_x, away_x, blocks, n_frames)
    home_gap, away_gap = _nearest_opponent(xy, home_slots, away_slots, blocks)

    # Most advanced outfield position along the attack: how far a side has pushed its last line.
    home_forward = xy[:, home_slots, 0] * direction[:, None]
    away_forward = xy[:, away_slots, 0] * -direction[:, None]
    home_front = np.nanmax(np.where(np.isfinite(home_forward), home_forward, -np.inf), axis=1)
    away_front = np.nanmax(np.where(np.isfinite(away_forward), away_forward, -np.inf), axis=1)
    home_front = np.where(np.isfinite(home_front), home_front, np.nan)
    away_front = np.where(np.isfinite(away_front), away_front, np.nan)

    # -- team_ball: home minus away, zero where the ball is not seen ---------------------------
    clipped = np.clip(away_d - home_d, -15.0, 15.0) / 15.0
    close1 = np.exp(-home_d) - np.exp(-away_d)
    close2 = np.exp(-home_d / 2.5) - np.exp(-away_d / 2.5)
    close3 = np.exp(-home_d / 6.0) - np.exp(-away_d / 6.0)
    slow = np.exp(-_zero_nan(ball_speed) / 5.0)
    control_diff = home_control - away_control
    side = np.sign(away_d - home_d)
    ball_side = ((ball[:, 0] / length - 0.5) * 2.0) * direction
    columns = [
        ("distance_diff", clipped),
        ("close_1m", close1),
        ("close_2_5m", close2),
        ("close_6m", close3),
        ("near_count", (home_near - away_near) / 5.0),
        ("control", control_diff),
        ("close_slow", close2 * slow),
        ("close_fast", close2 * (1.0 - slow)),
        ("side_12", _centred(side, 12, blocks)),
        ("side_25", _centred(side, 25, blocks)),
        ("side_50", _centred(side, 50, blocks)),
        ("side_past_25", window_mean(side, -25, -1, blocks)),
        ("side_future_25", window_mean(side, 1, 25, blocks)),
        ("control_25", _centred(control_diff, 25, blocks)),
        ("control_past", window_mean(control_diff, -25, -1, blocks)),
        ("control_future", window_mean(control_diff, 1, 25, blocks)),
        ("ball_half", ball_side),
    ]
    team_ball = np.column_stack([_zero_nan(values) for _, values in columns])
    team_ball[~visible] = 0.0
    ball_names = [name for name, _ in columns]

    # -- carried control: usable on every frame, the bridge across ball gaps -------------------
    carried = []
    carry_names = []
    for frames_decay in CARRY_FRAMES:
        carried.append(last_sign * np.exp(-last_age / float(frames_decay)))
        carried.append(next_sign * np.exp(-next_age / float(frames_decay)))
        carry_names += ["last_{}".format(frames_decay), "next_{}".format(frames_decay)]

    # -- team_shape: home minus away, every frame ----------------------------------------------
    raw_shape = [
        ("spread_x", home_spread_x - away_spread_x),
        ("spread_y", home_spread_y - away_spread_y),
        ("area", (home_spread_x * home_spread_y - away_spread_x * away_spread_y) / 100.0),
        ("height", (home_x * direction + away_x * direction) / length),
        ("speed", home_speed - away_speed),
        ("forward_speed", (home_vx + away_vx) * direction),
        ("opponent_gap", home_gap - away_gap),
        ("front", (home_front - away_front) / length),
    ]
    shape_columns = []
    shape_names = []
    for name, values in raw_shape:
        for half_width in SHAPE_WINDOWS:
            shape_columns.append(_zero_nan(_centred(values, half_width, blocks)))
            shape_names.append("{}_{}".format(name, half_width))
    team_shape = np.column_stack(carried + shape_columns)
    shape_names = carry_names + shape_names

    # -- dead: symmetric between the teams -----------------------------------------------------
    all_speed = _masked_stats(speed)[0]
    fast_share = _masked_stats(np.where(np.isfinite(speed), (speed > 2.5).astype(np.float32), np.nan))[0]
    seen = visible.astype(np.float64)
    since, until, gap_length = _gap_position(visible, blocks)
    margin_x = np.minimum(ball[:, 0], length - ball[:, 0]).astype(np.float64)
    margin_y = np.minimum(ball[:, 1], width - ball[:, 1]).astype(np.float64)
    margin = np.minimum(margin_x, margin_y)
    nearest_any = np.fmin(home_d, away_d)
    separation = np.abs(home_x - away_x) / length
    spread_all = _masked_stats(xy[:, :, 0])[1] / length

    dead_columns = []
    dead_names = []

    def add(name, values):
        dead_columns.append(_zero_nan(values))
        dead_names.append(name)

    for half_width in DEAD_WINDOWS:
        add("speed_{}".format(half_width), _centred(all_speed, half_width, blocks))
        add("fast_share_{}".format(half_width), _centred(fast_share, half_width, blocks))
        add("seen_{}".format(half_width), _centred(seen, half_width, blocks))
        add("ball_speed_{}".format(half_width), _zero_nan(_centred(ball_speed, half_width, blocks)))
    add("speed_past", window_mean(all_speed, -75, -1, blocks))
    add("speed_future", window_mean(all_speed, 1, 75, blocks))
    add("speed_rising", window_mean(all_speed, 1, 75, blocks) - window_mean(all_speed, -75, -1, blocks))
    add("slow_players", np.exp(-_zero_nan(_centred(all_speed, 50, blocks)) / 1.0))
    add("gap_length", np.log1p(gap_length))
    add("gap_long", (gap_length > 125).astype(np.float64))
    add("gap_since", np.log1p(since))
    add("gap_until", np.log1p(until))
    add("seen", seen)
    add("margin", np.where(visible, np.exp(-_zero_nan(margin) / 2.0), 0.0))
    add("outside", np.where(visible, (margin < 0.0).astype(np.float64), 0.0))
    add("ball_still", np.where(visible, np.exp(-_zero_nan(ball_speed) / 1.0), 0.0))
    add("nearest_any", np.where(visible, np.clip(_zero_nan(nearest_any), 0.0, DISTANCE_CAP) / DISTANCE_CAP, 0.0))
    add("separation", _centred(separation, 25, blocks))
    add("spread_all", _centred(spread_all, 25, blocks))
    add("tracked", np.isfinite(xy[:, :, 0]).sum(axis=1) / 22.0)
    add("no_control_past", np.exp(-last_age / 125.0))
    dead = np.column_stack(dead_columns)

    names = {"team_ball": ball_names, "team_shape": shape_names, "dead": dead_names}
    return Features(team_ball.astype(np.float32), team_shape.astype(np.float32),
                    dead.astype(np.float32), visible, names)
