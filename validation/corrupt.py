"""Degrade a clean public game until it behaves like the competition's automatic tracking.

The public games are corrected data, and they differ from the competition match in one way that
would make validating on them meaningless if left alone: their ball is almost never visible when it
is dead, so "the ball is missing" nearly answers the dead-ball question. In the public five minutes
of the competition match, two of the four dead-ball stretches have the ball visible throughout, and
two of the four long gaps fall in the middle of live play.

The noise model below reinstates that, and calibrate() fits its strengths to numbers the organisers
publish rather than to anything gated: the four baselines' full-match scores in EVALUATION.md, and
the visibility of the ball readable from the committed baseline_raw.csv. A public game is a fair
stand-in when the same baseline scores on it what it scores on the real match.

    python validation/corrupt.py                # search the noise strengths, print the best fits

Every corruption is seeded, so a validation run is reproducible.
"""

import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from possession.tracking import HOME, Match  # noqa: E402

# Params
FPS = 25
# Share of each dead stretch's frames settled at the restart spot, once the ball has been carried.
RESTART_HOLD_SHARE = 0.4
# Mean length in frames of a lost player track and of a cross-team identity switch.
DROPOUT_MEAN_FRAMES = 125
SWITCH_MEAN_FRAMES = 125
MAX_GAP_FRAMES = 25 * 60
# Long gaps only: real gaps below 5 s are almost absent, see calibrate_noise.py.
MIN_GAP_FRAMES = 100
FALLBACK_GAP_FRAMES = (130, 180, 270, 400, 550, 800, 1200)

PUBLISHED = {0: (0.6532, 0.0486), 25: (0.6752, 0.1996), 50: (0.6724, 0.2433), 125: (0.6356, 0.2242)}
TARGET_VISIBLE = 0.593

# Result of running this file: the noise strengths whose baseline scores best match the published
# ones. These are what the validation of the method uses.
CALIBRATED = {"dead_visible": 0.25, "alive_gap_share": 0.10, "switches_per_min": 1.0,
              "ball_false_share": 0.08, "pos_sigma": 0.25, "ball_sigma": 0.3,
              "tracked_mean": 20.5}


def real_gap_lengths():
    """Lengths of the competition match's long ball gaps, from the committed baseline_raw.csv."""
    path = os.path.join(ROOT, "evaluation", "results", "baseline_raw.csv")
    if not os.path.isfile(path):
        return np.array(FALLBACK_GAP_FRAMES)
    raw = pm.label_codes(pm.read_label_csv(path)) != pm.D
    out = []
    for start, stop in ((11185, 90652), (110808, 190198)):
        window = raw[start:stop]
        bounds = np.concatenate(([0], np.flatnonzero(window[1:] != window[:-1]) + 1, [len(window)]))
        for index in range(len(bounds) - 1):
            if not window[bounds[index]]:
                out.append(bounds[index + 1] - bounds[index])
    out = np.array(out)
    return np.clip(out[out >= MIN_GAP_FRAMES], MIN_GAP_FRAMES, MAX_GAP_FRAMES)


def _spans(mask, start, stop):
    """[(a, b)] maximal True runs of mask[start:stop], in absolute frames."""
    window = mask[start:stop]
    if not len(window):
        return []
    edges = np.diff(np.concatenate(([0], window.astype(np.int8), [0])))
    return list(zip((np.flatnonzero(edges == 1) + start).tolist(),
                    (np.flatnonzero(edges == -1) + start).tolist()))


def _show_ball_when_dead(ball, labels, blocks, share, rng):
    """Carry the ball from where it went dead to where it restarts, on a share of the stoppages."""
    dead = labels == pm.D
    for start, stop in blocks:
        for a, b in _spans(dead, start, stop):
            if rng.random_sample() >= share:
                continue
            before = np.flatnonzero(~np.isnan(ball[start:a, 0]))
            after = np.flatnonzero(~np.isnan(ball[b:stop, 0]))
            if not len(before) and not len(after):
                continue
            first = ball[start + before[-1]] if len(before) else ball[b + after[0]]
            last = ball[b + after[0]] if len(after) else first
            hold = int((b - a) * RESTART_HOLD_SHARE)
            moving = max(1, (b - a) - hold)
            ramp = np.minimum(np.arange(b - a) / float(moving), 1.0)[:, None]
            ball[a:b] = first + (last - first) * ramp


def _drop_ball_when_alive(ball, labels, blocks, share, gap_lengths, rng):
    """Remove the ball from long stretches of live play, until `share` of live frames is gone."""
    alive = (labels == pm.H) | (labels == pm.A)
    for start, stop in blocks:
        target = int(alive[start:stop].sum() * share)
        removed, attempts = 0, 0
        while removed < target and attempts < 2000:
            attempts += 1
            length = int(rng.choice(gap_lengths))
            a = int(rng.randint(start, max(start + 1, stop - length)))
            removed += int(alive[a:a + length].sum())
            ball[a:a + length] = np.nan


def _misplace_ball(ball, xy, blocks, share, rng):
    """Stick the ball onto a random player for short stretches: a detector locked on a false hit."""
    for start, stop in blocks:
        target = int((stop - start) * share)
        done = 0
        while done < target:
            length = int(rng.randint(10, 75))
            a = int(rng.randint(start, max(start + 1, stop - length)))
            slot = int(rng.randint(xy.shape[1]))
            track = xy[a:a + length, slot]
            usable = ~np.isnan(track[:, 0]) & ~np.isnan(ball[a:a + length, 0])
            ball[a:a + length][usable] = track[usable]
            done += length


def _drop_players(xy, blocks, tracked_mean, rng):
    """Lose player tracks for a few seconds at a time, until tracked_mean objects remain."""
    for start, stop in blocks:
        present = ~np.isnan(xy[start:stop, :, 0])
        current = present.sum(axis=1).mean()
        if current <= tracked_mean:
            continue
        to_remove = int((current - tracked_mean) * (stop - start))
        removed = 0
        while removed < to_remove:
            length = int(rng.exponential(DROPOUT_MEAN_FRAMES)) + 5
            a = int(rng.randint(start, max(start + 1, stop - length)))
            slot = int(rng.randint(xy.shape[1]))
            removed += int((~np.isnan(xy[a:a + length, slot, 0])).sum())
            xy[a:a + length, slot] = np.nan


def _switch_identities(xy, team, blocks, per_minute, rng):
    """Swap a home track with a nearby away track for a few seconds, as a tracker does in a duel."""
    home_slots = np.flatnonzero(team == HOME)
    away_slots = np.flatnonzero(team != HOME)
    for start, stop in blocks:
        count = int(per_minute * (stop - start) / (FPS * 60.0))
        for _ in range(count):
            frame = int(rng.randint(start, stop - 1))
            delta = (xy[frame, home_slots][:, None, :] - xy[frame, away_slots][None, :, :])
            distance = np.sqrt((delta ** 2).sum(axis=2))
            if not np.isfinite(distance).any():
                continue
            i, j = np.unravel_index(np.nanargmin(distance), distance.shape)
            length = int(rng.exponential(SWITCH_MEAN_FRAMES)) + 5
            end = min(stop, frame + length)
            home_slot, away_slot = home_slots[i], away_slots[j]
            swap = xy[frame:end, home_slot].copy()
            xy[frame:end, home_slot] = xy[frame:end, away_slot]
            xy[frame:end, away_slot] = swap


def corrupt(match, labels, seed, dead_visible=0.5, alive_gap_share=0.1, switches_per_min=2.0,
            ball_false_share=0.0, pos_sigma=0.25, ball_sigma=0.3, tracked_mean=20.5):
    """A new Match with automatic-tracking noise. The labels are unchanged and not returned."""
    rng = np.random.RandomState(seed)
    xy = match.xy.copy()
    ball = match.ball_xy.copy()
    gap_lengths = real_gap_lengths()

    _show_ball_when_dead(ball, labels, match.blocks, dead_visible, rng)
    _drop_ball_when_alive(ball, labels, match.blocks, alive_gap_share, gap_lengths, rng)
    if ball_false_share > 0:
        _misplace_ball(ball, xy, match.blocks, ball_false_share, rng)
    _switch_identities(xy, match.team, match.blocks, switches_per_min, rng)
    _drop_players(xy, match.blocks, tracked_mean, rng)

    xy = xy + rng.normal(0.0, pos_sigma, xy.shape).astype(np.float32)
    ball = ball + rng.normal(0.0, ball_sigma, ball.shape).astype(np.float32)
    return Match(xy, ball, match.team, match.blocks, match.pitch, name=match.name + "_noisy")


def calibrate(games, seeds=(0,)):
    """Search noise strengths; print the ones whose baseline best reproduces the published scores."""
    from calibrate_noise import nearest_team_codes, smooth_codes

    grid = {"dead_visible": (0.25, 0.5, 0.75),
            "alive_gap_share": (0.05, 0.1, 0.2, 0.3),
            "switches_per_min": (1.0, 3.0, 6.0),
            "ball_false_share": (0.0, 0.08, 0.15)}
    names = sorted(grid)
    results = []
    for values in itertools.product(*(grid[name] for name in names)):
        params = dict(zip(names, values))
        scores = {window: [] for window in PUBLISHED}
        visible = []
        for match, labels in games:
            for seed in seeds:
                noisy = corrupt(match, labels, seed, **params)
                graded = labels != pm.X
                visible.append(np.mean(~np.isnan(noisy.ball_xy[graded, 0])))
                codes = nearest_team_codes(noisy)
                for window in PUBLISHED:
                    outcome = pm.evaluate(labels, smooth_codes(codes, window))
                    scores[window].append((outcome["macro_f1"], outcome["spell_f1_mean"]))
        means = {window: np.mean(scores[window], axis=0) for window in PUBLISHED}
        loss = sum((means[w][0] - PUBLISHED[w][0]) ** 2 + (means[w][1] - PUBLISHED[w][1]) ** 2
                   for w in PUBLISHED) + (np.mean(visible) - TARGET_VISIBLE) ** 2
        results.append((loss, params, means, float(np.mean(visible))))
        print(".", end="", flush=True)
    print()

    results.sort(key=lambda item: item[0])
    for loss, params, means, visible in results[:5]:
        print("loss {:.5f}  visible {:.3f}  {}".format(loss, visible, params))
        for window in sorted(PUBLISHED):
            print("    +/-{:<4d} macroF1 {:.3f} (real {:.3f})  spellF1 {:.3f} (real {:.3f})".format(
                window, means[window][0], PUBLISHED[window][0],
                means[window][1], PUBLISHED[window][1]))
    return results


if __name__ == '__main__':
    from open_games import GAMES, load_game

    loaded = []
    for game in GAMES:
        match_, labels_, _ = load_game(game)
        loaded.append((match_, labels_))
    calibrate(loaded)
