"""How close are the public games to the competition match? Measured, not assumed.

Everything chosen on the public games is only worth something if they behave like the competition
match, so this compares the two on everything the organisers have published: the visibility pattern
of the ball (readable off the committed evaluation/results/baseline_raw.csv), and the scores of the
reference baseline, which is the same algorithm in both columns.

    python validation/calibrate_noise.py

Reads no gated data.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from open_games import GAMES, load_game  # noqa: E402
from possession.tracking import HOME  # noqa: E402

# Params
# The competition's blocks, and the baseline's published full-match scores (EVALUATION.md).
COMPETITION_BLOCKS = ((11185, 90652), (110808, 190198))
PUBLISHED = {0: (0.6532, 0.0486), 25: (0.6752, 0.1996), 50: (0.6724, 0.2433), 125: (0.6356, 0.2242)}
WINDOWS = (0, 25, 50, 125)
GAP_BINS = ((1, 25), (26, 125), (126, 250), (251, 750), (751, 10 ** 7))


def run_lengths(mask, blocks):
    """(lengths of True runs, lengths of False runs), never across a block boundary."""
    on, off = [], []
    for start, stop in blocks:
        window = mask[start:stop]
        if not len(window):
            continue
        bounds = np.concatenate(([0], np.flatnonzero(window[1:] != window[:-1]) + 1, [len(window)]))
        for index in range(len(bounds) - 1):
            (on if window[bounds[index]] else off).append(bounds[index + 1] - bounds[index])
    return np.array(on), np.array(off)


def nearest_team_codes(match):
    """The reference baseline's step 1 on any Match: pm.H / pm.A where the ball is seen, else D."""
    delta = match.xy - match.ball_xy[:, None, :]
    distance = np.sqrt((delta ** 2).sum(axis=2))
    distance[np.isnan(distance)] = np.inf
    nearest = distance.argmin(axis=1)
    codes = np.where(match.team[nearest] == HOME, pm.H, pm.A).astype(np.int8)
    codes[~np.isfinite(distance.min(axis=1))] = pm.D
    return codes


def smooth_codes(codes, window):
    """The reference baseline's step 2: a centred majority vote, D where nobody was seen."""
    if window <= 0:
        return codes.copy()
    home = np.concatenate(([0], np.cumsum(codes == pm.H)))
    away = np.concatenate(([0], np.cumsum(codes == pm.A)))
    index = np.arange(len(codes))
    low = np.maximum(0, index - window)
    high = np.minimum(len(codes), index + window + 1)
    home_count = home[high] - home[low]
    away_count = away[high] - away[low]
    out = np.full(len(codes), pm.D, dtype=np.int8)
    seen = (home_count + away_count) > 0
    out[seen] = np.where(home_count[seen] >= away_count[seen], pm.H, pm.A)
    return out


def gap_line(gaps):
    total = max(1, gaps.sum())
    parts = []
    for low, high in GAP_BINS:
        members = (gaps >= low) & (gaps <= high)
        parts.append("{}-{}: n{} {:.3f}".format(low, high if high < 10 ** 7 else "", members.sum(),
                                                gaps[members].sum() / float(total)))
    return "  ".join(parts)


if __name__ == '__main__':
    raw = pm.label_codes(pm.read_label_csv(
        os.path.join(ROOT, "evaluation", "results", "baseline_raw.csv")))
    visible = raw != pm.D
    _, gaps = run_lengths(visible, COMPETITION_BLOCKS)
    graded = sum(stop - start for start, stop in COMPETITION_BLOCKS)
    print("COMPETITION  visible {:.3f}  gaps {} median {:.0f}  flips/min {:.1f}".format(
        sum(visible[a:b].sum() for a, b in COMPETITION_BLOCKS) / float(graded), len(gaps),
        np.median(gaps),
        sum(int((raw[a:b][raw[a:b] != pm.D][1:] != raw[a:b][raw[a:b] != pm.D][:-1]).sum())
            for a, b in COMPETITION_BLOCKS) / (graded / 1500.0)))
    print("  gaps  " + gap_line(gaps))

    for game in GAMES:
        match, labels, _ = load_game(game)
        runs = pm.valid_runs(labels)
        seen = ~np.isnan(match.ball_xy[:, 0])
        _, gaps = run_lengths(seen, match.blocks)
        codes = nearest_team_codes(match)
        graded_mask = labels != pm.X
        flips = sum(int((codes[a:b][codes[a:b] != pm.D][1:] != codes[a:b][codes[a:b] != pm.D][:-1]).sum())
                    for a, b in match.blocks)
        print("\nOPEN GAME {}  visible {:.3f}  gaps {} median {:.0f}  flips/min {:.1f}  tracked {:.1f}".format(
            game, seen[graded_mask].mean(), len(gaps), np.median(gaps),
            flips / (graded_mask.sum() / 1500.0),
            np.mean((~np.isnan(match.xy[graded_mask][:, :, 0])).sum(axis=1))))
        print("  gaps  " + gap_line(gaps))
        for code, name in ((pm.H, "H"), (pm.A, "A"), (pm.D, "D")):
            members = labels == code
            print("  gt {}  frames {:6d}  ball visible {:.3f}".format(name, members.sum(), seen[members].mean()))
        alive_seen = seen & ((labels == pm.H) | (labels == pm.A))
        print("  nearest-team accuracy on alive visible frames {:.3f}".format(
            np.mean(codes[alive_seen] == labels[alive_seen])))
        for window in WINDOWS:
            result = pm.evaluate(labels, smooth_codes(codes, window))
            print("  baseline +/-{:<4d} macroF1 {:.4f} (real {:.4f})  spellF1 {:.4f} (real {:.4f})  "
                  "poss {:4d}/{:4d}".format(window, result["macro_f1"], PUBLISHED[window][0],
                                           result["spell_f1_mean"], PUBLISHED[window][1],
                                           result["n_pred_spells"], result["n_gt_spells"]))
