"""Improved baseline combining nearest-player with smart tagging priors.

This improves on the simple baseline by:
1. Using Smart Tagging to identify dead-ball intervals (set pieces, phases)
2. Combining ball track with tactical phase hints
3. Better handling of possession structure

The key insight: during set pieces and tactical phases, the team tag from smart
tagging is a strong prior that should influence our label choice.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from games import GAME, GAMES, require, source_frames

# Params
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "evaluation", "results")

# The four published variants: (name, half-width of the majority-vote window, in frames).
VARIANTS = (
    ("baseline_raw", 0),
    ("baseline_smooth25", 25),
    ("baseline_smooth50", 50),
    ("baseline_smooth125", 125),
)

# Constants
FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING = "N"
HOME, AWAY, NOBODY = 0, 1, -1
LABELS = np.array(["H", "A", "D"])


def team_of_each_track(meta_path):
    """Array over track slots: 0 if home, 1 if away."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))

    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return np.array([HOME if team_id == home_id else AWAY for team_id in slots], dtype=np.int8)


def team_names(meta_path):
    """(home, away) club names."""
    root = ET.parse(meta_path).getroot()
    return tuple(team.findtext("Name") for team in root.iter("Team"))


def nearest_player_labels(raw_path, team_of_track, n_frames, pitch):
    """Per-frame team of the player closest to the ball. NOBODY where no answer."""
    length, width = pitch
    out = np.full(n_frames, NOBODY, dtype=np.int8)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            ball = parts[2]
            if ball[0] == MISSING:
                continue

            comma = ball.index(",")
            ball_x = float(ball[:comma]) * length
            ball_y = float(ball[comma + 1:ball.index(",", comma + 1)]) * width

            best, best_team = None, NOBODY
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] == MISSING:
                    continue
                comma = group.index(",")
                dx = float(group[:comma]) * length - ball_x
                dy = float(group[comma + 1:group.index(",", comma + 1)]) * width - ball_y
                distance = dx * dx + dy * dy
                if best is None or distance < best:
                    best, best_team = distance, team_of_track[slot]

            out[int(parts[0]) - 1] = best_team
    return out


def smooth(codes, window):
    """Majority vote over a centred window."""
    if window <= 0:
        return np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D"))

    home_cumulative = np.concatenate(([0], np.cumsum(codes == HOME)))
    away_cumulative = np.concatenate(([0], np.cumsum(codes == AWAY)))

    index = np.arange(len(codes))
    low = np.maximum(0, index - window)
    high = np.minimum(len(codes), index + window + 1)
    home = home_cumulative[high] - home_cumulative[low]
    away = away_cumulative[high] - away_cumulative[low]

    labels = np.full(len(codes), "D", dtype="<U1")
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], "H", "A")
    return labels


def write_submission(path, labels):
    """Write a frame,label CSV."""
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label)
                          for frame, label in enumerate(labels))
    return path


def load_smart_tagging(data_dir=None):
    """Load smart tagging events and convert to frames."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, data_dir)

    # Load smart tagging XML
    smart_tagging_path = os.path.join(data_dir, "tactical", "BAR-ATH_tactical_smart_tagging.xml")
    if not os.path.exists(smart_tagging_path):
        print("Smart tagging file not found:", smart_tagging_path)
        return None

    fps = GAMES[GAME]["fps"]
    events = []
    for instance in ET.parse(smart_tagging_path).iter("instance"):
        start_s = float(instance.findtext("start"))
        end_s = float(instance.findtext("end"))
        events.append({
            "code": instance.findtext("code"),
            "start_s": start_s,
            "end_s": end_s,
            "start_frame": int(round(start_s * fps)),
            "end_frame": int(round(end_s * fps)),
            "tags": {label.findtext("group"): label.findtext("text")
                     for label in instance.findall("label")},
        })
    return events


def get_team_from_smart_tagging(events, frame):
    """Get the team tag from smart tagging at a given frame."""
    # Find overlapping intervals
    for event in events:
        if event["start_frame"] <= frame < event["end_frame"]:
            team = event["tags"].get("Team")
            # Map team name to H/A (home is first team in metadata)
            home_name, away_name = team_names(meta_path)
            if team == home_name:
                return HOME
            elif team == away_name:
                return AWAY
    return None


def improved_baseline_with_smart_tagging(data_dir=None, window=50):
    """Build improved baseline using smart tagging priors.

    Strategy:
    1. Use ball track as primary signal when available
    2. Use smart tagging team hints during dead-ball periods (set pieces, phases)
    3. During conflicts, prefer the more direct evidence (ball track > smart tagging)
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, data_dir)
    n_frames = source_frames("tactical", GAME)

    home, away = team_names(meta_path)
    print("teams: H = {}, A = {}".format(home, away))
    print("reading {} ...".format(os.path.basename(raw_path)))

    # Get nearest player labels (ball track only)
    codes_ball = nearest_player_labels(raw_path, team_of_each_track(meta_path), n_frames,
                                        GAMES[GAME]["pitch"])

    seen = int(np.sum(codes_ball != NOBODY))
    print("  ball located on {} of {} frames ({:.1f}%)\n".format(
        seen, n_frames, 100.0 * seen / n_frames))

    # Load smart tagging events
    events = load_smart_tagging(data_dir)
    if events is None:
        print("Smart tagging not available, using ball track only")
        return smooth(codes_ball, window)

    print("loaded {} smart tagging events".format(len(events)))

    # Combine ball track with smart tagging hints
    codes_combined = np.full(n_frames, NOBODY, dtype=np.int8)

    for frame in range(n_frames):
        ball_team = codes_ball[frame]
        smart_team = get_team_from_smart_tagging(events, frame)

        if ball_team != NOBODY and smart_team is not None:
            # Both signals available
            if ball_team == smart_team:
                # Agreement - use either signal
                codes_combined[frame] = ball_team
            elif ball_team == NOBODY:
                # Ball not visible, use smart tagging
                codes_combined[frame] = smart_team
            else:
                # Conflict - prefer ball track (more direct evidence)
                codes_combined[frame] = ball_team
        elif ball_team == NOBODY and smart_team is not None:
            # Only smart tagging available (dead ball period)
            codes_combined[frame] = smart_team
        else:
            # Ball visible, no smart tagging hint - use ball track
            codes_combined[frame] = ball_team

    return smooth(codes_combined, window)


def build(game=GAME, data_dir=None):
    """Per-frame team codes for the whole video."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)

    home, away = team_names(meta_path)
    print("teams: H = {}, A = {}".format(home, away))
    print("reading {} ...".format(os.path.basename(raw_path)))

    codes = nearest_player_labels(raw_path, team_of_each_track(meta_path), n_frames,
                                  GAMES[game]["pitch"])
    seen = int(np.sum(codes != NOBODY))
    print("  ball located on {} of {} frames ({:.1f}%)\n".format(
        seen, n_frames, 100.0 * seen / n_frames))
    return codes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--window", type=int, nargs="+",
                        help="half-widths to write instead of the four published variants")
    parser.add_argument("--out", default=RESULTS_DIR, help="where to write")
    parser.add_argument("--data-dir", default=None, help="path to data folder (overrides MTAH_DATA_DIR)")
    args = parser.parse_args()

    codes = build(data_dir=args.data_dir)
    variants = ([("baseline_smooth{}".format(w) if w else "baseline_raw", w)
                 for w in args.window] if args.window else list(VARIANTS))

    for name, window in variants:
        labels = smooth(codes, window)
        path = write_submission(os.path.join(args.out, "{}.csv".format(name)), labels)
        counts = {label: int(np.sum(labels == label)) for label in LABELS}
        print("{:<20} window +/-{:<4} H {:>6}  A {:>6}  D {:>6}".format(
            name, window, counts["H"], counts["A"], counts["D"]))
        print("{:<20} -> {}".format("", os.path.relpath(path, RESULTS_DIR)))

    print("\nNow score them:  python evaluation/run_evaluation.py")
