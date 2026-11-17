"""Metrica's public sample games as Match objects, with frame labels derived from their events.

These games are the validation set. Their labels are not the competition's labels: the competition
reads ownership off corrected tracking, these read it off a human-coded event stream. They agree on
what a possession is closely enough to choose hyperparameters, and report_labels() below prints the
statistics that make that comparison checkable.

Label rules, in the terms CHALLENGE.md uses:

  - the team of the last on-ball event owns the ball, so a loose ball stays with the last owner
  - a challenge is not an on-ball event: a touch without control does not move possession
  - every set piece is the end of a stoppage, and the stoppage began at the ball going out, the
    foul, the goal or the whistle, whichever the events record, or else at the last on-ball event
"""

import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from possession.tracking import AWAY, HOME, Match, read_epts  # noqa: E402

# Params
DATA_DIR = os.path.join(ROOT, "external_data", "metrica")
PITCH = (105.0, 68.0)
GAMES = (1, 2, 3)
OWNERSHIP_TYPES = ("PASS", "SHOT", "SET PIECE", "RECOVERY", "BALL LOST", "CARRY")
# Skipped while walking back from a set piece to find the last on-ball event.
OFF_BALL_TYPES = ("CARD", "FAULT RECEIVED", "CHALLENGE")
STOP_LOST_SUBTYPES = ("OFFSIDE", "HAND BALL", "END HALF", "REFEREE HIT", "WOODWORK")
STOP_SHOT_SUBTYPES = ("-OUT", "GOAL")
# Longest believable stoppage before a restart. Longer means the events are sparse rather than
# the ball being dead, so the walk-back is capped to keep one gap from swallowing live play.
MAX_STOPPAGE_FRAMES = 25 * 150


def _game_path(game, suffix):
    return os.path.join(DATA_DIR, "Sample_Game_{}".format(game),
                        "Sample_Game_{}_{}".format(game, suffix))


def _read_csv_tracking(path):
    """(frames, periods, xy, ball_xy) from one team's tracking CSV. Three header rows."""
    table = pd.read_csv(path, skiprows=2)
    period = table["Period"].values.astype(np.int64)
    frames = table["Frame"].values.astype(np.int64)
    positions = table.iloc[:, 3:].values.astype(np.float32)
    n_slots = positions.shape[1] // 2 - 1
    xy = positions[:, :2 * n_slots].reshape(-1, n_slots, 2)
    return frames, period, xy, positions[:, -2:]


def _load_csv_game(game):
    frames, period, home_xy, ball = _read_csv_tracking(
        _game_path(game, "RawTrackingData_Home_Team.csv"))
    away_frames, _, away_xy, _ = _read_csv_tracking(
        _game_path(game, "RawTrackingData_Away_Team.csv"))
    if not np.array_equal(frames, away_frames):
        raise ValueError("game {}: home and away tracking disagree on frames".format(game))

    n_frames = int(frames.max())
    xy = np.full((n_frames, home_xy.shape[1] + away_xy.shape[1], 2), np.nan, dtype=np.float32)
    xy[frames - 1] = np.concatenate([home_xy, away_xy], axis=1)
    ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)
    ball_xy[frames - 1] = ball
    xy *= np.array(PITCH, dtype=np.float32)
    ball_xy *= np.array(PITCH, dtype=np.float32)

    team = np.array([HOME] * home_xy.shape[1] + [AWAY] * away_xy.shape[1], dtype=np.int8)
    blocks = []
    for value in np.unique(period):
        members = frames[period == value]
        blocks.append((int(members.min()) - 1, int(members.max())))

    table = pd.read_csv(_game_path(game, "RawEventsData.csv"))
    events = pd.DataFrame({
        "team": np.where(table["Team"].str.lower() == "home", HOME, AWAY),
        "type": table["Type"].str.upper(),
        "subtype": table["Subtype"].fillna("").str.upper(),
        "start": table["Start Frame"].astype(int) - 1,
        "end": table["End Frame"].astype(int) - 1,
    })
    return xy, ball_xy, team, blocks, events


def _count_lines(path):
    with open(path) as handle:
        return sum(1 for _ in handle)


def _load_epts_game(game):
    frames_total = _count_lines(_game_path(game, "tracking.txt"))
    xy, ball_xy, team = read_epts(_game_path(game, "tracking.txt"),
                                  _game_path(game, "metadata.xml"), frames_total, PITCH)
    with open(_game_path(game, "events.json")) as handle:
        records = json.load(handle)["data"]

    # The metadata lists the home side first, which is the id tracking.read_metadata calls HOME.
    root = ET.parse(_game_path(game, "metadata.xml")).getroot()
    home_id = [element.get("id") for element in root.iter("Team")][0]

    rows, period_starts = [], {}
    for record in records:
        subtypes = record.get("subtypes")
        if isinstance(subtypes, dict):
            subtypes = [subtypes]
        start = int(record["start"]["frame"]) - 1
        rows.append({"team": HOME if record["team"]["id"] == home_id else AWAY,
                     "type": record["type"]["name"].upper(),
                     "subtype": "-".join(item["name"] for item in (subtypes or [])).upper(),
                     "start": start,
                     "end": int(record["end"]["frame"]) - 1})
        period_starts.setdefault(int(record["period"]), []).append(start)

    # The second half starts at the first event of the second period.
    second = min(period_starts[2])
    return xy, ball_xy, team, [(0, second), (second, frames_total)], pd.DataFrame(rows)


def _is_stop(row):
    """Frame at which this event stops play, or None if it does not."""
    if row["type"] == "BALL OUT":
        return row["end"]
    if row["type"] == "FAULT RECEIVED":
        return row["start"]
    if row["type"] == "CHALLENGE" and "FAULT" in row["subtype"]:
        return row["start"]
    if row["type"] == "SHOT" and any(text in row["subtype"] for text in STOP_SHOT_SUBTYPES):
        return row["end"]
    if row["type"] == "BALL LOST" and any(text in row["subtype"] for text in STOP_LOST_SUBTYPES):
        return row["end"]
    return None


def labels_from_events(events, blocks, n_frames):
    """pm.H / pm.A / pm.D per frame, pm.X outside the halves."""
    labels = np.full(n_frames, pm.X, dtype=np.int8)
    rows = events.sort_values(["start"], kind="mergesort").to_dict("records")

    for start, stop in blocks:
        inside = [row for row in rows if start <= row["start"] < stop]
        owners = [row for row in inside if row["type"] in OWNERSHIP_TYPES]
        if not owners:
            continue

        # Owner: forward fill from each on-ball event, the first owner covering the head.
        change_frames = np.array([row["start"] for row in owners], dtype=np.int64)
        change_teams = np.array([pm.H if row["team"] == HOME else pm.A for row in owners],
                                dtype=np.int8)
        index = np.searchsorted(change_frames, np.arange(start, stop), side="right") - 1
        labels[start:stop] = change_teams[np.maximum(index, 0)]

        # Stoppage: back from every set piece to what stopped play.
        for position, row in enumerate(inside):
            if row["type"] != "SET PIECE":
                continue
            stop_frames, last_ball_end = [], None
            for earlier in reversed(inside[:position]):
                frame = _is_stop(earlier)
                if frame is not None:
                    stop_frames.append(frame)
                if earlier["type"] not in OFF_BALL_TYPES:
                    last_ball_end = earlier["end"]
                    break
            if stop_frames:
                dead_from = min(stop_frames)
            elif last_ball_end is not None:
                dead_from = last_ball_end
            else:
                continue  # the opening kick-off of a half has nothing before it
            dead_from = max(dead_from, row["start"] - MAX_STOPPAGE_FRAMES, start)
            if dead_from < row["start"]:
                labels[dead_from:row["start"]] = pm.D

        # After the last on-ball event of a half nothing is in play.
        last_end = max(row["end"] for row in owners)
        if last_end + 1 < stop:
            labels[last_end + 1:stop] = pm.D
    return labels


def load_game(game):
    """(Match, labels, events) for one public game."""
    if game == 3:
        xy, ball_xy, team, blocks, events = _load_epts_game(game)
    else:
        xy, ball_xy, team, blocks, events = _load_csv_game(game)
    labels = labels_from_events(events, blocks, xy.shape[0])
    return Match(xy, ball_xy, team, blocks, PITCH, name="open{}".format(game)), labels, events


def report_labels():
    """The label statistics that can be set beside the public ground truth's."""
    for game in GAMES:
        match, labels, _ = load_game(game)
        runs = pm.valid_runs(labels)
        stats = pm.possession_stats(labels, runs)
        graded = labels != pm.X
        minutes = graded.sum() / 25.0 / 60.0
        dead = labels == pm.D
        spells = pm.possession_spells(labels, runs)
        lengths = np.array([b - a for a, b, _ in spells]) / 25.0
        print("game {}  {:6.1f} min  blocks {}  poss {:4d} ({:.2f}/min)  mean {:5.1f}s  median "
              "{:5.1f}s  <4s {:.2f}  dead {:.3f}  home {:.3f}  ball visible {:.3f}".format(
                  game, minutes, match.blocks, stats["n_spells"], stats["n_spells"] / minutes,
                  stats["mean_duration_s"], np.median(lengths), (lengths < 4).mean(),
                  dead[graded].mean(), stats["home_share"],
                  np.mean(~np.isnan(match.ball_xy[graded, 0]))))


if __name__ == '__main__':
    report_labels()
