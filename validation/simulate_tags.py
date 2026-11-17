"""Smart-Tagging-shaped entries for the public games, which have none of their own.

Only used to check that the tag evidence helps and to set its weight. The real tags' accuracy is not
known, so these are made deliberately imperfect: some set pieces are missed, windows are jittered,
phases cover part of a possession, and a share of the team tags are wrong.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from possession.tagging import AWAY_NAME, HOME_NAME  # noqa: E402

# Params
FPS = 25
SET_PIECE_RECALL = 0.75
SET_PIECE_HALF_WINDOW = int(2.5 * FPS)
SET_PIECE_JITTER = FPS
PHASE_RECALL = 0.7
PHASE_MIN_FRAMES = 5 * FPS
WRONG_TEAM = 0.1
FALSE_PHASES_PER_GAME = 30


def _team_tag(sign):
    return {"Team": HOME_NAME if sign > 0 else AWAY_NAME}


def simulate(labels, events, blocks, seed, set_piece_recall=SET_PIECE_RECALL,
             phase_recall=PHASE_RECALL, wrong_team=WRONG_TEAM):
    """[{code, start_frame, end_frame, tags}] for one game."""
    rng = np.random.RandomState(seed)
    out = []

    for row in events.to_dict("records"):
        if row["type"] != "SET PIECE":
            continue
        kind = row["subtype"]
        opens_half = "KICK OFF" in kind and any(a == row["start"] for a, _ in blocks)
        # The real file tags both kick-offs (DATA.md), and the decoder leans on them.
        if not opens_half and rng.random_sample() > set_piece_recall:
            continue
        centre = row["start"] + int(rng.normal(0.0, SET_PIECE_JITTER))
        sign = 1.0 if row["team"] == 0 else -1.0
        tags = _team_tag(sign) if "THROW" not in kind else {"Team": "N/A"}
        if "KICK OFF" in kind:
            tags["Type"] = "Kick Off (Start)" if opens_half else "Kick Off (After Goal)"
        out.append({"code": "SET PIECES", "start_frame": centre - SET_PIECE_HALF_WINDOW,
                    "end_frame": centre + SET_PIECE_HALF_WINDOW, "tags": tags})

    runs = pm.valid_runs(labels)
    for a, b, code in pm.possession_spells(labels, runs):
        if b - a < PHASE_MIN_FRAMES or rng.random_sample() > phase_recall:
            continue
        cover = rng.uniform(0.6, 1.0)
        start = a + int((b - a) * rng.uniform(0.0, 1.0 - cover))
        stop = start + int((b - a) * cover)
        sign = 1.0 if code == pm.H else -1.0
        if rng.random_sample() < wrong_team:
            sign = -sign
        out.append({"code": "PROGRESSION", "start_frame": start, "end_frame": stop,
                    "tags": _team_tag(sign)})

    n_frames = len(labels)
    for _ in range(FALSE_PHASES_PER_GAME):
        start = int(rng.randint(blocks[0][0], n_frames - 250))
        out.append({"code": "PROGRESSION", "start_frame": start,
                    "end_frame": start + int(rng.randint(50, 250)),
                    "tags": _team_tag(1.0 if rng.random_sample() < 0.5 else -1.0)})
    return out
