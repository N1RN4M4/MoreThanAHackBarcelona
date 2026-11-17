"""Smart Tagging as per-frame evidence: a prior on the team, and on where stoppages end.

DATA.md's warnings are applied directly. A tag is evidence added to the decoder rather than a label
copied across, so its errors can be outvoted. `Team` can be N/A, and every throw-in is, so those
entries say nothing about the team. Tag values are matched case-insensitively.

Entries are the dicts starter/load_smart_tagging.py returns:
    {code, start_frame, end_frame, tags: {group: text}}
"""

import numpy as np

# Params
HOME_NAME = "barcelona"
AWAY_NAME = "atletic club"
# The team these codes name is the team with the ball.
ATTACKING_CODES = ("build up", "progression", "creating chances", "attacking transition",
                   "ball in final third", "ball in the box", "players in the box", "long ball")
# The team these codes name is the team without it.
DEFENDING_CODES = ("defending in attacking third", "defending in middle third",
                   "defending in defensive third", "defensive transition")
SET_PIECE_CODE = "set pieces"
KICK_OFF_START = "kick off (start)"
# Set-piece windows are about 5 s around the restart (starter/load_smart_tagging.py). The first part
# leans dead and the rest leans alive, with the restarting team named.
RESTART_SHARE = 0.5
# Set-piece frames carry more team evidence than a phase: the restarting team has the ball.
SET_PIECE_TEAM_WEIGHT = 3.0
# The stoppage before a half's kick-off: the longest believable lead-in in frames (10 minutes), half
# the tag's window, and evidence strong enough that nothing else outvotes it.
MAX_KICK_OFF_LEAD = 25 * 600
KICK_OFF_HALF_WINDOW = int(2.5 * 25)
KICK_OFF_DEAD_EVIDENCE = 100.0


def _team_sign(tags, home_name=HOME_NAME, away_name=AWAY_NAME):
    team = (tags.get("Team") or "").strip().lower()
    if team == home_name:
        return 1.0
    if team == away_name:
        return -1.0
    return 0.0


def tag_evidence(tags, n_frames, blocks):
    """(team, dead) per frame. team is positive for home, dead is positive for a stoppage."""
    team = np.zeros(n_frames, dtype=np.float64)
    dead = np.zeros(n_frames, dtype=np.float64)
    if not tags:
        return team, dead

    for entry in tags:
        code = (entry["code"] or "").strip().lower()
        start = max(0, entry["start_frame"])
        stop = min(n_frames, entry["end_frame"])
        if stop <= start:
            continue
        sign = _team_sign(entry["tags"])

        if code in ATTACKING_CODES:
            team[start:stop] += sign
            dead[start:stop] -= 1.0
        elif code in DEFENDING_CODES:
            team[start:stop] -= sign
            dead[start:stop] -= 1.0
        elif code == SET_PIECE_CODE:
            middle = start + int((stop - start) * RESTART_SHARE)
            dead[start:middle] += 1.0
            dead[middle:stop] -= 1.0
            team[middle:stop] += sign * SET_PIECE_TEAM_WEIGHT

    # Everything before a half's tagged kick-off is a stoppage. This is the published trap in the
    # second half (FAQ.md): graded from 110808, kicked off at 113742. A window is placed by its
    # midpoint, and a long lead-in is refused: a kick-off tagged late in a half belongs to the next
    # one, and must not turn the half before it dead.
    for start, stop in blocks:
        kick_offs = [(entry["start_frame"] + entry["end_frame"]) // 2 for entry in tags
                     if (entry["tags"].get("Type") or "").strip().lower() == KICK_OFF_START]
        kick_offs = [frame for frame in kick_offs if start <= frame < stop]
        if kick_offs and min(kick_offs) - start <= MAX_KICK_OFF_LEAD:
            dead[start:min(kick_offs) - KICK_OFF_HALF_WINDOW] += KICK_OFF_DEAD_EVIDENCE
    return team, dead
