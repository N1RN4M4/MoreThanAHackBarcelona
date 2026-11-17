"""The tracking as arrays, from the FIFA/EPTS export or from anything else shaped like it.

Everything downstream works on one container, Match, so the method runs unchanged on the
competition ATD and on the public validation games in validation/open_games.py.

Reading 199319 lines of text takes a minute or two, so the parsed arrays are cached as an .npz
under cache/. That cache is a frame-level export of the match tracking: cache/ is gitignored and
it must stay out of any public repository (RULES.md).
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

# Params
HOME, AWAY = 0, 1
FIELD_SEP = ":"
GROUP_SEP = ";"
VALUE_SEP = ","
FPS = 25
# Half-width, in frames, of the central difference behind derived speeds. +/-2 frames is 0.16 s
# of baseline, which is enough to stop position noise from dominating the speed.
SPEED_HALF_WIDTH = 2
# Faster than this is a tracker jump, not a human or a ball.
MAX_PLAYER_SPEED = 12.0
MAX_BALL_SPEED = 45.0


class Match(object):
    """One match as arrays. Positions are metres, the pitch runs from (0, 0) to pitch.

    xy          (n_frames, n_slots, 2) float32, NaN where the slot is not tracked
    ball_xy     (n_frames, 2) float32, NaN where the ball is not tracked
    team        (n_slots,) int8, HOME or AWAY
    blocks      [(start, stop)] half-open frame ranges of each half
    pitch       (length, width) in metres
    tags        optional Smart Tagging entries, see tagging.py
    """

    def __init__(self, xy, ball_xy, team, blocks, pitch, tags=None, name="match"):
        self.xy = xy
        self.ball_xy = ball_xy
        self.team = np.asarray(team, dtype=np.int8)
        self.blocks = list(blocks)
        self.pitch = tuple(pitch)
        self.tags = tags
        self.name = name

    @property
    def n_frames(self):
        return self.xy.shape[0]

    def graded_mask(self):
        """Boolean over frames: inside a half."""
        mask = np.zeros(self.n_frames, dtype=bool)
        for start, stop in self.blocks:
            mask[start:stop] = True
        return mask


def derived_speed(xy, blocks, max_speed):
    """Speed in m/s from positions, NaN where either end of the difference is missing.

    Used for players and ball on every dataset instead of the export's own speed column. The
    validation games have no such column, and a feature that is computed the same way everywhere
    is one that transfers from them to the competition match.
    """
    squeeze = xy.ndim == 2
    if squeeze:
        xy = xy[:, None, :]
    speed = np.full(xy.shape[:2], np.nan, dtype=np.float32)
    width = SPEED_HALF_WIDTH
    for start, stop in blocks:
        if stop - start <= 2 * width:
            continue
        delta = xy[start + 2 * width:stop] - xy[start:stop - 2 * width]
        speed[start + width:stop - width] = (
            np.sqrt((delta ** 2).sum(axis=2)) * FPS / float(2 * width))
    speed[speed > max_speed] = np.nan
    return speed[:, 0] if squeeze else speed


def read_metadata(meta_path):
    """(team_of_slot, slot_orders) from an EPTS metadata XML.

    slot_orders is [(first_frame, last_frame, [slot index, ...])], one entry per data format
    specification. The competition export has a single specification listing every track in file
    order. The public Game 3 has several, because its substitutions change the columns of the raw
    file part-way through, so position in a line cannot be assumed to mean the same player for the
    whole match.
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))

    players = list(root.iter("Player"))
    if not players:
        raise ValueError("{}: no Player entries".format(meta_path))
    # The first team in the file is the home side: the FIFA/EPTS convention, see starter/baseline.py.
    team_of_slot = np.array([HOME if player.get("teamId") == team_ids[0] else AWAY
                             for player in players], dtype=np.int8)
    slot_of_player = {player.get("id"): index for index, player in enumerate(players)}

    channel_owner = {}
    for channel in root.iter("PlayerChannel"):
        channel_owner[channel.get("id")] = channel.get("playerId")

    slot_orders = []
    for spec in root.iter("DataFormatSpecification"):
        order = []
        for ref in spec.iter("PlayerChannelRef"):
            channel = ref.get("playerChannelId")
            owner = channel_owner.get(channel)
            if channel and channel.endswith("_x") and owner in slot_of_player:
                order.append(slot_of_player[owner])
        slot_orders.append((int(spec.get("startFrame")), int(spec.get("endFrame")), order))
    return team_of_slot, slot_orders


def _group_xy(group):
    """(x, y) of one 'x,y[,speed...]' group, or None if the group holds no position."""
    if not group or group[0] == "N":
        return None
    values = group.split(VALUE_SEP)
    if len(values) < 2 or not values[0] or not values[1]:
        return None
    x, y = float(values[0]), float(values[1])
    if x != x or y != y:
        return None
    return x, y


def read_epts(raw_path, meta_path, n_frames, pitch):
    """(xy, ball_xy, team_of_slot) in metres, indexed by video frame = frame_id - 1."""
    team_of_slot, slot_orders = read_metadata(meta_path)
    length, width = pitch
    xy = np.full((n_frames, len(team_of_slot), 2), np.nan, dtype=np.float32)
    ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)

    spec_index = 0
    default_order = list(range(len(team_of_slot)))
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame_id = int(parts[0])
            frame = frame_id - 1
            if not 0 <= frame < n_frames:
                continue
            while (spec_index + 1 < len(slot_orders)
                   and frame_id > slot_orders[spec_index][1]):
                spec_index += 1
            groups = parts[1].split(GROUP_SEP)
            order = slot_orders[spec_index][2] if slot_orders else default_order
            # A specification that does not name every column says nothing usable about any of
            # them: fall back on file order, which is what starter/baseline.py assumes throughout.
            if len(order) != len(groups):
                order = default_order
            for column, group in enumerate(groups):
                position = _group_xy(group)
                if position is not None:
                    slot = order[column]
                    xy[frame, slot, 0] = position[0] * length
                    xy[frame, slot, 1] = position[1] * width
            position = _group_xy(parts[2])
            if position is not None:
                ball_xy[frame, 0] = position[0] * length
                ball_xy[frame, 1] = position[1] * width
    return xy, ball_xy, team_of_slot


def load_cached(cache_path, builder):
    """(xy, ball_xy, team) from cache_path, building and saving them on the first call."""
    if os.path.isfile(cache_path):
        with np.load(cache_path) as saved:
            return saved["xy"], saved["ball_xy"], saved["team"]
    xy, ball_xy, team = builder()
    folder = os.path.dirname(cache_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    np.savez(cache_path, xy=xy, ball_xy=ball_xy, team=team)
    return xy, ball_xy, team
