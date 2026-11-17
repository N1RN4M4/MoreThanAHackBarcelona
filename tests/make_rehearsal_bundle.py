"""Write a fake data bundle in the competition's exact format, from a public game.

    python tests/make_rehearsal_bundle.py <folder> [--game 1]

The real bundle is gated, so the code that reads it could not otherwise be run before Cempaka does.
This lays a noisy public game onto the competition's frame numbers and writes the same files under
the same names: 199319 lines of "frame:x,y,speed;...:x,y,speed", 46 slots, EPTS metadata, SportsCode
tags. Point MTAH_DATA_DIR at the folder and every command runs as it will on the real thing. The
labels of the public game are written beside it so that the result can be scored.

This exercises the plumbing. The score it produces is not independent evidence of anything, because
the committed model was fitted on the same public games.
"""

import argparse
import os
import sys
from xml.sax.saxutils import escape

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "validation"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from corrupt import CALIBRATED, corrupt  # noqa: E402
from games import GAMES, GAME, asset_paths  # noqa: E402
from open_games import PITCH, load_game  # noqa: E402
from possession.tagging import AWAY_NAME, HOME_NAME  # noqa: E402
from simulate_tags import simulate  # noqa: E402

# Params
N_FRAMES = GAMES[GAME]["n_video_frames"]
N_SLOTS = 46
FIRST_HALF_START = 11185
SECOND_HALF_KICK_OFF = 113742
SECOND_HALF_GRADED = 110808
SEED = 202


def lay_out(match, labels):
    """(xy, ball, labels, offset per block) on the competition's frame numbers, normalised to 0-1."""
    (first_a, first_b), (second_a, second_b) = match.blocks
    home = np.flatnonzero(match.team == 0)
    away = np.flatnonzero(match.team != 0)
    slots = np.concatenate([home, away])
    xy = np.full((N_FRAMES, N_SLOTS, 2), np.nan, dtype=np.float32)
    ball = np.full((N_FRAMES, 2), np.nan, dtype=np.float32)
    out = np.full(N_FRAMES, pm.X, dtype=np.int8)
    scale = np.array(PITCH, dtype=np.float32)

    n_first = min(first_b - first_a, 90652 - FIRST_HALF_START)
    n_second = min(second_b - second_a, 190198 - SECOND_HALF_KICK_OFF)
    spans = ((first_a, FIRST_HALF_START, n_first), (second_a, SECOND_HALF_KICK_OFF, n_second))
    for source, target, count in spans:
        xy[target:target + count, :len(slots)] = match.xy[source:source + count][:, slots] / scale
        ball[target:target + count] = match.ball_xy[source:source + count] / scale
        out[target:target + count] = labels[source:source + count]

    # Dead before the second half's kick-off, as FAQ.md describes, and to the end of each half.
    out[FIRST_HALF_START + n_first:90652] = pm.D
    out[SECOND_HALF_GRADED:SECOND_HALF_KICK_OFF] = pm.D
    out[SECOND_HALF_KICK_OFF + n_second:190198] = pm.D
    return xy, ball, out, len(home), {first_a: FIRST_HALF_START - first_a,
                                      second_a: SECOND_HALF_KICK_OFF - second_a}


def _speeds(xy):
    """Metres per second, in the third value of each group, NaN where a neighbour is missing."""
    speed = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    delta = (xy[1:] - xy[:-1]) * np.array(PITCH, dtype=np.float32)
    speed[1:] = np.sqrt((delta ** 2).sum(axis=-1)) * 25.0
    return speed


def write_raw(path, xy, ball):
    speed = _speeds(xy)
    ball_speed = _speeds(ball)
    with open(path, "w", newline="\n") as handle:
        for frame in range(N_FRAMES):
            row = xy[frame].tolist()
            speeds = speed[frame].tolist()
            groups = ["NaN,NaN,NaN" if x != x else "{:.5f},{:.5f},{:.2f}".format(x, y, s if s == s else 0.0)
                      for (x, y), s in zip(row, speeds)]
            bx, by = ball[frame].tolist()
            ball_group = "NaN,NaN,NaN" if bx != bx else "{:.5f},{:.5f},{:.2f}".format(
                bx, by, float(ball_speed[frame]) if ball_speed[frame] == ball_speed[frame] else 0.0)
            handle.write("{}:{}:{}\n".format(frame + 1, ";".join(groups), ball_group))


def write_meta(path, n_home):
    players = []
    channels = []
    refs = []
    for slot in range(N_SLOTS):
        team = "HOMEID" if slot < n_home else "AWAYID"
        pid = "P{}".format(slot + 1)
        players.append('<Player id="{}" teamId="{}"><Name>Track {}</Name><ShirtNumber>{}</ShirtNumber></Player>'.format(
            pid, team, slot + 1, slot + 1))
        for kind in ("x", "y", "s"):
            channels.append('<PlayerChannel id="player{}_{}" playerId="{}" channelId="{}"/>'.format(
                slot + 1, kind, pid, kind))
        refs.append('<SplitRegister separator=","><PlayerChannelRef playerChannelId="player{0}_x"/>'
                    '<PlayerChannelRef playerChannelId="player{0}_y"/>'
                    '<PlayerChannelRef playerChannelId="player{0}_s"/></SplitRegister>'.format(slot + 1))
    with open(path, "w", newline="\n") as handle:
        handle.write('<?xml version="1.0" encoding="utf-8"?>\n<main><Metadata>'
                     '<Teams><Team id="HOMEID"><Name>FC Barcelona</Name></Team>'
                     '<Team id="AWAYID"><Name>Athletic Club</Name></Team></Teams>'
                     '<Players>{}</Players><Devices><Device><Sensors><Sensor id="position">'
                     '<Channels><Channel id="x"/><Channel id="y"/></Channels></Sensor></Sensors>'
                     '<PlayerChannels>{}</PlayerChannels></Device></Devices>'
                     '<DataFormatSpecifications><DataFormatSpecification startFrame="1" endFrame="{}" separator=":">'
                     '<StringRegister name="frameCount"/><SplitRegister separator=";">{}</SplitRegister>'
                     '</DataFormatSpecification></DataFormatSpecifications></Metadata></main>\n'.format(
                         "".join(players), "".join(channels), N_FRAMES, "".join(refs)))


def write_tags(path, tags, offsets, blocks):
    """SportsCode XML, in seconds from the first frame, as the real file is."""
    rows = []
    for index, entry in enumerate(tags):
        block_start = blocks[1][0] if entry["start_frame"] >= blocks[1][0] else blocks[0][0]
        shift = offsets[block_start]
        labels = "".join("<label><group>{}</group><text>{}</text></label>".format(
            escape(group), escape(text.title() if group == "Team" else text))
            for group, text in entry["tags"].items())
        rows.append("<instance><ID>{}</ID><start>{:.2f}</start><end>{:.2f}</end><code>{}</code>{}</instance>".format(
            index + 1, (entry["start_frame"] + shift) / 25.0, (entry["end_frame"] + shift) / 25.0,
            escape(entry["code"]), labels))
    with open(path, "w", newline="\n") as handle:
        handle.write('<?xml version="1.0" encoding="utf-8"?>\n<file><ALL_INSTANCES>{}</ALL_INSTANCES></file>\n'.format(
            "".join(rows)))


def build(folder, game=1):
    """Write the bundle under folder. Returns the path of the labels CSV for scoring it."""
    match, labels, events = load_game(game)
    noisy = corrupt(match, labels, SEED, **CALIBRATED)
    tags = simulate(labels, events, match.blocks, SEED)

    xy, ball, out, n_home, offsets = lay_out(noisy, labels)
    paths = asset_paths(data_dir=folder)
    os.makedirs(os.path.dirname(paths["atd_raw"]), exist_ok=True)
    write_raw(paths["atd_raw"], xy, ball)
    write_meta(paths["atd_meta"], n_home)
    write_tags(paths["smart_tagging"], tags, offsets, match.blocks)

    truth = os.path.join(folder, "rehearsal_labels.csv")
    with open(truth, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, "HADX"[code]) for frame, code in enumerate(out.tolist()))
    return truth


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder")
    parser.add_argument("--game", type=int, default=1)
    args = parser.parse_args()
    print("labels for scoring: {}".format(build(args.folder, args.game)))
