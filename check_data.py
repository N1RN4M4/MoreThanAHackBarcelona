"""Does the data on this machine match what the pipeline was written for? Run this first.

    python check_data.py

The pipeline was written without access to the match data, from the published format description and
public sample games. This compares the two wherever that can be done without looking at the data.
Every line printed is PASS or FAIL with an aggregate beside it, and nothing frame-level is ever
printed, so the output is safe to share. Exit code 0 means every check passed.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from games import GAME, GAMES, asset_paths, half_video_blocks, source_frames  # noqa: E402
from possession.tracking import HOME, read_metadata  # noqa: E402

# Params
# Published in DATA.md.
EXPECTED_FRAMES = 199319
EXPECTED_VISIBLE_ALL = 0.569
EXPECTED_VISIBLE_GRADED = 0.593
EXPECTED_TRACKED_GRADED = 20.5
EXPECTED_SLOTS = 46
EXPECTED_TAGS = 638
SECOND_HALF_KICK_OFF = 113742
TOLERANCE_SHARE = 0.01
TOLERANCE_TRACKED = 0.5
# How often this parser's nearest team must equal the committed baseline's. Not 1.0: the two compute
# distances at different float widths, so an exact tie can break the other way, but only rarely.
MIN_BASELINE_AGREEMENT = 0.995

_failures = []


def report(name, passed, detail=""):
    print("{}  {:<34} {}".format("PASS" if passed else "FAIL", name, detail))
    if not passed:
        _failures.append(name)


def check_files(paths):
    missing = [name for name, path in paths.items() if not os.path.isfile(path)]
    report("files present", not missing, "missing: {}".format(missing) if missing else str(len(paths)))
    return not missing


def check_line_format(raw_path):
    """Shape of the raw file, from counts alone. Never prints a line of it."""
    n_lines = bad_lines = 0
    ball_values, player_values, player_groups = set(), set(), set()
    first_id = last_id = None
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split(":")
            n_lines += 1
            if len(parts) != 3:
                bad_lines += 1
                continue
            frame_id = int(parts[0])
            first_id = frame_id if first_id is None else first_id
            last_id = frame_id
            groups = parts[1].split(";")
            player_groups.add(len(groups))
            if n_lines % 1000 == 0:
                player_values.update(len(group.split(",")) for group in groups)
                ball_values.add(len(parts[2].split(",")))
    report("line count", n_lines == EXPECTED_FRAMES, str(n_lines))
    report("lines split into 3 fields", bad_lines == 0, "{} bad".format(bad_lines))
    report("frame ids run 1..n", first_id == 1 and last_id == n_lines,
           "{}..{}".format(first_id, last_id))
    report("slots per line", player_groups == {EXPECTED_SLOTS}, str(sorted(player_groups)))
    print("info  values per player group {}   per ball group {}".format(
        sorted(player_values), sorted(ball_values)))


def check_metadata(meta_path):
    """Team order and, by name only, any data channel beyond position and speed."""
    root = ET.parse(meta_path).getroot()
    team_of_slot, slot_orders = read_metadata(meta_path)
    n_home = int((team_of_slot == HOME).sum())
    report("two teams, every slot assigned", len(team_of_slot) == EXPECTED_SLOTS,
           "{} slots, {} home, {} away".format(len(team_of_slot), n_home, len(team_of_slot) - n_home))
    names = [team.findtext("Name") for team in root.iter("Team")]
    print("info  teams in file order: {}".format(names))
    report("home team is listed first", bool(names) and "barcelona" in (names[0] or "").lower(),
           str(names[:1]))
    print("info  format specifications {}, columns named {}".format(
        len(slot_orders), sorted({len(order) for _, _, order in slot_orders})))
    # If the export ever carried the ball's state or its owner, it would be listed here.
    channels = sorted({element.get("id", "").split("_")[-1] for element in root.iter("Channel")})
    print("info  channel kinds in the metadata: {}".format(channels))
    sensors = sorted({element.get("id", "") for element in root.iter("Sensor")})
    print("info  sensors in the metadata: {}".format(sensors))


def check_tracking(game, data_dir=None):
    """Visibility and object counts against DATA.md, and the parser against the committed baseline."""
    from calibrate_noise import nearest_team_codes
    from make_submission import load_competition_match

    match = load_competition_match(game, data_dir)
    visible = np.isfinite(match.ball_xy[:, 0])
    graded = match.graded_mask()
    share_all = visible.mean()
    share_graded = visible[graded].mean()
    report("ball visible, all frames", abs(share_all - EXPECTED_VISIBLE_ALL) < TOLERANCE_SHARE,
           "{:.3f} (DATA.md {:.3f})".format(share_all, EXPECTED_VISIBLE_ALL))
    report("ball visible, graded frames", abs(share_graded - EXPECTED_VISIBLE_GRADED) < TOLERANCE_SHARE,
           "{:.3f} (DATA.md {:.3f})".format(share_graded, EXPECTED_VISIBLE_GRADED))
    tracked = np.isfinite(match.xy[graded][:, :, 0]).sum(axis=1).mean()
    report("objects tracked, graded frames", abs(tracked - EXPECTED_TRACKED_GRADED) < TOLERANCE_TRACKED,
           "{:.1f} (DATA.md {:.1f})".format(tracked, EXPECTED_TRACKED_GRADED))

    finite = match.xy[graded][np.isfinite(match.xy[graded][:, :, 0])]
    length, width = match.pitch
    inside = np.mean((finite[:, 0] > -5) & (finite[:, 0] < length + 5)
                     & (finite[:, 1] > -5) & (finite[:, 1] < width + 5))
    report("players are on the pitch", inside > 0.99, "{:.4f} inside".format(inside))
    ball = match.ball_xy[graded & visible]
    outside = np.mean((ball[:, 0] < 0) | (ball[:, 0] > length) | (ball[:, 1] < 0) | (ball[:, 1] > width))
    print("info  share of seen ball positions beyond the lines: {:.4f}".format(outside))

    baseline_path = os.path.join(ROOT, "evaluation", "results", "baseline_raw.csv")
    baseline = pm.label_codes(pm.read_label_csv(baseline_path))
    agreement = np.mean(nearest_team_codes(match) == baseline)
    report("parser agrees with baseline_raw.csv", agreement >= MIN_BASELINE_AGREEMENT,
           "{:.5f} of frames".format(agreement))

    tags = match.tags
    report("Smart Tagging entries", len(tags) == EXPECTED_TAGS, str(len(tags)))
    kick_offs = [entry for entry in tags
                 if (entry["tags"].get("Type") or "").strip().lower() == "kick off (start)"]
    contains = any(entry["start_frame"] <= SECOND_HALF_KICK_OFF <= entry["end_frame"]
                   for entry in kick_offs)
    report("a kick-off tag covers the 2nd half's", contains, "{} kick-off tags".format(len(kick_offs)))
    names = sorted({(entry["tags"].get("Team") or "") for entry in tags})
    print("info  Team tag values: {}".format(names))
    report("Team tags are the expected names",
           {"barcelona", "atletic club"} <= {name.strip().lower() for name in names}, "")


if __name__ == '__main__':
    sys.path.insert(0, os.path.join(ROOT, "validation"))
    paths = asset_paths(GAME)
    if check_files({name: paths[name] for name in ("atd_meta", "atd_raw", "smart_tagging")}):
        report("frame count in games.py", source_frames("tactical") == EXPECTED_FRAMES, "")
        report("graded blocks", half_video_blocks(GAME) == [(1, 11185, 90651), (2, 110808, 190197)], "")
        report("pitch in games.py", GAMES[GAME]["pitch"] == (105.3, 68.0), str(GAMES[GAME]["pitch"]))
        check_metadata(paths["atd_meta"])
        check_line_format(paths["atd_raw"])
        check_tracking(GAME)

    print()
    if _failures:
        print("{} CHECK(S) FAILED: {}".format(len(_failures), _failures))
        print("Do not run make_submission.py until each of these is understood.")
        sys.exit(1)
    print("all checks passed")
