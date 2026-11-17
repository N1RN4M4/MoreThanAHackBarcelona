"""Produce the possession file from the provided data. The one command that regenerates it.

    python make_submission.py --team <team_name>
    python evaluation/validate_submission.py evaluation/results/<team_name>.csv

Reads the tactical ATD and Smart Tagging from data/ (or MTAH_DATA_DIR), applies the committed model
and parameters, and writes evaluation/results/<team_name>.csv. Nothing is hand-set per frame.

Everything printed here is an aggregate: counts, scores, a checksum. The output of this script is
safe to share. The cache/ directory and the CSV it writes are frame-level and are not: they stay on
this machine and out of any public repository.
"""

import argparse
import hashlib
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from possession import pipeline  # noqa: E402
from possession.tracking import Match, load_cached, read_epts  # noqa: E402

# Params
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
CACHE_DIR = os.path.join(ROOT, "cache")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
# What a plausible result looks like. The first two come from the public window's rate of 3.4
# possessions per minute over 106 minutes of play, widened generously; these are a check on the run
# and are not tuned against. A result outside them means something went wrong, not that it is bad.
PLAUSIBLE_POSSESSIONS = (150, 600)
PLAUSIBLE_MEAN_SECONDS = (6.0, 30.0)
PLAUSIBLE_DEAD_SHARE = (0.10, 0.60)
# The published spellF1 of the best baseline on the public window. Below it, something is broken.
PUBLIC_WINDOW_FLOOR = 0.3256


def load_competition_match(game=GAME, data_dir=None, use_cache=True):
    """The tactical recording as a Match, Smart Tagging included."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    pitch = GAMES[game]["pitch"]

    def builder():
        print("parsing {} (a minute or two) ...".format(os.path.basename(raw_path)), flush=True)
        return read_epts(raw_path, meta_path, n_frames, pitch)

    if use_cache:
        cache_name = "{}_tactical_{}.npz".format(game, os.path.getsize(raw_path))
        xy, ball_xy, team = load_cached(os.path.join(CACHE_DIR, cache_name), builder)
    else:
        xy, ball_xy, team = builder()

    blocks = [(first, last + 1) for _, first, last in half_video_blocks(game)]
    from starter.load_smart_tagging import load as load_tags
    tags = load_tags(game, data_dir)
    return Match(xy, ball_xy, team, blocks, pitch, tags=tags, name=game)


def write_submission(path, labels):
    """frame,label, with \\n line endings on every platform so the bytes are the same everywhere."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    letters = pipeline.LABEL_LETTERS
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, letters[code])
                          for frame, code in enumerate(labels.tolist()))


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarise(labels, blocks, public_gt_path=PUBLIC_GT):
    """Aggregates only. Returns the list of sanity problems found; empty means none."""
    problems = []
    spells = pm.possession_spells(labels, blocks)
    graded = sum(stop - start for start, stop in blocks)
    n_dead = sum(int((labels[start:stop] == pm.D).sum()) for start, stop in blocks)
    lengths = np.array([stop - start for start, stop, _ in spells], dtype=np.float64) / 25.0
    mean_seconds = float(lengths.mean()) if len(lengths) else 0.0
    dead_share = n_dead / float(graded)
    print("possessions {}   mean {:.1f} s   median {:.1f} s   dead share {:.3f}".format(
        len(spells), mean_seconds, float(np.median(lengths)) if len(lengths) else 0.0, dead_share))

    def check(name, value, bounds):
        if not bounds[0] <= value <= bounds[1]:
            problems.append("{} {} is outside {}".format(name, value, bounds))

    check("possession count", len(spells), PLAUSIBLE_POSSESSIONS)
    check("mean possession seconds", round(mean_seconds, 2), PLAUSIBLE_MEAN_SECONDS)
    check("dead share", round(dead_share, 3), PLAUSIBLE_DEAD_SHARE)

    if public_gt_path and os.path.isfile(public_gt_path):
        truth = pm.label_codes(pm.read_label_csv(public_gt_path))
        result = pm.evaluate(truth, labels)
        print("public 5 minutes (a correctness check, never a target): spellF1 {:.4f}  macroF1 {:.4f}  "
              "possessions {}/{}".format(result["spell_f1_mean"], result["macro_f1"],
                                         result["n_pred_spells"], result["n_gt_spells"]))
        if result["spell_f1_mean"] < PUBLIC_WINDOW_FLOOR:
            problems.append("public-window spellF1 {:.4f} is below the best baseline's {:.4f}".format(
                result["spell_f1_mean"], PUBLIC_WINDOW_FLOOR))
    return problems


def run(team, data_dir=None, out_dir=RESULTS_DIR, use_cache=True, public_gt_path=PUBLIC_GT):
    """Write the file, print the aggregates. Returns (path, checksum, sanity problems)."""
    match = load_competition_match(data_dir=data_dir, use_cache=use_cache)
    print("computing evidence and decoding ...", flush=True)
    labels = pipeline.predict(match)

    path = os.path.join(out_dir, "{}.csv".format(team))
    write_submission(path, labels)
    problem = pm.validate_submission(pm.read_label_csv(path), source_frames("tactical"))
    if problem:
        raise ValueError("the written file failed validation: {}".format(problem))

    problems = summarise(labels, match.blocks, public_gt_path)
    checksum = sha256_of(path)
    print("wrote {}\nsha256 {}".format(path, checksum))
    return path, checksum, problems


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", required=True, help="the name on the board; names the file")
    parser.add_argument("--out", default=RESULTS_DIR, help="folder to write into")
    parser.add_argument("--no-cache", action="store_true", help="parse the tracking afresh")
    args = parser.parse_args()

    _, _, found = run(args.team, out_dir=args.out, use_cache=not args.no_cache)
    if found:
        print("\nSANITY CHECK FAILED -- do not submit this file:")
        for item in found:
            print("  " + item)
        sys.exit(1)
    print("\nsanity checks passed")
