"""Which part of the method, if any, behaves differently on the real match? Aggregates only.

    python diagnose_public_window.py

The models were fitted on public games, so this checks each one separately on the one stretch of the
competition match with known labels. The ranges below are what the same models score on public games
they were not fitted on, under the calibrated noise, with room left for the window's small size.

Read it as a fault-finder and never as a way of choosing between versions: 17 possessions cannot do
that (EVALUATION.md). An accuracy far below its range, and above all one below 0.5, means something
is wired wrongly, such as a sign or a team order. It does not mean a parameter wants adjusting.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from make_submission import PUBLIC_GT, load_competition_match  # noqa: E402
from possession import pipeline  # noqa: E402
from possession.models import load_models  # noqa: E402

# Params
# (name, held-out accuracy on public games, the level below which to stop and report).
TEAM_BALL = ("team from ball, seen frames", "0.89-0.91", 0.65)
TEAM_SHAPE = ("team from shape, unseen frames", "0.72-0.78", 0.50)
DEAD = ("dead or alive", "0.85-0.87", 0.65)
TAG_TEAM = ("tag team agrees with truth", "0.80-0.95 (simulated tags)", 0.55)
MIN_FRAMES = 50


def line(spec, value, count):
    name, expected, floor = spec
    if count < MIN_FRAMES:
        print("info  {:<32} too few frames ({}) to say".format(name, count))
        return True
    passed = value >= floor
    print("{}  {:<32} {:.3f}  on {} frames  (public games: {})".format(
        "PASS" if passed else "FAIL", name, value, count, expected))
    return passed


def main(gt_path):
    truth = pm.label_codes(pm.read_label_csv(gt_path))
    match = load_competition_match()
    evidence = pipeline.compute_evidence(match, load_models(pipeline.MODEL_PATH))

    graded = truth != pm.X
    alive = (truth == pm.H) | (truth == pm.A)
    home = truth == pm.H
    seen = evidence.visible & alive
    unseen = ~evidence.visible & alive
    tagged = alive & (evidence.tag_team != 0)

    print("graded frames {}, ball seen on {:.3f} of live and {:.3f} of dead ones".format(
        int(graded.sum()), evidence.visible[alive].mean(), evidence.visible[truth == pm.D].mean()))
    results = [
        line(TEAM_BALL, np.mean((evidence.team_ball[seen] > 0) == home[seen]) if seen.any() else 0.0,
             int(seen.sum())),
        line(TEAM_SHAPE, np.mean((evidence.team_shape[unseen] > 0) == home[unseen]) if unseen.any() else 0.0,
             int(unseen.sum())),
        line(DEAD, np.mean((evidence.dead[graded] > 0) == (truth[graded] == pm.D)), int(graded.sum())),
        line(TAG_TEAM, np.mean((evidence.tag_team[tagged] > 0) == home[tagged]) if tagged.any() else 0.0,
             int(tagged.sum())),
    ]

    predicted = pipeline.decode(evidence, pipeline.load_params())
    result = pm.evaluate(truth, predicted)
    print("decoded: spellF1 {:.4f}  macroF1 {:.4f}  possessions {}/{}  alive/dead {:.3f}  team {:.3f}".format(
        result["spell_f1_mean"], result["macro_f1"], result["n_pred_spells"], result["n_gt_spells"],
        result["alive_dead_f1"], result["team_acc_both_alive"]))
    return all(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gt", default=PUBLIC_GT, help="labels to compare with; the public window by default")
    args = parser.parse_args()
    if not main(args.gt):
        print("\nSomething is wired wrongly. Report the FAIL lines, do not tune around them.")
        sys.exit(1)
    print("\nno part looks broken")
