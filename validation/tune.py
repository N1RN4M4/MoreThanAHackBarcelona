"""Choose the decoder's parameters on the public games, and say how well that generalises.

    python validation/tune.py                  # about 10 minutes on many cores
    python validation/tune.py --trials 40      # a quick look

The search is scored on several noise levels at once, so a choice has to work when the ball is
harder to see than the calibration assumes as well as when it matches. The figure to believe is the
nested one: parameters chosen on two games and scored on the third. The public five minutes of the
competition match take no part in any of this (EVALUATION.md).
"""

import argparse
import json
import multiprocessing
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import harness as hz  # noqa: E402
import possession_metrics as pm  # noqa: E402
from possession import features as feature_module  # noqa: E402
from possession import pipeline  # noqa: E402
from simulate_tags import simulate  # noqa: E402

# Params
# The calibrated noise, then harder: more of the ball lost in play, and more dead balls in view.
SCENARIOS = {
    "calibrated": hz.CALIBRATED,
    "dead_ball_visible": dict(hz.CALIBRATED, dead_visible=0.5),
    "more_gaps": dict(hz.CALIBRATED, alive_gap_share=0.2),
}
# Log-uniform ranges of the random search.
RANGES = {
    "team_ball_temperature": (0.02, 1.0),
    "team_shape_temperature": (0.0003, 0.3),
    "dead_temperature": (0.01, 1.0),
    "switch_cost": (0.3, 12.0),
    "to_dead_cost": (0.3, 12.0),
    "from_dead_cost": (0.3, 12.0),
}
BIAS_RANGE = (-0.3, 0.3)
TAG_TEAM_WEIGHT = 0.02
TAG_DEAD_WEIGHT = 0.05

_EVIDENCE = None


def _init_worker(evidence):
    global _EVIDENCE
    _EVIDENCE = evidence


def _evaluate_trial(params):
    """spellF1 per (game, scenario) under one set of parameters."""
    out = {}
    for key, (evidence, labels) in _EVIDENCE.items():
        predicted = pipeline.decode(evidence, params)
        out[key] = pm.spell_scores(labels, predicted, evidence.blocks)["mean_f1"]
    return params, out


def sample_params(rng):
    params = {name: float(np.exp(rng.uniform(np.log(low), np.log(high))))
              for name, (low, high) in sorted(RANGES.items())}
    params["dead_bias"] = float(rng.uniform(*BIAS_RANGE))
    params["tag_team_weight"] = TAG_TEAM_WEIGHT
    params["tag_dead_weight"] = TAG_DEAD_WEIGHT
    return params


def build_evidence(use_tags):
    """{(game, scenario): (Evidence from models fitted on the other games, labels)}."""
    train = []
    for game in hz.GAMES:
        for seed in hz.TRAIN_SEEDS:
            match, labels, _ = hz.noisy_copy(game, seed)
            train.append((game, feature_module.build(match), labels))

    evidence = {}
    for game in hz.GAMES:
        models = hz.fit_models([(built, labels) for g, built, labels in train if g != game])
        for name, noise in SCENARIOS.items():
            match, labels, events = hz.noisy_copy(game, hz.TEST_SEED, noise)
            if use_tags:
                match.tags = simulate(labels, events, match.blocks, hz.TEST_SEED)
            evidence[(game, name)] = (pipeline.compute_evidence(match, models), labels)
        print("evidence ready for game {}".format(game), flush=True)
    return evidence


def nested_report(results):
    """Choose on two games, score on the third, for each third."""
    keys = list(results[0][1])
    scores = []
    for game in hz.GAMES:
        train_keys = [key for key in keys if key[0] != game]
        test_keys = [key for key in keys if key[0] == game]
        best = max(results, key=lambda item: np.mean([item[1][key] for key in train_keys]))
        line = {key[1]: best[1][key] for key in test_keys}
        scores.append(line)
        print("held-out game {}: ".format(game)
              + "  ".join("{} {:.4f}".format(name, value) for name, value in sorted(line.items())))
    for name in SCENARIOS:
        print("  mean {:<20} {:.4f}".format(name, np.mean([line[name] for line in scores])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-tags", action="store_true", help="tune without simulated tags")
    parser.add_argument("--out", default=pipeline.PARAMS_PATH)
    args = parser.parse_args()

    rng_ = np.random.RandomState(args.seed)
    trials = [sample_params(rng_) for _ in range(args.trials)]
    evidence_ = build_evidence(not args.no_tags)

    with multiprocessing.Pool(args.workers, initializer=_init_worker, initargs=(evidence_,)) as pool:
        outcomes = []
        for index, outcome in enumerate(pool.imap_unordered(_evaluate_trial, trials, chunksize=4)):
            outcomes.append(outcome)
            if (index + 1) % 50 == 0:
                print("  {}/{}  best so far {:.4f}".format(
                    index + 1, len(trials),
                    max(np.mean(list(o[1].values())) for o in outcomes)), flush=True)

    # Trial order from the pool varies; sort so the choice among ties does not.
    outcomes.sort(key=lambda item: json.dumps(item[0], sort_keys=True))
    print("\nNESTED: chosen on two games, scored on the third")
    nested_report(outcomes)

    best_params, best_scores = max(outcomes, key=lambda item: np.mean(list(item[1].values())))
    print("\nALL GAMES: best mean {:.4f}".format(np.mean(list(best_scores.values()))))
    for key in sorted(best_scores):
        print("  game {} {:<20} {:.4f}".format(key[0], key[1], best_scores[key]))
    print(json.dumps(best_params, indent=1, sort_keys=True))
    with open(args.out, "w") as handle:
        json.dump({name: round(value, 6) for name, value in best_params.items()}, handle,
                  indent=1, sort_keys=True)
        handle.write("\n")
    print("wrote {}".format(os.path.relpath(args.out, ROOT)))
