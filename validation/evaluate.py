"""What the method scores on matches it has not seen, and what each part contributes.

    python validation/evaluate.py

Leave-one-game-out on the noisy public games, against the reference baseline on the same noisy
tracking. Every row uses the committed parameters in possession/params.json, which were chosen with
all the games in view, so for a single honest figure read the nested result of tune.py. These rows
are for comparing the components with each other, and a gap under about 0.04 is not a result
(EVALUATION.md).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import harness as hz  # noqa: E402
from calibrate_noise import nearest_team_codes, smooth_codes  # noqa: E402
from possession import features as feature_module  # noqa: E402
from possession import pipeline  # noqa: E402
from simulate_tags import simulate  # noqa: E402

# Params
BOOTSTRAP_DRAWS = 1000
SHAPE_COLUMNS = ("spread_*", "area_*", "height_*", "speed_*", "forward_speed_*", "opponent_gap_*", "front_*")
CARRY_COLUMNS = ("last_25", "last_125", "last_500", "next_25", "next_125", "next_500")
# Each variant: which columns are silenced, and how good the simulated tags are (None: no tags).
GOOD_TAGS = {}
BAD_TAGS = {"set_piece_recall": 0.5, "phase_recall": 0.4, "wrong_team": 0.3}
VARIANTS = (
    ("full method", (), GOOD_TAGS),
    ("no Smart Tagging", (), None),
    ("poor Smart Tagging", (), BAD_TAGS),
    ("no team shape", SHAPE_COLUMNS, GOOD_TAGS),
    ("no carried control", CARRY_COLUMNS, GOOD_TAGS),
    ("no ball-half column", ("ball_half",), GOOD_TAGS),
)


def main():
    params = pipeline.load_params()
    train = []
    for game in hz.GAMES:
        for seed in hz.TRAIN_SEEDS:
            match, labels, _ = hz.noisy_copy(game, seed)
            train.append((game, feature_module.build(match), labels))

    test = {}
    baseline = []
    for game in hz.GAMES:
        match, labels, events = hz.noisy_copy(game, hz.TEST_SEED)
        test[game] = (match, labels, events, feature_module.build(match))
        baseline.append(hz.score(labels, smooth_codes(nearest_team_codes(match), 50), BOOTSTRAP_DRAWS))
    print("{:<24} {:>7} {:>7} {:>7} {:>8} {:>8}   {}".format(
        "variant", "game 1", "game 2", "game 3", "spellF1", "macroF1", "possessions predicted / true"))
    print("{:<24} {:>7.4f} {:>7.4f} {:>7.4f} {:>8.4f} {:>8.4f}   {}".format(
        "baseline_smooth50", baseline[0]["spell_f1_mean"], baseline[1]["spell_f1_mean"],
        baseline[2]["spell_f1_mean"], np.mean([r["spell_f1_mean"] for r in baseline]),
        np.mean([r["macro_f1"] for r in baseline]),
        " ".join("{}/{}".format(r["n_pred_spells"], r["n_gt_spells"]) for r in baseline)))

    for name, drop, tag_quality in VARIANTS:
        results = []
        for game in hz.GAMES:
            models = hz.fit_models([(hz.zero_columns(built, drop), labels)
                                    for g, built, labels in train if g != game])
            match, labels, events, built = test[game]
            match.tags = (None if tag_quality is None
                          else simulate(labels, events, match.blocks, hz.TEST_SEED, **tag_quality))
            evidence = pipeline.compute_evidence(match, models, built=hz.zero_columns(built, drop))
            results.append(hz.score(labels, pipeline.decode(evidence, params), BOOTSTRAP_DRAWS))
        print("{:<24} {:>7.4f} {:>7.4f} {:>7.4f} {:>8.4f} {:>8.4f}   {}".format(
            name, results[0]["spell_f1_mean"], results[1]["spell_f1_mean"],
            results[2]["spell_f1_mean"], np.mean([r["spell_f1_mean"] for r in results]),
            np.mean([r["macro_f1"] for r in results]),
            " ".join("{}/{}".format(r["n_pred_spells"], r["n_gt_spells"]) for r in results)), flush=True)
        if name == "full method":
            for game, result in zip(hz.GAMES, results):
                print("    game {}: 90% interval {:.3f}-{:.3f}   {}".format(
                    game, result["spell_bootstrap_low"], result["spell_bootstrap_high"],
                    hz.describe(result)))


if __name__ == '__main__':
    main()
