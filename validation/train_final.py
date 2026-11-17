"""Fit the models the submission uses, on all the public games, and write possession/model.json.

    python validation/train_final.py

Nothing from the competition match goes into this. The coefficients are a plain JSON file, so the
organisers can read them, and regenerating them from the public games needs nothing gated.

It also prints whether each model's largest coefficients keep their sign when any one game is left
out. A coefficient that flips with the choice of games is noise, and one that holds is a statement
about football that has a chance of being true of this match too.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import harness as hz  # noqa: E402
from possession import features as feature_module  # noqa: E402
from possession.models import save_models  # noqa: E402
from possession.pipeline import MODEL_PATH  # noqa: E402

# Params
TOP_COLUMNS = 12


if __name__ == '__main__':
    samples = []
    names = None
    for game in hz.GAMES:
        for seed in hz.TRAIN_SEEDS:
            match, labels, _ = hz.noisy_copy(game, seed)
            built = feature_module.build(match)
            samples.append((game, built, labels))
            names = built.names

    final = hz.fit_models([(built, labels) for _, built, labels in samples])
    save_models(MODEL_PATH, final)
    print("wrote {}".format(os.path.relpath(MODEL_PATH, ROOT)))

    folds = {game: hz.fit_models([(built, labels) for g, built, labels in samples if g != game])
             for game in hz.GAMES}
    columns = {"team_ball": names["team_ball"] + names["team_shape"],
               "team_shape": names["team_shape"],
               "dead": names["dead"]}
    for head, head_columns in columns.items():
        coef = np.array(final[head]["coef"])
        fold_coef = np.array([folds[game][head]["coef"] for game in hz.GAMES])
        print("\n{}  intercept {:+.3f}".format(head, final[head]["intercept"]))
        for index in np.argsort(-np.abs(coef))[:TOP_COLUMNS]:
            stable = bool(np.all(np.sign(fold_coef[:, index]) == np.sign(coef[index])))
            print("  {:<22} {:+.3f}  {}".format(head_columns[index], coef[index],
                                               "stable" if stable else "SIGN FLIPS BETWEEN FOLDS"))
