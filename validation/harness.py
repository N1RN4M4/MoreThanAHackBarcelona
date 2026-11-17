"""Shared by the validation scripts: noisy public games, their features, and fitting the models.

Leave-one-game-out throughout: a game is never scored by models fitted on it, so the numbers say
something about a match the method has not seen, which is the situation it will be in.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from corrupt import CALIBRATED, corrupt  # noqa: E402
from open_games import GAMES, load_game  # noqa: E402
from possession import features as feature_module  # noqa: E402
from possession.models import fit_logistic  # noqa: E402

# Params
# Every 5th frame is plenty to fit on: neighbouring frames are nearly the same observation.
FIT_STRIDE = 5
TRAIN_SEEDS = (11, 12)
TEST_SEED = 101


def noisy_copy(game, seed, params=None):
    """(noisy Match, labels, events) for one public game."""
    match, labels, events = load_game(game)
    return corrupt(match, labels, seed, **(params or CALIBRATED)), labels, events


def fit_models(samples):
    """The three heads from [(Features, labels)], taking every FIT_STRIDE-th graded frame."""
    ball_X, ball_y, shape_X, shape_y, dead_X, dead_y = [], [], [], [], [], []
    for built, labels in samples:
        index = np.arange(len(labels))[::FIT_STRIDE]
        index = index[labels[index] != pm.X]
        alive = (labels[index] == pm.H) | (labels[index] == pm.A)

        frames = index[alive]
        y = (labels[frames] == pm.H).astype(np.float64)
        seen = built.visible[frames]
        ball_X.append(np.hstack([built.team_ball[frames[seen]], built.team_shape[frames[seen]]]))
        ball_y.append(y[seen])
        shape_X.append(built.team_shape[frames[~seen]])
        shape_y.append(y[~seen])

        dead_X.append(built.dead[index])
        dead_y.append((labels[index] == pm.D).astype(np.float64))

    return {
        "team_ball": fit_logistic(np.vstack(ball_X), np.concatenate(ball_y), intercept=False),
        "team_shape": fit_logistic(np.vstack(shape_X), np.concatenate(shape_y), intercept=False),
        "dead": fit_logistic(np.vstack(dead_X), np.concatenate(dead_y), intercept=True),
    }


def score(labels, predicted, bootstrap=0):
    return pm.evaluate(labels, predicted, bootstrap_n=bootstrap)


def zero_columns(built, drop):
    """A copy of Features with the named columns silenced, for measuring what they contribute."""
    copy = feature_module.Features(built.team_ball.copy(), built.team_shape.copy(), built.dead.copy(),
                                   built.visible, built.names)
    for matrix, key in ((copy.team_ball, "team_ball"), (copy.team_shape, "team_shape"),
                        (copy.dead, "dead")):
        for index, name in enumerate(built.names[key]):
            if any(name == item or (item.endswith("*") and name.startswith(item[:-1])) for item in drop):
                matrix[:, index] = 0.0
    return copy


def describe(result):
    return ("macroF1 {:.4f}  spellF1 {:.4f}  poss {:4d}/{:4d}  alive/dead {:.3f}  team {:.3f}".format(
        result["macro_f1"], result["spell_f1_mean"], result["n_pred_spells"], result["n_gt_spells"],
        result["alive_dead_f1"], result["team_acc_both_alive"]))
