"""Features, models and decoder, as one function from a Match to one label per frame."""

import json
import os

import numpy as np

from possession import decoder, features as feature_module
from possession.models import load_models, predict_logit
from possession.tagging import tag_evidence

# Params
HERE = os.path.dirname(os.path.abspath(__file__))
# Fitted by validation/train_final.py on the public games. Never on the competition match.
MODEL_PATH = os.path.join(HERE, "model.json")
# Chosen by validation/tune.py on the public games. Never on the public five minutes of the
# competition match, which cannot tell two versions of a method apart (EVALUATION.md).
PARAMS_PATH = os.path.join(HERE, "params.json")
# Rounded before decoding, so a difference in the last bits of a floating-point sum on another
# machine cannot move a decision: RULES.md asks for the same file.
EVIDENCE_DECIMALS = 4
# Labels are 0, 1, 2 here and in evaluation/possession_metrics.py, with X = 3 outside the halves.
X_LABEL = 3
LABEL_LETTERS = "HADX"


class Evidence(object):
    """The per-frame log-odds. Decoding is far cheaper than this, so tuning reuses one of these."""

    def __init__(self, team_ball, team_shape, dead, visible, tag_team, tag_dead, blocks):
        self.team_ball = team_ball
        self.team_shape = team_shape
        self.dead = dead
        self.visible = visible
        self.tag_team = tag_team
        self.tag_dead = tag_dead
        self.blocks = blocks


def load_params(path=PARAMS_PATH):
    with open(path) as handle:
        return json.load(handle)


def compute_evidence(match, models, built=None):
    """Evidence for one Match. `built` lets a caller reuse features it already has."""
    built = built or feature_module.build(match)
    team_ball = predict_logit(models["team_ball"],
                              np.hstack([built.team_ball, built.team_shape]))
    team_shape = predict_logit(models["team_shape"], built.team_shape)
    dead = predict_logit(models["dead"], built.dead)
    tag_team, tag_dead = tag_evidence(match.tags, match.n_frames, match.blocks)
    return Evidence(team_ball, team_shape, dead, built.visible, tag_team, tag_dead, match.blocks)


def decode(evidence, params):
    """Coded labels per frame: 0 H, 1 A, 2 D inside the halves, 3 X outside them."""
    n = len(evidence.dead)
    team = np.where(evidence.visible,
                    evidence.team_ball * params["team_ball_temperature"],
                    evidence.team_shape * params["team_shape_temperature"])
    team = team + evidence.tag_team * params["tag_team_weight"]
    dead = (evidence.dead * params["dead_temperature"] + params["dead_bias"]
            + evidence.tag_dead * params["tag_dead_weight"])

    table = np.round(np.column_stack([team, -team, dead]), EVIDENCE_DECIMALS)
    costs = decoder.transition_costs(params["switch_cost"], params["to_dead_cost"],
                                     params["from_dead_cost"])
    labels = np.full(n, X_LABEL, dtype=np.int8)
    for start, stop in evidence.blocks:
        labels[start:stop] = decoder.viterbi(table[start:stop], costs)
    return labels


def predict(match, models=None, params=None):
    """Coded labels for a whole Match, from the committed model and parameters by default."""
    models = models or load_models(MODEL_PATH)
    return decode(compute_evidence(match, models), params or load_params())
