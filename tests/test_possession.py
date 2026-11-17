"""Unit tests for possession/. Standard library and numpy only, and no match data.

    python -m unittest discover -s tests
"""

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from possession import decoder, features, pipeline  # noqa: E402
from possession.models import fit_logistic, predict_logit  # noqa: E402
from possession.tagging import tag_evidence  # noqa: E402
from possession.tracking import AWAY, HOME, Match, derived_speed, read_epts, read_metadata  # noqa: E402

META = """<?xml version="1.0" encoding="utf-8"?>
<main><Metadata>
<Teams><Team id="T1"><Name>Home</Name></Team><Team id="T2"><Name>Away</Name></Team></Teams>
<Players>
<Player id="P1" teamId="T1"/><Player id="P2" teamId="T2"/><Player id="P3" teamId="T1"/>
</Players>
<PlayerChannels>
<PlayerChannel id="p1_x" playerId="P1"/><PlayerChannel id="p1_y" playerId="P1"/>
<PlayerChannel id="p2_x" playerId="P2"/><PlayerChannel id="p2_y" playerId="P2"/>
<PlayerChannel id="p3_x" playerId="P3"/><PlayerChannel id="p3_y" playerId="P3"/>
</PlayerChannels>
<DataFormatSpecifications>
<DataFormatSpecification startFrame="1" endFrame="2">
<PlayerChannelRef playerChannelId="p1_x"/><PlayerChannelRef playerChannelId="p1_y"/>
<PlayerChannelRef playerChannelId="p2_x"/><PlayerChannelRef playerChannelId="p2_y"/>
</DataFormatSpecification>
<DataFormatSpecification startFrame="3" endFrame="4">
<PlayerChannelRef playerChannelId="p3_x"/><PlayerChannelRef playerChannelId="p3_y"/>
<PlayerChannelRef playerChannelId="p2_x"/><PlayerChannelRef playerChannelId="p2_y"/>
</DataFormatSpecification>
</DataFormatSpecifications>
</Metadata></main>
"""


def synthetic_match(n_frames=3000, seed=0):
    """Two teams drifting about, a ball that is sometimes lost. Nothing here resembles football."""
    rng = np.random.RandomState(seed)
    steps = rng.normal(0.0, 0.05, (n_frames, 22, 2))
    xy = (rng.uniform(10.0, 60.0, (22, 2)) + np.cumsum(steps, axis=0)).astype(np.float32)
    ball = (np.array([50.0, 30.0]) + np.cumsum(rng.normal(0.0, 0.2, (n_frames, 2)), axis=0)).astype(np.float32)
    ball[700:1100] = np.nan
    xy[400:600, 3] = np.nan
    team = np.array([HOME] * 11 + [AWAY] * 11, dtype=np.int8)
    return Match(xy, ball, team, [(0, 1500), (1500, n_frames)], (105.3, 68.0))


class ParserTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.meta = os.path.join(self.folder, "meta.xml")
        self.raw = os.path.join(self.folder, "raw.txt")
        with open(self.meta, "w") as handle:
            handle.write(META)

    def tearDown(self):
        shutil.rmtree(self.folder)

    def test_team_order_and_specifications(self):
        team, orders = read_metadata(self.meta)
        self.assertEqual(team.tolist(), [HOME, AWAY, HOME])
        self.assertEqual(orders, [(1, 2, [0, 1]), (3, 4, [2, 1])])

    def test_columns_follow_the_specification_for_that_frame(self):
        with open(self.raw, "w") as handle:
            handle.write("3:0.1,0.2;0.3,0.4:0.5,0.5\n")
        xy, ball, _ = read_epts(self.raw, self.meta, 4, (100.0, 50.0))
        np.testing.assert_allclose(xy[2, 2], [10.0, 10.0], atol=1e-3)  # first column is slot 3 here
        np.testing.assert_allclose(xy[2, 1], [30.0, 20.0], atol=1e-3)
        self.assertTrue(np.isnan(xy[2, 0, 0]))
        np.testing.assert_allclose(ball[2], [50.0, 25.0], atol=1e-3)

    def test_speed_column_and_missing_groups(self):
        with open(self.raw, "w") as handle:
            handle.write("1:0.1,0.2,7.5;NaN,NaN,NaN:NaN,NaN,NaN\n")
            handle.write("2:NaN,NaN,NaN;0.3,0.4,1.0:0.5,0.5,12.0\n")
            handle.write("not a tracking line\n")
        xy, ball, _ = read_epts(self.raw, self.meta, 4, (100.0, 50.0))
        np.testing.assert_allclose(xy[0, 0], [10.0, 10.0], atol=1e-3)
        self.assertTrue(np.isnan(xy[0, 1, 0]))
        self.assertTrue(np.isnan(ball[0, 0]))
        np.testing.assert_allclose(ball[1], [50.0, 25.0], atol=1e-3)

    def test_frame_id_is_one_based(self):
        with open(self.raw, "w") as handle:
            handle.write("1:0.1,0.2;0.3,0.4:0.5,0.5\n")
        xy, _, _ = read_epts(self.raw, self.meta, 4, (100.0, 50.0))
        self.assertFalse(np.isnan(xy[0, 0, 0]))
        self.assertTrue(np.isnan(xy[1, 0, 0]))


class DecoderTest(unittest.TestCase):
    costs = decoder.transition_costs(1.0, 1.0, 1.0)

    def test_follows_strong_evidence(self):
        emissions = np.zeros((30, 3))
        emissions[:10, decoder.H] = 1.0
        emissions[10:20, decoder.D] = 1.0
        emissions[20:, decoder.A] = 1.0
        path = decoder.viterbi(emissions, self.costs)
        self.assertEqual(path.tolist(), [0] * 10 + [2] * 10 + [1] * 10)

    def test_a_blip_does_not_earn_two_switches(self):
        emissions = np.zeros((30, 3))
        emissions[:, decoder.H] = 0.1
        emissions[15, decoder.A] = 0.5
        self.assertTrue((decoder.viterbi(emissions, self.costs) == decoder.H).all())

    def test_a_change_waits_for_the_new_team_on_a_tie(self):
        # H evidence, a ball in flight with no evidence at all, then A: the change belongs at the
        # first A frame, because possession stays with the last team that had it.
        emissions = np.zeros((30, 3))
        emissions[:10, decoder.H] = 1.0
        emissions[:10, decoder.A] = -1.0
        emissions[20:, decoder.A] = 1.0
        emissions[20:, decoder.H] = -1.0
        emissions[:, decoder.D] = -5.0
        path = decoder.viterbi(emissions, self.costs)
        self.assertEqual(int(np.flatnonzero(path == decoder.A)[0]), 20)

    def test_empty(self):
        self.assertEqual(len(decoder.viterbi(np.zeros((0, 3)), self.costs)), 0)


class FeaturesTest(unittest.TestCase):
    def test_window_mean_ignores_nan_and_stays_in_its_block(self):
        values = np.array([1.0, np.nan, 3.0, 100.0])
        out = features.window_mean(values, -1, 1, [(0, 3), (3, 4)])
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0, 100.0])

    def test_shapes(self):
        match = synthetic_match()
        built = features.build(match)
        for matrix in (built.team_ball, built.team_shape, built.dead):
            self.assertEqual(matrix.shape[0], match.n_frames)
            self.assertTrue(np.isfinite(matrix).all())
        self.assertEqual(built.team_ball.shape[1], len(built.names["team_ball"]))
        self.assertEqual(built.team_shape.shape[1], len(built.names["team_shape"]))
        self.assertEqual(built.dead.shape[1], len(built.names["dead"]))

    def test_swapping_the_teams_flips_team_columns_and_leaves_dead_alone(self):
        # The property that lets a model fitted on other clubs apply here: no column can tell which
        # side is the home side.
        match = synthetic_match()
        swapped = Match(match.xy, match.ball_xy, 1 - match.team, match.blocks, match.pitch)
        a, b = features.build(match), features.build(swapped)
        np.testing.assert_allclose(a.team_ball, -b.team_ball, atol=1e-3)
        np.testing.assert_allclose(a.team_shape, -b.team_shape, atol=1e-3)
        np.testing.assert_allclose(a.dead, b.dead, atol=1e-3)

    def test_ball_columns_are_silent_when_the_ball_is_lost(self):
        built = features.build(synthetic_match())
        self.assertTrue((built.team_ball[800:1000] == 0).all())
        self.assertTrue(built.visible[0])
        self.assertFalse(built.visible[800])


class ModelTest(unittest.TestCase):
    def test_recovers_a_separable_rule_and_is_exactly_repeatable(self):
        rng = np.random.RandomState(0)
        X = rng.normal(size=(4000, 3))
        y = (X[:, 0] - X[:, 1] > 0).astype(np.float64)
        one = fit_logistic(X, y, intercept=False)
        two = fit_logistic(X, y, intercept=False)
        self.assertEqual(one, two)
        self.assertGreater(np.mean((predict_logit(one, X) > 0) == y), 0.99)

    def test_no_intercept_means_antisymmetric(self):
        rng = np.random.RandomState(1)
        X = rng.normal(size=(1000, 2))
        y = (X[:, 0] + rng.normal(size=1000) > 0).astype(np.float64)
        model = fit_logistic(X, y, intercept=False)
        np.testing.assert_allclose(predict_logit(model, X), -predict_logit(model, -X))


class TaggingTest(unittest.TestCase):
    blocks = [(0, 1000), (2000, 3000)]

    def entry(self, code, start, stop, **tags):
        return {"code": code, "start_frame": start, "end_frame": stop, "tags": tags}

    def test_attacking_and_defending_codes_point_opposite_ways(self):
        team, dead = tag_evidence([self.entry("BUILD UP", 10, 20, Team="Barcelona"),
                                   self.entry("DEFENDING IN MIDDLE THIRD", 30, 40, Team="Barcelona")],
                                  3000, self.blocks)
        self.assertGreater(team[15], 0)
        self.assertLess(team[35], 0)
        self.assertLess(dead[15], 0)

    def test_na_team_says_nothing_about_the_team(self):
        team, _ = tag_evidence([self.entry("SET PIECES", 10, 20, Team="N/A", Type="Throw-in")],
                               3000, self.blocks)
        self.assertTrue((team == 0).all())

    def test_case_does_not_matter(self):
        team, _ = tag_evidence([self.entry("set pieces", 10, 20, Team="ATLETIC CLUB")], 3000, self.blocks)
        self.assertLess(team[18], 0)

    def test_frames_before_a_kick_off_are_a_stoppage(self):
        tags = [self.entry("SET PIECES", 2500, 2600, Team="N/A", Type="Kick Off (Start)")]
        _, dead = tag_evidence(tags, 3000, self.blocks)
        self.assertGreater(dead[2100], 50)
        self.assertLess(dead[2700], 50)
        self.assertLess(dead[500], 50)

    def test_a_kick_off_late_in_a_half_cannot_kill_that_half(self):
        # The bug validation found: a second-half kick-off tagged just before the boundary.
        blocks = [(0, 70000), (70000, 140000)]
        tags = [self.entry("SET PIECES", 69900, 70060, Team="N/A", Type="Kick Off (Start)")]
        _, dead = tag_evidence(tags, 140000, blocks)
        self.assertLess(dead[100], 50)

    def test_no_tags_is_no_evidence(self):
        team, dead = tag_evidence(None, 100, [(0, 100)])
        self.assertTrue((team == 0).all() and (dead == 0).all())


class PipelineTest(unittest.TestCase):
    def test_labels_are_x_outside_the_halves_and_scoreable(self):
        match = synthetic_match()
        match.blocks = [(100, 1400), (1600, 2900)]
        labels = pipeline.predict(match)
        self.assertEqual(len(labels), match.n_frames)
        self.assertTrue((labels[:100] == pipeline.X_LABEL).all())
        self.assertTrue((labels[1400:1600] == pipeline.X_LABEL).all())
        self.assertTrue((labels[100:1400] != pipeline.X_LABEL).all())
        self.assertEqual(pipeline.LABEL_LETTERS, "".join(pm.LABELS))

    def test_the_same_input_gives_the_same_labels(self):
        match = synthetic_match()
        np.testing.assert_array_equal(pipeline.predict(match), pipeline.predict(match))


class SpeedTest(unittest.TestCase):
    def test_a_jump_is_not_a_speed(self):
        xy = np.zeros((20, 2), dtype=np.float32)
        xy[:, 0] = np.arange(20) * 0.2  # 5 m/s at 25 fps
        speed = derived_speed(xy, [(0, 20)], 12.0)
        np.testing.assert_allclose(speed[5], 5.0, atol=1e-3)
        xy[10:, 0] += 50.0
        self.assertTrue(np.isnan(derived_speed(xy, [(0, 20)], 12.0)[10]))


if __name__ == '__main__':
    unittest.main()
