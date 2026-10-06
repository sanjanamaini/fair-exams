"""Unit tests for the pieces every number depends on: exam arithmetic, thresholds, ranks."""
from __future__ import annotations

import unittest

import numpy as np

from fairexams.panels import rank_table, score_at_rank, top_threshold, valid_scores


class Arithmetic(unittest.TestCase):
    def test_known_impossible_totals(self):
        v = valid_scores()
        for s in (717, 718, 719, 714, 713, 709):
            self.assertNotIn(s, v)
        for s in (720, 716, 715, 712, 711, 710, 708, 0, -180):
            self.assertIn(s, v)

    def test_range(self):
        v = valid_scores()
        self.assertEqual(min(v), -180)
        self.assertEqual(max(v), 720)


class Thresholds(unittest.TestCase):
    def test_top_threshold_respects_share(self):
        marks = np.array([100] * 90 + [200] * 5 + [300] * 5)
        self.assertEqual(top_threshold(marks, 0.10), 200)  # 10 of 100 score 200 or more
        self.assertEqual(top_threshold(marks, 0.05), 300)

    def test_ties_never_exceed_share(self):
        marks = np.array([100] * 95 + [500] * 5)
        self.assertEqual(top_threshold(marks, 0.04), 501)  # the tie group of 5 is too big: empty group
        self.assertEqual(top_threshold(marks, 0.05), 500)
        self.assertEqual(top_threshold(np.arange(1, 11), 0.2), 9)


class Ranks(unittest.TestCase):
    def test_rank_table(self):
        rt = rank_table(np.array([700, 700, 650, 600, 600, 600]))
        row = rt.set_index("score").loc[600]
        self.assertEqual((row["best_rank"], row["worst_rank"]), (4, 6))
        self.assertEqual(score_at_rank(rt, 3), 650)
        self.assertEqual(score_at_rank(rt, 5), 600)


if __name__ == "__main__":
    unittest.main()
