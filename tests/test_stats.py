"""Unit tests for the statistical core: each method recovers a known truth on synthetic data."""
from __future__ import annotations

import unittest

import numpy as np
from scipy import stats

from fairexams.bulk import bulk_prediction
from fairexams.stats import (bh, bh_cutoff, central_matching, hypergeom_tails, nested_icc, randomised_z,
                             storey_pi0)


class RandomisedZ(unittest.TestCase):
    def test_exactly_normal_under_the_null(self):
        rng = np.random.default_rng(1)
        N, K, n = 2000, 20, 300
        k = rng.hypergeometric(K, N - K, n, size=20000)
        p_ge, p_le, p_gt, p_eq = hypergeom_tails(k, N, K, n)
        z = randomised_z(p_gt, p_eq, rng.uniform(size=len(k)), p_lt=p_le - p_eq)
        self.assertAlmostEqual(float(np.mean(z)), 0.0, delta=0.03)
        self.assertAlmostEqual(float(np.std(z)), 1.0, delta=0.03)

    def test_deep_deficit_stays_finite(self):
        p_ge, p_le, p_gt, p_eq = hypergeom_tails(0, 50000, 25000, 700)
        z = randomised_z(p_gt, p_eq, 0.5, p_lt=p_le - p_eq)
        self.assertTrue(np.isfinite(z) and z < -20)


class ICC(unittest.TestCase):
    def test_recovers_known_icc(self):
        rng = np.random.default_rng(2)
        sums, sss, ns, cities = [], [], [], []
        for c in range(60):
            for j in range(8):
                n = int(rng.integers(200, 600))
                y = rng.normal(c * 0.1 + rng.normal(0, np.sqrt(0.05)), np.sqrt(0.95), size=n)
                sums.append(y.sum()); sss.append((y ** 2).sum()); ns.append(n); cities.append(c)
        rho, _ = nested_icc(np.array(sums), np.array(sss), np.array(ns), np.array(cities))
        self.assertAlmostEqual(rho, 0.05, delta=0.015)

    def test_zero_when_allotment_is_random(self):
        rng = np.random.default_rng(3)
        sums, sss, ns, cities = [], [], [], []
        for c in range(40):
            for j in range(6):
                y = rng.normal(c, 1, size=400)
                sums.append(y.sum()); sss.append((y ** 2).sum()); ns.append(400); cities.append(c)
        rho, _ = nested_icc(np.array(sums), np.array(sss), np.array(ns), np.array(cities))
        self.assertLess(rho, 0.003)


class Multiplicity(unittest.TestCase):
    def test_bh_matches_definition(self):
        p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
        q = bh(p)
        self.assertTrue(np.all(q[:2] <= 0.05) and np.all(q[2:] > 0.05))
        self.assertAlmostEqual(bh_cutoff(p, 0.05), 0.008)
        self.assertAlmostEqual(bh_cutoff(np.array([0.5, 0.9]), 0.05), 0.025)

    def test_storey_pi0_uniform(self):
        p = np.random.default_rng(4).uniform(size=20000)
        self.assertAlmostEqual(storey_pi0(p), 1.0, delta=0.03)


class EmpiricalNull(unittest.TestCase):
    def test_central_matching(self):
        z = np.random.default_rng(5).normal(0.5, 1.3, size=40000)
        d0, s0 = central_matching(z)
        self.assertAlmostEqual(d0, 0.5, delta=0.05)
        self.assertAlmostEqual(s0, 1.3, delta=0.06)


class Bulk(unittest.TestCase):
    def test_prediction_matches_when_centre_is_a_random_draw(self):
        rng = np.random.default_rng(6)
        city = np.round(rng.normal(250, 160, size=60000))
        own = rng.choice(city, size=3000, replace=False)
        share, mu, sigma = bulk_prediction(own, np.sort(city), 600)
        self.assertAlmostEqual(share, float((city >= 600).mean()), delta=0.006)
        self.assertAlmostEqual(mu, 0.0, delta=0.06)
        self.assertAlmostEqual(sigma, 1.0, delta=0.06)


if __name__ == "__main__":
    unittest.main()
