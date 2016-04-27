import numpy as np
import numpy.random as npr
import flymc as ff
import unittest
from model_setup import *

npr.seed(1)

class BoundsTest(object):
    TOL = 1e-6
    EPS = 1e-6 # Artificial gap between bound and likelihood
    NUM_TRIALS = 100
    N = 30
    D = 4
    K = 3

    def test_bounds(self):
        for model, th, z in self.random_setup():
            all_L = model._logL(th, z.bright)
            all_B = model._logB(th, z.bright)
            all_LBgap = model._LBgap(th, z.bright)
            for L, B, LBgap in zip(all_L, all_B, all_LBgap):
                self.assertGreater(L - B, 0)
                self.assertAlmostEqual(L - B, LBgap)

    def test_product_of_bounds(self):
        # The (log of the) product of bounds is allowed to have an arbitrary
        # constant added, so we have to look at diffs as th is changed.
        all_idxs = np.arange(self.N)
        for model, th, z in self.random_setup():
            th_A = self.random_th()
            B_prod_A = model._logBProduct(th_A)
            B_prod_explicit_A = np.sum(model._logB(th_A, all_idxs))

            th_B = self.random_th()
            B_prod_B = model._logBProduct(th_B)
            B_prod_explicit_B = np.sum(model._logB(th_B, all_idxs))
            self.assertAlmostEqual(B_prod_B - B_prod_A,
                                   B_prod_explicit_B - B_prod_explicit_A)

    def test_bound_tight_at_th_map(self):
        for i in range(self.N):
            th_map = self.random_th()
            z = self.random_z()
            model = self.random_model(th_map=th_map)
            all_idxs = np.arange(self.N)
            all_L = model._logL(th_map, z.bright)
            all_B = model._logB(th_map, z.bright)
            for L, B in zip(all_L, all_B):
                self.assertLess(B, L)
                self.assertGreater(B, L - 1.1 * self.EPS)

class LogisticModelTest(LogisticModelSetup, BoundsTest, unittest.TestCase):
    pass
class MulticlassLogisticModelTest(MulticlassLogisticModelSetup, BoundsTest, unittest.TestCase):
    pass
class RobustRegressionModelTest(RobustRegressionModelSetup, BoundsTest, unittest.TestCase):
    pass
