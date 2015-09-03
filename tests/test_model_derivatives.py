import numpy as np
import numpy.random as npr
import flymc as ff
from util import nd, nd_bounds
import unittest
from model_setup import *

npr.seed(1)

class DerivativesTest(object):
    PLACES = 4
    NUM_TRIALS = 10
    N = 30
    D = 4
    K = 3

    def test_prior(self):
        for model, th, z in self.random_setup():
            LB, UB = nd_bounds(model._logPrior, th)
            AD = model._D_logPrior(th)
            self.check_ordered(LB, AD, UB)

    def test_likelihood(self):
        for model, th, z in self.random_setup():
            LB, UB = nd_bounds(lambda th: np.sum(model._logL(th, z.bright)), th)
            AD = np.sum(model._D_logL(th, z.bright), axis=0)
            self.check_ordered(LB, AD, UB)
        
    def test_bound(self):
        for model, th, z in self.random_setup():
            LB, UB = nd_bounds(lambda th:
                   np.sum(model._logB(th, z.bright)), th)
            AD = np.sum(model._D_logB(th, z.bright), axis=0)
            self.check_ordered(LB, AD, UB)

    def test_LB_gap(self):
        for model, th, z in self.random_setup():
            LB, UB = nd_bounds(lambda th:
                   np.sum(model._LBgap(th, z.bright)), th)
            AD = np.sum(model._D_LBgap(th, z.bright), axis=0)
            self.check_ordered(LB, AD, UB)

    def test_bound_product(self):
        for model, th, z in self.random_setup():
            LB, UB = nd_bounds(lambda th: model._logBProduct(th), th)
            AD = model._D_logBProduct(th)
            self.check_ordered(LB, AD, UB)
        
    def test_marg_likelihood(self):
        for model, th, z in self.random_setup():
            LB, UB = nd_bounds(lambda th: model.log_p_marg(th), th)
            AD = model.D_log_p_marg(th)
            self.check_ordered(LB, AD, UB)
        
    def test_pseudo_likelihood(self):
        for model, th, z in self.random_setup():
            LB, UB = nd_bounds(lambda th: np.sum(model.log_pseudo_lik(th, z.bright)), th)
            AD = np.sum(model._D_log_pseudo_lik(th, z.bright), axis=0)
            self.check_ordered(LB, AD, UB)
        
    def test_joint_posterior(self):
        for model, th, z in self.random_setup():
            LB, UB = nd_bounds(lambda th: model.log_p_joint(th, z), th)
            AD =  model.D_log_p_joint(th, z)
            self.check_ordered(LB, AD, UB)

    def check_ordered(self, A, B, C):
        # Check that A < B < C
        for a, b, c in zip(A.ravel(), B.ravel(), C.ravel()):
            self.assertLess(a, b)
            self.assertLess(b, c)

    def check_close(self, A, B):
        rel_error = np.abs(A-B)/np.abs(A)
        self.assertEqual(A.shape, B.shape)
        self.assertAlmostEqual(np.max(rel_error), 0, places=self.PLACES)

class LogisticModelTest(LogisticModelSetup, DerivativesTest, unittest.TestCase):
    pass
class MulticlassLogisticModelTest(MulticlassLogisticModelSetup, DerivativesTest, unittest.TestCase):
    pass
class RobustRegressionModelTest(RobustRegressionModelSetup, DerivativesTest, unittest.TestCase):
    pass
