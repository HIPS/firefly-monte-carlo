import numpy as np
import numpy.random as npr
import flymc as ff
from util import mcmc_estimator, chain_generator_th
import unittest

npr.seed(1)

class StepAlgorithmsTest(object):
    def test_2D_gaussian(self):
        TOL = 0.2
        th_mean = np.array([0.2, -0.9])
        th_var = np.array([[0.7, 0.2],
                          [0.2, 0.7]])
        th_sq_mean = th_var + np.outer(th_mean, th_mean)
        th_var_inv = np.linalg.inv(th_var)
        th_init = np.array([10, 20])
 
        def gauss_prob(th, dummy):
            return -0.5 * (th - th_mean).dot(th_var_inv).dot(th - th_mean)

        def D_gauss_prob(th, dummy):
            return -th_var_inv.dot(th - th_mean)

        stepper = self.create_stepper(gauss_prob, D_gauss_prob)

        chain_gen = chain_generator_th(stepper, th_init)
        correct_moments = np.concatenate((th_mean, th_sq_mean.ravel()))
        print "correct moments:", correct_moments
        est_moments = mcmc_estimator(chain_gen, TOL)
        self.assertLess(np.max(np.abs(est_moments - correct_moments)), TOL)

class ThetaStepMHTest(StepAlgorithmsTest, unittest.TestCase):
    def create_stepper(self, prob, D_prob):
        stepsize = 1.5
        return ff.ThetaStepMH(prob, stepsize)

class ThetaStepLangevinTest(StepAlgorithmsTest, unittest.TestCase):
    def create_stepper(self, prob, D_prob):
        stepsize = 1.5
        return ff.ThetaStepLangevin(prob, D_prob, stepsize)

class ThetaStepSliceTest(StepAlgorithmsTest, unittest.TestCase):
    def create_stepper(self, prob, D_prob):
        linewidth = 2
        return ff.ThetaStepSlice(prob, linewidth)
