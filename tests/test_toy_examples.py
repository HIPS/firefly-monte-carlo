import numpy as np
from nose.plugins.attrib import attr
import sys

from util import *

sys.path.append('..')
import flymc as ff

def create_toy_logistic_model():
    x = np.array([[ 2.2,  1.7],
                  [-1.5,  2.9],
                  [ 1.4, -1.2]])
    t = np.array([1, 0, 1])
    th0 = 1.2
    y0 = 2
    return ff.LogisticModel(x, t, th0=th0, y0=y0)

correct_moments_toy_logistic = [1.323, -0.374, 2.439, -0.460, -0.460,
                                0.749, 0.152,  0.227, 0.088]

def create_toy_robust_regression_model():
    x = np.array([[-0.7, -0.8],
                  [-1.1,  1.5],
                  [-0.2, -0.2]])
    t = np.array([0.5, 1.6, -3])
    th0 = 2.2
    y0 = 0.5
    return ff.RobustRegressionModel(x, t, th0=th0, y0=y0)

correct_moments_robust_regression = [-0.559, 0.474, 1.181, 0.010, 0.010,
                                     0.775, 0.102, 0.113, 0.893]

def create_toy_multiclass_logistic_model():
    x = np.array([[ 0.4],
                  [ 1.7],
                  [-0.6]])
    t = np.array([0, 0, 1])
    K = 2
    th0 = 1.1
    y0 = 2.4
    return ff.MulticlassLogisticModel(x, t, K, th0=th0, y0=y0)

correct_moments_multiclass_logistic = [0.740, -0.740, 1.479, -0.269, -0.269,
                                       1.479, 0.185, 0.205, 0.148]

def check_flymc_converges(model, correct_moments, th_stepper):
    tol = 0.2
    q = 0.5
    z_stepper = ff.zStepMH(model.log_pseudo_lik , q)
    th_init = model.draw_from_prior()
    z_init = ff.BrightnessVars(model.N)
    chain_gen = chain_generator_th_z(th_stepper, z_stepper,
                                     th_init, z_init)
    est_moments = mcmc_estimator(chain_gen, tol)
    assert(np.all(np.abs(est_moments - correct_moments) < tol))

@attr('slow')
def test_logistic_model_MH():
    stepsize = 1
    model = create_toy_logistic_model()
    th_stepper = ff.ThetaStepMH(model.log_p_joint, stepsize)
    correct_moments = correct_moments_toy_logistic
    check_flymc_converges(model, correct_moments, th_stepper)

@attr('slow')
def test_logistic_model_slice():
    linewidth = 2
    model = create_toy_logistic_model()
    correct_moments = correct_moments_toy_logistic
    th_stepper = ff.ThetaStepSlice(model.log_p_joint, linewidth)
    check_flymc_converges(model, correct_moments, th_stepper)

@attr('slow')
def test_logistic_model_langevin():
    stepsize = 1
    model = create_toy_logistic_model()
    correct_moments = correct_moments_toy_logistic
    th_stepper = ff.ThetaStepLangevin(model.log_p_joint, model.D_log_p_joint, stepsize)
    check_flymc_converges(model, correct_moments, th_stepper)

@attr('slow')
def test_robust_regression_model_MH():
    stepsize = 1
    model = create_toy_robust_regression_model()
    th_stepper = ff.ThetaStepMH(model.log_p_joint, stepsize)
    correct_moments = correct_moments_robust_regression
    check_flymc_converges(model, correct_moments, th_stepper)

@attr('slow')
def test_multiclass_logistic_model_MH():
    stepsize = 1
    model = create_toy_multiclass_logistic_model()
    th_stepper = ff.ThetaStepMH(model.log_p_joint, stepsize)
    correct_moments = correct_moments_multiclass_logistic
    check_flymc_converges(model, correct_moments, th_stepper)

def main():
    # Computes the true moments of the posterior distributions by quadrature
    print "Multiclass logistic regression model moments:"
    print compute_moments_by_quadrature(create_toy_multiclass_logistic_model())
    print "Robust regression model moments:"
    print compute_moments_by_quadrature(create_toy_robust_regression_model())
    print "Logistic model moments:"
    print compute_moments_by_quadrature(create_toy_logistic_model())

if __name__ == "__main__":
    main()
