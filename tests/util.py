import numpy as np
import itertools as it
from scipy import integrate

EPS = 1e-4
REL_TOL = 1e-10

def nd(fun, th):
    # Symmetric finite differences numeric derivative
    deriv = np.zeros(th.shape)
    for in_dims in it.product(*map(range, th.shape)):
        unit_vect = np.zeros(th.shape)
        unit_vect[in_dims] = 1
        deriv[in_dims] = (fun(th + unit_vect * EPS) -
                          fun(th - unit_vect * EPS)) / (2 * EPS)
    return deriv

def nd_bounds(fun, th):
    # Takes a pre and post derivative which act as reasonable bounds
    # for the derivative itself. (Only violated if derivative is non-monotonic
    # over domain [th - EPS, th + EPS] which is very unlikely for random th and
    # small EPS.)
    right_deriv = np.zeros(th.shape)
    left_deriv = np.zeros(th.shape)
    for in_dims in it.product(*map(range, th.shape)):
        eps_vect = np.zeros(th.shape)
        eps_vect[in_dims] = EPS
        left_deriv[in_dims]  = (fun(th) - fun(th - eps_vect)) / EPS
        right_deriv[in_dims] = (fun(th + eps_vect) - fun(th)) / EPS

    LB = np.minimum(left_deriv, right_deriv)
    UB = np.maximum(left_deriv, right_deriv)
    LB = LB - (REL_TOL * np.abs(LB))
    UB = UB + (REL_TOL * np.abs(UB))
    return LB, UB 

def mcmc_estimator(chain_gen, tol=0.1, N_min=1024, N_max = 10**7):
    # Takes the generator chain_gen, which produces a sequence of vectors
    # generated from a markov chain, and runs it until convergence. It returns
    # an estimate of the means of the values and an attempted confidence interval.
    # e.g.
    # >>> import numpy.random as npr
    # >>> def gaussian_generator():
    # >>>     while True:
    # >>>         yield npr.randn()
    # >>> mcmc_estimator(gaussian_generator())
    # >>> 0.005
    samps = []
    tol2 = tol**2
    N_check = N_min
    print "Samples, converged, between_chain_var, mean"
    for n, x in enumerate(chain_gen):
        if n == N_check:
            converged, between_chain_var, mean = compute_convergence(samps)
            print n, converged, between_chain_var, mean
            if converged and np.all(between_chain_var < tol2):
                return mean
            else:
                N_check = N_check * 2
                samps = []

        samps.append(x)
        if n > N_max:
            break

    raise Exception("Failed to converge after %s iterations" % n)

def compute_convergence(samps, N_subchains = 16):
    # Produce a (hopefully conservative) estimate of the variance of the mean
    # of the chain.
    samps = np.array(samps)
    len_subchain = samps.shape[0]/N_subchains
    subchains = np.split(samps, N_subchains, axis=0)
    subchain_means = [np.mean(chain, axis=0) for chain in subchains]
    subchain_vars = [np.var(chain, axis=0) for chain in subchains]
    within_chain_var = np.mean(subchain_vars, axis=0)
    between_chain_var = np.var(subchain_means, axis=0)
    converged = np.all(between_chain_var < within_chain_var)
    means = np.mean(subchain_means, axis=0)
    return converged, between_chain_var, means

def chain_generator_th(stepper, th_init):
    th = th_init
    while True:
        th = stepper.step(th, None)
        th_flat = th.ravel()
        yield np.concatenate((th_flat, np.outer(th_flat, th_flat).ravel()))

def chain_generator_th_z(th_stepper, z_stepper, th_init, z_init):
    th = th_init
    z = z_init
    z_isbright = np.zeros(z.N)
    while True:
        th = th_stepper.step(th, z)
        z = z_stepper.step(th, z)
        th_flat = th.ravel()
        th_sq = np.outer(th_flat, th_flat).ravel()
        z_isbright[:] = 0
        z_isbright[z.bright] = 1
        yield np.concatenate((th_flat, th_sq, z_isbright))

def compute_expectation(model, fun):
    # Computes the expectation of fun(th) wrt the posterior distribution,
    # given by exp(model.log_p_marg(th))
    th_shape = model.th_shape
    L = np.prod(th_shape)
    lims = [[-np.inf, np.inf]] * L
    def integrand(*args):
        th = np.array(args).reshape(th_shape)
        return np.exp(model.log_p_marg(th)) * fun(th)

    return integrate.nquad(integrand, lims)[0]
    
def compute_moments_by_quadrature(model):
    N = model.N
    L = np.prod(model.th_shape)
    # Partition function for marginal distribution over theta
    Z = compute_expectation(model, lambda th : 1)
     
    th_mean = [compute_expectation(model, lambda th : th[i]) / Z
               for i in range(L)]
    th_sq = [compute_expectation(model, lambda th : th[i] * th[j]) / Z
             for i, j in it.product(range(L), range(L))]

    def bright_prob(th, idx):
        inv_pseudo_lik = np.exp(-model.log_pseudo_lik(th, [idx]))
        return 1 / (1 + inv_pseudo_lik)

    z_mean = [compute_expectation(model, lambda th: bright_prob(th, i)) / Z
              for i in range(N)]

    return th_mean + th_sq + z_mean

