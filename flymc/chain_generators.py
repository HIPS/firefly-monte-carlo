import flymc as ff
import numpy as np
import numpy.random as npr

def logistic_regression_chain(x, t, N_iter=100, stepsize=1, th0=1, q=0.1, y0=1, seed=None):

    # Set seed
    npr.seed(seed)

    # Obtain joint distributions over z and th and set step functions
    model = ff.LogisticModel(x, t, th0=th0, y0=y0)
    z__stepper = ff.zStepMH(model.log_pseudo_lik , q)
    th_stepper = ff.ThetaStepMH(model.log_p_joint, stepsize)

    # Initialize
    N, D = x.shape
    th = np.random.randn(D)*th0
    z = ff.BrightnessVars(N)

    # run chain
    th_chain = np.zeros((N_iter,  ) +  th.shape)
    z_chain  = np.zeros((N_iter, N), dtype=bool)

    for i in range(N_iter):
        th = th_stepper.step(th, z)
        z  = z__stepper.step(th ,z)
        # Record the intermediate results
        th_chain[i,:] = th.copy()
        z_chain[i,z.bright] = 1

    print "th0 = ", th0, "frac accepted is", th_stepper.frac_accepted,  "bright frac is:", np.mean(z_chain)
    return th_chain, z_chain
