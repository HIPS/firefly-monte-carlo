import numpy as np
import numpy.random as npr
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
import flymc as ff

# Set hyperparameters
N = 20                  # number of data points
D = 3                   # dimension of data points (plotting the data requires D=3)
stepsize = 0.75         # size of Metropolis-Hastings step in theta
th0 = 2.5               # scale of weights
y0 = 2                  # point at which to make bounds tight
q = 0.05                # Metropolis-Hastings proposal probability for z

# Cosmetic settings
mpl.rcParams['axes.linewidth'] = 3
mpl.rcParams['lines.linewidth'] = 7
mpl.rcParams['toolbar'] = "None"
mpl.rcParams['figure.facecolor'] = "1"

def main():
    # Generate synthetic data
    x = 2 * npr.rand(N,D) - 1  # data features, an (N,D) array
    x[:, 0] = 1
    th_true = 10.0 * np.array([0, 1, 1])
    y = np.dot(x, th_true[:, None])[:, 0]
    t = npr.rand(N) > (1 / ( 1 + np.exp(y)))  # data targets, an (N) array of 0s and 1s

    # Obtain joint distributions over z and th
    model = ff.LogisticModel(x, t, th0=th0, y0=y0)

    # Set up step functions
    th = np.random.randn(D) * th0
    z = ff.BrightnessVars(N)
    th_stepper = ff.ThetaStepMH(model.log_p_joint, stepsize)
    z__stepper = ff.zStepMH(model.log_pseudo_lik, q)

    plt.ion()
    ax = plt.figure(figsize=(8, 6)).add_subplot(111)
    while True:
        th = th_stepper.step(th, z)  # Markov transition step for theta
        z  = z__stepper.step(th ,z)  # Markov transition step for z
        update_fig(ax, x, y, z, th, t)
        plt.draw()
        plt.pause(0.05)

def update_fig(ax, x, y, z, th, t):
    b = np.zeros(N)
    b[z.bright] = 1

    bright1s = (   t  *    b ).astype(bool)
    bright0s = ((1-t) *    b ).astype(bool)
    dark1s   = (   t  * (1-b)).astype(bool)
    dark0s   = ((1-t) * (1-b)).astype(bool)
    ms, bms, mew = 45, 45, 5

    ax.clear()
    ax.plot(x[dark0s,1],   x[dark0s,2],  's', mec='Blue', mfc='None', ms=ms,  mew=mew)
    ax.plot(x[dark1s,1],   x[dark1s,2],  'o', mec='Red',  mfc='None', ms=ms,  mew=mew)
    ax.plot(x[bright0s,1], x[bright0s,2],'s', mec='Blue', mfc='Blue', ms=bms, mew=mew)
    ax.plot(x[bright1s,1], x[bright1s,2],'o', mec='Red',  mfc='Red',  ms=bms, mew=mew)

    X = np.arange(-3,3)
    th1, th2, th3 = th[0], th[1], th[2]
    Y = (-th1 - th2 * X) / th3

    ax.plot(X, Y, color='grey')
    lim = 1.15
    ax.set_xlim([-lim,lim])
    ax.set_ylim([-lim,lim])
    ax.set_yticks([])
    ax.set_xticks([])

if __name__ == "__main__":
    main()
