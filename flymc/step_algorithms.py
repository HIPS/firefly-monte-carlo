import numpy as np
import numpy.random as npr
from abc import ABCMeta
from abc import abstractmethod

class Stepper(object):
    
    __metaclass__ = ABCMeta

    @abstractmethod
    def step(th, z):
        pass

# --- Algorithms for stepping in th ---

class ThetaStepMH(Stepper):
    is_gradient_method = False

    def __init__(self, prob, stepsize):
        self.prob = prob
        self.stepsize = stepsize

    def step(self, th, z):
        th_new = th + npr.standard_normal(th.shape)*self.stepsize
        # Cache friendly order: evaluate old value first
        if np.log(npr.rand()) < - self.prob(th, z) + self.prob(th_new, z) :
            self.num_rejects = 0
            return th_new
        else:
            self.num_rejects = 1
            return th

class ThetaStepLangevin(Stepper):
    is_gradient_method = True

    def __init__(self, prob, D_prob, stepsize):
        self.prob = prob
        self.D_prob = D_prob
        self.stepsize = stepsize

    def step(self, th, z):
        s = self.stepsize
        randstep = npr.standard_normal(th.shape)
        th_new = th + 0.5*s**2*self.D_prob(th,z) + randstep*s
        # bonus for probability difference. Using cache-friendly order
        diff_probs = -self.prob(th, z)+self.prob(th_new, z)
        # penalty for having asymmetric proposals:
        randstep_back = (th - (th_new + 0.5*s**2*self.D_prob(th_new,z)))/s
        diff_proposal = 0.5*np.sum(randstep_back**2) - 0.5*np.sum(randstep**2)
        # M-H accept/reject:
        if np.log(npr.rand()) < diff_probs - diff_proposal:
            self.num_rejects = 0
            return th_new
        else:
            self.num_rejects = 1
            return th

class ThetaStepSlice(Stepper):
    is_gradient_method = False

    def __init__(self, logprob, line_width):
        self.line_width = line_width
        self.logprob = logprob

    def step(self, th, z):

        direction = npr.randn(len(th))
        direction = direction / np.sqrt(np.sum(direction**2))

        R = self.line_width*npr.rand()
        L = R - self.line_width
        curr_logprob = self.logprob(th, z) + np.log(npr.rand())
        # rejection sample along the horizontal line (because we don't know the bounds exactly)

        self.num_rejects = 0
        while True:
            x = (R - L)*npr.rand() + L  # uniformly sample between R and L
            th_new = th + x*direction
            new_logprob   = self.logprob(th_new, z)
            if new_logprob > curr_logprob:
                break # we have our sample
            elif x < 0:
                L = x 
            elif x > 0:
                R = x
            else:
                raise Exception("Slice sampler shrank to zero!")
            self.num_rejects += 1

        return th_new

# --- Algorithms for stepping in z ---

class zStepMH(Stepper):

    def __init__(self, log_bd_ratio, q):
        self.log_bd_ratio = log_bd_ratio
        self.q = q

    def step(self, th, z):
        # Propose a set of data points to switch state
        proposed_brighten = z.rand_dark(self.q)
        Nb = proposed_brighten.size
        proposed_darken   = z.bright
        Nd = proposed_darken.size

        # Consider the bright -> dark proposals
        log_p_accept = np.log(self.q) - self.log_bd_ratio(th, proposed_darken)
        idxs_accepted = proposed_darken[np.log(npr.rand(Nd)) <  log_p_accept]
        z.darken(idxs_accepted)

        # Consider the dark -> bright proposals
        log_p_accept = -np.log(self.q) + self.log_bd_ratio(th, proposed_brighten)
        idxs_accepted = proposed_brighten[np.log(npr.rand(Nb)) <  log_p_accept]
        z.brighten(idxs_accepted)

        return z

class zStepNone(Stepper):
    '''
    Does nothing, just for convenience
    '''

    def __init(self):
        pass

    def step(self, th, z):
        return z
