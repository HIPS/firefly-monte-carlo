import numpy as np
import numpy.random as npr
from abc import ABCMeta
from abc import abstractmethod

EPS = 1e-6 # Gap between bound and likelihood to tolerate floating point error

def log_logistic(x):
    # Overflow-avoiding version of the log logistic function.
    abs_x = np.abs(x)
    return 0.5 * (x - abs_x) - np.log(1+np.exp(-abs_x))

class CacheWithIdxs(object):
    def __init__(self, N, N_theta):
        self.size = N_theta
        self.values = np.zeros((N, N_theta))
        self.exists = np.zeros((N, N_theta), dtype=bool)
        self.lists  = [] # A list of lists containing the cached indices
        for i in range(N_theta): self.lists.append([])
        self.thetas = [None]*N_theta
        self.oldest = 0

    def retrieve(self, th, idxs):
        # Check whether it's in the cache and give the values if it is
        # NOTE: cache tests identity, not equality, so if the value of a
        # th object changes it will be messed up
        for i, th_cache in enumerate(self.thetas):
            if th_cache is not None and np.all(th == th_cache) and np.all(self.exists[idxs, i]):
                self.oldest = (i + 1) % len(self.thetas)
                return self.values[idxs, i]

    def store(self, th, idxs, new_values):
        for i, th_cache in enumerate(self.thetas):
            if th_cache is not None and th is th_cache:
                assert(np.all(th == th_cache)), "Value of th changed" # This can be turned off for performance
                vacant = np.where(np.logical_not(self.exists[idxs, i]))
                self.values[idxs[vacant], i] = new_values[vacant]
                self.exists[idxs[vacant], i] = 1
                self.lists[i] += list(idxs[vacant])
                return

        # if we didn't find it, we have a new theta
        i = self.oldest
        self.oldest = (self.oldest + 1) % len(self.thetas)
        self.thetas[i] = th.copy()
        self.exists[self.lists[i], i] = 0
        self.exists[idxs, i] = 1
        self.values[idxs, i] = new_values
        del self.lists[i][:]
        self.lists[i] += list(idxs)

class SimpleCache(object):
    def __init__(self, N_theta):
        self.size = N_theta
        self.values = np.zeros(N_theta)
        self.thetas = [None]*N_theta
        self.oldest = 0

    def retrieve(self, th):
        # Check whether it's in the cache and give the values if it is
        # NOTE: cache tests identity, not equality, so if the value of a
        # th object changes it will be messed up
        for i, th_cache in enumerate(self.thetas):
            if th_cache is not None and th is th_cache:
                assert(np.all(th == th_cache)), "Value of th changed" 
                self.oldest = (i + 1) % len(self.thetas)
                return self.values[i]

    def store(self, th, new_values):
        for i, th_cache in enumerate(self.thetas):
            if th_cache is not None and np.all(th == th_cache): return
        # if we didn't find it, we have a new theta
        i = self.oldest
        self.oldest = (self.oldest + 1) % len(self.thetas)
        self.thetas[i] = th.copy()
        self.values[i] = new_values

class Model(object):

    __metaclass__ = ABCMeta

    def __init__(self, cache_size=2):
        # To make things cache-friendly, should always evaluate the old value first
        self.pseudo_lik_cache = CacheWithIdxs(self.N, cache_size)
        self.p_marg_cache = SimpleCache(cache_size)
        self.num_lik_evals = 0
        self.num_D_lik_evals = 0

    def log_p_joint(self, th, z):
        # joint distribution over th and z
        return self._logPrior(th) + self._logBProduct(th) \
                  + np.sum(self.log_pseudo_lik(th, z.bright))

    def D_log_p_joint(self, th, z):
        # Derivative wrt theta of the joint distribution
        return self._D_logPrior(th) + self._D_logBProduct(th) \
                  + np.sum(self._D_log_pseudo_lik(th, z.bright), axis=0)

    def log_pseudo_lik(self, th, idxs):
        # Pseduo-likelihood: ratio of bright to dark 
        # proabilities of indices idxs at th
        # Check for cached value:
        cached_value = self.pseudo_lik_cache.retrieve(th, idxs)
        if cached_value is not None:
            # this is only to test the cache. Comment out for real use
            # assert np.all(cached_value == self._LBgap(th,idxs) + np.log(1-np.exp(-self._LBgap(th,idxs))) )
            return cached_value

        # Otherwise compute it:
        gap = self._LBgap(th,idxs)
        result = gap + np.log(1-np.exp(-gap)) # this way avoids overflow
        self.pseudo_lik_cache.store(th, idxs, result)
        self.num_lik_evals += len(idxs)
        return result

    def _D_log_pseudo_lik(self, th, idxs):
        # Derivative of pseudo-likelihood wrt theta
        gap = self._LBgap(th,idxs)
        D_LBgap = self._D_LBgap(th, idxs)
        self.num_D_lik_evals += len(idxs)
        return D_LBgap/(1-np.exp(-gap)).reshape((len(idxs),) + (1,)*th.ndim)
    
    def log_p_marg(self, th, z=None):
        # marginal posterior prob. Takes z as an optional agrument but doesn't use it
        cached_value = self.p_marg_cache.retrieve(th)
        if cached_value != None:
            # this is only to test the cache. Comment out for real use
            # assert cached_value == self._logPrior(th) + np.sum(self._logL(th, range(self.N)))
            return cached_value                        

        result = self._logPrior(th) + np.sum(self._logL(th, range(self.N)))
        self.p_marg_cache.store(th, result)
        self.num_lik_evals += self.N
        return result

    def D_log_p_marg(self, th, z=None):
        self.num_D_lik_evals += self.N
        return self._D_logPrior(th) + np.sum(self._D_logL(th, range(self.N)), axis=0)

    def log_lik_all(self, th):
        return np.sum(self._logL(th, range(self.N)))

    def reset(self):
        # resets the counters and cache for a fresh start
        self.pseudo_lik_cache = CacheWithIdxs(self.N, self.pseudo_lik_cache.size)
        self.p_marg_cache = SimpleCache(self.p_marg_cache.size)
        self.num_lik_evals = 0
        self.num_D_lik_evals = 0

    @abstractmethod
    def _logL(self, th, idxs):
        pass
    @abstractmethod
    def _D_logL(self, th, idxs):
        pass

    @abstractmethod
    def _logB(self, th, idxs):
        pass
    @abstractmethod
    def _D_logB(self, th, idxs):
        pass

    @abstractmethod
    def _LBgap(self, th, idxs):
        pass
    @abstractmethod
    def _D_LBgap(self, th, idxs):
        pass

    @abstractmethod
    def _logBProduct(self, th):
        pass
    @abstractmethod
    def _D_logBProduct(self, th):
        pass

    @abstractmethod
    def _logPrior(self, th):
        pass
    @abstractmethod
    def _D_logPrior(self, th):
        pass

    @abstractmethod
    def draw_from_prior(self):
        pass

class LogisticModel(Model):
    def __init__(self, x, t, th0=1, y0=1.5, th_map=None):
        '''
        x      : Data, a (N, D) array
        t      : Targets, a (N) array of 0s and 1s
        th0    : Scale of the prior on weights
        th_map : Size (D) array, an estimate of MAP, for tuning the bounds
        y0     : Point at which to make bounds tight (if th_map
                 not given)
        '''
        self.N, self.D = x.shape
        Model.__init__(self)

        self.dat = x*(2*t[:,None]-1)
        if th_map is None:          # create the same bound for all data points
            y0_vect = np.ones(self.N)*y0
        else:                       # create bounds to be tight at th_map
            y0_vect = np.dot(self.dat, th_map[:,None])[:,0]

        a, b, c = self._logistic_bound(y0_vect)
        self.coeffs = (a, b, c)
        # Compute sufficient statistics of data
        self.dat_sum  = np.sum(self.dat*b[:,None], 0)
        self.dat_prod = np.dot(self.dat.T, self.dat*a[:,None])

        # Other hyperparameters
        self.th0 = th0

        self.th_shape = (self.D,)

    def _logL(self, th, idxs):
        # logistic regression log likelihoods
        # returns an array of size idxs.size
        y = np.dot(self.dat[idxs,:],th[:,None])[:,0]
        return log_logistic(y)

    def _D_logL(self, th, idxs):
        # sum of derivative of log likelihoods of data points idxs
        y = np.dot(self.dat[idxs,:],th[:,None])[:,0]
        return self.dat[idxs,:]*(np.exp(-y)/(1+np.exp(-y)))[:,None]

    def _logB(self, th, idxs):
        # lower bound on logistic regression log likelihoods
        # returns an array of size idxs.size
        y = np.dot(self.dat[idxs,:],th[:,None])[:,0]
        a, b, c = self.coeffs
        return a[idxs]*y**2 + b[idxs]*y + c[idxs]

    def _LBgap(self, th, idxs):
        # sum of derivative of log likelihoods of data points idxs
        y = np.dot(self.dat[idxs,:],th[:,None])[:,0]
        L = log_logistic(y)
        a, b, c = self.coeffs
        B = a[idxs]*y**2 + b[idxs]*y + c[idxs] 
        return L - B

    def _D_logB(self, th, idxs):
        y = np.dot(self.dat[idxs,:],th[:,None])[:,0]
        a, b, c = self.coeffs
        return self.dat[idxs,:]*(2*a[idxs]*y + b[idxs])[:,None]

    def _D_LBgap(self, th, idxs):
        # sum of derivative of log likelihoods of data points idxs
        y = np.dot(self.dat[idxs,:],th[:,None])[:,0]
        a, b, c = self.coeffs
        scalar_LBgap = (np.exp(-y)/(1+np.exp(-y))) - (2*a[idxs]*y + b[idxs])
        return self.dat[idxs,:]*scalar_LBgap[:,None]

    def _logBProduct(self, th):
        # log of the product of all the lower bounds
        y  = np.dot(th, self.dat_sum)
        y2 = np.dot(th[None,:],np.dot(self.dat_prod,th[:,None]))
        return y2 + y # note: we're ignoring a constant here since we don't care about normalization
  
    def _D_logBProduct(self, th):
        return self.dat_sum + 2*np.dot(self.dat_prod,th[:,None])[:,0]

    def _logPrior(self, th):
        return -0.5*np.sum((th/self.th0)**2)
 
    def _D_logPrior(self, th):
        return -th/self.th0**2

    def draw_from_prior(self):
        return npr.randn(self.D)*self.th0

    def _logistic_bound(self, y0):
        # Coefficients of a quadratic lower bound to the log-logistic function
        # i.e    a*x**2 + b*x + c < log(  exp(x)/(1+exp(x))  ) 
        # y0 parameterizes a family of lower bounds to the logistic function
        # (the bound is tight at +/- y0)
        pexp = np.exp(y0)
        nexp = np.exp(-y0)
        f = pexp + nexp
        a = -0.25/y0*(pexp-nexp)/(2 + f)
        b = 0.5*np.ones(y0.size)
        c = -a*y0**2 - 0.5*np.log(2 + f) - EPS
        return (a, b, c)

class MulticlassLogisticModel(Model):
    def __init__(self, x, t, K, th0=1, y0=1.5, th_map=None):
        '''
        Softmax classification over K classes. The weight, th, are an array
        of size (K, D)
        Parameters:
        x      : Data, a (N, D) array
        t      : Targets, a (N) array of integers from 0 to K-1
        th0    : Scale of the prior on weights
        th_map : Size (K, D) array, an estimate of MAP, for tuning the bounds
        y0     : Point at which to make bounds tight (if th_map not given)
        '''
        assert K == max(t)-min(t) + 1

        self.N, self.D = x.shape
        Model.__init__(self)
        self.x = x
        self.t = t
        self.K = K
        self.t_hot = np.zeros((self.N, self.K)) # "1-hot" coding
        self.t_hot[np.arange(self.N),self.t] = 1

        # Creating y_vect, an array of size (N, K)
        if th_map is None:
            # create the same bound for all data points
            y0_vect = np.zeros((self.N, self.K))
            y0_vect[np.arange(self.N),t] = y0
        else:
            # create bounds to be tight at th_map
            y0_vect = np.dot(self.x, th_map.T)

        # self.b is (N, K) and self.c is (N)
        self.b, self.c = self._bohning_bound(y0_vect)

        # # Compute sufficient statistics of data
        self.xt_sum = np.zeros((self.K,self.D)) # (K, D) array
        for i in range(self.N): self.xt_sum[self.t[i],:] += self.x[i,:]
        self.xb_sum = np.dot(self.b.T, self.x)  # (K, D) array
        self.xx_sum = np.dot(self.x.T, self.x)  # (D, D) array

        # Other hyperparameters
        self.th0 = th0
        self.th_shape = (self.K, self.D)

    def _logL(self, th, idxs):
        y = self.x[idxs, :].dot(th.T)  # (len(idxs), K)
        r = np.arange(len(idxs))
        y_maxes = np.max(y, axis=1, keepdims=True)
        y = y - y_maxes
        return y[r,self.t[idxs]] - np.log(np.sum(np.exp(y),axis=1))

    def _logB(self, th, idxs):
        y = self.x[idxs, :].dot(th.T)
        r = np.arange(len(idxs))
        return y[r,self.t[idxs]] - 0.25*np.sum(y**2,axis=1) \
                                 + 0.25*np.sum(y,axis=1)**2/self.K \
                                 + np.sum(y*self.b[idxs,:],axis=1) \
                                 + self.c[idxs]

    def _LBgap(self, th, idxs):
        y = self.x[idxs, :].dot(th.T)  # (len(idxs), K)
        y_maxes = np.max(y, axis=1, keepdims=True)
        L = - np.log(np.sum(np.exp(y - y_maxes), axis=1)) - y_maxes[:,0]
        B = - 0.25*np.sum(y**2,axis=1) \
                                 + 0.25*np.sum(y,axis=1)**2/self.K \
                                 + np.sum(y*self.b[idxs,:],axis=1) \
                                 + self.c[idxs]
        return L-B

    def _D_logL(self, th, idxs):
        # return an array of size (len(idxs), K, D)
        exp_y = np.exp(self.x[idxs, :].dot(th.T))
        return      self.x[idxs,:][ : ,None, :  ]  \
            * ( self.t_hot[idxs,:][ : , :  ,None]  \
                           - exp_y[ : , :  ,None]  \
            /np.sum(exp_y, axis=1)[ : ,None,None]  )
            # size is:      (len(idxs), K  , D  )

    def _D_logB(self, th, idxs):        
        th_sumk = np.sum(th, axis=0)
        A_th = 0.25*(-th + th_sumk[None,:]/self.K) # size (K, D)
        return             self.x[idxs,:][ : ,None, :  ]   \
                   * ( self.t_hot[idxs,:][ : , :  ,None]   \
                       +   self.b[idxs,:][ : , :  ,None]   \
         + 2 * self.x[idxs,:].dot(A_th.T)[ : , :  ,None] )
          # size is:               (len(idxs), K  , D  )                

    def _D_LBgap(self, th, idxs):

        exp_y = np.exp(self.x[idxs, :].dot(th.T))
        th_sumk = np.sum(th, axis=0)
        A_th = 0.25*(-th + th_sumk[None,:]/self.K) # size (K, D)
        return             self.x[idxs,:][ : ,None, :  ]   \
                            *(    - exp_y[ : , :  ,None]  \
                   /np.sum(exp_y, axis=1)[ : ,None,None]  
                       -   self.b[idxs,:][ : , :  ,None]   \
         - 2 * self.x[idxs,:].dot(A_th.T)[ : , :  ,None] )
          # size is:               (len(idxs), K  , D  )                

    def _logBProduct(self, th):
        th_sumk = np.sum(th, axis=0)
        th_A_th = - 0.25*np.dot(th.T, th) \
                  + 0.25*th_sumk[None,:] * th_sumk[:,None]/self.K
        return np.sum(th      *self.xt_sum) \
             + np.sum(th_A_th *self.xx_sum) \
             + np.sum(th      *self.xb_sum)

    def _D_logBProduct(self, th):
        th_sumk = np.sum(th, axis=0)
        A_th = 0.25*(-th + th_sumk[None,:]/self.K) # size (K, D)
        return    self.xt_sum                   \
                + 2 * np.dot(A_th, self.xx_sum) \
                + self.xb_sum
        
    def _logPrior(self, th):
        return -0.5*np.sum(th**2)/self.th0**2

    def _D_logPrior(self, th):
        return -th/self.th0**2

    def draw_from_prior(self):
        return npr.randn(self.K, self.D)*self.th0

    def _bohning_bound(self, y0):
        exp_y0 = np.exp(y0)
        A_y0 =y0.dot( 0.5*np.diag(np.ones(self.K)) - 0.5*np.ones((self.K, self.K))/self.K  )

        g = exp_y0/np.sum(exp_y0, axis=1)[:,None]
        b = A_y0 - g
        c = - 0.5*np.sum(y0*A_y0, axis=1) \
            + np.sum(g*y0,axis=1) \
            - np.log(np.sum(exp_y0,axis=1)) \
            - EPS
        return b, c

    def frac_misclassified(self, th):
        y = self.x.dot(th.T)  # (len(idxs), K)
        predictions = np.argmax(y, axis=1)
        return 1- np.sum(np.equal(self.t, predictions))/float(len(self.t))


class RobustRegressionModel(Model):
    def __init__(self, x, t_raw, scale=1, v=4, th0=1, y0=0, th_map=None):
        '''
        x      : Data, a (N, D) array
        t      : Targets, a (N) array of 0s and 1s
        th0    : Scale of the prior on weights
        th_map : Size (D) array, an estimate of MAP, for tuning the bounds
        y0     : Point at which to make bounds tight (if th_map
                 not given)
        '''
        self.N, self.D = x.shape
        Model.__init__(self)

        self.v     = v     = float(v)
        self.y0    = y0    = float(y0)
        t = t_raw / scale
        self.logZ = - np.log(scale) # normalization to account for change in scale 

        self.x = x
        self.t = t

        if th_map is None:          # create the same bound for all data points
            y0_vect = np.ones(self.N)*y0
        else:                       # create bounds to be tight at th_map
            y0_vect = t - np.dot(self.x, th_map[:,None])[:,0]

        a, c = self._t_dist_bound(y0_vect, v)
        self.coeffs = (a, c)

        self.A = x.T.dot(x*a[:,None])
        self.B = ( -2*t[None,:].dot(x*a[:,None]) ).T

        self.th0 = th0
        self.th_shape = (self.D,)

    def _logL(self, th, idxs):
        # returns an array of size idxs.size
        y_pred = np.dot(self.x[idxs,:],th[:,None])[:,0]
        residuals = self.t[idxs] - y_pred
        return -0.5*(self.v+1)*np.log(1 + residuals**2/self.v) + self.logZ

    def _D_logL(self, th, idxs):
        v = self.v
        y_pred = np.dot(self.x[idxs,:],th[:,None])[:,0]
        residuals = self.t[idxs] - y_pred
        return -self.x[idxs, :] \
            * (- residuals*(v+1)/(v + residuals**2) )[:,None] 

    def _logB(self, th, idxs):
        # lower bound on logistic regression log likelihoods
        # returns an array of size idxs.size
        y_pred = np.dot(self.x[idxs,:],th[:,None])[:,0]
        residuals = self.t[idxs] - y_pred
        a, c = self.coeffs
        return a[idxs]*residuals**2 + c[idxs] + self.logZ
        
    def _D_logB(self, th, idxs):
        y_pred = np.dot(self.x[idxs,:],th[:,None])[:,0]
        residuals = self.t[idxs] - y_pred
        a, _ = self.coeffs
        return -self.x[idxs,:]*(2*a[idxs]*residuals)[:,None]
        
    def _LBgap(self, th, idxs):
        # sum of derivative of log likelihoods of data points idxs
        y_pred = np.dot(self.x[idxs,:],th[:,None])[:,0]
        residuals = self.t[idxs] - y_pred
        a, c = self.coeffs
        L = -0.5*(self.v+1)*np.log(1 + residuals**2/self.v)
        B = a[idxs]*residuals**2 + c[idxs]
        return L - B

    def _D_LBgap(self, th, idxs):
        v = self.v
        y_pred = np.dot(self.x[idxs,:],th[:,None])[:,0]
        residuals = self.t[idxs] - y_pred
        a, _ = self.coeffs
        return -self.x[idxs, :] \
            * ( - residuals*(v+1)/(v + residuals**2)\
                - residuals*a[idxs]*2              )[:,None] 

    def _logBProduct(self, th):
        # log of the product of all the lower bounds
        return th[None,:].dot(self.A.dot(th[:,None])) + self.B.T.dot(th[:,None])

    def _D_logBProduct(self, th):
        return 2*self.A.dot(th[:,None])[:,0] + self.B[:,0]

    def _logPrior(self, th):
        # sparse prior
        return - np.sum(np.abs(th))/self.th0
 
    def _D_logPrior(self, th):
        return -np.sign(th)/self.th0

    def draw_from_prior(self):
        return npr.exponential(size=self.D)*self.th0

    def _t_dist_bound(self, y0, v):
        v  = float(v)
        a = -0.5*(v+1)/(v+y0**2)
        c = -a*y0**2 -(v+1)/2*np.log(1+y0**2/v) - EPS
        return (a, c)
