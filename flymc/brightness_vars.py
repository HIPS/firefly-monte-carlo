import numpy as np

class BrightnessVars(object):
    '''
    Class to maintain a record of all N 'brightness' variables, z_i while
    while allowing the following methods to be efficiently performed:
    
    - Switching the state of M data points (O(M) time)
    - Selecting M random bright or dark data point (O(M) time)
    - Producing a list of all B bright data points (O(B) time)

    The class maintains two length N arrays and an integer B:
    B        : the current number of bright data points
    self.arr : contains the indices of the data points in an arbitrary
               order except that the bright ones are first.
    self.tab : a direct-address table. tab[i] contains the index of
               data point i in the array arr, so arr[tab[i]] == i .
    For example, if we had 6 data points, of which only 0 and 4 were bright,
    the arrays could be the following:
    
    self.tab = [0, 2,   5, 3, 1, 4]
    self.arr = [0, 4,   1, 3, 5, 2]
    self.B = 2        |  <- bright/dark boundary
    '''
    def __init__(self, N, bright_dpts=[]):
        # Initializes the set, with bright indices specifies by bright_dpts
        self.N = N
        self.B = 0
        self.tab = np.arange(N)
        self.arr = np.arange(N)
        self.brighten(bright_dpts)

    def brighten(self, dpts):
        # Makes data points dpts bright
        # dpts must not contain duplicates
        # locations of data points that need to be moved:
        locs = self.tab[dpts][self.tab[dpts] >= self.B]
        self._move_elements(locs, self.B, self.B + len(locs))
        self.B += len(locs)

    def darken(self, dpts):
        # Makes data points dpts dark
        # dpts must not contain duplicates
        # locations of data points that need to be moved:
        locs = self.tab[dpts][self.tab[dpts] < self.B]
        self._move_elements(locs, self.B - len(locs), self.B)
        self.B -= len(locs)
        
    def _move_elements(self, locs, low, high):
        # Moves the elemnents in arr[locs] to arr[range(low,high)], relocating
        # any displaced tenants to the newly vacated real estate
        old = locs[np.logical_or(locs < low , locs >= high)]
        correct = locs[np.logical_and(low <= locs, locs < high)]       
        mask = np.ones(high-low, dtype=bool) 
        mask[correct - low] = False     # mask the locations already correct
        new = np.arange(low,high)[mask]       

        self.arr[new], self.arr[old] = self.arr[old], self.arr[new]
        self.tab[self.arr[new]] = new
        self.tab[self.arr[old]] = old

    def rand_bright(self, p):
        # Returns bimoial(self.B, p) random bright datapoints without replacement
        idxs = self._bernoulli_trials(self.B, p)
        return self.arr[idxs]

    def rand_dark(self, p):
        # Returns bimoial(self.N - self.B, p) random dark datapoints without replacement
        idxs = self.B + self._bernoulli_trials(self.N-self.B, p)
        return self.arr[idxs]

    def _bernoulli_trials(self, M, p):
        # Simulates M Bernoulli trials with success probability p, returning
        # the indices (from 0 to M-1) of successes
        assert(p>0)
        geometric = np.random.geometric
        i = -1
        success_idxs = []
        while True:
            i += geometric(p)
            if i >= M : break
            success_idxs.append(i)
        return np.array(success_idxs, dtype=int)

    @property
    def bright(self): 
        # Returns all the bright datapoints
        return self.arr[:self.B].copy()

    @property
    def dark(self): 
        # Returns all the dark datapoints
        return self.arr[self.B:].copy()

    @property
    def isvalid(self):
        # Confirms self-consitency
        valid = np.sum(self.arr) == (self.N-1)*self.N/2 \
                and np.all(self.arr[self.tab]==np.arange(self.N))
        return valid
