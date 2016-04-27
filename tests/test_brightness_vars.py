import numpy as np
import numpy.random as npr
import flymc as ff
import unittest
npr.seed(1)

class BrightnessVarsTest(unittest.TestCase):
    N = 10
    EPS = 1e-10 # because _bernoulli_trials won't accept p==0
    def test_add_remove(self):
        all_idx = np.arange(self.N)
        bright = [0, 1, 9, 6, 5, 4]
        z = ff.BrightnessVars(self.N)
        z.brighten(bright)
        self.assertEqual(set(z.bright), set(bright))
        self.assertEqual(set(z.dark), set(all_idx) - set(bright))
        new_dark = [4, 9]
        z.darken(new_dark)
        self.assertEqual(set(z.bright), set(bright) - set(new_dark))
        self.assertEqual(set(z.dark), (set(all_idx) - set(bright)) | set(new_dark))

    def test_rand_p_eq_0(self):
        z = ff.BrightnessVars(self.N)
        all_idx = np.arange(self.N)
        bright = [0, 1, 9, 6, 5, 4]
        self.assertEqual(set(z.rand_bright(self.EPS)), set([]))
        self.assertEqual(set(z.rand_dark(self.EPS)), set([]))
        z.brighten(bright)
        self.assertEqual(set(z.rand_bright(self.EPS)), set([]))
        self.assertEqual(set(z.rand_dark(self.EPS)), set([]))
        z.brighten(all_idx)
        self.assertEqual(set(z.rand_bright(self.EPS)), set([]))
        self.assertEqual(set(z.rand_dark(self.EPS)), set([]))

    def test_rand_p_eq_1(self):
        z = ff.BrightnessVars(self.N)
        all_idx = np.arange(self.N)
        bright = [0, 1, 9, 6, 5, 4]
        self.assertEqual(set(z.rand_bright(1)), set([]))
        self.assertEqual(set(z.rand_dark(1)), set(all_idx))
        z.brighten(bright)
        self.assertEqual(set(z.rand_bright(1)), set(bright))
        self.assertEqual(set(z.rand_dark(1)), set(all_idx) - set(bright))
        z.brighten(all_idx)
        self.assertEqual(set(z.rand_bright(1)), set(all_idx))
        self.assertEqual(set(z.rand_dark(1)), set([]))

    def test_rand_stats(self):
        z = ff.BrightnessVars(self.N)
        all_idx = np.arange(self.N)
        bright = [0, 1, 9, 6, 5, 4]
        dark = list(set(all_idx) - set(bright))
        z.brighten(bright)
        num_trials = 10000
        p = 0.4
        dark_counts = np.zeros(self.N)
        bright_counts = np.zeros(self.N)
        for i in range(num_trials):
            dark_counts[z.rand_dark(p)] += 1
            bright_counts[z.rand_bright(p)] += 1

        # Counts, if allowed, should all be 4000 +/- 50 at the 1 s.d. level
        self.assertTrue(np.all(dark_counts[bright] == 0))
        self.assertTrue(np.all(bright_counts[dark] == 0))
        self.assertTrue(np.all(abs(dark_counts[dark] - 4000) < 200))
        self.assertTrue(np.all(abs(bright_counts[bright] - 4000) < 200))
