import numpy as np
import numpy.random as npr
import flymc as ff

class ModelSetup(object):
    def random_setup(self):
        for i in range(self.NUM_TRIALS):
            model = self.random_model()
            th = self.random_th()
            z = self.random_z()
            yield model, th, z

    def random_z(self):
        z = ff.BrightnessVars(self.N)
        z.brighten(np.where(npr.rand(self.N) > 0.5)[0])
        return z

class LogisticModelSetup(ModelSetup):
    def random_model(self, th_map=None):
        x = npr.randn(self.N, self.D)
        t = npr.randint(2, size=self.N)
        return ff.LogisticModel(x, t, th_map=th_map)

    def random_th(self):
        return np.random.randn(self.D)

class MulticlassLogisticModelSetup(ModelSetup):
    def random_model(self, th_map=None):
        x = npr.randn(self.N, self.D)
        t = npr.randint(self.K, size=self.N)
        return ff.MulticlassLogisticModel(x, t, self.K, th_map=th_map)

    def random_th(self):
        return np.random.randn(self.K, self.D)

class RobustRegressionModelSetup(ModelSetup):
    def random_model(self, th_map=None):
        x = npr.randn(self.N, self.D)
        t = 5 * npr.randn(self.N)
        return ff.RobustRegressionModel(x, t, scale=2, th_map=th_map)

    def random_th(self):
        return np.random.randn(self.D)
