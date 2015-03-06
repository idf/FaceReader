from expr.experiment_setup import Experiment
from facerec_py.facerec import normalization
from facerec_py.facerec.feature import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as nd

from skimage import data
from skimage.util import img_as_float
from skimage.filter._gabor import gabor_kernel


__author__ = 'Danyang'


class GaborFilter(AbstractFeature):
    def __init__(self, freq_r=(0.05, 0.15, 0.25), theta_r=6, sigma_tuple=(1, 3)):
        AbstractFeature.__init__(self)
        self._freq_t = freq_r
        self._theta_r = theta_r
        self._sigma_t = sigma_tuple

        self._kernels = []
        for theta in range(self._theta_r):
            theta = theta / float(self._theta_r) * np.pi
            for sigma in self._sigma_t:
                for frequency in self._freq_t:
                    kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
                    self._kernels.append(kernel)

    def compute(self, X, y):
        """

        :param X: The images, which is a Python list of numpy arrays.
        :param y: The corresponding labels (the unique number of the subject, person) in a Python list.
        :return:
        """
        # build the column matrix
        XC = asColumnMatrix(X)

        features = []
        for x in X:
            features.append(self.garbo_feature(x))
        return features

    def extract(self, X):
        """

        :param X: a single test data
        :return:
        """
        return self.garbo_feature(X)

    def garbo_feature(self, x):
        feats = np.zeros((len(self._kernels), 2), dtype=np.float32)
        for k, kernel in enumerate(self._kernels):
            filtered = nd.convolve(x, kernel, mode='wrap')
            feats[k, 0] = filtered.mean()
            feats[k, 1] = filtered.var()

        for i in xrange(2):
            feats[:, i] = normalization.zscore(feats[:, i])  # needed?

        return feats

    def raw_convolve(self, x):
        feats = np.zeros((len(self._kernels), x.size), dtype=np.float32)
        for k, kernel in enumerate(self._kernels):
            filtered = nd.convolve(x, kernel, mode='wrap')
            feats[k, :] = filtered.reshape(1, -1)
        return feats


    @property
    def prop(self):
        return self._prop

    def __repr__(self):
        return "GaborFilter (freq=%s, theta=%s)" % (str(self._freq_t), str(self._theta_r))


class LGBPHS(AbstractFeature):
    def __init__(self):
        """
        Un-weighted Local Gabor Binary Pattern Histogram Sequence
        :return:
        """
        super(LGBPHS, self).__init__()
        gabor = GaborFilter(theta_r=2, sigma_tuple=(1, ))
        gabor.garbo_feature = gabor.raw_convolve
        lbp = SpatialHistogram()

        self._model = ChainOperator(gabor, lbp)

    def compute(self, X, y):
        return self._model.compute(X, y)

    def extract(self, X):
        return self._model.extract(X)

    def __repr__(self):
        return "LGBPHS"


class GaborFilterFisher(AbstractFeature):
    def __init__(self):
        super(GaborFilterFisher, self).__init__()
        self._gabor = GaborFilter(theta_r=2, sigma_tuple=(1, ))  # decrease param; otherwise memory issue
        self._gabor.garbo_feature = self._gabor.raw_convolve  # replace
        self._fisher = Fisherfaces(14)

    def compute(self, X, y):
        model = ChainOperator(self._gabor, self._fisher)
        return model.compute(X, y)

    def extract(self, X):
        model = ChainOperator(self._gabor, self._fisher)
        return model.extract(X)

    def __repr__(self):
        return "GaborFilterFisher"

