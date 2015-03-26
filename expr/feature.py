import numpy as np
from scipy import ndimage as nd
from skimage.filters import gabor_kernel  # from skimage.filter._gabor import gabor_kernel

from facerec_py.facerec import normalization
from facerec_py.facerec.feature import *
# import cv2
from util.commons_util.decorators.algorithms import memoize, memoize_force

__author__ = 'Danyang'


class GaborFilterSki(AbstractFeature):
    """
    Implemented using skimage

    The frequency of the span-limited sinusoidal grating is given by Freq and its orientation is specified by Theta.
    Sigma is the scale parameter.
    """
    def __init__(self, freq_t=(0.05, 0.15, 0.25), theta_r=8, sigma_tuple=(1, 2, 4, 8, 16)):
        AbstractFeature.__init__(self)
        self._freq_t = freq_t
        self._theta_r = theta_r
        self._sigma_t = sigma_tuple

        self._kernels = []
        for theta in range(self._theta_r):
            theta = theta / float(self._theta_r) * np.pi
            for sigma in self._sigma_t:
                for frequency in self._freq_t:
                    kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
                    self._kernels.append(kernel)  # taking real part

    def compute(self, X, y):
        """
        convention: cap X, small y
        :param X: The images, which is a Python list of numpy arrays.
        :param y: The corresponding labels (the unique number of the subject, person) in a Python list.
        :return:
        """
        # build the column matrix
        XC = asColumnMatrix(X)

        features = []
        for x in X:
            features.append(self.filter(x))
        return features

    def extract(self, X):
        """

        :param X: a single test data
        :return:
        """
        return self.filter(X)

    def filter(self, x):
        feats = np.zeros((len(self._kernels), 2), dtype=np.float32)
        for k, kernel in enumerate(self._kernels):
            filtered = nd.convolve(x, kernel, mode='wrap', cval=0)
            feats[k, 0] = filtered.mean()
            feats[k, 1] = filtered.var()

        for i in xrange(2):
            feats[:, i] = normalization.zscore(feats[:, i])  # needed?

        return feats

    def convolve_to_col(self, x):
        feats = np.zeros((len(self._kernels), x.size), dtype=np.float32)
        for k, kernel in enumerate(self._kernels):
            filtered = nd.convolve(x, kernel, mode='reflect', cval=0)
            feats[k, :] = filtered.reshape(1, -1)
        return feats

    def raw_convolve(self, x):
        feats = np.zeros((len(self._kernels), x.shape[0], x.shape[1]), dtype=np.float32)
        for i, v in enumerate(self._kernels):
            filtered = nd.convolve(x, v, mode='reflect', cval=0)
            feats[i, :, :] = filtered
        return feats

    @property
    def prop(self):
        return self._prop

    def __repr__(self):
        return "GaborFilterSki (freq=%s, theta=%s)" % (str(self._freq_t), str(self._theta_r))

    def short_name(self):
        return "GaborFilter"


class GaborFilterCv2(AbstractFeature):
    """
    Implemented using cv2

    http://docs.opencv.org/trunk/modules/imgproc/doc/filtering.html#getgaborkernel
    """
    def __init__(self, orient_cnt=8, scale_cnt=5):
        AbstractFeature.__init__(self)
        self._orient_cnt = orient_cnt
        self._scale_cnt = scale_cnt
        self._kernels = []
        self.build_filters()

    def build_filters(self):
        self._kernels = []
        k_max = 199
        lambd = 10.0  # 10.0
        sigma = 4.0  # 4.0
        gamma = 0.5
        for theta in np.arange(0, np.pi, np.pi/self._orient_cnt):
            for scale in range(self._scale_cnt):
                ksize = int(k_max/2**scale+0.5)  # TODO
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
                self._kernels.append(kernel)

    def compute(self, X, y):
        """
        convention: cap X, small y
        :param X: The images, which is a Python list of numpy arrays.
        :param y: The corresponding labels (the unique number of the subject, person) in a Python list.
        :return:
        """
        # build the column matrix
        XC = asColumnMatrix(X)

        features = []
        for x in X:
            features.append(self.filter(x))
        return features

    def extract(self, X):
        """

        :param X: a single test data
        :return:
        """
        return self.filter(X)

    def filter(self, x):
        return self.fractalius_filter(x)

    def normalize(self, mul, kernel):
        return kernel / (mul*kernel.sum())

    def simple_filter(self, x):
        """
        Simple Gabor Convolve
        :param x: a single test data
        :return:
        """
        feats = np.zeros((len(self._kernels), x.shape[0], x.shape[1]), dtype=np.float32)
        for i, kernel in enumerate(self._kernels):
            k = self.normalize(1, kernel)
            filtered = cv2.filter2D(x, cv2.CV_8UC3, k)
            feats[i, :, :] = filtered
        return feats

    def fractalius_filter(self, x):
        """
        http://www.redfieldplugins.com/filterFractalius.htm

        only need to get the largest
        :param x: a single test data
        :return: features
        """
        feats = np.zeros((1, x.shape[0], x.shape[1]), dtype=np.float32)

        accum = np.zeros_like(x)
        for i, kernel in enumerate(self._kernels):
            k = self.normalize(1.5, kernel)
            filtered = cv2.filter2D(x, cv2.CV_8UC3, k)
            np.maximum(accum, filtered, accum)

        feats[0, :, :] = accum
        return feats

    def __repr__(self):
        return "GaborFilterCv2 (orient_count=%s, scale_count=%s)" % (str(self._orient_cnt), str(self._scale_cnt))

    def short_name(self):
        return "GaborFilter"


class MultiScaleSpatialHistogram(SpatialHistogram):
    def __init__(self, lbp_operator=ExtendedLBP(), sz=(8, 8)):
        super(MultiScaleSpatialHistogram, self).__init__(lbp_operator, sz)

    def spatially_enhanced_histogram(self, X):
        hists = []
        for x in X:
            hist = super(MultiScaleSpatialHistogram, self).spatially_enhanced_histogram(x)
            hists.append(hist)
        return np.asarray(hists)


class ConcatenatedSpatialHistogram(SpatialHistogram):
    def __init__(self, lbp_operator=ExtendedLBP(), sz=(8, 8)):
        super(ConcatenatedSpatialHistogram, self).__init__(lbp_operator, sz)

    def spatially_enhanced_histogram(self, X):
        hists = []
        for x in X:
            hist = super(ConcatenatedSpatialHistogram, self).spatially_enhanced_histogram(x)
            hists.extend(hist)
        return np.asarray(hists)


class LGBPHS(AbstractFeature):
    def __init__(self):
        """
        Un-weighted Local Gabor Binary Pattern Histogram Sequence
        :return:
        """
        super(LGBPHS, self).__init__()
        gabor = GaborFilterSki(theta_r=2, sigma_tuple=(1, ))
        gabor.filter = gabor.raw_convolve
        lbp = MultiScaleSpatialHistogram()

        self._model = ChainOperator(gabor, lbp)

    def compute(self, X, y):
        return self._model.compute(X, y)

    def extract(self, X):
        return self._model.extract(X)

    def __repr__(self):
        return "LGBPHS(%s)" % (repr(self._model))

    def short_name(self):
        return "LGBPHS"


class LGBPHS2(LGBPHS):
    def __init__(self):
        super(LGBPHS, self).__init__()
        # gabor = GaborFilterSki(freq_t=(0.15, ), theta_r=4)
        gabor = GaborFilterCv2(4, 2)
        lbp = ConcatenatedSpatialHistogram()

        self._model = ChainOperator(gabor, lbp)

    def __repr__(self):
        return "LGBPHS2(%s)"%(repr(self._model))

    def short_name(self):
        return "LGBPHS"


class GaborFilterFisher(AbstractFeature):
    def __init__(self):
        super(GaborFilterFisher, self).__init__()
        self._gabor = GaborFilterSki(theta_r=2, sigma_tuple=(1, ))  # decrease param; otherwise memory issue
        self._gabor.filter = self._gabor.convolve_to_col  # replace
        self._fisher = Fisherfaces(14)

    def compute(self, X, y):
        model = ChainOperator(self._gabor, self._fisher)
        return model.compute(X, y)

    def extract(self, X):
        model = ChainOperator(self._gabor, self._fisher)
        return model.extract(X)

    def __repr__(self):
        return "GaborFilterFisher"

    def short_name(self):
        return "GaborFisher"

