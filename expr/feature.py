import numpy as np
from scipy import ndimage as nd
from skimage.filters import gabor_kernel  # from skimage.filter._gabor import gabor_kernel

from facerec_py.facerec import normalization
from facerec_py.facerec.feature import *
import cv2
from facerec_py.facerec.lbp import *
from facerec_py.facerec.preprocessing import LBPPreprocessing

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
        self.orient_cnt = orient_cnt
        self.scale_cnt = scale_cnt
        self._kernels = []
        self.build_filters()

    def build_filters(self):
        self._kernels = []
        k_max = 399
        lambd = 10.0  # 10.0
        sigma = 4.0  # 4.0
        gamma = 0.5
        for theta in np.arange(0, np.pi, np.pi/self.orient_cnt):
            for scale in range(self.scale_cnt):
                ksize = int(k_max/2**scale+0.5)  # TODO
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0.5, ktype=cv2.CV_32F)  # psi: (0.5, 93.3%), (0.75, 93.33%)
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
        return self.__simple_filter(x)

    def normalize(self, mul, kernel):
        return kernel / (mul*kernel.sum())

    def __simple_filter(self, x):
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

    def __fractalius_filter(self, x):
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
        return "GaborFilterCv2 (orient_count=%s, scale_count=%s)" % (str(self.orient_cnt), str(self.scale_cnt))

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
    def __init__(self, lbp_operator=LPQ(radius=4), sz=(8, 8)):
        super(ConcatenatedSpatialHistogram, self).__init__(lbp_operator, sz)

    def spatially_enhanced_histogram(self, X):
        hists = []
        for x in X:  # reduce 1 dimension
            hist = super(ConcatenatedSpatialHistogram, self).spatially_enhanced_histogram(x)
            hists.extend(hist)
        return np.asarray(hists)


class ChainedFeature(AbstractFeature):
    def __init__(self, feature1, feature2):
        if not isinstance(feature1, AbstractFeature) or not isinstance(feature2, AbstractFeature):
            raise TypeError("model not correct, since must be AbstractFeature")

        self.feature = ChainOperator(feature1, feature2)

    def compute(self, X, y):
        return self.feature.compute(X, y)

    def extract(self, X):
        return self.feature.extract(X)

    def __repr__(self):
        return "%s(%s)"%(self.short_name(), repr(self.feature))

    def short_name(self):
        return self.__class__.__name__


class LGBPHS(ChainedFeature):
    def __init__(self):
        """
        Un-weighted Local Gabor Binary Pattern Histogram Sequence
        :return:
        """
        gabor = GaborFilterSki(theta_r=2, sigma_tuple=(1, ))
        gabor.filter = gabor.raw_convolve
        lbp_hist = MultiScaleSpatialHistogram()
        super(LGBPHS, self).__init__(gabor, lbp_hist)


class LGBPHS2(ChainedFeature):
    def __init__(self, n_orient=4, n_scale=2, lbp_operator=ExtendedLBP(radius=3)):  # alternatively LPQ
        gabor = GaborFilterCv2(n_orient, n_scale)
        lbp_hist = ConcatenatedSpatialHistogram(lbp_operator=lbp_operator)
        super(LGBPHS2, self).__init__(gabor, lbp_hist)


class GaborFisher(ChainedFeature):
    def __init__(self):
        gabor = GaborFilterSki(theta_r=2, sigma_tuple=(1, ))  # decrease param; otherwise memory issue
        gabor.filter = gabor.convolve_to_col  # replace
        fisher = Fisherfaces(14)
        super(GaborFisher, self).__init__(gabor, fisher)


class LbpFisher(ChainedFeature):
    def __init__(self, lbp_operator=ExtendedLBP(radius=11)):  # (6, 11, 14, 15, 19)
        lbp = LBPPreprocessing(lbp_operator=lbp_operator)  # preprocessing, not histogram
        fisher = Fisherfaces(14)
        super(LbpFisher, self).__init__(lbp, fisher)


class GaborLbpFisher(ChainedFeature):
    def __init__(self, n_orient=4, n_scale=2, lbp_operator=ExtendedLBP(radius=11)):
        # TODO
        gabor = GaborFilterCv2(n_orient, n_scale)
        lbp_fisher = LbpFisher(lbp_operator)
        super(GaborLbpFisher, self).__init__(gabor, lbp_fisher)