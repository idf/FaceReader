from facerec_py.facerec.feature import AbstractFeature
import numpy as np
from facerec_py.facerec.util import asColumnMatrix
from sklearn.decomposition import KernelPCA

__author__ = 'Danyang'


class KPCA(AbstractFeature):
    def __init__(self, num_components=0):
        AbstractFeature.__init__(self)
        self._num_components = num_components
        self._kpca = None

    def compute(self,X,y):
        """
        PCA over the entire images set
        dimension reduction for entire images set


        * Prepare the data with each column representing an image.
        * Subtract the mean image from the data.
        * Calculate the eigenvectors and eigenvalues of the covariance matrix.
        * Find the optimal transformation matrix by selecting the principal components (eigenvectors with largest eigenvalues).
        * Project the centered data into the subspace.
        Reference: http://en.wikipedia.org/wiki/Eigenface#Practical_implementation

        :param X: The images, which is a Python list of numpy arrays.
        :param y: The corresponding labels (the unique number of the subject, person) in a Python list.
        :return:
        """
        # build the column matrix
        XC = asColumnMatrix(X)
        y = np.asarray(y)

        # set a valid number of components
        if self._num_components <= 0 or (self._num_components > XC.shape[1]-1):
            self._num_components = XC.shape[1]-1  # one less dimension

        # center dataset
        self._mean = XC.mean(axis=1).reshape(-1,1)
        XC = XC - self._mean
        n_features = XC.shape[0]
        # get the features from the given data
        # http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html
        self._kpca = KernelPCA(n_components=60, kernel="poly", degree=3, coef0=0.0)

        self._kpca.fit(XC.T)

        features = []
        for x in X:
            features.append(self.extract(x))
        return features

    def extract(self,X):
        X = np.asarray(X).reshape(-1,1)
        return self.project(X)

    def project(self, X):
        """
        Need to transpose to fit the dim requirement of KernelPCA
        """
        X = X - self._mean
        return self._kpca.transform(X.T)

    @property
    def num_components(self):
        return self._num_components

    def __repr__(self):
        return "KernelPCA (num_components=%d)" % self._num_components

    def short_name(self):
        return "KernelPCA"