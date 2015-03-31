import numpy as np
import math
from sklearn.neighbors import NearestNeighbors as nn
from facerec_py.facerec.feature import *
from facerec_py.facerec.util import *

__author__ = "XingJia"

class train():
    def __init__(self):
        self.mat = []
        self.labels = []
        self.dim = 0
        self.N = 0
        self.totalClass = 0

class NDAFisher(AbstractFeature):
    def __init__(self, num_components=100):
        AbstractFeature.__init__(self)
        self._num_components = num_components

    def compute(self, X, y):
        nda = NDA()
        pca = PCA(self._num_components)
        model = ChainOperator(pca, nda)
        model.compute(X, y)
        self._eigenvectors = np.dot(pca.eigenvectors, nda.W.T)
        features = []
        for x in X:
            xp = self.project(x.reshape(-1, 1))
            features.append(xp)
        return features

    def extract(self, X):
        X = np.asarray(X).reshape(-1, 1)
        return self.project(X)

    def project(self, X):
        return np.dot(self._eigenvectors.T, X)



class NDA(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)

    def compute(self, X, y, K=1, useweights=0):
        """
        fit data into to get the NDA model
        :param X: The images, which is a Python list of numpy arrays.
        :param y: The corresponding labels (the unique number of the subject, person) in a Python list.
        :return:
        """
        self.train = train()
        self.train.mat = np.array(X)
        self.train.mat = asColumnMatrix(X) #columns for data, rows for dim
        self.train.labels = y   #labels of samples
        self.train.N = np.size(X, 0)  #total number of samples
        self.train.dim = np.size(X, 1) #number of dim per sample
        self.train.totalClass = len(np.unique(y)) #number of classes
        self.K = K

        self.dim = np.size(X, 1)
        self.origdim = self.train.dim
        self.N = np.size(X, 0)
        self.totalClass = self.train.totalClass
        self.meandata = np.mean(self.train.mat, 1, dtype='float64')
        self.train.mat = self.train.mat - np.dot(self.meandata, np.ones((1, self.train.N)))

        self.indIn = np.zeros((K, self.N))
        self.indEx = np.zeros((K, self.N))
        self.valIn = np.zeros((K, self.N))
        self.valEx = np.zeros((K, self.N))

        self.compute_within_class_matrix_whitening()
        self.mnnInn = self.compute_within_class_scatter_matrix()
        self.diffIntra = self.train.mat - self.mnnInn
        self.Wscat = np.dot(self.diffIntra, self.diffIntra.transpose())/self.diffIntra.shape[1]
        self.eval, self.evec = np.linalg.eig(self.Wscat)
        self.ind = np.argsort(self.eval, 0)
        self.eval = self.eval[self.ind]
        self.eval = np.flipud(self.eval)
        self.ind = np.flipud(self.ind)
        self.evec = self.evec[:, self.ind]
        self.wdim = np.max(np.nonzero(self.eval > math.pow(10, -8))).real
        self.evec = self.evec[:, 0:self.wdim]
        self.whiteMat = np.dot(np.diag(1/np.sqrt(self.eval[0:self.wdim])), self.evec.transpose())
        self.Wtr = np.dot(self.whiteMat, self.train.mat)

        self.compute_bet_class_cluster_dist()
        self.mnnEx = self.compute_bet_class_cluster_matrix()
        self.diffExtra = self.Wtr - self.mnnEx
        if useweights:
            self.weights = np.minimum(self.valIn[self.K-1, :], self.valEx[self.K-1, :])
            temp = np.ones((self.Wtr.shape[1],))
            temp = np.dot(temp, self.weights)
            temp = temp * self.diffExtra
            self.bscat = np.dot(temp, np.transpose(self.diffExtra)) / self.N
        else:
            self.bscat = np.dot(self.diffExtra, self.diffExtra.conj().transpose())/self.N
        self.eigval, self.evec = np.linalg.eig(self.bscat)

        self.ind = np.argsort(self.eigval)
        self.val = self.eigval[self.ind]
        self.ind = np.flipud(self.ind)
        self.eigval = self.eigval[self.ind]
        self.eigvec = self.evec[:, self.ind[0:min(self.dim, self.wdim)]]
        self.mat = np.dot(self.eigvec.conj().transpose(), self.Wtr)
        self.W = np.dot(self.eigvec.conj().transpose(), self.whiteMat)
        self.proymat = self.W

        features = []
        for x in X:
            # # print 'x.shape:', x.shape
            xp = self.project(x.reshape(-1, 1))
            features.append(xp)
        return features


    def compute_bet_class_cluster_dist(self):
        # print 'Calculate distances for between-class scatter...'
        for x in np.unique(self.train.labels):
            # # print 'class:', x
            who_cl = np.where(self.train.labels == x)[0]
            who_notcl = np.where((self.train.labels != x))[0]
            self.data_intra = self.Wtr[:, who_cl]
            self.data_extra = self.Wtr[:, who_notcl]

            knn = nn().fit(self.data_extra.transpose())
            self.dextra, self.indextra = knn.kneighbors(self.data_intra.transpose())
            self.dextra = self.dextra.transpose()
            self.indextra = self.indextra.transpose()
            self.indEx[:, who_cl] = who_notcl[self.indextra[1, :]]
            self.valEx[:, who_cl] = self.dextra[1, :]

    def compute_bet_class_cluster_matrix(self):
        if self.K == 1:
            mnnEx = self.Wtr[:, map(lambda x: int(x), self.indEx[0, :])]
        else:
            mnnEx = np.zeros((np.size(self.Wtr, 0), self.train.N))
            for n in range(0, self.train.N):
                mnnEx[:, n] == np.mean(self.Wtr[:, map(lambda x: int(x), self.indEx[:, n])], 1)
        return mnnEx

    def compute_within_class_matrix_whitening(self):
        # print 'Distances for within-class scatter... '
        for x in np.unique(self.train.labels):
            who_cl = np.where(self.train.labels == x)[0]
            self.data_intra = self.train.mat[:, who_cl]
            knn = nn().fit(self.data_intra.transpose())
            self.dintra, self.indintra = knn.kneighbors(self.data_intra.transpose())
            self.dintra = self.dintra.transpose()
            self.indintra = self.indintra.transpose()
            self.dintra[self.K, :] = []
            self.indintra[self.K, :] = []
            self.valIn[:, who_cl] = self.dintra[1, :]
            self.indIn[:, who_cl] = who_cl[self.indintra[1, :]]

    def compute_within_class_scatter_matrix(self):
        if self.K == 1:
            mnnInn = self.train.mat[:, map(lambda x: int(x), self.indIn[0])]
        else:
            mnnInn = np.zeros((self.train.dim, self.train.N))
            for n in range(0, self.train.N):
                mean = np.mean(self.train.mat[:, map(lambda x: int(x), self.indIn[:, n])], 1)
                for x in range(0, self.train.dim):
                    mnnInn[x, n] = mean[x]
        return mnnInn

    def extract(self, X):
        X = np.asarray(X).reshape(-1, 1)
        return self.project(X)

    def project(self, X):
        return np.dot(self.W, X)

    def __repr__(self):
        return "NDA"











