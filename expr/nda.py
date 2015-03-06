import numpy as np
import math
import sklearn.neighbors.NearestNeighbors as nn
from facerec_py.facerec.feature import *
class Nda(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)

    def compute(self, X, y, K=1, dim=10, useweights=0):
        """
        fit data into to get the NDA model
        :param X: The images, which is a Python list of numpy arrays.
        :param y: The corresponding labels (the unique number of the subject, person) in a Python list.
        :return:
        """
        self.train.mat = X
        self.train.labels = y
        self.train.N = np.size(X, 0)
        self.train.dim = np.size(X, 1)
        self.train.totalClass = len(np.unique(y))

        self.dim = np.size(X, 1)
        self.origdim = self.train.dim
        self.N = np.size(X, 0)
        self.totalClass = train.totalClass
        self.meandata = np.mean(self.train.mat, 2)
        self.train.mat = self.train.mat - self.meandata * np.ones(1, self.train.N)

        self.indIn = np.zeros(K, self.N)
        self.indEx = np.zeros(K, self.N)
        self.valIn = np.zeros(K, self.N)
        self.valEx = np.zeros(K, self.N)
        self.compute_within_class_matrix_whitening(self.train, K)
        self.minInn = self.compute_within_class_scatter_matrix(self, self.train)
        self.diffIntra = self.train.mat - self.mnnInn
        self.Wscat = np.dot(self.diffIntra, self.diffIntra.conj().transpose())/self.diffIntra.shape[1]
        self.eval, self.evec = np.linalg.eig(self.Wscat)
        self.eval = np.diag(self.eval)
        self.ind = np.argsort(self.eval)
        self.eval = np.flipud(self.eval[self.ind])
        self.ind = np.flipud(self.ind)
        self.evec = self.evec[:, self.ind]
        self.wdim = np.max(np.nonzero(self.eval>math.pow(10, -8)))
        self.evec = self.evec[:, 1:self.wdim]
        self.whiteMat = np.dot(np.diag(1/np.sqrt(eval[:self.wdim])), self.evec.conj().transpose())
        self.Wtr = np.dot(self.whiteMat, self.train.mat)
        self.comput_bet_class_cluster_dist()
        self.mnnEx = self.compute_within_class_scatter_matrix()
        self.diffExtra = self.Wtf - self.mnnEx

        self.bscat = np.dot(self.diffExtra, self.diffExtra.conj().transpose())/train.N
        self.eigval, self.evec = np.linalg.eig(self.bscat)
        self.eigval = np.diag(self.eigval)
        self.ind = np.argsort(self.eigval)
        self.val = self.eigval[self.ind]
        self.ind = np.flipud(self.ind)
        self.eigval = self.eigval[self.ind]
        self.eigvec = self.evec[:, self.ind[1:min(self.dim, self.wdim)]]
        self.mat = np.dot(self.eigvec.conj().transpose(), self.Wtr)
        self.W = np.dot(self.eigvec.conj().transpose(), self.whiteMat)
        self.proymat = self.W

    def comput_bet_class_cluster_dist(self):
        print 'Distances for between-class scatter...'
        for x in range(0,self.totalClass):
            print(x)
            who_cl = np.nonzero(self.train.labels == x)
            who_notcl = np.nonzero((self.train.labels != x))
            self.data_intra = self.Wtr[:, who_cl]
            self.data_extra = self.Wtr[:, who_notcl]

            knn = nn().fit(self.data_extra)
            self.dextra, self.indextra = knn.kneighbors(self.data_intra)
            self.indEx[:,who_cl] = who_notcl[self.indextra]
            self.valEx[:,who_cl] = self.dextra

    def comput_bet_class_cluster_matrix(self):
        if self.K == 1:
            mnnEx = self.Wtr[:, self.indEx[1, :]]
        else:
            mnnEx = np.zeros(np.size(self.Wtr, 0), self.train.N)
            for n in range(0, self.train.N):
                mnnEx[:,n] == np.mean(self.Wtr[:, self.indEx[:, n]], 1)


    def compute_within_class_matrix_whitening(self, train, K):
        print 'Distances for within-class scatter... '
        for x in range(0, self.totalClass):
            print(x,'class:')
            who_cl = np.find(train.labels == x)
            self.data_intra = train.mat[:who_cl]
            # cuantos = len(who_cl)
            knn = nn().fit(self.data_intra)
            self.dintra, self.indintra = knn.kneighbors(self.data_intra)
            self.dintra[self.K, :] = []
            self.indintra[self.K, :] = []
            self.indIn[:, who_cl] = who_cl[self.indintra]
            self.valIn[:, who_cl] = self.dintra

    def compute_within_class_scatter_matrix(self):
        if self.K == 1:
            mnnInn = self.train.mat[:, self.indIn]
        else:
            mnnIn = np.zeros(self.train.dim, self.train.N)
            for n in range(0, self.train.N):
                mnnIn[:,n] = np.mean(self.train.mat[:, self.indIn[:, n]], 2)
        return mnnInn

    def extract(self, X):
        return self.proymat

    def __repr__(self):
        return "NDA"











