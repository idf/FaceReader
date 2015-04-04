from expr.feature import *

__author__ = 'Danyang'


class WeightedHS(SpatialHistogram):  # per gabor
    def __init__(self, lbp_operator=ExtendedLBP(3), sz=(8, 8), X=None, y=None):
        super(WeightedHS, self).__init__(lbp_operator, sz)  # must be placed before
        self.weights = {}
        self.Hs = {}

        self.L = self.calculate_L(X)
        self.X = X
        self.y = y

    def init_cache(self):
        self.weights = {}
        self.Hs = {}

    def compute(self, X, y):
        raise NotImplementedError("Not and won't be implemented.")

    def hist_intersect(self, p, q):
        p = np.asarray(p).flatten()
        q = np.asarray(q).flatten()
        sim = np.sum(np.minimum(p, q))
        return sim

    def get_weight(self, row, col):
        args = (row, col)
        if args not in self.weights:
            self.weights[args] = self._get_weight(row, col)
        return self.weights[args]

    def _get_weight(self, row, col):
        C = len(np.unique(self.y))
        m_I = 0
        for i in np.unique(self.y):
            N_i = self.get_N(i)
            Hs_i = self.get_Hs(i, row, col)
            for k in xrange(len(Hs_i)):
                for j in xrange(k):
                    m_I += self.hist_intersect(Hs_i[j], Hs_i[k])
            m_I *= 2.0/(N_i*(N_i-1))
        m_I *= 1.0/C

        S2_I = 0
        for i in np.unique(self.y):
            Hs_i = self.get_Hs(i, row, col)
            for k in xrange(len(Hs_i)):
                for j in xrange(k):
                    S2_I += (self.hist_intersect(Hs_i[j], Hs_i[k])-m_I)**2

        m_E = 0
        for i in np.unique(self.y):
            for j in np.unique(self.y[self.y != i]):
                N_i = self.get_N(i)
                N_j = self.get_N(j)
                Hs_i = self.get_Hs(i, row, col)
                Hs_j = self.get_Hs(j, row, col)
                for h_i in Hs_i:
                    for h_j in Hs_j:
                        m_E += self.hist_intersect(h_i, h_j)
                m_E *= 1.0/(N_i*N_j)
        m_E *= 2.0/(C*(C-1))

        S2_E = 0
        for i in np.unique(self.y):
            for j in np.unique(self.y[self.y!=i]):
                Hs_i = self.get_Hs(i, row, col)
                Hs_j = self.get_Hs(j, row, col)
                for h_i in Hs_i:
                    for h_j in Hs_j:
                        S2_E += (self.hist_intersect(h_i, h_j)-m_E)**2

        weight = (m_I-m_E)**2/(S2_I+S2_E)   # histogram normalization won't affect weight
        return weight

    def get_N(self, i):
        return len(self.X[self.y==i])

    def calculate_L(self, X):
        L = []
        for x in X:
            L.append(self.lbp_operator(x))
        return np.asarray(L)

    def get_Hs(self, label, row, col):
        args = (label, row, col)
        if args not in self.Hs:
            self.Hs[args] = self._get_Hs(label, row, col)
        return self.Hs[args]

    def _get_Hs(self, label, row, col):
        L = self.L[self.y==label]
        Hs = []
        for l in L:
            lbp_height, lbp_width = l.shape
            grid_rows, grid_cols = self.sz
            py = int(np.floor(lbp_height / grid_rows))
            px = int(np.floor(lbp_width / grid_cols))

            C = l[row * py:(row + 1) * py, col * px:(col + 1) * px]
            Hs.append(super(WeightedHS, self)._get_histogram(C, row, col))
        return Hs

    def _get_histogram(self, C, row, col, normed=True):
        return self.get_weight(row, col)* \
               super(WeightedHS, self)._get_histogram(C, row, col)


class ConcatendatedWeightedHS(SpatialHistogram):
    def __init__(self, lbp_operator=ExtendedLBP(radius=3), sz=(8, 8)):
        super(ConcatendatedWeightedHS, self).__init__(lbp_operator, sz)
        self.weights = None

    def compute(self, X, y):
        self.weights = self.construct_wights(X, y)
        super(ConcatendatedWeightedHS, self).compute(X, y)

    def construct_wights(self, X, y):
        X = np.asarray(X)
        n_gabors = X.shape[1]

        weights = [WeightedHS(X=X[:, i, :, :], y=y) for i in xrange(n_gabors)]  # manipulate high dimensional data
        return weights

    def spatially_enhanced_histogram(self, X):
        hists = []
        for gabor_idx, x in enumerate(X):  # reduce 1 dimension, gabor dimension
            hist = self.weights[gabor_idx].spatially_enhanced_histogram(x)
            hists.extend(hist)
        return np.asarray(hists)


class WeightedLGBPHS(ChainedFeature):
    def __init__(self, n_orient=4, n_scale=2, lbp_operator=ExtendedLBP(radius=3)):  # alternatively LPQ
        # TODO speed up
        gabor = GaborFilterCv2(n_orient, n_scale)
        lbp_hist = ConcatendatedWeightedHS(lbp_operator=lbp_operator)
        super(WeightedLGBPHS, self).__init__(gabor, lbp_hist)
