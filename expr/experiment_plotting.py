from experiment_setup import *
import numpy as np
__author__ = 'Danyang'


class Plotter(object):
    def __init__(self):
        pass

    def _plot(self, models):
        expr = Experiment()
        for model in models:
            cv = expr.experiment(model, threshold_up=1)
            expr.plot_roc(cv)
        expr.show_plot()


class PlotterPCA(Plotter):
    def plot_components(self):
        models = []
        for num_component in xrange(10, 150, 30):  # TODO
            models.append(PCA(num_component))

        self._plot(models)

    def plot_energy(self):
        models = []

        class PCA_energy(PCA):
            def short_name(self):
                return "PCA: %.2f%%"%(self.energy_percentage*100)

        for num_component in xrange(20, 110, 40):  # TODO
            models.append(PCA_energy(num_component))

        self._plot(models)


class PlotterFisher(Plotter):
    def plot_components(self):
        models = []
        for num_components in xrange(1, 16, 3):  # TODO
            models.append(Fisherfaces(num_components))

        self._plot(models)


class PlotterKnn(object):
    def plot_kNN(self):
        """
        _plot the graph of varying k of kNN
        :return:
        """
        pca = PCA(40)
        expr = Experiment()

        plt.figure("PCA precision for different k in kNN")
        plt.xlabel("k of kNN")
        plt.ylabel("precision")

        xys = []
        for k in xrange(1, 41):  # TODO
            cv = expr.experiment(pca, threshold_up=0, kNN_k=k, debug=False)
            xys.append((k, cv.validation_results[0].precision))

        plt.plot([elt[0] for elt in xys], [elt[1] for elt in xys])
        plt.show()


class PlotterLgbphs(Plotter):
    def plot_lbp_algorihtms(self):
        class LgbphsSub(LGBPHS2):
            def short_name(self):
                return "%s"%self.feature.model2.lbp_operator.short_name()

        models = []
        for lbp in (OriginalLBP(), ExtendedLBP(radius=3), LPQ(radius=4)):
            models.append(LgbphsSub(lbp_operator=lbp))

        self._plot(models)

    def plot_scales(self, r=(1, 5, 9)):  # 1~5
        class LgbphsSub(LGBPHS2):
            def short_name(self):
                return "scale: %s" % self.feature.model1.scale_cnt

        self._plot([LgbphsSub(n_scale=i, lbp_operator=OriginalLBP()) for i in r])

    def plot_orientations(self, r=xrange(2, 9, 2)):  # 1~8
        class LgbphsSub(LGBPHS2):
            def short_name(self):
                return "orient: %s" % self.feature.model1.orient_cnt

        self._plot([LgbphsSub(n_orient=i, lbp_operator=OriginalLBP()) for i in r])

    def plot_histogram(self):
        pass


class PlotterKernelPCA(Plotter):
    def plot_rbf(self, r=(10000.0/(200*200), 0.5, 0.75, 1.0)):  # TODO
        class KPCASub(KPCA):
            def short_name(self):
                return "%s, gamma=%.4f"%(self._kernel, self._gamma)

        self._plot([KPCASub(kernel="rbf", gamma=i) for i in r])

    def plot_poly_degree(self):
        """
        varying degrees for poly
        """
        models = []
        class KPCA_poly(KPCA):
            def short_name(self):
                return "poly (degree: %d)" %self._degree
            
        for degree in xrange(1, 6):
            models.append(KPCA_poly(50, "poly", degree))
        self._plot(models)

    def plot_kernels(self):
        """
        varying kernels
        :return:
        """
        kernels = ["poly", "sigmoid", "cosine", "rbf"]
        models = []
        for kernel in kernels:
            models.append(KPCA(50, kernel))
        self._plot(models)

    # others plotting TODO
    def plot_poly_coef0(self, r = (0.0, 20.0, 40.0, 60.0, 80.0)):
        """
        varying coef0 values for poly
        """
        class KPCA_coef0(KPCA):
            def short_name(self):
                return "poly (coef0: %.2f)" %self._coef0
        self._plot([KPCA_coef0(kernel="poly", coef0 = i) for i in r])

    def plot_poly_gamma(self, r = (10.0, 40.0, 70.0, 100.0, 130.0)):
        """
        varying gamma values for poly
        """
        class KPCA_gamma(KPCA):
            def short_name(self):
                return "poly (gamma: %.2f)" %self._gamma
        self._plot([KPCA_gamma(kernel = "poly", gamma = i) for i in r])

    def plot_sigmoid(self, r = (10.0, 20.0, 30.0, 40.0, 50.0)):
        """
        varying gamma values for sigmoid
        """
        class KPCA_sigmoid(KPCA):
            def short_name(self):
                return "sigmoid (gamma: %.2f)"%self._gamma
        self._plot([KPCA_sigmoid(kernel = "sigmoid", gamma = i) for i in r])


if __name__=="__main__":
    # PlotterPCA().plot_energy()
    PlotterKernelPCA().plot_rbf()
    # PlotterPCA().plot_components()
    # PlotterFisher().plot_components()
    # PlotterKnn().plot_kNN()
    # PlotterLgbphs().plot_orientations()
