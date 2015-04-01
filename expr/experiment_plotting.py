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
                # energy = np.count_nonzero(self._eigenvalues)
                percentage = (np.sum(self._eigenvalues))/self._total_energy
                return "PCA: %.6f"%percentage

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
        for k in xrange(1, 41):
            cv = expr.experiment(pca, threshold_up=0, kNN_k=k, debug = False)
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



if __name__=="__main__":
    # PlotterPCA().plot_energy()
    PlotterPCA().plot_components()
    # PlotterFisher().plot_components()
    # PlotterKnn().plot_kNN()
    # PlotterLgbphs().plot_orientations()
