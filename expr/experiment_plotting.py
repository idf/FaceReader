from experiment_setup import *

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
        for num_component in xrange(30, 50, 10):  # TODO
            models.append(PCA(num_component))

        self._plot(models)

    def plot_energy(self):
        models = []

        class PCA_energy(PCA):
            def short_name(self):
                return "PCA: %.6f"%self.energy_percentage()

        for num_component in xrange(20, 100, 5):  # TODO
            models.append(PCA_energy(num_component))

        self._plot(models)


class PlotterFisher(Plotter):
    def plot_components(self):
        models = []
        for num_components in xrange(1, 15):  # TODO
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
            cv = expr.experiment(pca, threshold_up=0, kNN_k=k, debug=False)
            xys.append((k, cv.validation_results[0].precision))

        plt.plot([elt[0] for elt in xys], [elt[1] for elt in xys])
        plt.show()


class PlotterLgbphs(Plotter):
    def plot_lbp_algorihtms(self):
        class LgbphsSub(LGBPHS2):
            def short_name(self):
                return "%s"%self.model.model2.lbp_operator.short_name()

        models = []
        for lbp in (OriginalLBP(), ExtendedLBP(radius=3), LPQ(radius=4)):
            models.append(LgbphsSub(lbp_operator=lbp))

        self._plot(models)

    def plot_scales(self, r=(1, 5, 9)):  # 1~5
        class LgbphsSub(LGBPHS2):
            def short_name(self):
                return "scale: %s" % self.model.model1.scale_cnt

        self._plot([LgbphsSub(n_scale=i, lbp_operator=OriginalLBP()) for i in r])

    def plot_orientations(self, r=(1, 3, 5, 8)):  # 1~8
        class LgbphsSub(LGBPHS2):
            def short_name(self):
                return "orient: %s" % self.model.model1.orient_cnt

        self._plot([LgbphsSub(n_orient=i, lbp_operator=OriginalLBP()) for i in r])

    def plot_histogram(self):
        pass



if __name__=="__main__":
    PlotterPCA().plot_components()
    # PlotterLgbphs().plot_orientations()