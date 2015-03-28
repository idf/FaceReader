from experiment_setup import *

__author__ = 'Danyang'


def plot(models):
    expr = Experiment()
    for model in models:
        cv = expr.experiment(model, threshold_up=1)
        expr.plot_roc(cv)
    expr.show_plot()


def plot_PCA():
    models = []
    for num_component in xrange(30, 80, 10):  # TODO
        models.append(PCA(num_component))
    plot(models)


def plot_PCA_eneger():
    models = []

    class PCA_energy(PCA):
        def short_name(self):
            return "PCA: %.6f"%self.energy_percentage()

    for num_component in xrange(20, 100, 5):  # TODO
        models.append(PCA_energy(num_component))
    plot(models)


def plot_Fisher():
    models = []
    for num_components in xrange(1, 15):  # TODO
        models.append(Fisherfaces(num_components))
    plot(models)


def plot_kNN():
    """
    plot the graph of varying k of kNN
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


if __name__=="__main__":
    # plot_PCA()
    plot_kNN()