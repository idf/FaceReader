import sys
from expr.weighted_hs import WeightedLGBPHS

from facerec_py.facerec.distance import *
from facerec_py.facerec.classifier import NearestNeighbor
from facerec_py.facerec.model import PredictableModel, FeaturesEnsemblePredictableModel
from facerec_py.facerec.validation import KFoldCrossValidation, shuffle
from facerec_py.facerec.visual import subplot
from facerec_py.facerec.util import minmax_normalize
from expr.read_dataset import read_images
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from expr.feature import *
from util.commons_util.logger_utils.logger_factory import LoggerFactory
from scipy.interpolate import spline
import numpy as np

__author__ = 'Danyang'


class Drawer(object):
    def __init__(self, smooth=False):
        plt.figure("ROC")
        plt.axis([0, 0.5, 0.5, 1.001])
        # ax = pyplot.gca()
        # ax.set_autoscale_on(False)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        # colors: http://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
        plt.rc('axes', color_cycle=['r', 'g', 'b', 'c', 'm', 'y', 'k',
                                    'darkgreen', 'chocolate', 'darksalmon', 'darkseagreen', 'yellowgreen'])
        self.is_smooth = smooth
        self._rocs = []

    def show(self):
        plt.legend(handles=self._rocs)
        plt.show()

    def plot_roc(self, cv):
        """
        :type cv: KFoldCrossValidation
        :param cv:
        :return:
        """
        # Extract FPR
        FPRs = [r.FPR for r in cv.validation_results]
        TPRs = [r.TPR for r in cv.validation_results]

        # add (0, 0), and (1, 1)
        FPRs.append(0.0)
        TPRs.append(0.0)
        FPRs.append(1.0)
        TPRs.append(1.0)

        if self.is_smooth:
            FPRs, TPRs = self.smooth(FPRs, TPRs)

        # Plot ROC
        roc, = plt.plot(FPRs, TPRs, label=cv.model.feature.short_name())
        self._rocs.append(roc)

    def smooth(self, x, y):
        x = np.array(x)
        y = np.array(y)
        x, idx = np.unique(x, return_index=True)  # otherwise singular matrix
        y = y[idx]

        x_sm = np.linspace(x.min(), x.max(), 60)  # evenly spaced numbers over a specified interval.
        y_sm = spline(x, y, x_sm)
        return x_sm, y_sm


class Experiment(object):
    def __init__(self, smooth=False, froze_shuffle=False):
        self.logger = LoggerFactory().getConsoleLogger("facerec")
        self._drawer = Drawer(smooth)
        self.X, self.y = shuffle(*self.read())  # shuffle once
        self.froze_shuffle = froze_shuffle  # whether to froze the subsequent shuffling in validation

    def read(self):
        # This is where we write the images, if an output_dir is given
        # in command line:
        out_dir = None
        # You'll need at least a path to your image data, please see
        # the tutorial coming with this source code on how to prepare
        # your image data:
        if len(sys.argv) < 2:
            print "USAGE: experiment_setup.py </path/to/images>"
            sys.exit()
        # Now read in the image data. This must be a valid path!
        X, y = read_images(sys.argv[1])

        X = np.asarray(X)
        y = np.asarray(y)
        return X, y

    def plot_fisher_original(self, X, model):
        E = []
        for i in xrange(min(model.feature.eigenvectors.shape[1], 16)):
            e = model.feature.eigenvectors[:, i].reshape(X[0].shape)
            E.append(minmax_normalize(e, 0, 255, dtype=np.uint8))
        # Plot them and store the plot to "python_fisherfaces_fisherfaces.png"
        subplot(title="Fisherfaces", images=E, rows=4, cols=4, sptitle="Fisherface", colormap=cm.jet,
                filename="fisherfaces.png")
        # Close current figure
        plt.close()

    def plot_fisher(self, X, model, r=3, c=5):
        """
        draw fisher face components
        color map: http://matplotlib.org/examples/color/colormaps_reference.html
        :param X: images
        :param model: fisher face model
        :param r: number of rows
        :param c: number of cols
        :return:
        """
        E = []
        for i in xrange(min(model.feature.eigenvectors.shape[1], r*c)):
            e = model.feature.eigenvectors[:, i].reshape(X[0].shape)
            E.append(minmax_normalize(e, 0, 255, dtype=np.uint8))

        # Plot them and store the plot to "python_fisherfaces_fisherfaces.png"
        subplot(title="Fisherface Components", images=E, rows=r, cols=c, sptitle="fisherface", colormap=cm.rainbow,
                filename="fisherfaces.png")
        plt.close()


    def experiment(self, feature=Fisherfaces(), plot=None, dist_metric=EuclideanDistance(), threshold_up=0, kNN_k=1, number_folds=None, debug=True):
        """
        Define the Fisherfaces as Feature Extraction method

        :param feature: feature extraction
        :param plot: function to plot
        :param dist_metric: distance metric
        :param threshold_up: threshold for ROC
        :param kNN_k: k for kNN classifier
        :param debug: if true, display the images of wrongly classified face
        :return:
        """
        # Define a 1-NN classifier with Euclidean Distance:
        classifier = NearestNeighbor(dist_metric=dist_metric, k=kNN_k)
        # Define the model as the combination
        model = self._get_model(feature, classifier)
        # Compute the Fisherfaces on the given data (in X) and labels (in y):
        model.compute(self.X, self.y)
        # Then turn the first (at most) 16 eigenvectors into grayscale
        # images (note: eigenvectors are stored by column!)
        if plot:
            plot(self.X, model)
        # Perform a k-fold cross validation
        # Perform a k-fold cross validation
        if number_folds is None:
            number_folds = len(np.unique(self.y))
            if number_folds>15: number_folds = 10


        cv = KFoldCrossValidation(model, k=number_folds, threshold_up=threshold_up, froze_shuffle=self.froze_shuffle, debug=debug)
        # cv = LeaveOneOutCrossValidation(model)
        cv.validate(self.X, self.y)

        # And print the result:
        print cv
        if debug:
            self.logger.info("Cross validation completed; press any key on any image to continue")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return cv

    def _get_model(self, feature, classifier):
        return PredictableModel(feature=feature, classifier=classifier)

    def show_plot(self):
        """
        Plot the graph at the end
        :return:
        """
        self._drawer.show()

    def plot_roc(self, cv):
        """
        Plot a individual result
        :param cv:
        :return:
        """
        self._drawer.plot_roc(cv)


class FeaturesEnsembleExperiment(Experiment):
    def _get_model(self, features, classifier):
        return FeaturesEnsemblePredictableModel(features, classifier)


def draw_roc(expr):
    """
    set threshold_up=1
    :param expr:
    :return:
    """
    cv = expr.experiment(Fisherfaces(14), threshold_up=1)
    expr.plot_roc(cv)
    cv = expr.experiment(PCA(50), threshold_up=1)
    expr.plot_roc(cv)
    cv = expr.experiment(SpatialHistogram(), dist_metric=HistogramIntersection(), threshold_up=1)
    expr.plot_roc(cv)

    expr.show_plot()


def ensemble_lbp_fisher():
    # features = [Fisherfaces(i) for i in xrange(14, 19)]
    features = [LbpFisher(ExtendedLBP(i)) for i in (3, 6, 10, 11, 14, 15, 19)]
    expr = FeaturesEnsembleExperiment()
    expr.experiment(features, debug=False)


if __name__ == "__main__":
    expr = Experiment(froze_shuffle=True)
    # draw_roc(expr)
    # expr.experiment(SpatialHistogram(), dist_metric=HistogramIntersection())
    # expr.experiment(LGBPHS2(), dist_metric=HistogramIntersection())
    # expr.experiment(PCA(50), plot=expr.plot_fisher, debug=False)
    # expr.experiment(Fisherfaces(15), plot=expr.plot_fisher, debug=False)
    # expr.experiment(Identity(), debug=False)
    # expr.experiment(LbpFisher(), debug=False)
    # expr.experiment(LbpFisher(), debug=False)
    # ensemble_lbp_fisher()
    # expr.experiment(WeightedLGBPHS(), debug=False)