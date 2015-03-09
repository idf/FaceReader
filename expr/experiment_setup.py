import sys

from facerec_py.facerec.feature import *
from facerec_py.facerec.distance import EuclideanDistance
from facerec_py.facerec.classifier import NearestNeighbor
from facerec_py.facerec.model import PredictableModel
from facerec_py.facerec.validation import KFoldCrossValidation
from facerec_py.facerec.visual import subplot
from facerec_py.facerec.util import minmax_normalize
from expr.read_dataset import read_images
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as pyplot
import re
from expr.feature import *
from util.commons_util.logger_utils.logger_factory import LoggerFactory

__author__ = 'Danyang'


class Experiment(object):
    def __init__(self):
        self.logger = LoggerFactory().getConsoleLogger("facerec")

    def plot_fisher(self, X, model):
        E = []
        for i in xrange(min(model.feature.eigenvectors.shape[1], 16)):
            e = model.feature.eigenvectors[:, i].reshape(X[0].shape)
            E.append(minmax_normalize(e, 0, 255, dtype=np.uint8))
        # Plot them and store the plot to "python_fisherfaces_fisherfaces.pdf"
        subplot(title="Fisherfaces", images=E, rows=4, cols=4, sptitle="Fisherface", colormap=cm.jet,
                filename="fisherfaces.png")
        # Close current figure
        pyplot.close()


    def experiment(self, feature=Fisherfaces(), plot=None):
        """
         Define the Fisherfaces as Feature Extraction method:
        :param feature:
        :return:
        """
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
        [X, y] = read_images(sys.argv[1])
        # Define a 1-NN classifier with Euclidean Distance:
        classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=1)
        # Define the model as the combination
        model = PredictableModel(feature=feature, classifier=classifier)
        # Compute the Fisherfaces on the given data (in X) and labels (in y):
        model.compute(X, y)
        # Then turn the first (at most) 16 eigenvectors into grayscale
        # images (note: eigenvectors are stored by column!)
        if plot:
            plot(X, model)
        # Perform a 10-fold cross validation
        cv = KFoldCrossValidation(model, k=10, threshold_up=0)
        cv.validate(X, y)

        # And print the result:
        print cv
        return cv

    def plot_roc(self, cv):
        # Convert to string
        temp_result = repr(cv)
        # Extract FPR
        raw_fpr = re.findall(r'FPR=\d+\.\d+%', temp_result)
        string_fpr = repr(raw_fpr)
        temp_fpr = re.findall(r'\d+\.\d+', string_fpr)
        fpr = map(float, temp_fpr)
        myDividend = 100.00
        true_fpr = [x / myDividend for x in fpr]
        # Extract TPR
        raw_tpr = re.findall(r'TPR=\d+\.\d+%', temp_result)
        string_tpr = repr(raw_tpr)
        temp_tpr = re.findall(r'\d+\.\d+', string_tpr)
        tpr = map(float, temp_tpr)
        true_tpr = [x / myDividend for x in tpr]
        print true_fpr
        print true_tpr
        # Plot ROC
        pyplot.figure(2)
        # pyplot.axis([-0.01, 1.0, 0.95, 1.01])
        # ax = pyplot.gca()
        # ax.set_autoscale_on(False)
        pyplot.plot(true_fpr, true_tpr, 'g')
        pyplot.show()

if __name__ == "__main__":
    expr = Experiment()
    expr.experiment(Fisherfaces(14), expr.plot_fisher)
    # expr.experiment(SpatialHistogram(), None)
    # expr.experiment(PCA(50), None)
    # expr.experiment(GaborFilter(), None)
    # expr.experiment(GaborFilterFisher(), None)
    # expr.experiment(LGBPHS(), None)
