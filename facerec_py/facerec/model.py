from facerec_py.facerec.feature import AbstractFeature
from facerec_py.facerec.classifier import AbstractClassifier
from facerec_py.facerec.normalization import inverse_dissim


class PredictableModel(object):
    def __init__(self, feature, classifier):
        if not isinstance(feature, AbstractFeature):
            raise TypeError("feature must be of type AbstractFeature!")
        if not isinstance(classifier, AbstractClassifier):
            raise TypeError("classifier must be of type AbstractClassifier!")

        self.feature = feature
        self.classifier = classifier

    def compute(self, X, y):
        features = self.feature.compute(X, y)
        self.classifier.compute(features, y)

    def predict(self, X):
        q = self.feature.extract(X)
        return self.classifier.predict(q)

    def binary_predict(self, X, lbl, sim=inverse_dissim):
        q = self.feature.extract(X)
        return self.classifier.binary_predict(q, lbl, sim)


    def __repr__(self):
        feature_repr = repr(self.feature)
        classifier_repr = repr(self.classifier)
        return "PredictableModel (feature=%s, classifier=%s)" % (feature_repr, classifier_repr)
