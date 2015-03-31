import copy
import collections

from facerec_py.facerec.feature import AbstractFeature
from facerec_py.facerec.classifier import AbstractClassifier


class AbstractPredictableModel(object):
    def compute(self, X, y):
        raise NotImplementedError("Every AbstractPredictableModel must implement the compute method.")

    def predict(self, X):
        raise NotImplementedError("Every AbstractPredictableModel must implement the predict method.")

    def __repr__(self):
        return "AbstractPredictableModel"


class PredictableModel(AbstractPredictableModel):
    def __init__(self, feature, classifier):
        super(PredictableModel, self).__init__()
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

    def __repr__(self):
        feature_repr = repr(self.feature)
        classifier_repr = repr(self.classifier)
        return "PredictableModel (feature=%s, classifier=%s)" % (feature_repr, classifier_repr)


class FeaturesEnsemblePredictableModel(AbstractPredictableModel):
    def __init__(self, features, classifier):
        super(FeaturesEnsemblePredictableModel, self).__init__()
        for feature in features:
            if not isinstance(feature, AbstractFeature):
                raise TypeError("feature must be of type AbstractFeature!")
        if not isinstance(classifier, AbstractClassifier):
            raise TypeError("classifier must be of type AbstractClassifier!")

        self.features = features
        self.classifiers = [copy.deepcopy(classifier) for _ in features]

    def compute(self, X, y):
        for i in xrange(len(self.features)):
            feats = self.features[i].compute(X, y)
            self.classifiers[i].compute(feats, y)

    def predict(self, X):
        qs = [feature.extract(X) for feature in self.features]
        ps = [self.classifiers[i].predict(qs[i]) for i in xrange(len(qs))]
        # majority voting
        dic = collections.defaultdict(int)
        for elt in ps:
            dic[elt[0]] += 1
        maxa, label = -1, -1
        for k, v in dic.items():
            if v>maxa:
                maxa = v
                label = k

        for elt in ps:
            if elt[0]==label:
                return elt

        return None

    def __repr__(self):
        feature_repr = repr(self.features)
        classifier_repr = repr(self.classifiers[0])
        return "PredictableModel (features=%s, classifier=%s)" % (feature_repr, classifier_repr)
