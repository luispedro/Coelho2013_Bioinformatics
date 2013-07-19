import numpy as np
from milk.supervised.base import base_adaptor, supervised_model
import milk
from computefeatures import project


class codebook_model(supervised_model):
    def __init__(self, centroids, base, normalise=True):
        self.centroids = centroids
        self.base = base
        self.normalise = normalise

    def apply(self, features):
        from milk.unsupervised.kmeans import assign_centroids
        f0,f1 = features
        features = assign_centroids(f0, self.centroids, histogram=True, normalise=self.normalise)
        if f1 is not None and len(f1):
            features = np.concatenate((features, f1))
        return self.base.apply(features)

class precluster_learner_plus_features(object):

    def __init__(self, k=None, kfrac=None):
        self.k = k
        self.kfrac = kfrac
        self.sample = 16

    def train(self, features, labels, **kwargs):
        from milk.supervised.gridsearch import gridminimise
        from milk.supervised import svm
        c_features = np.concatenate([f for f,_ in features if f.size])
        c_features = c_features[::self.sample]

        learner = milk.defaultlearner()
        k = (self.k if self.k is not None else len(features)//self.kfrac)
        _,codebook = milk.kmeans(c_features, k=k, R=123)
        features = project.f(features, codebook)
        model = learner.train(features, labels)
        return codebook_model(codebook, model)
