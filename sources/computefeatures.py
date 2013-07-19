from jug import TaskGenerator, Task
from jug.compound import CompoundTaskGenerator
import pyslic
import numpy as np
import milk


@CompoundTaskGenerator
def computeallfeatures(images, ref_variant, base):
    local = computebasefeatures(images, [ref_variant])
    if base is None:
        base = [[] for _ in images]
    else:
        base = computebasefeatures(images, base)
    return zip(local, base)


def computebasefeatures(images, base):
    return [Task(pyslic.computefeatures, im, base, region=1) for im in images]


@TaskGenerator
def features1centroids(features, k, ri):
    def _subsample(array, frac, R):
        n = len(array)
        if n == 0:
            return array
        np.random.seed(R)
        return array[np.random.random(n) <= frac]
    from milk.unsupervised.kmeans import kmeans, assign_centroids
    surfs = [_subsample(s, 1./16., i) for i,(s,_) in enumerate(features)]
    csurfs = np.concatenate(surfs)
    _,centroids = kmeans(csurfs, k, R=(1024*k+ri))
    return centroids

@TaskGenerator
def project(features, centroids):
    from milk.unsupervised.kmeans import assign_centroids
    return np.array([
        np.concatenate([
            assign_centroids(s, centroids, histogram=True, normalise=True),
            other
            ])
        for s,other in features])

@TaskGenerator
def save_profile(name, profiles):
    def acc(cmat):
        return cmat.trace()/float(cmat.sum())
    results = [(k,ri,acc(cmat),aic) for k,ri,cmat,aic in profiles]
    results = np.array(results)
    np.save('results/%s-profile.npy' % name, results)

@TaskGenerator
def aic(features, centroids):
    from milk.unsupervised.gaussianmixture import AIC
    from milk.unsupervised.kmeans import assign_centroids
    assignments = []
    feats = []
    for fs,_ in features:
        assignments.extend(assign_centroids(fs, centroids))
        feats.append(fs)
    return AIC(np.concatenate(feats), assignments, centroids)
