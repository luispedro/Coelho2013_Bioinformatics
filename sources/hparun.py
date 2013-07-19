from collections import defaultdict
import milk
from hpa import load, surfref, surf, field_features
from jug import Task, bvalue
from jug.utils import identity
from jug import CachedFunction, TaskGenerator
from jug.compound import CompoundTaskGenerator
from jug.mapreduce import currymap
from jug import mapreduce, iteratetask
from milk.supervised.multi_label import one_by_one
from learner import precluster_learner_plus_features
import milk.ext.jugparallel
from milk.measures.nfoldcrossvalidation import foldgenerator
from copy import deepcopy
import numpy as np
from itertools import chain

concatenate = TaskGenerator(np.concatenate)
hstack = TaskGenerator(np.hstack)
vstack = TaskGenerator(np.vstack)
to_array = TaskGenerator(np.array)

@TaskGenerator
def sample_features(f, s, n):
    import numpy as np
    import random
    random.seed(s)
    f = random.sample(f, n)
    return np.array(f)


@TaskGenerator
def all_labels(images):
    alllabels = set()
    for im in images:
        label = im.label[0]
        alllabels.update(label.split(','))
    return list(sorted(alllabels))

@CompoundTaskGenerator
def all_field_features(iimages):
    images = bvalue(iimages)
    return to_array(mapreduce.map(field_features, images, map_step=16))

@TaskGenerator
def label_origins(images):
    allgenes = list(set(im.gene[0] for im in images))
    allgenes.sort()
    allgenes = {g:i for i,g in enumerate(allgenes)}
    labels = []
    origins = []
    for im in images:
        labels.append(im.label)
        origins.append(allgenes[im.gene[0]])
    return np.array(labels), np.array(origins)

@TaskGenerator
def add_base_features(features, base):
    if base is None:
        return np.asarray([(f,[]) for f in features], dtype=object)
    raise NotImplementedError

@TaskGenerator
def train_test(features, labels, train, test, learner, ret_cmat=False):
    from milk.measures import confusion_matrix
    learner = deepcopy(learner)
    features = np.array(features)
    labels = np.array(labels, dtype=object)
    model = learner.train(features[train], labels[train])
    predicted = model.apply_many(features[test])
    if ret_cmat:
        return confusion_matrix(predicted, labels[test])
    return predicted, labels[test]

@TaskGenerator
def as_cmatrix(accs):
    import numpy as np
    from itertools import chain
    predicted,real = accs
    universe = set(chain(*chain(predicted, real)))
    cmat = np.zeros((2,2), int)
    for ps,rs in zip(predicted,real):
        for ell in universe:
            y = int(ell in ps)
            x = int(ell in rs)
            cmat[y,x] += 1
    return cmat


@TaskGenerator
def full_cmat(accs):
    from milk.measures import confusion_matrix
    predicted,real = accs
    return confusion_matrix(predicted, real)

@TaskGenerator
def summarize_acc(accs):
    cmat = sum(accs)
    return cmat.astype(float).trace()/cmat.sum()


@TaskGenerator
def single_label(images):
    e2labels = {}

    blacklist = []
    ims = []
    for im in images:
        if len(im.label) == 1 and im.gene[0] is not None:
            im = deepcopy(im)
            im.label = im.label[0][4:]
            if im.gene[0] in e2labels and e2labels[im.gene[0]] != im.label:
                blacklist.append(im.gene)
            else:
                e2labels[im.gene[0]] = im.label
                ims.append(im)
    blacklist = set(blacklist)
    return [im for im in ims if im.gene[0] not in blacklist]


@TaskGenerator
def shuffle_images(images):
    import milk.utils
    R = milk.utils.get_pyrandom(11)
    images = images[:]
    R.shuffle(images)
    return images

kmeans = TaskGenerator(milk.kmeans)
@TaskGenerator
def project_centroids(s, centroids):
    from milk.unsupervised.kmeans import assign_centroids
    return assign_centroids(s, centroids, histogram=True, normalise=True)

@TaskGenerator
def sample(array, frac, R):
    n = len(array)
    if n == 0:
        return array
    np.random.seed(R)
    return array[np.random.random(n) <= frac]


@TaskGenerator
def select_sample(sampled, train):
    return np.concatenate([s for s,t in zip(sampled,train) if t])

@TaskGenerator
def select_features(sampled, train):
    return np.array([s for s,t in zip(sampled,train) if t])

learner = milk.supervised.defaultlearner.defaultlearner()
plearner = precluster_learner_plus_features(kfrac=4)

iimages = Task(load)
iimages = single_label(iimages)

accs = defaultdict(list)
accs2 = defaultdict(list)

@TaskGenerator
def zip_features(fs0, fs1):
    if fs1 is None:
        return np.array([(f,[]) for f in fs0], dtype=object)
    return np.array([(f0,f1) for f0,f1 in zip(fs0,fs1)], dtype=object)

@TaskGenerator
def k_from_train(train):
    return sum(train)//4

@TaskGenerator
def project_all_centroids(surfs, centroids):
    print 'in function'
    return np.array([project_centroids.f(s, centroids) for s in surfs])

@CompoundTaskGenerator
def project_block_centroids(surfs, centroids):
    from jug.mapreduce import _break_up as breakup
    blocks = []
    for ss in breakup(surfs, 256):
        blocks.append(project_all_centroids(ss, centroids))
    return vstack(blocks)

@CompoundTaskGenerator
def cluster_project(concatenated, surfs, k, ke):
    centroids = kmeans(concatenated, k=ke, R=((k if k is not None else -1)*7+12324))
    centroids = centroids[1]
    return project_block_centroids(surfs, centroids)


@TaskGenerator
def just_2_breakup(origins):
    groups = origins % 2
    seen = defaultdict(int)
    oindex = []
    for orig in origins:
        index = seen[orig]
        oindex.append(index)
        seen[orig] += 1
    oindex = np.array(oindex)
    return np.array([
            groups == 0,
            oindex ==0])

@TaskGenerator
def two_fold(features, labels, groups):
    from milk.measures import confusion_matrix
    features = np.asanyarray(features)
    labels = np.asanyarray(labels)

    learner = milk.defaultlearner()
    final = 0
    for f in [False, True]:
        selected = (groups == f)
        model = learner.train(features[selected], labels[selected])
        predicted = model.apply_many(features[~selected])
        cmat = confusion_matrix(predicted, labels[~selected])
        final = cmat + final
    return final

@CompoundTaskGenerator
def two_fold_ref(base_features, surfs, labels, groups):
    sampled = identity(currymap(sample, [(s,1./16,(3+i)) for i,s in enumerate(surfs)], map_step=32))
    groups = bvalue(groups)

    cmats = []
    for f in [False, True]:
        selected = (groups == f)
        concatenated = select_sample(sampled, selected)
        k = selected.sum()//4
        ke = k
        features = cluster_project(concatenated, surfs, k=k, ke=ke)

        features = hstack([features,base_features])
        cmats.append(
            train_test(features, labels, selected, ~selected, learner, True)
        )
    return summarize_acc(cmats)


images = bvalue(iimages)
lo = label_origins(iimages)
labels,origins = bvalue(lo)
splits = just_2_breakup(lo[1])
base_features = all_field_features(iimages)
bo = two_fold(base_features, labels, splits[0])
bno = two_fold(base_features, labels, splits[1])


all_surf_refs = mapreduce.map(surfref, images, map_step=256)
all_surfs = mapreduce.map(surf, images, map_step=256)
so = two_fold_ref(base_features, all_surf_refs, labels, splits[0])
sno = two_fold_ref(base_features, all_surf_refs, labels, splits[1])


use_origins=True
labels,origins = bvalue(label_origins(iimages))
base_features = all_field_features(iimages)
base_result = milk.ext.jugparallel.nfoldcrossvalidation(base_features, labels, origins=(origins if use_origins else None), learner=deepcopy(learner))

images = bvalue(iimages)
nr_images = len(images)
for name,surfs in [
            ('surf-ref', mapreduce.map(surfref, images, map_step=256)),
            ('surf', mapreduce.map(surf, images, map_step=256))]:
    sampled = identity(currymap(sample, [(s,1./16,(3+i)) for i,s in enumerate(surfs)], map_step=32))

    ks = [None]
    if name == 'surf-ref':
        ks = list(chain(ks, xrange(32,513)))

    for train,test in foldgenerator(labels, origins=origins, nfolds=10):
        train = identity(train.copy())
        test = identity(test.copy())

        concatenated = select_sample(sampled, train)
        for k in ks:
            ke = (k if k is not None else k_from_train(train))
            features = cluster_project(concatenated, surfs, k=k, ke=ke)
            accs[name,k].append(
                train_test(features, labels, train, test, learner, True)
            )
            features = hstack([features, base_features])
            accs2[name,k].append(
                train_test(features, labels, train, test, learner, True)
            )
accs_acc = dict([(k,summarize_acc(v))
                    for k,v in accs.items()])
accs2_acc = dict([(k,summarize_acc(v))
                    for k,v in accs2.items()])

@TaskGenerator
def line_results(baseline, surf, surf_ref, surf_ref_added):
    import milk.measures
    n = len(images)
    baseline = baseline[0].trace()
    surf = int(n * surf)
    surf_ref = int(n * surf_ref)
    combined = int(n * surf_ref_added)
    p = milk.measures.bayesian_significance(n,baseline, combined)
    if p > 0.01:
        p = '%.2f' % p
    elif p <= 0.0:
        p = '0.0'
    else:
        e = -np.floor(np.log10(p))
        p *= 10**e
        p = r'$%.1f \cdot 10^{-%d}$' % (p,int(e))

    def f(c):
        if c is None: return '-'
        r = 0.1 * round(1000.*c/n)
        if c == best:
            return r'\textbf{%s}' % r
        return r
    best = max(baseline,surf,surf_ref,combined)
    with open('../paper/tables/result-hpa.tex','w') as output:
        print >> output, r'%(name)s & %(baseline)s & %(surf)s & %(surf_ref)s & %(combined)s & %(p)s \\' % {
                    'name' : 'HPA',
                    'baseline': f(baseline),
                    'surf' : f(surf),
                    'surf_ref': f(surf_ref),
                    'combined' : f(combined),
                    'p': p,
        }


@TaskGenerator
def check_significance(oname, per_image, per_protein):
    import milk.measures
    n = len(images)
    p = milk.measures.bayesian_significance(n, int(n*per_protein), int(n*per_image))
    if p > 0.01:
        p = '%.2f' % p
    elif p <= 0.0:
        p = '0.0'
    else:
        e = -np.floor(np.log10(p))
        p *= 10**e
        p = r'%.1f 10^{-%d}' % (p,int(e))
    with open(oname,'w') as output:
        print >>output, per_protein, per_image, p

@TaskGenerator
def print_statistics(labels):
    n = len(labels)
    nl = len(set(labels))
    with open('../paper/tables/hpa-statistics.tex','w') as output:
        print >>output, 'HPA & %s & %s & \citealp{Barbe2008} \\\\' % (n,nl)

@TaskGenerator
def acc_of(cmat):
    return cmat.astype(float).trace()/cmat.sum()

@TaskGenerator
def save_profile(oname, images, profile):
    import pickle
    profile = np.array(profile)
    n = len(images)
    pickle.dump((n,profile), open(oname, 'w'))

@TaskGenerator
def print_two_table(bo, bno, so, sno):
    with open('../paper/tables/hpa-per-image.tex','w') as output:
        output.write('''\
HPA & per image  & {bno:.3} & {sno:.3} \\\\
HPA & per protein  & {bo:.3} & {so:.3} \\\\
'''.format(bo=(100*bo),bno=(100*bno),so=(100*so),sno=(100*sno)))

bo = acc_of(bo)
bno = acc_of(bno)
line_results(base_result, accs_acc['surf',None], accs_acc['surf-ref',None], accs2_acc['surf-ref',None])
check_significance('results/significance-base_2.txt', bo, bno)
check_significance('results/significance-surf-ref_2.txt', so, sno)
print_two_table(bo, bno, so, sno)
print_statistics(labels)


profile = [(k,accs2_acc['surf-ref',k]) for k in xrange(32,513)]
save_profile('results/hpa-surf-ref-field-profile.pkl', iimages, profile)

profile = [(k,accs_acc['surf-ref',k]) for k in xrange(32,513)]
save_profile('results/hpa-surf-ref-None-profile.pkl', iimages, profile)

