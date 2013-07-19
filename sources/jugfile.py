from jug import Task, CachedFunction, barrier, value
from jug.utils import identity
from computefeatures import computeallfeatures, computebasefeatures, features1centroids, project, save_profile, aic
from utils import load_directory, save_cmatrix, print_stats, save_comparisons, format_results_table, format_rt_compares, save_sizes, format_results_table_csv
import milk.ext.jugparallel
from learner import precluster_learner_plus_features
import pyslic
from copy import copy

datasets = [
    ('HeLa2D', 'hela', True, 'slf7dna', False),
    ('RT-widefield-no-origin', 'labeled-widefield', True, 'field-dna+', False),
    ('RT-widefield', 'labeled-widefield', True, 'field-dna+', True),
    ('RT-confocal-no-origin', 'labeled-confocal', True, 'field-dna+', False),
    ('RT-confocal', 'labeled-confocal', True, 'field-dna+', True),
    ('LOCATE-transfected', 'SubCellLoc/Transfected', True, 'field-dna+', False),
    ('LOCATE-endogenous', 'SubCellLoc/Endogenous', True, 'field-dna+', False),
    ('binucleate', 'binucleate', False, 'field+', False),
    ('CHO', 'cho', False, 'field+', False),
    ('Terminal Bulb', 'terminalbulb', False, 'field+', False),
    ('RNAi', 'rnai', False, 'field+', False),
    ]

print_stats(datasets)

run_twice = ['RT-widefield']

compares = []
rt_compares = []
sizes = {}
for name,directory,has_dna,base,use_origins in datasets:
    images = CachedFunction(load_directory,'../data/'+directory)
    sizes[name] = len(images)
    learner = precluster_learner_plus_features(kfrac=4)
    origins = None
    if use_origins:
        origins = [im.origin for im in images]
    labels = [im.label for im in images]
    surfs = ['surf']
    if has_dna:
        surfs.append('surf-ref')
    four = {}
    labels = identity(labels)
    for s in surfs:
        for use_base in [None, base]:
            features = computeallfeatures(images, s, use_base)
            features = identity(features)
            cmatrix = milk.ext.jugparallel.nfoldcrossvalidation(features, labels, origins=origins, learner=copy(learner))
            four[s,use_base is not None] = cmatrix
            save_cmatrix('%s-%s-%s.txt' % (name,s,use_base), cmatrix)

            if s == surfs[-1] and ('no-origins' not in name):
                cmats = []
                for k in xrange(32,385):
                    n = 1 + (name in run_twice)
                    for ri in xrange(n):
                        centroids = features1centroids(features, k, ri)
                        nfeatures = project(features, centroids)
                        cmatrix = milk.ext.jugparallel.nfoldcrossvalidation(nfeatures, labels, origins=origins)
                        cmats.append((k,ri,cmatrix[0], aic(features,centroids)))
                save_profile('%s-%s-%s' % (name,s,use_base), cmats)

    base_features = computebasefeatures(images, base)
    baseline = milk.ext.jugparallel.nfoldcrossvalidation(base_features, labels, origins=origins)

    save_cmatrix('%s-base.txt' % name, baseline)

    concatenated = four[surfs[-1], True]
    if 'no-origins' not in name:
        compares.append(
            (name, baseline, four['surf', False], four.get(('surf-ref', False)), concatenated)
            )
    if name.startswith('RT-'):
        rt_compares.append(
            (name, baseline, concatenated)
        )


compares = save_comparisons(compares)
format_results_table('../paper/tables/compare-all.tex', compares)
format_results_table_csv('../paper/tables/compare-all.csv', compares)
format_rt_compares('../paper/tables/compare-rt.tex', rt_compares)
save_sizes(sizes)


