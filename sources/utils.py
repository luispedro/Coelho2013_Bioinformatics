from jug import TaskGenerator

def maymkdir(dirname):
    from os import mkdir
    try:
        mkdir(dirname)
    except OSError:
        pass

@TaskGenerator
def save_cmatrix(outputname, cmatrix):
    import pyslic
    cmatrix,names = cmatrix
    acc = lambda cmat: cmat.trace()/float(cmat.sum())
    maymkdir('results')
    with open('results/' + outputname, 'w') as output:
        print >>output, pyslic.utils.format_confusion_matrix(cmatrix, names)
        print >>output
        print >>output, "Overall accuracy: %s" % (acc(cmatrix))


def load_rt(base):
    import pyslic
    from os import path, listdir
    j = path.join
    images = []
    for label in listdir(base):
        dir = j(base,label)
        if not path.isdir(dir):
            continue
        ni = 0
        for f in listdir(dir):
            if 'protein' in f:
                ci,ii,_ = f.split('-')
                f = j(dir,f)
                img = pyslic.Image(protein=f, dna=f.replace('protein','dna'))
                img.label = label
                images.append(img)
                img.id = (label, (int(ci),int(ii)))
                img.origin = label+'/'+ci
                ni += 1
    return images

def _hela_load_function(path):
    from imread import imread
    import numpy as np
    im = imread(path)
    if im.dtype == np.bool_:
        return im.astype(np.uint8)
    return im

def load_directory(base):
    if 'SubCellLoc' in base:
        from brisbane import load_brisbane
        return load_brisbane(base)
    if 'hela' in base:
        import pyslic
        images = pyslic.image.io.dirtransversal.dirtransversal(base)
        for im in images:
            im.load_function = _hela_load_function
        return images
    if 'rt-' in base or 'labeled-' in base:
        return load_rt(base)
    stopfiles = set(['.DS_Store'])
    import pyslic
    from os import listdir, path
    images = []
    for label in listdir(base):
        if not path.isdir(base + '/' + label):
            continue
        files = listdir(base + '/' + label)
        files = [f for f in files if f not in stopfiles]
        for i,fname in enumerate(sorted(files)):
            im = pyslic.Image(protein=path.join(base,label,fname))
            im.label = label
            im.id = i,label
            images.append(im)
    return images

@TaskGenerator
def print_stats(datasets):
    maymkdir('tables')
    output = file('tables/datasets-statistic.tex','w')
    for name,directory,has_dna,base,_ in datasets:
        images = load_directory('../data/'+directory)
        labels = [im.label for im in images]
        print >>output, "%s & %s & %s\\\\" % (name, len(images), len(set(labels)))
    output.close()

@TaskGenerator
def save_comparisons(compares):
    accuracies = []
    for name,baseline,local0,local1,combined in compares:
        n = baseline[0].sum()
        baseline = baseline[0].trace()
        alt0 = local0[0].trace()
        alt1 = (local1[0].trace() if local1 is not None else None)
        combined = combined[0].trace()
        accuracies.append((name, n, baseline, alt0, alt1, combined))
    return accuracies


@TaskGenerator
def format_results_table(outputname, compares):
    import numpy as np
    import milk.measures
    import milk
    with open(outputname, 'w') as output:
        for name,n,baseline,loc0,loc1,combined in compares:
            p = milk.measures.bayesian_significance(n,baseline, combined)
            if p > 0.01:
                p = '%.2f' % p
            elif p <= 0.0:
                p = '0.0'
            else:
                e = -np.floor(np.log10(p))
                p *= 10**e
                p = r'$%.1f \cdot 10^{-%d}$' % (p,int(e))

            best = max(baseline, loc0, (0 if loc1 is None else loc1), combined)
            def f(c):
                if c is None: return '-'
                r = 0.1 * round(1000.*c/n)
                if c == best:
                    return r'\textbf{%s}' % r
                return r
            baseline = f(baseline)
            surf = f(loc0)
            surf_ref = f(loc1)
            combined = f(combined)
            loc1 = (0 if loc1 is None else loc1)
            print >> output, r'%(name)s & %(baseline)s & %(surf)s & %(surf_ref)s & %(combined)s & %(p)s \\' % locals()

@TaskGenerator
def format_results_table_csv(outputname, compares):
    import numpy as np
    import milk.measures
    import milk
    with open(outputname, 'w') as output:
        for name,n,baseline,loc0,loc1,combined in compares:
            p = milk.measures.bayesian_significance(n,baseline, combined)
            best = max(baseline, loc0, (0 if loc1 is None else loc1), combined)
            def f(c):
                if c is None: return 'NaN'
                return c/float(n)
            baseline = f(baseline)
            surf = f(loc0)
            surf_ref = f(loc1)
            combined = f(combined)
            loc1 = (0 if loc1 is None else loc1)
            print >> output, r'%(name)s\t%(baseline)s\t%(surf)s\t%(surf_ref)s\t%(combined)s\t%(p)s' % locals()

@TaskGenerator
def format_rt_compares(outputname, compares):
    def f(r):
        cmat = r[0]
        return 0.1 * round( 1000. * cmat.trace() / cmat.sum() )
    with open(outputname, 'w') as output:
        for name, baseline, combined in compares:
            method = 'per protein'
            if 'no-origin' in name:
                name = name[:-len('-no-origin')]
                method = 'per image'
            baseline = f(baseline)
            combined = f(combined)

            print >> output, r'%s & %s & %s & %s \\' % (name, method, baseline, combined)

@TaskGenerator
def save_sizes(sizes):
    import pickle
    pickle.dump(sizes, open('results/dataset-sizes.pkl','w'))
