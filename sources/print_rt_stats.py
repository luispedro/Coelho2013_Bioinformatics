from utils import load_rt
from collections import defaultdict

abbr = {
	'unlabeled': 'UL',
    'nuclear': 'N',
    'nucleoli' : 'NO',
    'mitochondria': 'M',
    'Golgi': 'G',
    'ER' : 'ER',
    'cytoplasmic': 'Cyto',
    'membrane': 'PM',
    'lysosome': 'Lyso',
    'cytoskeleton': 'Cytosk'
}

order = [
        'UL',
        'NO',
        'N',
        'M',
        'G',
        'Cyto',
        'PM',
        'Lyso',
        'Cytosk',
        'ER',
        ]
rev_abbr = dict((v,k) for k,v in abbr.items())

def print_for(base, do_head=True, header=None):
    images = load_rt(base)

    labels = set(im.label for im in images)
    nr_proteins = defaultdict(set)
    nr_images = defaultdict(int)
    for im in images:
        label,(ci,_) = im.id
        nr_proteins[label].add(ci)
        nr_images[label] += 1
    nr_proteins = dict((ell,len(cs)) for ell,cs in nr_proteins.iteritems())
    nr_images = dict(nr_images)

    if do_head:
        print "              &",
        for label in order:
            print "%7s &" % label,
        print r'\\'
        print r'\midrule'
    if header:
        print header

    print 'Nr.\ proteins   &',
    for label in order:
        label = rev_abbr[label]
        print "%7s &" % nr_proteins[label],
        
    print r'\\'

    print 'Nr.\ images   &',
    for label in order:
        label = rev_abbr[label]
        print "%7s &" % nr_images[label],
        
    print r'\\'

header=r'\multicolumn{8}{l}{Widefield}\\'
print_for('../data/rt-widefield/', True, header)
print r'\midrule'
print r'\multicolumn{8}{l}{Confocal}\\'
print_for('../data/rt-confocal/', False)

