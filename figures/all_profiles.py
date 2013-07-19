from output import setup_default, output
from matplotlib import pyplot as plt
import numpy as np
import pickle
from sys import argv

COLOUR = (len(argv) > 1) and argv[1] == '--color'

datasets = [
    ('HeLa2D', 'hela', True, 'slf7dna', False),
    ('RT-widefield', 'rt-widefield', True, 'field-dna+', True),
    ('RT-confocal', 'rt-confocal', True, 'field-dna+', True),
    ('LOCATE-transfected', 'SubCellLoc/Transfected', True, 'field-dna+', False),
    ('LOCATE-endogenous', 'SubCellLoc/Endogenous', True, 'field-dna+', False),
    ('binucleate', 'binucleate', False, 'field+', False),
    ('CHO', 'cho', False, 'field+', False),
    ('Terminal Bulb', 'terminalbulb', False, 'field+', False),
    ('RNAi', 'rnai', False, 'field+', False),
    ]

sizes = pickle.load(open('../sources/results/dataset-sizes.pkl'))
for name,directory,has_dna,base,use_origins in datasets:
    n = sizes[name]
    s = 'surf-ref' if has_dna else 'surf'
    for use_base in [None,base]:
        datafile = '../sources/results/%s-%s-%s-profile.npy' % (name, s, use_base)
        data = np.load(datafile)
        setup_default(.54)

        k = (.9*n)//4
        if COLOUR:
            plt.plot(data.T[0], 100*data.T[2], 'ro')
            plt.plot([k,k+0.01], [90*data.T[2].min(), 110*data.T[2].max()], 'g-')
            cs = 'color'
        else:
            plt.plot(data.T[0], 100*data.T[2], 'o', mfc='#cccccc', mew=2)
            plt.plot([k,k+0.01], [90*data.T[2].min(), 110*data.T[2].max()], 'k-')
            cs = 'gs'
        plt.ylim([90*data.T[2].min(), min(100, 110*data.T[2].max())])
        #plt.xlabel('Nr. clusters (k)')
        #plt.ylabel('Accuracy (%)')
        output('profile-' + str(use_base) + '-' + name.replace(' ', '_') + '-' + cs)
