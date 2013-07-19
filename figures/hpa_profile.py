from output import setup_default, output
from matplotlib import pyplot as plt
import pickle
import numpy as np

from sys import argv
COLOUR = (len(argv) > 1) and argv[1] == '--color'
if COLOUR:
    cs = '-color'
else:
    cs = '-gs'


for use_base in ['field', 'None']:
    n, data = pickle.load(open('../sources/results/hpa-surf-ref-{}-profile.pkl'.format(use_base)))

    setup_default(.54)
    k = (.9*n)//4

    if COLOUR:
        plt.plot(data.T[0], 100*data.T[1], 'ro')
        plt.plot([k,k+0.01], [90*data.T[1].min(), 110*data.T[1].max()], 'g-')
    else:
        plt.plot(data.T[0], 100*data.T[1], 'o', mfc='#cccccc', mew=2)
        plt.plot([k,k+0.01], [90*data.T[1].min(), 110*data.T[1].max()], 'k-')
    plt.ylim([90*data.T[1].min(), min(100, 110*data.T[1].max())])
    name = 'HPA'
    output('profile-' + use_base + '-' + name.replace(' ', '_')+cs)
