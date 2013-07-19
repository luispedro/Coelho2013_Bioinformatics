import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpltools import style
style.use('ggplot')
data = pd.read_table('compare-all.csv', sep='\t', index_col=0)
data = data.select(lambda s: 'no-origin' not in s)

delta = data.combined - data.baseline
delta *= 100
delta.sort()
plt.rcParams.update({'figure.figsize': [16,12]})

for i,f in enumerate(data.p[delta.index]):
    if f >= .05:
        val = 'n.s'
    elif f >= .001:
    	val = '*'
    elif f >= 10e-5:
        val = '**'
    else:
        val = '***'
    plt.text(i, delta[i] - 1, val, horizontalalignment='center')
    
plt.plot([-1,11],[0,0], 'k-', lw=4)
#plt.xticks(np.arange(11), delta.index)
plt.xticks(np.arange(11), delta.index, fontsize=14)
plt.yticks(np.arange(-5,26,5), fontsize=14)

plt.ylim(-8,25)
#plt.ylabel('Difference to baseline (in percentage points of accuracy)')
plt.ylabel('Difference to baseline (in percentage points of accuracy)', fontsize=16)
plt.text(0, 16, 'n.s.: non-significant (P > 5%)\n*: 0.1% < p < 5%\n**: 10e-5 < p < 0.1%\n***:  p < 10e-5\n\n(Note that significance depends on both difference\n      and the size of dataset [not shown])', fontsize=20)

X = np.arange(len(delta))

plt.plot(delta, 'o', label='combined', ms=16)

delta2 = data.surf - data.baseline
plt.plot(X + .05, delta2[delta.index]*100, 'o', label='SURF', ms=16)

delta2 = data.surf_ref - data.baseline
plt.plot(X - .05, delta2[delta.index]*100, 'o', label='SURF-ref', ms=16)
plt.legend()

plt.plot([-1,11],[5,5], 'k:', lw=4)
plt.plot([-1,11],[-5,-5], 'k:', lw=4)
plt.savefig('results.pdf')

