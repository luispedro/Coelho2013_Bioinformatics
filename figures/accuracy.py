from output import setup_default, output
import numpy as np
import re
from matplotlib import pyplot as plt
from sys import argv
COLOUR = (len(argv) > 1) and argv[1] == '--color'
if COLOUR:
    cs = 'color'
else:
    cs = 'gs'

pat = re.compile('Overall accuracy: ([.0-9]+)$')

def read_accuracy(dset, fset):
    for line in file('../sources/results/%s-%s.txt' % (dset,fset)):
        if pat.match(line):
            value = pat.match(line).group(1)
            return float(value)
        
dset = 'RT-widefield'
fset = 'surf-ref-None'

baseline     = read_accuracy(dset, 'base')

setup_default(.8)
data = np.load('../sources/results/%s-%s-profile.npy' % (dset,fset))
const = np.ones_like(data.T[2])
if COLOUR:
    plt.plot(data.T[0], 100*data.T[2], 'o', label='trials')
    plt.plot(data.T[0], 100*const*baseline, ':', lw=3, label='baseline')
else:
    plt.plot(data.T[0], 100*data.T[2], 'o', label='trials', mfc='#cccccc', mew=2)
    plt.plot(data.T[0], 100*const*baseline, 'k-', lw=3, label='baseline')

plt.xlabel(r'$k$')
plt.ylabel(r'$\rm{accuracy (\%)}$')
#plt.legend()
output('accuracy-rt-widefield-'+cs)


setup_default(.8)
x = data.T[3]/1000.
if COLOUR:
    plt.plot(x, 100*data.T[2],'o')
    plt.plot(x, 100*const*baseline, ':', lw=3, label='baseline')
    cs = 'color'
else:
    plt.plot(x, 100*data.T[2], 'o', label='trials', mfc='#cccccc', mew=2)
    plt.plot(x, 100*const*baseline, 'k-', lw=3, label='baseline')
    cs = 'gs'
plt.xlabel(r'$\rm{AIC} \,(\!\times 1000)$')
plt.ylabel(r'$\rm{accuracy (\%)}$')
output('accuracy-aic-rt-widefield-'+cs)

