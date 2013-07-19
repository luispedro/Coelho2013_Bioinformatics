import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpltools import style
style.use('ggplot')
data = pd.read_table('compare-all.csv', sep='\t', index_col=0)
from matplotlib import pyplot as plt

line_color = '#647704'
circle_color = '#430a01'

data = np.array([
    [data.baseline['RT-widefield-no-origin'],
        data.baseline['RT-widefield'],
        data.combined['RT-widefield']],
    [data.baseline['RT-confocal-no-origin'],
        data.baseline['RT-confocal'],
        data.combined['RT-confocal']],
    [0.761,
        data.baseline['HPA'],
        data.combined['HPA']]])

plt.xlim(-.2,2.2)
plt.ylim(55,87)
plt.plot(100*data.T, '-', lw=8, color=line_color)
plt.plot(100*data.T, 'o', mew=12, color=circle_color)
plt.xticks([0,1,2], ["per image", "per protein", "per protein (with local features)"])
plt.ylabel('Accuracy (%)')
plt.savefig('generalization-cross-val3.pdf')


plt.clf()
data = data[:,:2]
plt.ylim(55,87)
plt.xlim(-.2,2.2)
plt.plot(100*data.T, '-', lw=8, color=line_color)
plt.plot(100*data.T, 'o', mew=12, color=circle_color)
plt.text(1.1, 100* data[0,1], 'RT-Widefield')
plt.text(1.1, 100* data[1,1], 'RT-Confocal')
plt.text(1.1, 100*data[2,1], 'Human Protein Atlas (HPA)')
plt.xticks([0,1], ["per image", "per protein"])
plt.ylabel('Accuracy (%)')
plt.savefig('generalization-cross-val2.pdf')
