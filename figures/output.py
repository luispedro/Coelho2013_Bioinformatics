#vim: set ts=4 sts=4 expandtab sw=4:
from sys import argv
import matplotlib
matplotlib.use('SVG')
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import sqrt
mpl.rcdefaults()

_golden_mean = (sqrt(5)-1.)/2.

bronzetan = '#BD3600'
smokered = '#D95550'
overcome = '#F89B0F'
pondgreen = '#647704'
crystalgreen = '#90a905'
toastedchilipowder = '#430a01'
purplepumpkin = '#6d2243'
circus392 = '#FFDC68'

def setup_default(size,ratio=_golden_mean):
    """
    setup_default(size)

    size is a fraction of \\textwidth
    """
    TEXTWIDTH = 341. #points
    fig_width_pt = size * TEXTWIDTH
    pt_per_inch = 72.27
    fig_width = fig_width_pt/pt_per_inch
    fig_height =fig_width*ratio
    fig_size = [fig_width,fig_height]

    params = {
        'text.fontsize': 10,
        'text.usetex': True,

        'xtick.labelsize': 8,
        'xtick.color' : toastedchilipowder,

        'ytick.labelsize': 8,
        'ytick.color' : toastedchilipowder,

        'lines.linewidth': 1.,
        'lines.markeredgewidth': 0.,

        'savefig.dpi': 600,

        'figure.figsize': fig_size,

        'axes.labelsize': 10,
        'axes.color_cycle' : [
                bronzetan, pondgreen, purplepumpkin, circus392, smokered, crystalgreen
            ],
        'axes.grid' : True,
        'axes.labelsize': 'small',
        'axes.facecolor': '#ffffff',
        'axes.edgecolor': toastedchilipowder,
        'axes.labelcolor': '#000000'
        }
    plt.rcParams.update(params)
    plt.clf()
    #plt.axes([0.25,0.25,0.95-0.25,0.95-0.25])

_formats = ['svg']
def output(filename, clear=True):
    if (len(argv) > 1 and argv[1] == '--show'):
        plt.show()
    else:
        filename = 'generated/' + filename
        for format in _formats:
            plt.savefig("%s.%s" % (filename,format), bbox_inches='tight', pad_inches=0)
    if clear:
        plt.clf()

