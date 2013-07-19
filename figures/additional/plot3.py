import mahotas as mh
from pylab import *
from mpltools import style

style.use('ggplot')

basedir = '../../RANDTAG-LABELED-DATASETS/labeled-widefield/'

def false_color(base,n0,n1):
    g = mh.imread(basedir+'{}/{:02}-{:03}-protein.tiff'.format(base,n0,n1))
    r = mh.imread(basedir+'{}/{:02}-{:03}-dna.tiff'.format(base,n0,n1))
    return mh.as_rgb(r,g,None)

def features_for(imfile):
    return mh.features.haralick(mh.stretch(mh.imread(imfile))//4).mean(0)[[1,12]]

golgi = np.array([features_for(basedir+'Golgi/00-{:03}-protein.tiff'.format(i)) for i in xrange(25)])
nuc = np.array([features_for(basedir+'nuclear/01-{:03}-protein.tiff'.format(i)) for i in xrange(12)])
golgi2 = np.array([features_for(basedir+'Golgi/01-{:03}-protein.tiff'.format(i)) for i in xrange(14)])

scatter(golgi[:,0], golgi[:,1], c='r', label='Golgi (1)', s=64)
scatter(nuc[:,0], nuc[:,1], c='b', label='Nuclear', s=64)
xlabel('Haralick feature 2')
ylabel('Haralick feature 12')
plot([-5,15],[.7,1.1], 'k-',lw=3)
legend()
savefig('plot2.pdf')

scatter(golgi2[:,0], golgi2[:,1], c='g', label='Golgi (2)', s=64)
legend()
savefig('plot3.pdf')
