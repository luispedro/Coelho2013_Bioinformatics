import mahotas as mh
import numpy as np
basedir = '../../RANDTAG-LABELED-DATASETS/labeled-widefield/'
def false_color(base,n0,n1):
    g = mh.imread(basedir+'{}/{:02}-{:03}-protein.tiff'.format(base,n0,n1))
    r = mh.imread(basedir+'{}/{:02}-{:03}-dna.tiff'.format(base,n0,n1))
    return mh.as_rgb(r,g,None)


golgi = false_color('Golgi', 0, 0)
h,w,_ = golgi.shape
b = 256

for name, ci in [
        ('Golgi', 0),
        ('Golgi', 1),
        ('nuclear', 1),
        ]:
    canvas = np.zeros((2*h+b, 2*w+b, 3), np.uint8)
    canvas.fill(255)
    canvas[:h,:w] = false_color(name, ci, 0)
    canvas[:h,w+b:] = false_color(name, ci, 1)
    canvas[h+b:,:w] = false_color(name, ci, 2)
    canvas[h+b:,w+b:] = false_color(name, ci, 3)
    canvas = canvas[::2,::2]
    mh.imsave('{}{}.jpg'.format(name,ci), canvas)

