import pyslic
from os import listdir
import re

pat = '[0-9]{6}_([0-9]{2})_NOMX_(.+)_(dapi|myc)\.tif'

def load_brisbane(base):
    images = []
    files = set(listdir(base))
    for f in files:
        m = re.match(pat, f)
        idx,tag,kind = m.groups()
        if kind == 'myc':
            brother = f.replace('dapi','myc')
            if brother not in files:
                assert tag == "Nuclear_DAPI"
                brother = f
            brother = '%s/%s' % (base,brother)
            f = '%s/%s' % (base,f)
            im = pyslic.Image(protein=brother, dna=f)
            im.label = tag
            im.id = (tag,int(idx))
            images.append(im)
    return images
