from imcol.image import FileImage

HPA_BASEDIR = '/share/images/HPA'

def _fix_path(p):
    p = HPA_BASEDIR + p[len('/share/images/HPA'):]
    return p

def load_antibodies():
    from scipy.io import loadmat
    mat = loadmat('../data/hpa/IF_atlas_5_regionfeatures_all_A431R2-2_5folds_Info_EnsEMBL_ID.mat')
    abids = mat['antibodyids'].squeeze()
    ensids= mat['EnsEMBL_ID'].squeeze()
    uni = mat['proteinnames_uniprot2'].squeeze()

    ab2eu = {}
    for k,e,u in zip(abids, ensids, uni):
        assert k not in ab2eu
        e = (e[0] if e else None)
        u = (u[0] if u else None)
        ab2eu[k] = (e,u)
    return ab2eu

def load():
    from scipy.io import loadmat
    ab2eu = load_antibodies()
    seen = set()
    mat = loadmat('../data/hpa/IF_atlas_5_regionfeatures_all_A431R2-2_5folds_Info.mat')
    antibodyids = mat['antibodyids'].squeeze()
    readlist = mat['readlist'].squeeze()
    readlist_nuc = mat['readlist_nuc'].squeeze()
    readlist_er = mat['readlist_er'].squeeze()
    readlist_tub = mat['readlist_tub'].squeeze()
    readlist_mask = mat['readlist_mask'].squeeze()

    labels = mat['classlabels']
    labels -= 1 # Matlab is 1-based
    classes = [tuple(*c) for c in  mat['classes']]

    images = []
    for ab,prot,nuc,er,tub,mask,lab in \
            zip(antibodyids, readlist, readlist_nuc, readlist_er, readlist_tub, readlist_mask, labels):
        if prot[0] in seen:
            continue
        seen.add(prot[0])
        im = FileImage({
                'protein': _fix_path(prot[0]),
                'dna': _fix_path(nuc[0]),
                'er': _fix_path(er[0]),
                'tubulin': _fix_path(tub[0]),
                'mask': _fix_path(mask[0]),
        })
        label, = classes[lab]
        im.label = label.split(',')
        im.gene = ab2eu[ab]
        images.append(im)
    return images


def surfref(im):
    from mahotas.features import surf
    import numpy as np

    spoints = surf.interest_points(im.get('protein'), max_points=1024)
    return np.hstack([
            surf.descriptors(im.get('protein'), spoints, descriptor_only=True),
            surf.descriptors(im.get('dna'), spoints, descriptor_only=True),
            surf.descriptors(im.get('er'), spoints, descriptor_only=True),
            surf.descriptors(im.get('tubulin'), spoints, descriptor_only=True),
        ])

def surf(im):
    from mahotas.features import surf
    import numpy as np

    spoints = surf.interest_points(im.get('protein'), max_points=1024)
    return surf.descriptors(im.get('protein'), spoints, descriptor_only=True)

def field_features(im):
    import pyslic
    img = pyslic.Image()
    img.channels = {'protein':'<protein>', 'dna':'<dna>'}
    img.channeldata = {'protein':im.get("protein"), 'dna':im.get('dna')}
    im.unload()
    img.loaded = True
    return pyslic.computefeatures(img, 'field-dna+')
