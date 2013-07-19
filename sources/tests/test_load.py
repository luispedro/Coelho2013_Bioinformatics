from brisbane import load_brisbane
from utils import load_directory

def test_hela():
    assert len(load_directory('../data/hela')) == 862

def test_brisbane():
    assert len(set([im.label for im in load_brisbane('../data/SubCellLoc/Transfected/')]))  == 11
    assert len(set([im.label for im in load_brisbane('../data/SubCellLoc/Endogenous/')]))  == 10
