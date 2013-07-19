#!/bin/bash
wget -N http://ome.grc.nia.nih.gov/iicbu2008/cho.tar.gz
wget -N http://ome.grc.nia.nih.gov/iicbu2008/binucleate.tar.gz
wget -N http://ome.grc.nia.nih.gov/iicbu2008/celegans.tar.gz
wget -N http://ome.grc.nia.nih.gov/iicbu2008/terminalbulb.tar.gz
wget -N http://ome.grc.nia.nih.gov/iicbu2008/rnai.tar.gz

wget -N http://murphylab.web.cmu.edu/data/HeLa10Class2DImages_16bit_dna_protein_png.tgz

wget -N http://locate.imb.uq.edu.au/info_files/SubCellLoc.zip

for t in *.tar.gz; do
    dir=`basename $t .tar.gz`
    mkdir -p $dir
    cd $dir
    tar xzf ../$t
    cd ..
done

for z in *.zip; do
    unzip -u $z
done

mv HeLa10Class2DImages_16bit_dna_protein_png hela
