Source code  for paper *Determining the subcellular location of new proteins
from microscope images using local features* by Coelho et al. in
`Bioinformatics <http://dx.doi.org/10.1093/bioinformatics/btt392>`__

This repository is **for reproduction of the results in the paper**. If you
want to apply the methods to your data, check out the `tutorial on doing so
<http://murphylab.web.cmu.edu/software/2013_Bioinformatics_LocalFeatures/tutorial.html>`__.
It is a step by step manual on applying the methods to your data.

Dependencies
------------

::

    sudo apt-get install python python-pip python-virtualenv
    sudo apt-get install dvipng

Instructions
------------

1. For the Human Protein Atas data, please download from
   http://murphylab.web.cmu.edu/software/2012_PLoS_ONE_Reannotation/

    Edit the file ``sources/hpa.py`` to point to where you downloaded all the data.

2. Get the randtag data from
   http://murphylab.web.cmu.edu/software/2013_Bioinformatics_LocalFeatures/ or
   from `Data Dryad <http://datadryad.org/resource/doi:10.5061/dryad.2vm70>`__

3. The remaining data should be automatically downloaded when you run::

    doitall.sh

This will also run the computation.

If you want to take advantage of multiple processors, edit the file
``doitall.sh`` and set the ``NR_CPUS`` variable. Note that the whole
computation (i) takes a very long time (days) on a single core and (ii) is
designed to take full advantage of multiples cores.

Citation
--------

For referring to this work, please cite:

   *Determining the subcellular location of new proteins from microscope images
   using local features* by Luis Pedro Coelho, Joshua D. Kangas, Armaghan Naik,
   Elvira Osuna-Highley, Estelle Glory-Afshar, Margaret Fuhrman, Ramanuja
   Simha, Peter B. Berget, Jonathan W. Jarvik, and Robert F.  Murphy (2013).
   Bioinformatics, [`DOI <http://dx.doi.org/10.1093/bioinformatics/btt392>`__]
