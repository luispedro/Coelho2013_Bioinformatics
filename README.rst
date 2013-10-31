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
http://murphylab.web.cmu.edu/software/2013_Bioinformatics_LocalFeatures/ or from


3. The remaining data should be automatically downloaded when you run::

    doitall.sh

This will also run the computation.

If you want to take advantage of multiple processors, edit the file
``doitall.sh`` and set the ``NR_CPUS`` variable. Note that the whole
computation (i) takes a very long time (days) on a single core and (ii) is
designed to take full advantage of multiples cores.

Citation
--------

If you are using the results of this computation or referring to the ideas in
the paper, please cite:

If you are using this `mahotas <http://mahotas.rtfd.org>`__ based
implementation, please cite the software paper too:


