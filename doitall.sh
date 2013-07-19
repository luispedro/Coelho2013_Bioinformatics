#!/usr/bin/env bash

NR_CPUS=1

if [[ -z "$VIRTUAL_ENV" ]] ; then
    echo "creating virtualenv..."
    virtualenv --system-site-packages virtualenv
    source  virtualenv/bin/activate
fi

echo "Installing Python dependencies with pip:"
pip install \
    pyslic \
    mahotas \
    milk \
    Jug \
    numpy \
    scipy \
    matplotlib
python -c 'import imcol' || pip install git+https://github.com/luispedro/imcol.git

echo "Downloading data"

cd data;
    sh ./retrieve.sh
cd ..

echo "Running code"
echo 'This can take a while if you only have one CPU (maybe longer than 24 hours!)'
echo 'If you spawn multiple `jug` jobs, it will be faster'
echo 'Stop the process now and edit the NR_CPUS variable in the script'


cd sources
    for i in `seq $NR_CPUS`; do
        jug execute --aggressive-unload &
    done
    jug sleep-until
cd ..


cd sources
    for i in `seq $NR_CPUS`; do
        jug execute hparun.py --aggressive-unload &
    done
    jug sleep-until hparun.py
cd ..

echo "Building figures"
cd figures
    for p in *py; do
        python $p
    done
cd ..

echo "Done"

