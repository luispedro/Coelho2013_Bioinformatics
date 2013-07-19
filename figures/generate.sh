#!/usr/bin/env zsh
mkdir -p generated
for p in *.py; python $p
for p in *.py; python $p --color

cd generated
for s in *svg; inkscape $s --export-eps=`basename $s svg`eps
