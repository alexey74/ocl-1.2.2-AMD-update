#!/bin/bash

set -ex

clear
pname=$(realpath .)
cd /tmp
rm -f ocl.tgz && tar -C $(dirname $pname) -zcf ./ocl.tgz $(basename $pname)
octave-cli --eval 'pkg uninstall ocl; pkg install ocl.tgz; pkg test ocl'
cd -
octave-cli tests/*.m
