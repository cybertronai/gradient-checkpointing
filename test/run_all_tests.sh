#!/bin/sh
#
# On MacOS
# export CUDA_HOME=/usr/local/cuda
# export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/extras/CUPTI/lib
# export LD_LIBRARY_PATH=$DYLD_LIBRARY_PATH
# export PATH=$DYLD_LIBRARY_PATH:$PATH
#gi
# tests/run_all_tests.sh
#
# On Ubuntu:
#
# tests/run_all_tests.sh
#
#
BASEDIR="."
# for util.py which is 1 level up
export PYTHONPATH="$BASEDIR/..:$PYTHONPATH"
$BASEDIR/tf.sh $BASEDIR/memory_test.py
$BASEDIR/tf.sh $BASEDIR/util_test.py
$BASEDIR/tf.sh $BASEDIR/linearize_test.py  # not converted to CPU-only
$BASEDIR/tf.sh $BASEDIR/resnet_test.py
$BASEDIR/tf.sh $BASEDIR/keras_test.py

