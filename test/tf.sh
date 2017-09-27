#!/bin/sh
# Run python script, filtering out TensorFlow logging
# https://github.com/tensorflow/tensorflow/issues/566#issuecomment-259170351
python $* 3>&1 1>&2 2>&3 3>&- | grep -v "LOG_MEMORY" | grep -v "I tensorflow" | grep -v "pciBusID" | grep -v "totalMemory" | grep -v "deprecated" | grep -v "memoryClockRate"
