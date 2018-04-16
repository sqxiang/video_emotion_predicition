#!/bin/bash

TOOLS=../../build/tools

export HDF5_DISABLE_VERSION_CHECK=1
export PYTHONPATH=.

GLOG_logtostderr=1  $TOOLS/caffe test -model test_lstm_scratch_VAA.prototxt -weights models/snapshots_lstm_VAA_scratch_valence_iter_26880.caffemodel -iterations 116 -gpu 0
#-weights single_frame_all_layers_hyb_RGB_iter_5000.caffemodel  
echo "Done."
