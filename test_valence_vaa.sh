#!/bin/bash

TOOLS=../../build/tools

export HDF5_DISABLE_VERSION_CHECK=1
export PYTHONPATH=.

GLOG_logtostderr=1  $TOOLS/caffe test -model test_lstm_VAA.prototxt -weights models/snapshots_lstm_VAA_valence_iter_22272.caffemodel -iterations 116 
#-weights single_frame_all_layers_hyb_RGB_iter_5000.caffemodel  
echo "Done."
