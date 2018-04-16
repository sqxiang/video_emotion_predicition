#!/bin/bash

TOOLS=../../build/tools

export HDF5_DISABLE_VERSION_CHECK=1
export PYTHONPATH=.

GLOG_logtostderr=1  $TOOLS/caffe test -model test_lstm_RGB.prototxt -weights models/snapshots_lstm_RGB_iter_5000.caffemodel -iterations 116 -gpu 0 
#-weights single_frame_all_layers_hyb_RGB_iter_5000.caffemodel  
echo "Done."
