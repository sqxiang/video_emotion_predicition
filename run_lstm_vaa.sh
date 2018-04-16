#!/bin/bash

TOOLS=../../build/tools

export HDF5_DISABLE_VERSION_CHECK=1
export PYTHONPATH=.

GLOG_logtostderr=1  $TOOLS/caffe train -solver lstm_solver_vaa.prototxt -weights vaa_caffenet.caffemodel -gpu 0 2>&1 | tee logs/logfile-$(date "+%y-%m-%d_%H_%M_%S").log
#-weights single_frame_all_layers_hyb_RGB_iter_5000.caffemodel  
echo "Done."
