#!/bin/bash
   ../../build/tools/extract_features models/snapshots_lstm_RGB_iter_30000.caffemodel test_feature_RGB.prototxt fc6 features_RGB_test 116 leveldb GPU 0
