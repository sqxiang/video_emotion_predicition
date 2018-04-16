import lmdb

import os, re ,fileinput,math
import numpy as np
import sys
caffe_root = '/home/sqxiang/caffe/'
sys.path.insert(0,caffe_root + 'python')

import caffe
from caffe import io

train_path = "../idx/trainvideos.txt"
emotion_path = "train_emotion_pro"

def Z_ScoreNormalization(x):  
    x = (x - np.average(x)) / np.std(x);  
    return x; 

for line in fileinput.input(train_path):
    video = line.strip()
    filepath = os.path.join(emotion_path,video)
    txt_paths = []
    for emotion_txt in os.listdir(filepath):
        txt_paths.append(os.path.join(filepath,emotion_txt))
    txt_paths.sort()
    print txt_paths
    for txt_path in txt_paths:
        with open(txt_path,"r") as tf:
            emotions = tf.readlines()
            emotions = emotions[0].split(",")
            emotions = map(float,emotions)
            emotions = Z_ScoreNormalization(emotions)
            

