# coding:utf-8
import sys
caffe_root = "/home/sqxiang/caffe"
sys.path.insert(0,caffe_root+'python')
import caffe

import caffe.proto.caffe_pb2 as caffe_pb2      # 载入caffe.proto编译生成的caffe_pb2文件

# 载入模型
caffemodel_filename = 'rgbd/s00_dpt.caffemodel'
model = caffe_pb2.NetParameter()        # 为啥是NetParameter()而不是其他类，呃，目前也还没有搞清楚，这个是试验的
f = open(caffemodel_filename, 'rb')
model.ParseFromString(f.read())
f.close()

# noob阶段，只知道print输出
layers = model.layer
print len(layers)
print 'name: "%s"'%model.name
layer_id=-1
for layer in layers:
    layer_id = layer_id + 1
    print 'layer {'
    print '  name: "%s"'%layer.name
    print '  type: "%s"'%layer.type
    print '  shape: "%s"'%layer.shape
