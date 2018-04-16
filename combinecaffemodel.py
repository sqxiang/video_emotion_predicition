import sys
caffe_root = "/home/sqxiang/caffe/"
sys.path.insert(0,caffe_root+'python')
import caffe
caffe.set_mode_cpu()
#net = caffe.Net(caffe_root+"models/bvlc_reference_caffenet/deploy.prototxt","bvlc_reference_caffenet.caffemodel",caffe.TEST)
#net = caffe.Net("vaa_deploy.prototxt","bvlc_reference_caffenet.caffemodel",caffe.TEST)
net_googlenet = caffe.Net("googlenet_deploy.prototxt","bvlc_googlenet.caffemodel",caffe.TEST)

for name,blob in net_googlenet.blobs.iteritems():
    print name + "\t" + str(blob.data.shape)
