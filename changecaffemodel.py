import sys
caffe_root = "/home/sqxiang/caffe/"
sys.path.insert(0,caffe_root+'python')
import caffe

net = caffe.Net(caffe_root+"models/bvlc_reference_caffenet/deploy.prototxt","bvlc_reference_caffenet.caffemodel",caffe.TEST)
net_vaa = caffe.Net("vaa_deploy.prototxt","bvlc_reference_caffenet.caffemodel",caffe.TEST)
mean_file = "imagenet_mean.binaryproto" 

blob = caffe.proto.caffe_pb2.BlobProto()
data = open( mean_file , 'rb' ).read()
# parsing source data
blob.ParseFromString(data)
# convert to npy format
arr = np.array( caffe.io.blobproto_to_array(blob) )
mean_arr = arr[0].mean(1).mean(1)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mean_arr)            # subtract the dataset-mean value in each channel
#transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

net.blobs['data'].reshape(1,3,227,227)


image = caffe.io.load_image('')
transformed_image = transformer.preprocess('data', image)
