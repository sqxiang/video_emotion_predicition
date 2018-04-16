import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
sys.path.insert(0,"../../python")
import caffe
model_weights = "models/snapshots_lstm_RGB_iter_20000.caffemodel"
model_def = "deploy_lstm_RGB.prototxt"
net = caffe.Net(model_def,model_weights,caffe.TEST)

mean_file = "imagenet_mean.binaryproto"

def vis_square(data,path):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
    and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
               
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
    (0, 1), (0, 1))                 # add some space between filters
    + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data); plt.axis('off')
    plt.savefig(path)
    plt.close()

if __name__ == "__main__":
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
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
    
    image = caffe.io.load_image('ACCEDE00443/ACCEDE00443.010.jpg')
    transformed_image = transformer.preprocess('data', image)
    #cv2.imwrite("b.jpg",transformed_image[0])
    
    net.blobs["data"].data[...] = transformed_image
    
    net.forward()
    
    for layer_name,blob in net.blobs.iteritems():
        print layer_name + '\t' + str(blob.data.shape)
    
    filters = net.params['conv1'][0].data
    blob1 = net.blobs['conv1'].data[0]
    vis_square(filters.transpose(0, 2, 3, 1),"conv1.jpg")
    vis_square(blob1,"blob1.jpg")
    filters = net.params['conv2'][0].data
    blob1 = net.blobs['conv2'].data[0]
    vis_square(blob1,"blob2.jpg")
    vis_square(filters[:48].reshape(48**2,5,5),"conv2.jpg")

#    filters = net.params['conv3'][0].data
#    blob1 = net.blobs['conv3'].data[0]
#    vis_square(blob1,"blob3.jpg")
#    vis_square(filters,"conv3.jpg")
#
#    filters = net.params['conv4'][0].data
#    blob1 = net.blobs['conv4'].data[0]
#    vis_square(blob1,"blob4.jpg")
#    vis_square(filters.transpose(0, 2, 3, 1),"conv4.jpg")
#
    filters = net.params['conv5'][0].data
    blob1 = net.blobs['conv5'].data[0]
    vis_square(blob1,"blob5.jpg")
    vis_square(filters[:10].reshape(10*192,3,3),"conv5.jpg")

    blob1 = net.blobs['pool5'].data[0]
    vis_square(blob1,"blobpool5.jpg")



    feat = net.blobs['fc6'].data[0]
    plt.subplot(2, 1, 1)
    plt.plot(feat.flat)
    plt.subplot(2, 1, 2)
    _ = plt.hist(feat.flat[feat.flat > 0], bins=100)
    plt.savefig("blobfc6.jpg")
    plt.close()
    feat = net.params['fc6'][0].data
    plt.subplot(2, 1, 1)
    plt.plot(feat.flat)
    plt.subplot(2, 1, 2)
    _ = plt.hist(feat.flat[feat.flat > 0], bins=100)
    plt.savefig("fc6.jpg")
