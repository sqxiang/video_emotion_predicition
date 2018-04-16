import os
import caffe
import numpy as np
import glob

root='/home/sqxiang/lisa-caffe-public/'
deploy=root + 'video_lstm/'
caffe_root = root
import sys
sys.path.insert(0,caffe_root + 'python')

caffe.set_mode_gpu()
caffe.set_device(0)

import pickle

RGB_video_path = 'test_finalframes30/'
Audio_path = 'test_finalaudioimages30/'
mean_file = 'imagenet_mean.binaryproto'

video = sys.argv[1]



#Initialize transformers

def initialize_transformer(image_mean, is_flow):
  shape = (8*18, 3, 227, 227)
  transformer = caffe.io.Transformer({'data': shape})
  channel_mean = np.zeros((3,227,227))
  for channel_index, mean_val in enumerate(image_mean):
    channel_mean[channel_index, ...] = mean_val
  transformer.set_mean('data', channel_mean)
#  transformer.set_raw_scale('data', 255)
  transformer.set_channel_swap('data', (2, 1, 0))
  transformer.set_transpose('data', (2, 0, 1))
  return transformer

blob = caffe.proto.caffe_pb2.BlobProto()
data = open( mean_file , 'rb' ).read()
# parsing source data
blob.ParseFromString(data)
# convert to npy format
arr = np.array( caffe.io.blobproto_to_array(blob) )		
mean_arr = arr[0].mean(1).mean(1)
	
transformer_RGB = initialize_transformer(mean_arr, False)

# Extract list of frames in video
RGB_frames = glob.glob('%s%s/*.*' %(RGB_video_path, video))
audio_frames = glob.glob('%s%s/*.*' %(Audio_path, video))

RGB_frames.sort(lambda x,y:cmp(int(x.split(".")[-2]),int(y.split(".")[-2])))
audio_frames.sort(lambda x,y:cmp(int(x.split(".")[-2]),int(y.split(".")[-2])))
	
print RGB_frames

#classify video with LRCN model
def LRCN_classify_video(frames_1, frames_2, net, transformer, is_flow):
  clip_length = 18
  input_images = []
  audio_images = []
  for im in frames_1:
    input_im = caffe.io.load_image(im)
#    if (input_im.shape[0] < 240):
    input_im = caffe.io.resize_image(input_im, (256,256))
    input_images.append(input_im)

  for im in frames_2:
    input_im = caffe.io.load_image(im)
#    if (input_im.shape[0] < 240):
    input_im = caffe.io.resize_image(input_im, (256,256))
    audio_images.append(input_im)


  vid_length = len(input_images)
  input_data = []
  audio_data = []
  
  input_data.extend(input_images[0:clip_length])
  audio_data.extend(audio_images[0:clip_length])
  output_predictions = np.zeros((36,3))
  clip_input_rgb = caffe.io.oversample(input_data,[227,227])
  clip_input_audio = caffe.io.oversample(audio_data,[227,227])
  clip_clip_markers = np.zeros((clip_input_rgb.shape[0],1,1,1))
  print clip_clip_markers.shape
#    if is_flow:  #need to negate the values when mirroring
#      clip_input[5:,:,:,0] = 1 - clip_input[5:,:,:,0]
  caffe_in = np.zeros(np.array(clip_input_rgb.shape)[[0,3,1,2]], dtype=np.float32)
  for ix, inputs in enumerate(clip_input_rgb):
    caffe_in[ix] = transformer.preprocess('data',inputs)
  caffe_in_audio = np.zeros(np.array(clip_input_audio.shape)[[0,3,1,2]], dtype=np.float32)
  for ix, inputs in enumerate(clip_input_audio):
    caffe_in_audio[ix] = transformer.preprocess('data',inputs)

  out = net.forward_all(data=caffe_in, data2=caffe_in_audio, clip_markers=np.array(clip_clip_markers))
  output_predictions[0:36] = np.mean(out['probs'],1)
  return np.mean(output_predictions,0).argmax(), output_predictions


#Models and weights
lstm_model = 'deploy_fusion_lstm.prototxt'
fusion_weight = 'snapshots_lstm_VAA_iter_11500.caffemodel'

lstm_net =  caffe.Net(lstm_model, fusion_weight, caffe.TEST)
class_RGB_LRCN, predictions_RGB_LRCN = \
         LRCN_classify_video(RGB_frames, audio_frames,lstm_net, transformer_RGB,False)

print class_RGB_LRCN,predictions_RGB_LRCN
