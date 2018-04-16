import numpy as np

from keras.preprocessing import sequence
from keras.datasets import mnist
from keras.models import Sequential
#from keras.initializations import norRemal, identity
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.optimizers import RMSprop, Adadelta,SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Activation, TimeDistributedDense, Dropout, Reshape, Flatten,Masking
from keras.layers.wrappers import TimeDistributed
from keras.utils import np_utils
from keras.models import model_from_json
import cPickle as pickle
from keras.callbacks import EarlyStopping
#import json
import sys  
import leveldb  
import chardet
reload(sys)
import caffe
import cPickle as pickle
sys.setdefaultencoding('utf-8')
print sys.getdefaultencoding()
def read_data(levelpath):  
    db = leveldb.LevelDB(levelpath)  
   # db = leveldb.LevelDB("./features_fc7/")  
    i = 0
    dataX = []
    data = []
    for k in db.RangeIter(include_value = False):  
        i = i+1
        uni = db.Get(k)  
        datum = caffe.io.caffe_pb2.Datum()
        datum.ParseFromString(uni)
        arr = caffe.io.datum_to_array(datum)
        print i
        dataX.extend(arr.flatten("C"))
        data.append(dataX)
        dataX = []
    return data
 

# for reproducibility
np.random.seed(2016)  

embbeding = read_data("features_RGB")
z = np.array([[0.0]*4096])
embbeding = np.concatenate((z,embbeding),axis=0)
#print embbeding.shape
#print embbeding[0]
x_train_raw,y_train_raw = pickle.load(open("train_idx.pkl","rb"))
#print len(x_train),len(x_train[0])
test_embbeding = read_data("features_RGB_test")
test_embbeding = np.concatenate((z,test_embbeding),axis =0)

x_test_raw,y_test_raw = pickle.load(open("test_idx.pkl","rb"))

maxlen = 18
timesteps = 18
embedding_dims = len(embbeding[0])
dims = len(embbeding[0])
print "dims: ",dims
train_length = 6144
batch_size = 64
nb_epoches = 50
test_length = 4756

#print len(x_train),'train sequences'
#print 'pad sequences (samples x time)'
#X_train = sequence.pad_sequences(x_train,maxlen=maxlen)
#print 'X_train shape:',X_train.shape
#print X_train[0]

model = Sequential()
#model.add(Masking(mask_value= 0.0,input_shape=(18, 4096)))
model.add(LSTM(256,return_sequences=False,input_shape=(18,dims)))
model.add(Dropout(0.5))
model.add(Dense(128,init='uniform'))
model.add(Activation("tanh"))
model.add(Dropout(0.5))
model.add(Dense(3,init='uniform'))
model.add(Activation("softmax"))

sgd = SGD(lr=0.01, decay=1e-7, momentum=.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])


X_train = np.zeros((train_length,timesteps,dims))
y_train = []
	
for i in range(0,train_length):

	output = np.zeros((timesteps,dims))
	indices = x_train_raw[i]
	examplesY = int(y_train_raw[i])
	if examplesY == -1:
		examplesY = 2
	examples = embbeding[indices]
	numToAdd = len(indices)
	output[0:numToAdd,:] = examples
	X_train[i,:,:] = output
	y_train.append(examplesY)

y_train = np.array(y_train)
y_train_oh = np_utils.to_categorical(y_train,3) 
print "X_train shape: ",X_train.shape
print "y_train shape: ",y_train.shape		


#early_stopping = EarlyStopping(monitor='val_loss',min_delta=0.1,patience=10)
	
#	y_train_pred = model.predict_classes(X_train, verbose=0)  
#	print y_train_pred

X_test = np.zeros((test_length,timesteps,dims))
y_test = []

for i in range(0,test_length):
	output = np.zeros((timesteps,dims))
	indices = x_test_raw[i]
	examplesY = int(y_test_raw[i])
	if examplesY == -1:
		examplesY  = 2
	examples = test_embbeding[indices]
	numToAdd = len(indices)
	output[0:numToAdd,:] = examples
	X_test[i,:,:] = output
	y_test.append(examplesY)

X_test = np.array(X_test)
y_test = np.array(y_test)
#x_val  = X_test[0:2000]
#y_val = y_test[0:2000]
#y_val_oh = np_utils.to_categorical(y_val,3)
#X_test = X_test[2000:]
#y_test = y_test[2000:] 

#model.fit(X_train,y_train_oh,batch_size=batch_size,validation_data=(x_val,y_val_oh),nb_epoch=nb_epoches,verbose=1)
model.fit(X_train,y_train_oh,batch_size=batch_size,nb_epoch=nb_epoches,verbose=1)

y_train_pred = model.predict_classes(X_train, verbose=0)  
print('First 3 predictions: ', y_train_pred[:3])  
  
train_acc = float(np.sum(y_train == y_train_pred, axis=0)) / float(X_train.shape[0])  

print('Training accuracy: %.2f%%' % (train_acc * 100))  
  
y_test_pred = model.predict_classes(X_test, verbose=0)  
test_acc = float(np.sum(y_test == y_test_pred, axis=0)) / float(X_test.shape[0])  
print('Test accuracy: %.2f%%' % (test_acc * 100))  
