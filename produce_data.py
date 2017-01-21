
import time

import numpy as np
import cPickle as cp
from sliding_window import sliding_window

# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 113

# Hardcoded number of classes in the gesture recognition problem
NUM_CLASSES = 18

# Hardcoded length of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_LENGTH = 24

# activity length
ACTIVITY_LENGTH=30

# Length of the input sequence after convolutional operations
FINAL_SEQUENCE_LENGTH = 8

# Hardcoded step of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_STEP = 12

# Batch Size
#BATCH_SIZE = 100

# Number filters convolutional layers
NUM_FILTERS = ACTIVITY_LENGTH

# Size filters convolutional layers
FILTER_SIZE = 5

# Number of unit in the long short-term recurrent layers
NUM_UNITS_LSTM = 128

#number of unit in the outer rnn
OUTER_UNITS_LSTM = 64

NUM_EPOCHS=500

def load_dataset(filename):

    f = file(filename, 'rb')
    data = cp.load(f)
    f.close()

    X_train, y_train = data[0]
    X_test, y_test = data[1]

    print(" ..from file {}".format(filename))
    print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    return X_train, y_train, X_test, y_test

print("Loading data...")
X_train, y_train, X_test, y_test = load_dataset('data/oppChallenge_gestures.data')

print("X_train shape:{}".format(X_train.shape))
print("y_train shape:{}".format(y_train.shape))

if 1 in y_train:
    print("1 in y_train")
else:
    print("1 not in y_train")

assert NB_SENSOR_CHANNELS == X_train.shape[1]
def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x,(ws,data_x.shape[1]),(ss,1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y,ws,ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.float32)


# process the train data
X_train,y_train=opp_sliding_window(X_train,y_train,SLIDING_WINDOW_LENGTH,SLIDING_WINDOW_STEP)
print(" ..after sliding window (train): inputs {0}, targets {1}".format(X_train.shape, y_train.shape))

#in order to use activity_length sequense activities to forecast the next activity
#we should make data one more time

def third_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x,(ws,data_x.shape[1],data_x.shape[2]),(ss,data_x.shape[1],1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y,ws,ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.float32)


X_train,y_train=third_sliding_window(X_train,y_train,30,1)
X_train=X_train.reshape(-1,X_train.shape[1],1,X_train.shape[2],X_train.shape[3])
print(" ..after.. after sliding window (train): inputs {0}, targets {1}".format(X_train.shape, y_train.shape))
#inputs (46466, 30, 1, 24, 113), targets (46466,)
X_test, y_test = opp_sliding_window(X_test, y_test, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
print(" ..after sliding window (testing): X_test {0}, y_test {1}".format(X_test.shape, y_test.shape))

X_test,y_test=third_sliding_window(X_test,y_test,30,1)
X_test=X_test.reshape(-1,X_test.shape[1],1,X_test.shape[2],X_test.shape[3])
print(" ..after.. after sliding window (train): inputs {0}, targets {1}".format(X_test.shape, y_test.shape))
#inputs (9865, 30,1, 24, 113), targets (9865,)
X_val=X_train[30000:36000]
y_val=y_train[30000:36000]

# X_val=X_val.reshape(-1,3000,1,24,113)
# y_val=y_val.reshape(-1,100)

X_train=X_train[0:30000]
y_train=y_train[0:30000]

# X_train=X_train.reshape(-1,3000,1,24,113)
# y_train=y_train.reshape(-1,100)

X_test=X_test[0:8000]
y_test=y_test[0:8000]

# X_test=X_test.reshape(-1,3000,1,24,113)
# y_test=y_test.reshape(-1,100)

import h5py,os

file_train=h5py.File('train.h5','w')
file_train.create_dataset('data',data=X_train,compression="gzip",compression_opts=9)
file_train.create_dataset('label',data=y_train,compression="gzip",compression_opts=9)
file_train.close()

file_val=h5py.File('val.h5','w')
file_val.create_dataset('data',data=X_val,compression="gzip",compression_opts=9)
file_val.create_dataset('label',data=y_val,compression="gzip",compression_opts=9)
file_val.close()


file_test=h5py.File('test.h5','w')
file_test.create_dataset('data',data=X_test,compression="gzip",compression_opts=9)
file_test.create_dataset('label',data=y_test,compression="gzip",compression_opts=9)
file_test.close()

myTrainFile=h5py.File('train.h5','r')
trainData=myTrainFile['data']
print("data.shape is {}".format(trainData.shape))
trainLabel=myTrainFile['label']
print("label.shape is {}".format(trainLabel.shape))
myTrainFile.close()

myValFile=h5py.File('val.h5','r')
valData=myValFile['data']
print("data.shape is {}".format(valData.shape))
valLabel=myValFile['label']
print("label.shape is {}".format(valLabel.shape))
myValFile.close()


myTestFile=h5py.File('test.h5','r')
testData=myTestFile['data']
print("data.shape is {}".format(testData.shape))
testLabel=myTestFile['label']
print("label.shape is {}".format(testLabel.shape))
myTestFile.close()
