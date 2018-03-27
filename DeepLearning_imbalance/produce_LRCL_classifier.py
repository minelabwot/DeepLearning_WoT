import math
import numpy as np
import dask.array as da
import cPickle as cp
from sliding_window import sliding_window
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import h5py,os
from sklearn.utils import check_random_state, safe_indexing,shuffle

# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 48

# Hardcoded length of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_LENGTH = 48

SLIDING_WINDOW_STEP = 2

# activity length
ACTIVITY_LENGTH=20

ACTIVITY_STEP=1

def load_dataset(filename):

    f = file(filename, 'rb')
    data = cp.load(f)
    f.close()

    data_X, data_y = data[0]

    print(" ..from file {}".format(filename))
    print(" ..reading instances: train {0}".format(data_X.shape))

    data_X = data_X.astype(np.float32)
    # The targets are casted to int8 for GPU compatibility.
    data_y = data_y.astype(np.uint8)

    return data_X, data_y

print("Loading data...")
data_X, data_y = load_dataset('plant_1a_origin.data')

print("data_X shape:{}".format(data_X.shape))
print("data_y shape:{}".format(data_y.shape))

assert NB_SENSOR_CHANNELS == data_X.shape[1]

def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x,(ws,data_x.shape[1]),(ss,1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y,ws,ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.float32)

# process the train data
data_X,data_y=opp_sliding_window(data_X,data_y,SLIDING_WINDOW_LENGTH,SLIDING_WINDOW_STEP)
print(" ..after sliding window (train): inputs {0}, targets {1}".format(data_X.shape, data_y.shape))

print('after under and over sample')
print(Counter(data_y).values())
print('sum {0}'.format(sum(Counter(data_y).values())))


#in order to use activity_length sequense activities to forecast the next activity
#we should make data one more time

def third_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x,(ws,data_x.shape[1],data_x.shape[2]),(ss,data_x.shape[1],1))
    data_y = sliding_window(data_y,ws,ss)
    return data_x.astype(np.float32), data_y.astype(np.float32)

# def test_sliding_window(data_x, data_y, ws, ss):
#     data_x = sliding_window(data_x,(ws,data_x.shape[1],data_x.shape[2]),(ss,data_x.shape[1],1))
#     data_y = np.asarray([[i[-1]] for i in sliding_window(data_y,ws,ss)])
#     return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.float32)

data_X, data_y = third_sliding_window(data_X,data_y,ACTIVITY_LENGTH,ACTIVITY_STEP)
data_X=data_X.reshape(data_X.shape[0], data_X.shape[1], 1, data_X.shape[2],data_X.shape[3])
print(" ..after.. after sliding window (train): inputs {0}, targets {1}".format(data_X.shape, data_y.shape))
# inputs (23037, 20, 1, 48, 48), targets (23037, 20)
y_label=data_y[:,-1]
print('y_label.shape is {0}'.format(y_label.shape))
print y_label.shape[0]
# split normal data and anomaly data 
idx_under = np.empty((0, ), dtype=int)
random_state = check_random_state(None)
for target_class in np.unique(y_label):
    if target_class == 7:
        index_normal_class = slice(None)
        # n_samples = target_class.shape[0]
        # index_normal_class = random_state.choice(
        #     range(np.count_nonzero(y_label == target_class)),
        #     size=n_samples)
    else:
        index_normal_class = slice(0)
        # index_normal_class = random_state.choice(
        #     range(np.count_nonzero(y_label == target_class)),
        #     size=1)
    idx_under = np.concatenate((idx_under, np.flatnonzero(y_label == target_class)[index_normal_class]), axis=0)

normal_X = safe_indexing(data_X, idx_under)
normal_y = safe_indexing(data_y,idx_under)
normal_label=safe_indexing(y_label,idx_under)
print('only normal data')
print(sorted(Counter(normal_label).items()))

num_normal = int(math.ceil(0.9 * normal_X.shape[0]))
normal_train_X = normal_X[:num_normal]
normal_train_y = normal_y[:num_normal]
normal_test_X = normal_X[num_normal:]
normal_test_y = normal_y[num_normal:]

idx_under = np.empty((0, ), dtype=int)
random_state = check_random_state(None)
for target_class in np.unique(y_label):
    if target_class == 7:
        index_anomaly_class = slice(0)
    else:
        index_anomaly_class = slice(None)
    idx_under = np.concatenate((idx_under, np.flatnonzero(y_label == target_class)[index_anomaly_class]), axis=0)

anomaly_X = safe_indexing(data_X, idx_under)
anomaly_y=safe_indexing(data_y,idx_under)
anomaly_label=safe_indexing(y_label,idx_under)
print('only anomaly data')
print(sorted(Counter(anomaly_label).items()))

anomaly_X,anomaly_y=shuffle(anomaly_X,anomaly_y)
num_anomaly = int(math.ceil(0.8 * anomaly_X.shape[0]))
anomaly_train_X = anomaly_X[:num_anomaly]
anomaly_train_y = anomaly_y[:num_anomaly]
anomaly_test_X = anomaly_X[num_anomaly:]
anomaly_test_y = anomaly_y[num_anomaly:]

testSet_X = np.concatenate((normal_test_X,anomaly_test_X),axis=0)
testSet_y = np.concatenate((normal_test_y,anomaly_test_y),axis=0)
print('test set counter')
print(sorted(Counter(testSet_y[:,-1]).items()))
X_test,y_test = shuffle(testSet_X,testSet_y,random_state=0)

file_test = h5py.File('test_LRCL_informed_window'+str(SLIDING_WINDOW_LENGTH)+'.h5','w')
file_test.create_dataset('data',data=X_test,compression="gzip",compression_opts=9)
file_test.create_dataset('label',data=y_test,compression="gzip",compression_opts=9)
file_test.close()

num_test_normal = int(math.ceil(0.4 * X_test.shape[0]))
val_X = X_test[:num_test_normal]
val_y = y_test[:num_test_normal]
print('val set counter')
print(sorted(Counter(val_y[:,-1]).items()))

file_val=h5py.File('val_LRCL_informed_window'+str(SLIDING_WINDOW_LENGTH)+'.h5','w')
file_val.create_dataset('data',data=val_X,compression="gzip",compression_opts=9)
file_val.create_dataset('label',data=val_y,compression="gzip",compression_opts=9)
file_val.close()

normal_train_X = da.from_array(normal_train_X,chunks=128)
anomaly_train_X = da.from_array(anomaly_train_X,chunks=128)
normal_train_y = da.from_array(normal_train_y,chunks=128)
anomaly_train_y = da.from_array(anomaly_train_y,chunks=128)

trainSet_X = da.concatenate([normal_train_X,anomaly_train_X],axis=0)
trainSet_y = da.concatenate([normal_train_y,anomaly_train_y],axis=0)

# trainSet_X,trainSet_y = shuffle(trainSet_X,trainSet_y,random_state=0)
y_label=np.array(trainSet_y[:,-1])
print('train set counter')
print(sorted(Counter(y_label).items()))
trainSetName='train_LRCL_informed_window'+str(SLIDING_WINDOW_LENGTH)+'.h5'
trainSet_X = da.from_array(trainSet_X,chunks=128)
trainSet_y = da.from_array(trainSet_y,chunks=128)
da.to_hdf5(trainSetName,{'data':trainSet_X,'label':trainSet_y})

trainIndex=trainSet_X.shape[0]

myTrainFile=h5py.File('train_LRCL_informed_window'+str(SLIDING_WINDOW_LENGTH)+'.h5','r')
trainData=myTrainFile['data']
print("data.shape is {}".format(trainData.shape))
trainLabel=myTrainFile['label']
print("label.shape is {}".format(trainLabel.shape))
myTrainFile.close()

myValFile=h5py.File('val_LRCL_informed_window'+str(SLIDING_WINDOW_LENGTH)+'.h5','r')
valData=myValFile['data']
print("data.shape is {}".format(valData.shape))
valLabel=myValFile['label']
print("label.shape is {}".format(valLabel.shape))
myValFile.close()

myTestFile=h5py.File('test_LRCL_informed_window'+str(SLIDING_WINDOW_LENGTH)+'.h5','r')
testData=myTestFile['data']
print("data.shape is {}".format(testData.shape))
testLabel=myTestFile['label']
print("label.shape is {}".format(testLabel.shape))
myTestFile.close()
