import h5py
import numpy as np
import dask.array as da
from sklearn.utils import check_random_state, safe_indexing,shuffle
from collections import Counter
SLIDING_WINDOW_LENGTH=100
# dask is used to precess the large array.
# otherelse, the memory usage easily jumps to out of threshould
myTrainFile=h5py.File('train_LRCL_informed_window'+str(SLIDING_WINDOW_LENGTH)+'.h5','r')
trainSet_X=myTrainFile['data']
trainSet_y=myTrainFile['label']
trainSet_X=np.array(trainSet_X)
trainSet_y=np.array(trainSet_y)
y_label = trainSet_y[:,-1]

for seed in xrange(1,6):
    idx_under = np.empty((0, ), dtype=int)
    random_state = check_random_state(None)
    for target_class in np.unique(y_label):
        if target_class == 7:
            n_samples = 10000
            index_class = random_state.choice(
                range(np.count_nonzero(y_label == target_class)),
                size=n_samples)
        else:
            index_class = slice(None)
        idx_under = np.concatenate((idx_under, np.flatnonzero(y_label == target_class)[index_class]), axis=0)

    sub_X = safe_indexing(trainSet_X, idx_under)
    sub_y = safe_indexing(trainSet_y,idx_under)
    sub_label=safe_indexing(y_label,idx_under)

    print('sub counter')
    print(sorted(Counter(sub_label).items()))

    trainSet_X_seed = da.from_array(sub_X,chunks=128)
    trainSet_y_seed = da.from_array(sub_y,chunks=128)
    da.to_hdf5('train_LRCL_informed_window'+str(SLIDING_WINDOW_LENGTH)+'_seed'+str(seed)+'.h5',{'data':trainSet_X_seed,'label':trainSet_y_seed})
