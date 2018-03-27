import numpy as np
import cPickle as cp
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing


# NORM_MAX_THRESHOLDS=[847,1787,101,797,849,110,139,3,847,1787,101,797,849,110,139,3,
# 					847,1787,101,797,849,110,139,3,847,1787,101,797,849,110,139,3,
# 					847,1787,101,797,849,110,139,3,847,1787,101,797,849,110,139,3]
# NORM_MIN_THRESHOLDS=[-847,-1787,-101,-797,-849,-110,-139,-3,
# 					-847,-1787,-101,-797,-849,-110,-139,-3,
# 					-847,-1787,-101,-797,-849,-110,-139,-3,
# 					-847,-1787,-101,-797,-849,-110,-139,-3,
# 					-847,-1787,-101,-797,-849,-110,-139,-3,
# 					-847,-1787,-101,-797,-849,-110,-139,-3]

data = np.loadtxt(open('plant_1a_dataset.csv','rb'),delimiter=',')
# shape is (113202,49)

# def normalize(data,max_list,min_list):
	# max_list,min_list=np.array(max_list),np.array(min_list)
	# diffs=max_list - min_list
	# for i in np.arange(data.shape[1]):
	# 	data[:,i]=(data[:,i]-min_list[i])/diffs[i]
	
	# data[data>1]=0.99
	# data[data<0]=0.00


	# return data


def process_dataset_file(data):
	data_x=data[:,:-1]
	data_y=data[:,-1]
	# data_x=normalize(data_x,NORM_MAX_THRESHOLDS,NORM_MIN_THRESHOLDS)
	data_x=preprocessing.scale(data_x)
	print('data_x mean')
	print(data_x.mean(axis=0))
	print('data_x std')
	print(data_x.std(axis=0))

	data_y=data_y.astype(int)
	#class 7 is normal type
	data_y[data_y<1]=7
	print('data_y 0->7')
	print(sorted(Counter(data_y).items()))

	# dataset is segmented into train and test
	
	obj=[(data_x,data_y)]
	f=open('plant_1a.data','wb')
	cp.dump(obj,f,protocol=cp.HIGHEST_PROTOCOL)
	f.close()

process_dataset_file(data)






