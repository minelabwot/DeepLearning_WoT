import numpy as np

data=np.array([])

for x in xrange(1,7):
	if x==6:
		df = np.loadtxt(open('data/plant_1a_noduplicatation_component'+str(x)+'.csv','rb'),delimiter=',',usecols=(3,4,5,6,7,8,9,10,11))
	else:
		df = np.loadtxt(open('data/plant_1a_noduplicatation_component'+str(x)+'.csv','rb'),delimiter=',',usecols=(3,4,5,6,7,8,9,10))

	if x==1:
		data=df
	else:
		data=np.hstack([data,df])

np.savetxt('plant_1a_dataset.csv',data,fmt="%d" ,delimiter=',')
