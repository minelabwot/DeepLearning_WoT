import numpy as np
import cPickle as cp
import time
import os.path
import sys

# check and process the data in each component
lastValue=0
intimeStamp=0
removeIndex=[]
lackIndex=[]
mulLackIndex=[]
otherIndex=[]

# there should be a data folder to provide the raw data.
# and there are 6 files in this folder.(like plant_1a_component{1}.csv)
if os.path.exists('data/plant_1a_component1.csv'):
	print "start processing"
else:
	sys.exit('Error:\nyou have no data set in data folder.\n Please add the dataset first.\n You can find the whole dataset from PHM2015 challenge')

for x in xrange(1,7):
	with open('data/plant_1a_component'+str(x)+'.csv','r') as infile ,open('data/plant_1a_com'+str(x)+'.csv','w') as outfile:
		for line in infile:
			words=line.split(",")
			if intimeStamp == 0:
				intimeStamp = int(words[1])
				print "first intimeStamp {0}".format(intimeStamp)
				outfile.write(line)
				#time.sleep(5)
			else:
				value=int(words[1])
				#print "value is {0}".format(value)
				#time.sleep(5)
				if value==intimeStamp:
					print "repeat data:{0}".format(value)

				else: 
					outfile.write(line)
					intimeStamp=value

	timeStamp=0
	with open('data/plant_1a_com'+str(x)+'.csv','r') as inNoDupfile ,open('data/plant_1a_noduplicatation_component'+str(x)+'.csv','w') as outNoDupfile:
		for line in inNoDupfile:
			words=line.split(",")
			if timeStamp == 0:
				timeStamp = int(words[1])
				print "first timeStamp {0}".format(timeStamp)
				outNoDupfile.write(line)
				#time.sleep(5)
			else:
				value=int(words[1])
				#print "value is {0}".format(value)
				#time.sleep(5)
				if value==timeStamp+90:
					outNoDupfile.write(line)
					timeStamp=value

				elif value>(int(timeStamp)+90) & (value-timeStamp)%90==0:
					multiple=(value-timeStamp)/90
					print "multiple is {0}".format(multiple)
					#time.sleep(5)
					temp=words
					for x in xrange(1,multiple):
						temp[1]=int(timeStamp)+90*x
						print "lack multiple data:{0}".format(temp[1])
						outNoDupfile.write(','.join(str(x) for x in temp))
					outNoDupfile.write(line)
					timeStamp=value

				else:
					print "other data:{0}".format(value)
					time.sleep(5)


				






# re-grouping