import numpy as np
import cPickle as cp
import time

# check and process the data in each component
lastValue=0
timeStamp=0
removeIndex=[]
lackIndex=[]
mulLackIndex=[]
otherIndex=[]

# remove the duplicated records

with open('data/plant_1a_com2.csv','r') as infile ,open('data/plant_1a_noduplicatation_component2.csv','w') as outfile:
	for line in infile:
		words=line.split(",")
		if timeStamp == 0:
			timeStamp = int(words[1])
			print "first timeStamp {0}".format(timeStamp)
			outfile.write(line)
			#time.sleep(5)
		else:
			value=int(words[1])
			#print "value is {0}".format(value)
			#time.sleep(5)
			if value==timeStamp+90:
				outfile.write(line)
				timeStamp=value

			elif value>(int(timeStamp)+90) & (value-timeStamp)%90==0:
				multiple=(value-timeStamp)/90
				print "multiple is {0}".format(multiple)
				#time.sleep(5)
				temp=words
				for x in xrange(1,multiple):
					temp[1]=int(timeStamp)+90*x
					print "lack multiple data:{0}".format(temp[1])
					outfile.write(','.join(str(x) for x in temp))
				outfile.write(line)
				timeStamp=value

			else:
				print "other data:{0}".format(value)
				time.sleep(5)





# re-grouping