#!/usr/bin/python

'''
todo:
fix weight decay
'''

import sys
import time
import datetime
import os
import tensorflow as tf
import argparse
from PIL import Image
import numpy as np

#from InstanceParams import *
from util import *

#--parse args
argParser = argparse.ArgumentParser(description="start/resume training")
argParser.add_argument("-l","--logDev",dest="logDev",action="store_true")
argParser.add_argument("-g","--gpu",dest="gpu",action="store",default=0,type=int)
argParser.add_argument("-i","--iterations",dest="iterations",action="store",default=0,type=int)
cmdArgs = argParser.parse_args()
#------settings-------
printFrequency = 50
snapshotFrequency = 500

batchSize = 1
iterations = 400000
startIteration = 0
#---------------------import data

with tf.device("/gpu:"+str(cmdArgs.gpu)):
	trainingData = TrainingData(batchSize)

	#----------------------------------------------------

	'''
	#import data
	datasetRoot = "/home/jjyu/KITTI2012/"
	frame0Path = datasetRoot+"train_im0.txt"
	frame1Path = datasetRoot+"train_im1.txt"

	with open(frame0Path) as f:
		imagePairs0 = [x[:-3] for x in f.readlines()]

	with open(frame1Path) as f:
		imagePairs1 = [x[:-3] for x in f.readlines()]

	with tf.device("/gpu:"+str(cmdArgs.gpu)):
		trainingData = TrainingData(imagePairs0,imagePairs1,batchSize)
	'''

	frame0 = trainingData.frame0
	frame1 = trainingData.frame1

#start
with sessionSetup(cmdArgs) as sess:
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	#start queueing
	trainingData.dataQueuer.start_queueing(sess)

	#run
	for i in range(startIteration, iterations):
 		#run training
		result = sess.run([frame0,frame1])

		msk = np.asarray(np.clip(result[0]+0.5,0,1)*225,np.uint8)
		msk = np.squeeze(msk)
		im = Image.fromarray(msk)
		im.show()

		msk = np.asarray(np.clip(result[1]+0.5,0,1)*225,np.uint8)
		msk = np.squeeze(msk)
		im = Image.fromarray(msk)
		im.show()

		raw_input("press to continue")

		sys.stdout.flush()
