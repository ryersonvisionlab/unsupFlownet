import sys
import time
import datetime
import os
import tensorflow as tf
import argparse
import json
import socket

from util import *

# parse command line args
argParser = argparse.ArgumentParser(description="start/resume training")
argParser.add_argument("-l","--logDev",dest="logDev",action="store_true")
argParser.add_argument("-g","--gpu",dest="gpu",action="store",default=0,type=int)
argParser.add_argument("-i","--iterations",dest="iterations",action="store",default=0,type=int)
argParser.add_argument("-r","--resume",dest="resume",action="store_true")
cmdArgs = argParser.parse_args()

# multi gpu management
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(cmdArgs.gpu)

# load instance params, set instance path for logs and snapshots
with open("hyperParams.json") as f:
	instanceParams = json.load(f)
logPath = "logs/"
snapshotPath = "snapshots/"

# training settings helpers
printFrequency = instanceParams["printFreq"]
snapshotFrequency = instanceParams["snapFreq"]
batchSize = instanceParams["batchSize"]

iterations = instanceParams["iterations"]
baseLearningRate = instanceParams["baseLR"]
learningRate = baseLearningRate
snapshotFrequency = instanceParams["snapshotFreq"]

# iteration override
if cmdArgs.iterations > 0:
	print "overriding max training iterations from commandline argument"
	iterations = cmdArgs.iterations

#check for resume
resume, startIteration, snapshotFiles = checkResume(snapshotPath,logPath, cmdArgs)

# build graph
with tf.device("/gpu:"+str(cmdArgs.gpu)):
	trainingData = TrainingData(batchSize,instanceParams)
with tf.device("/gpu:"+str(cmdArgs.gpu)):
	# init
	with tf.variable_scope("netShare"):
		networkBodyF = NetworkBody(trainingData,instanceParams)
	with tf.variable_scope("netShare",reuse=True):
		networkBodyB = NetworkBody(trainingData,instanceParams,flipInput=True)

	trainingLoss = TrainingLoss(instanceParams,networkBodyF,networkBodyB,trainingData)
	solver,learningRateTensor = attachSolver(trainingLoss.loss)

	# loss scheduling
	recLossBWeightTensor = trainingLoss.recLossBWeight

# merge summaries
merged = tf.summary.merge_all()

# saver
saver = tf.train.Saver(max_to_keep=0)

# start
with sessionSetup(cmdArgs) as sess:
	if resume:
		saver.restore(sess,snapshotPath+snapshotFiles[-1][:-6])
	else:
		sess.run(tf.initialize_all_variables())

	trainingData.dataQueuer.start_queueing(sess)

	#start summary writer
	summary_writer = tf.summary.FileWriter(logPath, sess.graph)

	#run
	lastPrint = time.time()
	for i in range(startIteration, iterations):
		# scheduled values
		learningRate = learningRateSchedule(baseLearningRate, i)
		recLossBWeight = unsupLossBSchedule(i)

 		#run training
		feed_dict = {
			learningRateTensor: learningRate,
			recLossBWeightTensor: recLossBWeight,
		}
		summary,result,totalLoss = sess.run([merged,solver,trainingLoss.loss], feed_dict=feed_dict)

		if (i+1) % printFrequency == 0:
			timeDiff = time.time() - lastPrint
			itPerSec = printFrequency/timeDiff
			remainingIt = iterations - i
			eta = remainingIt/itPerSec
			print("Iteration "+str(i+1)+": loss: "+str(totalLoss)+", iterations per second: "+str(itPerSec)+", ETA: "+str(datetime.timedelta(seconds=eta)))+", lr: "+str(learningRate)

			summary_writer.add_summary(summary,i+1)
			summary_writer.flush()
			lastPrint = time.time()

		if (i+1) % snapshotFrequency == 0:
			saver.save(sess,"snapshots/iter_"+str(i+1).zfill(16)+".ckpt")

		sys.stdout.flush()

	# close queing
	trainingData.dataQueuer.close(sess)
