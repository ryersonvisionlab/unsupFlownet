import os
import time
import datetime
import tensorflow as tf
from PIL import Image
import numpy as np
import argparse
import json

from util import *

# parse args
argParser = argparse.ArgumentParser(description="runs validation and visualizations")
argParser.add_argument("-f",dest="showFlow",action="store_true")
argParser.add_argument("-e",dest="showError",action="store_true")
argParser.add_argument("-w",dest="showWarp",action="store_true")
argParser.add_argument("-a",dest="testAll",action="store_true")
argParser.add_argument("-t",dest="use2015",action="store_true")
argParser.add_argument("-i","--iteration",dest="iteration",action="store",default=0,type=int)
argParser.add_argument("-g","--gpu",dest="gpu",action="store",default=0,type=int)
cmdArgs = argParser.parse_args()

# multi gpu management
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(cmdArgs.gpu)

# load instance params
with open("hyperParams.json") as f:
	instanceParams = json.load(f)

# find latest snapshot
snapshotFiles = os.listdir("snapshots")
snapshotFiles = [filename for filename in snapshotFiles if filename[-11:] == ".ckpt.index"]
snapshotFiles.sort()

if cmdArgs.iteration > 0:
	iter = cmdArgs.iteration
	print "testing "+str(iter)
elif len(snapshotFiles) > 0:
	iter = int(snapshotFiles[-1][5:-11])
	print "testing "+ str(iter)
else:
	print "No snapshots found"
	exit()

# kitti2015 override
if cmdArgs.use2015:
	instanceParams["dataset"] = "kitti2015"

# import data
if instanceParams["dataset"] == "kitti2012":
	datasetRoot = "/home/jjyu/KITTI2012/"
	frame0Path = datasetRoot+"datalists/valPath1.txt";
	frame1Path = datasetRoot+"datalists/valPath2.txt";
	if cmdArgs.testAll:
		flowPath = datasetRoot+"valPathFloAll.txt";
	else:
		flowPath = datasetRoot+"valPathFlo.txt";
	desiredHeight = 384
	desiredWidth = 1280
elif instanceParams["dataset"] == "kitti2015":
	datasetRoot = "/home/jjyu/datasets/KITTI2015/"
	frame0Path = datasetRoot+"datalists/val_im1.txt";
	frame1Path = datasetRoot+"datalists/val_im2.txt";
	if cmdArgs.testAll:
		flowPath = datasetRoot+"datalists/val_flo_all.txt";
	else:
		flowPath = datasetRoot+"datalists/val_flo.txt";
	desiredHeight = 384
	desiredWidth = 1280
elif instanceParams["dataset"] == "sintel":
	datasetRoot = "/home/jjyu/datasets/Sintel/"
	frame0Path = datasetRoot+"datalists/val_im1.txt";
	frame1Path = datasetRoot+"datalists/val_im2.txt";
	flowPath = datasetRoot+"datalists/val_flow.txt";
	desiredHeight = 448
	desiredWidth = 1024
else:
	print "unknown dataset"
	exit()

with open(frame0Path) as f:
	imagePairs0 = [x[:-1] for x in f.readlines()]

with open(frame1Path) as f:
	imagePairs1 = [x[:-1] for x in f.readlines()]

with open(flowPath) as f:
	imageFlows = [x[:-1] for x in f.readlines()]

iterations = len(imageFlows)
testData = TestData(imagePairs0,imagePairs1,imageFlows,1,desiredHeight,desiredWidth)

# build graph
with tf.device("/gpu:"+str(cmdArgs.gpu)):
	with tf.variable_scope("netShare"):
		networkBody = NetworkBody(testData,instanceParams)
flowFinal = networkBody.flows[0]
epe, errorMap = epeEval(flowFinal,testData.flow,testData.flowMask)

# visualize
flowViz = flowToRgb(flowFinal)
errorMap = errorMap/20
transformGrid = flowTransformGrid(flowFinal)
mean = [0.448553, 0.431021, 0.410602]
mean = tf.expand_dims(tf.expand_dims(tf.expand_dims(mean,0),0),0)
warped = flowWarp(testData.frame1["rgb"]+mean,flowFinal)

# saver
saver = tf.train.Saver()

# config tensorflow
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

# start testing
epes = []

with tf.Session(config=config) as sess:
	saver.restore(sess,"snapshots/iter_"+str(iter).zfill(16)+".ckpt")

	#run
	lastPrint = time.time()
	for i in range(iterations):
		feed_dict = {
			testData.im0File: imagePairs0[i],
			testData.im1File: imagePairs1[i],
			testData.flowFile: imageFlows[i],
		}
		result = sess.run([epe,flowViz,errorMap,warped,testData.height,testData.width],feed_dict=feed_dict)
		h = result[4]
		w = result[5]
		flattened = [result[0][0]]
		epes += flattened
		print sum(epes)/float(len(epes))

		if cmdArgs.showFlow:
			arr = np.minimum(np.asarray(result[1]),1)
			arr = np.maximum(arr,0)
			arr = np.squeeze(np.asarray(arr*255,np.uint8))
			im = Image.fromarray(arr[:h,:w,:])
			im.show()
			raw_input("press to continue")

		if cmdArgs.showError:
			arr = np.minimum(np.asarray(result[2]),1)
			arr = np.maximum(arr,0)
			arr = np.squeeze(np.asarray(arr*255,np.uint8))
			im = Image.fromarray(arr[:h,:w])
			im.show()
			raw_input("press to continue")

		if cmdArgs.showWarp:
			arr = np.minimum(np.asarray(result[3]),1)
			arr = np.maximum(arr,0)
			arr = np.squeeze(np.asarray(arr*255,np.uint8))
			im = Image.fromarray(arr[:h,:w,:])
			im.show()
			raw_input("press to continue")
