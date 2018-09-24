#!/usr/bin/python

import tensorflow as tf
from components import *

class TestData:
	def __init__(self,filenames0,filenames1,flowFilenames,batchSize,desiredHeight,desiredWidth):
		with tf.variable_scope(None,default_name="ImagePairData"):
			self.im0File = tf.placeholder(tf.string,[])
			self.im1File = tf.placeholder(tf.string,[])
			self.flowFile = tf.placeholder(tf.string,[])

			#read data
			rawData = tf.read_file(self.im0File)
			imData0 = tf.image.decode_png(rawData, channels=3)

			rawData = tf.read_file(self.im1File)
			imData1 = tf.image.decode_png(rawData, channels=3)

			rawData = tf.read_file(self.flowFile)
			flowData = tf.image.decode_png(rawData, channels=3,dtype=tf.uint16)

			#convert to floats, and divide
			mean = [0.448553, 0.431021, 0.410602]
			mean = tf.expand_dims(tf.expand_dims(mean,0),0)
			imData0 = tf.cast(imData0,tf.float32)/255 - mean
			imData1 = tf.cast(imData1,tf.float32)/255 - mean
			flowData = tf.cast(flowData,tf.float32)

			#convert flow data
			flowDataU = (tf.slice(flowData,[0,0,0],[-1,-1,1])-32768)/64
			flowDataV = (tf.slice(flowData,[0,0,1],[-1,-1,1])-32768)/64
			flowDataM = tf.slice(flowData,[0,0,2],[-1,-1,1])
			flowData = tf.concat([flowDataU,flowDataV,flowDataM],2)

			#pad to fix image size
			height = tf.cast(tf.shape(flowData)[0],tf.int32)
			width = tf.cast(tf.shape(flowData)[1],tf.int32)
			heightDiff = desiredHeight - height
			widthDiff = desiredWidth - width

			padding = [[0,heightDiff],[0,widthDiff],[0,0]]
			imData0 = tf.pad(imData0,padding)
			imData1 = tf.pad(imData1,padding)
			flowData = tf.pad(flowData,padding)

			#fix image size, be careful here!
			imData0.set_shape([desiredHeight,desiredWidth,3])
			imData1.set_shape([desiredHeight,desiredWidth,3])
			flowData.set_shape([desiredHeight,desiredWidth,3])

			#place data into batches
			imData0 = tf.expand_dims(imData0,0)
			imData1 = tf.expand_dims(imData1,0)
			flowData = tf.expand_dims(flowData,0)

			#split flow and flowmask
			flow = tf.slice(flowData,[0,0,0,0],[-1,-1,-1,2])
			flowMask = tf.slice(flowData,[0,0,0,2],[-1,-1,-1,1])

			self.frame0 = {
				"rgb": imData0
			}

			self.frame1 = {
				"rgb": imData1
			}

			self.flow = flow
			self.flowMask = flowMask
			self.flowMaskAll = tf.ones_like(flowMask)
			self.height = height
			self.width = width
