#!/usr/bin/python

import tensorflow as tf
from components import *

class NetworkBody:
	def __init__(self,trainingData,instanceParams,flipInput=False):
		frame0 = trainingData.frame0["rgb"]
		frame1 = trainingData.frame1["rgb"]
		if flipInput:
			combined = tf.concat([frame1,frame0],3)
		else:
			combined = tf.concat([frame0,frame1],3)

		self.resnet = instanceParams["resnet"]
		self.flowScale = instanceParams["flowScale"]

		self.buildNetwork(combined,resnet=self.resnet)

	def buildNetwork(self,inputs,resnet):
		#contractive
		conv = convLayerRelu(inputs,7,64,2)

		if resnet:
			conv2 = resBlock(conv,5,128)
			conv3_1 = resBlock(conv2,5,256)
			conv4_1 = resBlock(conv3_1,3,512)
			conv5_1 = resBlock(conv4_1,3,512)
			conv6_1 = resBlock(conv5_1,3,1024)
		else:
			conv2 = convLayerRelu(conv,5,128,2)

			conv3 = convLayerRelu(conv2,5,256,2)
			conv3_1 = convLayerRelu(conv3,3,256,1)

			conv4 = convLayerRelu(conv3_1,3,512,2)
			conv4_1 = convLayerRelu(conv4,3,512,1)

			conv5 = convLayerRelu(conv4_1,3,512,2)
			conv5_1 = convLayerRelu(conv5,3,512,1)

			conv6 = convLayerRelu(conv5_1,3,1024,2)
			conv6_1 = convLayerRelu(conv6,3,1024,1)

		# expansive
		with tf.variable_scope(None,default_name="predict_flow6"):
			predict_flow6 = convLayer(conv6_1,3,2,1)

		deconv5 = deconvLayerRelu(conv6_1,4,512,2)
		concat5 = flowRefinementConcat(deconv5,conv5_1,predict_flow6)
		with tf.variable_scope(None,default_name="predict_flow5"):
			predict_flow5 = convLayer(concat5,3,2,1)

		deconv4 = deconvLayerRelu(concat5,4,256,2)
		concat4 = flowRefinementConcat(deconv4,conv4_1,predict_flow5)
		with tf.variable_scope(None,default_name="predict_flow4"):
			predict_flow4 = convLayer(concat4,3,2,1)

		deconv3 = deconvLayerRelu(concat4,4,128,2)
		concat3 = flowRefinementConcat(deconv3,conv3_1,predict_flow4)
		with tf.variable_scope(None,default_name="predict_flow3"):
			predict_flow3 = convLayer(concat3,3,2,1)

		deconv2 = deconvLayerRelu(concat3,4,64,2)
		concat2 = flowRefinementConcat(deconv2,conv2,predict_flow3)
		with tf.variable_scope(None,default_name="predict_flow2"):
			predict_flow2 = convLayer(concat2,3,2,1)

		deconv1 = deconvLayerRelu(concat2,4,32,2)
		concat1 = flowRefinementConcat(deconv1,conv,predict_flow2)
		with tf.variable_scope(None,default_name="predict_flow1"):
			predict_flow1 = convLayer(concat1,3,2,1)

		deconv0 = deconvLayerRelu(concat1,4,16,2)
		concat0 = flowRefinementConcat(deconv0,inputs,predict_flow1)
		with tf.variable_scope(None,default_name="predict_flow0"):
			predict_flow0 = convLayer(concat0,3,2,1)

		predict_flow0 = predict_flow0 * self.flowScale

		self.flow0 = predict_flow0
		self.flows = [predict_flow0]
