import tensorflow as tf
from components import *
from photoLoss import *

def gradLoss(flow,downsampledGrad0,downsampledGrad1,alpha,beta):
	"""
	like photoloss but use image gradients instead
	"""
	with tf.variable_scope(None,default_name="gradLoss"):
		return photoLoss(flow,downsampledGrad0,downsampledGrad1,alpha,beta)
