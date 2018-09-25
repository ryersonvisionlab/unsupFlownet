from DataPreProcessor import *

class UniqueCrop(DataPreProcessor):
	'''
	crops every image it processes differently
	allows different dimensions for every image
	expects 3d tensor, single image
	'''
	def __init__(self,crop_shape,distribution="uniform"):
		_VALID_DISTRIBUTIONS = ["uniform"]

		assert distribution in _VALID_DISTRIBUTIONS, "Invalid distribution specified"
		self.distribution = distribution # only uniform for now!

		assert len(crop_shape) == 2
		self.h = crop_shape[0]
		self.w = crop_shape[1]

	def attach_graph(self,data_in):
		with tf.variable_scope(None,default_name="unique_crop"):
			# get crop range
			dataShape = tf.shape(data_in)
			h = dataShape[0]
			w = dataShape[1]

			max_h_offset = h - self.h
			max_w_offset = w - self.w

			# generate crop, unifrom
			rand_h = tf.random_uniform([],0,max_h_offset,dtype=tf.int32)
			rand_w = tf.random_uniform([],0,max_w_offset,dtype=tf.int32)

			out = tf.slice(data_in,[rand_h,rand_w,0],[self.h,self.w,-1])

			return out

	def get_data_shape(self,data_shape_in):
		return [self.crop_h,self.crop_w,data_shape_in[2]]

	def get_data_type(self,data_type_in):
		return data_type_in
