from DataPreProcessor import *

class SharedCrop(DataPreProcessor):
	'''
	crops every image it processes the same
	every image sharing the crop must be the same dimension
	expects 3d tensor, single image
	'''
	def __init__(self,crop_shape,ref_data,distribution="uniform"):
		_VALID_DISTRIBUTIONS = ["uniform"]

		assert distribution in _VALID_DISTRIBUTIONS, "Invalid distribution specified"
		self.distribution = distribution # only uniform for now!

		assert len(crop_shape) == 2
		self.crop_h = crop_shape[0]
		self.crop_w = crop_shape[1]

		ref_shape = tf.shape(ref_data)
		self.in_h = ref_shape[0]
		self.in_w = ref_shape[1]

		#create shared offset values
		with tf.variable_scope(None,default_name="random_crop_pos"):
			max_h_offset = self.in_h - self.crop_h
			max_w_offset = self.in_w - self.crop_w

			# generate crop, unifrom
			rand_h = tf.random_uniform([],0,max_h_offset,dtype=tf.int32)
			rand_w = tf.random_uniform([],0,max_w_offset,dtype=tf.int32)

		# expose tensors
		self.rand_h = rand_h
		self.rand_w = rand_w

	def attach_graph(self,dataIn):
		with tf.variable_scope(None,default_name="shared_crop"):
			"""
			Todo: assert data shape
			"""
			dataShape = tf.shape(dataIn)
			inH = dataShape[0]
			inW = dataShape[1]

			tf.assert_equal(inH,self.in_h,message="data height not equal to reference data height")
			tf.assert_equal(inW,self.in_w,message="data width not equal to reference data width")

			out = tf.slice(dataIn,[self.rand_h,self.rand_w,0],[self.crop_h,self.crop_w,-1])

			return out

	def get_data_shape(self,data_shape_in):
		return [self.crop_h,self.crop_w,data_shape_in[2]]

	def get_data_type(self,data_type_in):
		return data_type_in
