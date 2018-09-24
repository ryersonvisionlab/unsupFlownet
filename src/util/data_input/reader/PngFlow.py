from DataReader import *

"""
maybe we should expose the valid pixel mask??
"""

class PngFlow(DataReader):
	def __init__(self,dataset_root,data_list_path):
		with tf.variable_scope(None,default_name="png_flow_data_reader"):
			DataReader.__init__(self,dataset_root,data_list_path)

			#graph setup
			flow_path = tf.placeholder(dtype=tf.string)
			png = tf.image.decode_png(tf.read_file(flow_path),dtype=tf.uint16, channels=3)
			png = tf.cast(png,tf.float32)
			flow_png = png[:,:,0:2]
			flow_mask_png = tf.expand_dims(tf.cast(tf.greater(png[:,:,2],0),tf.float32),2)

			#get h,w
			flow_shape = tf.shape(png)
			h = flow_shape[0]
			w = flow_shape[1]

			#convert flow
			flow_scale = tf.cast([[[h,w]]],tf.float32)
			flow = ((2.0/(2**16-1))*flow_png) - 1
			flow *= flow_scale
			flow *= flow_mask_png

			#expose tensors
			self.data_path = flow_path
			self.data_out = flow

			self.data_type = tf.float32
			self.data_shape = [-1,-1,2]
