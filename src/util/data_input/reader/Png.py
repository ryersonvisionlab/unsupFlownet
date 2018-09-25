from DataReader import *

class Png(DataReader):
	"""
	Manages the fetching of single peices of data on the cpu onto the gpu
	"""
	def __init__(self,dataset_root,data_list_path,channels=3,dtype=tf.uint8):
		with tf.variable_scope(None,default_name="image_data_reader"):
			DataReader.__init__(self,dataset_root,data_list_path)

			# graph setup
			img_path = tf.placeholder(dtype=tf.string)
			img = tf.image.decode_png(tf.read_file(img_path), channels=channels, dtype=dtype)

			# expose tensors
			self.data_path = img_path
			self.data_out = img
			self.data_type = dtype
			self.data_shape = [-1,-1,channels]
