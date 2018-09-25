import tensorflow as tf
import numpy as np

class DataReader(object):
	"""
	reads in data according to a datalist of paths
	handles reading of datalist
	default: creates input graph structure in TF, feed data is only path to data file
	get_feed_dict can be overriden to add custom processing on the CPU
	"""
	def __init__(self,dataset_root,data_list_path):
		self.dataset_root = dataset_root
		self.data_list_path = data_list_path

		# read in image list
		with open(data_list_path) as f:
			self.data_list_path = [x[:-1] for x in f.readlines()]

		self.n_data = len(self.data_list_path)

		# defaults that are expected
		self.data_out = None
		self.data_type = None
		self.data_shape = None

	def get_feed_dict(self,index):
		path = 	self.dataset_root + self.data_list_path[index]
		return {self.data_path: path}
