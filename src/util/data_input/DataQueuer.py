#!/usr/bin/python

import tensorflow as tf
import numpy as np
import threading
from scipy import misc
from random import randint
#from components import *
import itertools
import pre_processor

class DataQueuer(object):
	"""
	Manages the concurrent fetching of data on the cpu onto the gpu
	does not guarantee that images are appended in order
	"""

	def __init__(self,data_readers, pre_processors=[],n_threads=1,random_fetch = True):
		self.n_threads = n_threads
		self.random_fetch = random_fetch
		self.data_readers = data_readers

		with tf.variable_scope(None,default_name="queuer"):
			#ensure all readers have the same number of data
			self.n_data = data_readers[0].n_data
			self.next_data = 0
			for reader in data_readers:
				assert self.n_data == reader.n_data, "readers do not read the same number of data"

			assert len(data_readers) == len(pre_processors), "number of pre_processor sets not equal number of data readers"

			#thread lock
			self.lock = threading.Lock()

			#get outputs from readers
			data_outputs = []
			data_shapes = []
			data_types = []
			for it,reader in enumerate(data_readers):
				#pass reader through proprocessor
				processorList = pre_processors[it]

				curDataSource = reader.data_out
				curDataShape = reader.data_shape
				curDataType = reader.data_type
				for proc in processorList:
					curDataSource = proc.attach_graph(curDataSource)
					curDataShape = proc.get_data_shape(curDataShape)
					curDataType = proc.get_data_type(curDataType)
				data_outputs.append(curDataSource)
				data_shapes.append(curDataShape)
				data_types.append(curDataType)

			#queue
			self.queue = tf.FIFOQueue(shapes=data_shapes, dtypes=data_types, capacity=n_threads*8)
			self.enqueue_op = self.queue.enqueue(data_outputs)

	def close(self,sess):
		"""
		Args:
		Returns:
		"""
		close_op = self.queue.close(cancel_pending_enqueues=True)
		sess.run(close_op)
		for thread in self.threads:
			thread.join()

	def get_next_data_index(self):
		with self.lock:
			index = self.next_data
			self.next_data += 1

			if self.next_data >= self.n_data:
				self.next_data = 0

			return index

	def get_next_data_index_random(self):
		with self.lock:
			index = randint(0,self.n_data-1)

			return index

	def thread_main(self, sess):
		while True:
			if self.random_fetch:
				index = self.get_next_data_index_random()
			else:
				index = self.get_next_data_index()

			#get feed dicts from readers
			feed_dict = {}
			for reader in self.data_readers:
				feed_dict.update(reader.get_feed_dict(index))

			try:
				sess.run([self.enqueue_op],feed_dict=feed_dict)
			except tf.errors.CancelledError:
				exit()

	def start_queueing(self,sess):
		self.threads = []
		for i in range(self.n_threads):
			thread = threading.Thread(target=self.thread_main, args=(sess,))
			thread.daemon = True
			self.threads.append(thread)
			thread.start()
