import tensorflow as tf
from components import *
import data_input

class TrainingData:
	"""
	handles queuing and preprocessing prior to and after batching
	"""
	def __init__(self,batchSize,instanceParams,shuffle=True):
		with tf.variable_scope(None,default_name="ImagePairData"):
			borderThicknessH = instanceParams["borderThicknessH"]
			borderThicknessW = instanceParams["borderThicknessW"]
			if instanceParams["dataset"] == "kitti2012" or instanceParams["dataset"] == "kitti2015":
				datasetRoot = "/home/jjyu/KITTI2012/"
				frame0Path = datasetRoot+"datalists/train_im0.txt"
				frame1Path = datasetRoot+"datalists/train_im1.txt"
				desiredHeight = 320
				desiredWidth = 1152
			elif instanceParams["dataset"] == "sintel":
				datasetRoot = "/home/jjyu/datasets/Sintel/"
				frame0Path = datasetRoot+"datalists/train_raw_im0.txt"
				frame1Path = datasetRoot+"datalists/train_raw_im1.txt"
				desiredHeight = 384
				desiredWidth = 960
			else:
				assert False, "unknown dataset: " + instanceParams["dataset"]


			# create data readers
			frame0Reader = data_input.reader.Png(datasetRoot,frame0Path,3)
			frame1Reader = data_input.reader.Png(datasetRoot,frame1Path,3)

			#create croppers since kitti images are not all the same size
			cropShape = [desiredHeight,desiredWidth]
			cropper = data_input.pre_processor.SharedCrop(cropShape,frame0Reader.data_out)

			dataReaders = [frame0Reader,frame1Reader]
			DataPreProcessors = [[cropper],[cropper]]
			self.dataQueuer = data_input.DataQueuer(dataReaders,DataPreProcessors,n_threads=batchSize*4)

			# place data into batches, order of batches matches order of datareaders
			batch = self.dataQueuer.queue.dequeue_many(batchSize)

			## queuing complete

			# mean subtraction
			mean = [[[[0.448553, 0.431021, 0.410602]]]]
			img0raw = tf.cast(batch[0],tf.float32)/255.0 - mean
			img1raw = tf.cast(batch[1],tf.float32)/255.0 - mean

			## async section done ##

			#image augmentation
			photoParam = photoAugParam(batchSize,0.7,1.3,0.2,0.9,1.1,0.7,1.5,0.00)
			imData0aug = photoAug(img0raw,photoParam) - mean
			imData1aug = photoAug(img1raw,photoParam) - mean

			# artificial border augmentation
			borderMask = validPixelMask(tf.stack([1, \
				img0raw.get_shape()[1], \
				img0raw.get_shape()[2], \
				1]),borderThicknessH,borderThicknessW)

			imData0aug *= borderMask
			imData1aug *= borderMask

			#LRN skipped
			lrn0 = tf.nn.local_response_normalization(img0raw,depth_radius=2,alpha=(1.0/1.0),beta=0.7,bias=1)
			lrn1 = tf.nn.local_response_normalization(img1raw,depth_radius=2,alpha=(1.0/1.0),beta=0.7,bias=1)

			#gradient images
			imData0Gray = rgbToGray(img0raw)
			imData1Gray = rgbToGray(img1raw)

			imData0Grad = gradientFromGray(imData0Gray)
			imData1Grad = gradientFromGray(imData1Gray)

			# ----------expose tensors-----------

			self.frame0 = {
				"rgb": imData0aug,
				"rgbNorm": lrn0,
				"grad": imData0Grad
			}

			self.frame1 = {
				"rgb": imData1aug,
				"rgbNorm": lrn1,
				"grad": imData1Grad
			}

			self.validMask = borderMask
