import tensorflow as tf
from components import *


def smoothLoss2ndMaskCorrection(validMask):
	"""
	makes correct mask for smoothness based on a valid pixel mask
	if any invalid pixel is within the inclusion kernel, ignore
	"""

	inclusionKernel = tf.transpose(tf.constant([\
		[ \
			[ \
				[0,0,0,0,0],\
				[0,0,0,0,0],\
				[0,0,1,1,1],\
				[0,0,1,0,0],\
				[0,0,1,0,0]\
			] \
		] \
	],dtype=tf.float32),perm=[3,2,1,0])

	maskCor = tf.nn.conv2d(validMask,inclusionKernel,[1,1,1,1],padding="SAME")
	maskCor = tf.greater_equal(maskCor,4.95)
	maskCor = tf.cast(maskCor,tf.float32)

	return maskCor

def smoothLoss2nd(flow,alpha,beta,validPixelMask=None,img0Grad=None,boundaryAlpha=0):
	kernel = tf.transpose(tf.constant([\
		[ \
			[ \
				[0,0,0],\
				[0,1,-1],\
				[0,0,0]\
			] \
		], \
		[ \
			[ \
				[0,0,0],\
				[0,1,0],\
				[0,-1,0]\
			] \
		] \
	],dtype=tf.float32),perm=[3,2,1,0])

	with tf.variable_scope(None,default_name="smooth2ndLoss"):
		u = tf.slice(flow,[0,0,0,0],[-1,-1,-1,1])
		v = tf.slice(flow,[0,0,0,1],[-1,-1,-1,-1])

		flowShape = flow.get_shape()

		# first order
		neighborDiffU = tf.nn.conv2d(u,kernel,[1,1,1,1],padding="SAME")
		neighborDiffV = tf.nn.conv2d(v,kernel,[1,1,1,1],padding="SAME")

		# 2nd order
		neighborDiffU_x = tf.nn.conv2d(tf.expand_dims(neighborDiffU[:,:,:,0],-1),kernel,[1,1,1,1],padding="SAME")
		neighborDiffU_y = tf.nn.conv2d(tf.expand_dims(neighborDiffU[:,:,:,1],-1),kernel,[1,1,1,1],padding="SAME")
		neighborDiffV_x = tf.nn.conv2d(tf.expand_dims(neighborDiffV[:,:,:,0],-1),kernel,[1,1,1,1],padding="SAME")
		neighborDiffV_y = tf.nn.conv2d(tf.expand_dims(neighborDiffV[:,:,:,1],-1),kernel,[1,1,1,1],padding="SAME")

		diffs = tf.concat([neighborDiffU_x,neighborDiffU_y,neighborDiffV_x,neighborDiffV_y],3)
		dists = tf.reduce_sum(tf.abs(diffs),axis=3,keep_dims=True)
		robustLoss = charbonnierLoss(dists,alpha,beta,0.001)

		if not img0Grad == None:
			dMag = tf.sqrt(tf.reduce_sum(img0Grad**2, axis=3, keep_dims=True))
			mask = tf.exp(-boundaryAlpha*dMag)
			robustLoss *= mask

			# debug
			tf.summary.image("boundaryMask", mask)

		if validPixelMask is None:
			return robustLoss
		else:
			correctedMask = smoothLoss2ndMaskCorrection(validPixelMask)
			return robustLoss*correctedMask
