import tensorflow as tf
from components import *

def smoothLossMaskCorrection(validMask):
	"""
	makes correct mask for smoothness based on a valid pixel mask
	if any invalid pixel is within the inclusion kernel, ignore
	"""

	inclusionKernel = tf.transpose(tf.constant([\
		[ \
			[ \
				[0,0,0],\
				[0,1,1],\
				[0,1,0]\
			] \
		] \
	],dtype=tf.float32),perm=[3,2,1,0])

	maskCor = tf.nn.conv2d(validMask,inclusionKernel,[1,1,1,1],padding="SAME")
	maskCor = tf.greater_equal(maskCor,2.95)
	maskCor = tf.cast(maskCor,tf.float32)

	return maskCor

def smoothLoss(flow,alpha,beta,validPixelMask=None,img0Grad=None,boundaryAlpha=0):
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

	with tf.variable_scope(None,default_name="smoothLoss"):
		u = tf.slice(flow,[0,0,0,0],[-1,-1,-1,1])
		v = tf.slice(flow,[0,0,0,1],[-1,-1,-1,-1])

		flowShape = flow.get_shape()

		neighborDiffU = tf.nn.conv2d(u,kernel,[1,1,1,1],padding="SAME")
		neighborDiffV = tf.nn.conv2d(v,kernel,[1,1,1,1],padding="SAME")

		diffs = tf.concat([neighborDiffU,neighborDiffV],3)
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
			return robustLoss*smoothLossMaskCorrection(validPixelMask)
