import tensorflow as tf
from components import *
from photoLoss import *
from gradLoss import *
from smoothLoss import *
from smoothLoss2nd import *
from asymmetricSmoothLoss import *

def unsupFlowLoss(flow,flowB,frame0,frame1,validPixelMask,instanceParams):
	with tf.variable_scope(None,default_name="unsupFlowLoss"):
		# hyperparams
		photoAlpha = instanceParams["photoParams"]["robustness"]
		photoBeta = instanceParams["photoParams"]["scale"]

		smoothReg = instanceParams["smoothParams"]["weight"]

		smooth2ndReg = instanceParams["smooth2ndParams"]["weight"]
		smooth2ndAlpha = instanceParams["smooth2ndParams"]["robustness"]
		smooth2ndBeta = instanceParams["smooth2ndParams"]["scale"]

		gradReg = instanceParams["gradParams"]["weight"]
		gradAlpha = instanceParams["gradParams"]["robustness"]
		gradBeta = instanceParams["gradParams"]["scale"]

		boundaryAlpha = instanceParams["boundaryAlpha"]
		lossComponents = instanceParams["lossComponents"]

		# helpers
		rgb0 = frame0["rgbNorm"]
		rgb1 = frame1["rgbNorm"]
		grad0 = frame0["grad"]
		grad1 = frame1["grad"]

		# masking from simple occlusion/border
		occMask = borderOcclusionMask(flow) # occ if goes off image
		occInvalidMask = validPixelMask*occMask # occluded and invalid

		# loss components
		photo = photoLoss(flow,rgb0,rgb1,photoAlpha,photoBeta)
		grad = gradLoss(flow,grad0,grad1,gradAlpha,gradBeta)
		imgGrad = None
		if lossComponents["boundaries"]:
			imgGrad = grad0

		smoothMasked = asymmetricSmoothLoss(flow,instanceParams,occMask,validPixelMask,imgGrad,boundaryAlpha)
		smooth2ndMasked = smoothLoss2nd(flow,smooth2ndAlpha,smooth2ndBeta,validPixelMask,imgGrad,boundaryAlpha)

		# apply masking
		photoMasked = photo * occInvalidMask
		gradMasked = grad * occInvalidMask

		# average spatially
		photoAvg = tf.reduce_mean(photoMasked,reduction_indices=[1,2])
		gradAvg = tf.reduce_mean(gradMasked,reduction_indices=[1,2])
		smoothAvg = tf.reduce_mean(smoothMasked,reduction_indices=[1,2])
		smooth2ndAvg = tf.reduce_mean(smooth2ndMasked,reduction_indices=[1,2])

		# weight loss terms
		gradAvg = gradAvg*gradReg
		smoothAvg = smoothAvg*smoothReg
		smooth2ndAvg = smooth2ndAvg*smooth2ndReg

		# summaries ----------------------------
		tf.summary.scalar("photoLoss",tf.reduce_mean(photoAvg))
		tf.summary.scalar("smoothLoss",tf.reduce_mean(smoothAvg))

		# final loss
		finalLoss = photoAvg + smoothAvg
		if lossComponents["smooth2nd"]:
			tf.summary.scalar("smooth2ndLoss",tf.reduce_mean(smooth2ndAvg))
			finalLoss += smooth2ndAvg
		if lossComponents["gradient"]:
			tf.summary.scalar("gradLoss",tf.reduce_mean(gradAvg))
			finalLoss += gradAvg
		return finalLoss
