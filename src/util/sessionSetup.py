import tensorflow as tf

def sessionSetup(cmdArgs):
	"""
	container for tensorflow config
	"""
	# config
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	if cmdArgs.logDev:
		config.log_device_placement = True

	return tf.Session(config=config)
