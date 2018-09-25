
def learningRateSchedule(baseLR, iteration):
	"""
	returns learning rate given training iteration
	"""
	if iteration > 500000:
		return baseLR/8
	elif iteration > 400000:
		return baseLR/4
	elif iteration > 300000:
		return baseLR/2
	elif iteration > 200000:
		return baseLR
	elif iteration > 100000:
		return baseLR
	else:
		return baseLR
