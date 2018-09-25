
def unsupLossBSchedule(iteration):
	"""
	Schedule for weighting loss between forward and backward loss
	"""
	if iteration > 400000:
		return 0.5
	elif iteration > 300000:
		return 0.5
	elif iteration > 200000:
		return 0.5
	elif iteration > 190000:
		return 0.5
	elif iteration > 180000:
		return 0.4
	elif iteration > 170000:
		return 0.3
	elif iteration > 160000:
		return 0.2
	elif iteration > 150000:
		return 0.1
	else:
		return 0.0
