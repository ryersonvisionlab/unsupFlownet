import os

def checkResume(snapshotPath, logPath, cmdArgs):
	"""
	managed snapshots and training resuming
	"""
	snapshotFiles = os.listdir(snapshotPath)
	snapshotFiles = [filename for filename in snapshotFiles if filename[-11:] == ".ckpt.index"]
	snapshotFiles.sort()

	resume = False
	startIteration = 0

	if len(snapshotFiles) > 0:
		choice = ""

		"if resume flag set, predetermine resume"
		if cmdArgs.resume:
			choice = "y"

		while choice != "y" and choice != "n":
			print "Snapshot files detected ("+snapshotFiles[-1]+") would you like to resume? (y/n)"
			print "Old snapshots and logs will be removed if not resuming!"
			choice = raw_input().lower()

		if choice == "y":
			resume = True
			startIteration = int(snapshotFiles[-1][5:-11])
			print "resuming from iteration " + str(startIteration)
		else:
			print "removing old snapshots and logs, training from scratch"
			resume = False
			snapshotFiles = os.listdir(snapshotPath)
			logFiles = os.listdir(logPath)
			for file in snapshotFiles:
				os.remove(snapshotPath+file)
			for file in logFiles:
				os.remove(logPath+file)
	else:
		print "No snapshots found, training from scratch"

	return resume, startIteration, snapshotFiles
