# keeps a directory empty! for training with a full hard drive.
# REMEMBER TO SHUT THIS OFF BEFORE YOU SHUT TRAINING OFF!!!!
import os
from time import sleep
while True:
	files = os.listdir('.')
	for i in files:
		if "t7" in i:
			os.system("rm " + str(i))
			print "Deleted: " + str(i)
	sleep(60)
	
