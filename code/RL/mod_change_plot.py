import numpy as np
import glob
import os
import csv

def collect(directory):
	import re
	def getID(x):
		return int(re.findall("\d+",x)[-1])
	path = os.path.join(directory, "modulator*.npy")
	files = glob.glob(path)
	files = sorted(files, key=getID)
	# log writer
	savepath = os.path.normpath(os.path.join(directory, os.pardir)) # savepath is the parent dir of log path, without 'log'
	log_mod = open(os.path.join(savepath, 'modulators.csv'), 'a')
	writer = csv.writer(log_mod, delimiter=',')
	for f in files:
		mod = np.load(f)
		mod = np.squeeze(mod).tolist()
		step = getID(f)
		mod.insert(0,step)
		writer.writerows([mod])
	log_mod.close()


if __name__ == '__main__':
    import time
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', default='/home/Data/Dianjing/lensnet_baseline_rgb_traj_search/test_base2/log/', action='store', help='log directory.')

    args = parser.parse_args()
    if os.path.basename(os.path.normpath(args.path)) != 'log': 
        args.path = os.path.join(args.path, 'log') # Auto-fill the "log" to models directory
    collect(args.path)