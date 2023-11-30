import sys
sys.path.append("C:\\Users\\ldj\\Documents\\MachineLearning_NN\\practice\\VisualRL\\draft\\figs\\data\\replot_temp")
from temp import *
import glob
if __name__ == "__main__":
	files = glob.glob("mod_*.png")
	for name in files:
		img = Image.open(name)
		img = trim(img)
		print(name)
		img.save(name)