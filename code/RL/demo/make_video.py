import glob
import imageio
import re
import matplotlib.pyplot as plt
import cv2
def getID(string):
	return int(re.findall("\d+",string)[-1])
if __name__ == "__main__":
	video_path = "demo.mp4"
	fps = 1
	files = glob.glob("[0-9].png")
	images = []
	for name in files:
		print(name)
		img = imageio.imread(name)
		images.append(img)
	start_img = images[0]
	images = [start_img] * fps + images
	end_img = images[-1]
	images = images + [end_img] * fps*6
	imageio.mimsave(video_path, images, fps=fps)