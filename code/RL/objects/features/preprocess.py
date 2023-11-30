from PIL import Image, ImageChops
import numpy as np
"""
remove margin and resize to reasonable size
"""
def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
def scale(img, ratio):
    size = np.array(img.size)
    size = (size*ratio).astype('int')
    img = img.resize(size)
    return img

def process():
    import glob
    img_list = glob.glob("*.png")
    for f in img_list:  
        img = Image.open(f)
        img = trim(img)
        ratio = np.random.rand()*0.4 + 0.2
        print(ratio)
        if f!='bush.png':img = scale(img, ratio)
        img.save(f)

#process()

def process_target_feature():
    #import glob
    #img_list = glob.glob("target.png")
    img = Image.open("../target.png")
    img = trim(img)
    ratios = np.random.uniform(0.1, 0.6, size=20)
    for idx, r in enumerate(ratios):
        print(idx, r)
        scale(img, r).save(f"feature_{idx}.png")

process_target_feature()