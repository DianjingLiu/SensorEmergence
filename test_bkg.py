"""
This document is to construct a more complex background image, while keeping the objects recognizable
"""
from env_baseline import *
import matplotlib.pyplot as plt
from PIL import Image
env = visRL(n_obstacles=3)
env.reset()
bkg = Image.open('./objects/bkg.png')
def add_grid(img, ngrid=[8,8], grid_color = [0,0,0]):
    img = np.array(img)
    dx = img.shape[0]//ngrid[0]
    dy = img.shape[1]//ngrid[1]
    img[:,::dy,:] = grid_color
    img[::dx,:,:] = grid_color
    return Image.fromarray(img)
############## Processing object images ###################
def reduce(img, ratio):
    size = np.array(img.size)
    size = (size*ratio).astype('int')
    img = img.resize(size)
    return img
def trim(im): # crop margin
    from PIL import Image, ImageChops
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
def add_element(fig):
    # load objects and remove margin
    path = "./objects"
    bush1   = trim( Image.open(os.path.join(path, "bush.png"))      )
    animal  = trim( Image.open(os.path.join(path, "Object_1.png"))  )
    rock    = trim( Image.open(os.path.join(path, "Object_2.png"))  )
    bush2   = trim( Image.open(os.path.join(path, "Object_6.png"))  )
    grass1  = trim( Image.open(os.path.join(path, "Object_9.png"))  )
    grass2  = trim( Image.open(os.path.join(path, "Object_10.png")) )
    grass3  = trim( Image.open(os.path.join(path, "Object_11.png")) )
    grass4  = trim( Image.open(os.path.join(path, "Object_12.png")) )
    bush3   = trim( Image.open(os.path.join(path, "Object_13.png")) )
    bush4   = trim( Image.open(os.path.join(path, "Object_14.png")) )
    # adjust img size
    bush1  = reduce(bush1 , 1)
    animal = reduce(animal, 0.3)
    rock   = reduce(rock  , 0.3)
    bush2  = reduce(bush2 , 0.3)
    grass1 = reduce(grass1, 0.3)
    grass2 = reduce(grass2, 0.2)
    grass3 = reduce(grass3, 0.2)
    grass4 = reduce(grass4, 0.2)
    bush3  = reduce(bush3 , 0.2)
    bush4  = reduce(bush4 , 0.2)
    # set object offset and position
    window = np.array(fig.size)//8
    shift ={
        "bush1 " : np.array([ 0, 0 ]),
        "animal" : np.array([ 0, 54 ]),
        "rock  " : np.array([ 20, 40 ]),
        "bush2 " : np.array([ 54, 54 ]),
        "grass1" : np.array([ 54, 54 ]),
        "grass2" : np.array([ 0, 0 ]),
        "grass3" : np.array([ -10, -10 ]),
        "grass4" : np.array([ 0, 0 ]),
        "bush3 " : np.array([ 40, 40 ]),
        "bush4 " : np.array([ -15, -15 ]),
    }
    grid = {
        "bush1 " : np.array([ 1, 1 ]),
        "animal" : np.array([ 5, 4 ]),
        "rock  " : np.array([ 5, 4 ]),
        "bush2 " : np.array([ 0, 2 ]),
        "grass1" : np.array([ 5, 4 ]),
        "grass2" : np.array([ 2, 6 ]),
        "grass3" : np.array([ 4, 2 ]),
        "grass4" : np.array([ 5, 4 ]),
        "bush3 " : np.array([ 3, 3 ]),
        "bush4 " : np.array([ 5, 2 ]),
    }
    position = {}
    for element in shift.keys():
        position[element] = tuple( shift[element] + np.multiply(grid[element], window) )
    # paste to background
    fig.paste(bush1 , position['bush1 '], bush1 )
    fig.paste(rock  , position['rock  '], rock  )
    fig.paste(animal, position['animal'], animal)
    fig.paste(bush2 , position['bush2 '], bush2 )
    fig.paste(grass1, position['grass1'], grass1)
    fig.paste(grass2, position['grass2'], grass2)
    fig.paste(grass3, position['grass3'], grass3)
    fig.paste(grass4, position['grass4'], grass4)
    fig.paste(bush3 , position['bush3 '], bush3 )
    fig.paste(bush4 , position['bush4 '], bush4 )
    
    #set_trace()

add_element(bkg)
bkg.save('./objects/bkg2.png')
env.bkg = bkg
bkg = env.reset_map()
bkg = add_grid(bkg)
bkg.show() 
bkg = bkg.resize((64,64))
bkg.show()