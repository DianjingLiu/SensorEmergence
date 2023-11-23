import numpy as np
import gym
from gym import Env, spaces
import cv2
import skimage.measure
import random
import glob
import os
from PIL import Image
from pdb import set_trace
class visRL(Env):
    """
    The background map is compressed at initialization. This will speedup the training. If need better visulization, we need save a clear version of image
    Update 2020.3.27: add blocks to the map. When resetting the game, can choose either reset the blocks or not. 
    Update 2020.4.22: Use the new desert map. Use PIL library to add foreground objects to the map. Discard obstacle handling class.
                      Note that pasting all objects to the 512*512 map takes 0.01s. No need to speedup plotting part.
                      Note that 'env.range' is changed to 'env.mapsize'. Accordingly change this parameter name in "run.py" and "utils.py"
    """
    def __init__(self, figpath='./objects/', bkg_path=None, map_size = [8, 8], bkg_size = [64, 64], n_obstacles=0,gray_img=False, n_random_features=0):
        """
        load background image. set parameters for gameplay and visulization
        """
        if type(map_size) == int: map_size = [map_size, map_size]
        self.mapsize = np.array(map_size)
        # Load background and foreground images
        self.object_lib = self.load_objects(figpath, bkg_path)
        self._n_random_features = n_random_features # Number of random background features. If 0, will not add random features.
        self.bkg_features = self.load_bkg_features(os.path.join(figpath, 'features'))
        # Plotting parameters
        shape = list(bkg_size) + [1 if gray_img else 3] # output shape is [64,64,3]. Note that shape[1] corresponds to horizontal axis x, shape[0] corresponds to vertical axis y.
        self.windowsize = np.divide(self.bkg.size, map_size).astype(int)
        self.n_obstacles = n_obstacles
        self.output_size = tuple(bkg_size)
        self.gray_img = gray_img # If True, will output gray-scale image
        # Game play parameters
        self.max_step = 300
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low= np.zeros(shape),
                                            high=np.ones(shape), 
                                            dtype=np.float32)
        self.move = np.array([[0,-1], [0,1], [-1,0], [1,0]])
        # checker helps check if the game is playable
        self.checker = checker()
        self.reset(reset_obstacles=True)
    ############## functions for game play ########################
    def reset(self,x0=None,x=None, reset_obstacles=True, max_dist=np.inf):
        self.n_step = 0
        self.mining = False
        self.hit = False
        # Reset obstacles if necessary
        if reset_obstacles:
            avoid = [] # a list of positions to avoid when resetting obstacles
            if x  is not None: avoid.append(tuple(x))
            if x0 is not None: avoid.append(tuple(x0))
            self.reset_obstacles(self.n_obstacles, avoid=avoid)
        # Initialize reward position
        if x0 is not None:
            self.x0 = x0
            assert self.is_valid(x0)==0, "Given x0 should not overlap with preset obstacles"
        else:
            self.x0 = self.random_position()
        # Initialize agent position
        if x is not None:
            self.x = x
            assert self.is_valid(x)==0, "Given initial x should not overlap with preset obstacles"
        else:
            self.x  = self.random_position()
            while np.linalg.norm(self.x0-self.x, 1) > max_dist:
                self.x = self.random_position()
        # Check if the agent can reach the target. If yes, reset the map and return state. If not, reset again until the game is valid.
        if self.checker.check(self):
            self.map = self.reset_map()
            return self.get_state()
        else:
            return self.reset(x0, x, reset_obstacles, max_dist)
    def step(self, action):
        self.n_step += 1 
        self.mining = action==4
        if not self.mining:
            new_position = self.x + self.move[action]
            #self.x = np.clip(self.x, [0,0], self.mapsize-1)
            flag = self.is_valid(new_position) # flag: 0 -- valid; 1 -- collide with obstacles; 2 -- out of map range
            if flag==0: # valid new position
                self.x = new_position
                self.hit = False
            elif flag==1: # collide, add penalty, do not move
                self.hit = True
            else: # hit boundary, do not apply penalty
                self.hit = False
        #print(self.x, action)
        reward = self.calc_reward()
        s = self.get_state()
        done = self.if_done(reward)
        info = {}
        return s, reward, done, info

    def calc_reward(self):
        # continued reward
        #dist = np.linalg.norm(self.x0-self.x)
        dist = np.linalg.norm(self.x0-self.x, 1) # L1 norm
        if self.mining and dist == 0:
            return 5
        else:
            return 0 - self.hit*0.5

    def if_done(self, reward):
        if self.n_step>=self.max_step: 
            return True
        else:
            return abs(reward)>=5

    ############### functions handling obstacles #################
    def reset_obstacles(self, n, avoid=[]):
        # ganerate n random obstacle positions
        # avoid: a list of positions where obstacles should not exist. This is in case the given initial x and x0 overlap with randomly generated obstacles
        self.obstacles = set() # a set of obstacle positions. Each position is a tuple
        while len(self.obstacles) < n:
            new = tuple(np.random.randint(self.mapsize[0], size=2))
            if not new in avoid:
                self.obstacles.add(new)
    def clear_obstacles(self):
        self.obstacles = set()
    def is_valid(self, position):
        """
        decide if the position is valid or not. The position can be valid only if it is within the map range and does not overlap with obstacles
        input: position -- a array-like integer vector with length 2.
        return: int. If the returned flag == 0, the input position is valid; if flag==1, the position collide with obstacles; if flag=2, position out of map range
        """
        position = tuple(position)
        # check if within map range
        in_range = (0<=position[0]<self.mapsize[0]) and (0<=position[1]<self.mapsize[1])
        # check if collide with obstacles
        collide = position in self.obstacles
        if in_range and not collide:
            return 0
        elif collide:
            return 1
        else:
            return 2
    def random_position(self):
        # generate a valid random position
        while True:
            position = np.array([np.random.randint(0, self.mapsize[0]), np.random.randint(0, self.mapsize[1])])
            if self.is_valid(position)==0:
                return position
    def draw_obstacles(self, fig):
        # draw the obstacles to the input map
        for pos in self.obstacles:
            pos = np.array(pos)
            pixel = np.multiply(pos+0.5, self.windowsize).astype(int)
            pixel = tuple(pixel)
            fig = cv2.cv2.drawMarker(fig, pixel, 1, markerType=cv2.MARKER_TILTED_CROSS, markerSize=6, thickness=1, line_type=cv2.LINE_AA)
            #cv2.circle(fig, pixel, 3, color=1,thickness=2)
            #pt1, pt2 = self.get_loc(pos)
            #cv2.rectangle(fig, pt1, pt2 , color=1,thickness=1)
        return fig
    ###################### functions for plotting and visulization ####################
    """
    Note that plotting the image and resize only takes ~0.01s.
    Functions:
        load_bkg_features(): load some objects to add to the background. The objects are irrelevant to the game play.
        load_objects(): called in __init__(). It is a function to load the images of objects and their shift positions.
        get_loc() is a function to return the foreground position
        reset_map(): called in reset(). Add target and obstacles to the background, save image to self.map. 
        plot() add agent label to the self.map, returns the high resolution state image
        get_state() resize the image to low resolution for training.
        render() displays the high resolution image
    attributes:
        object_lib -- a dictionary to save all foreground objects. Including agent, target, obstacles. 
                      Each item is a tuple (image, shift), where image is the object image, shift is the 
                      relative position to help calculate the foreground location in the background map.
    """
    def load_objects(self, path, bkg_path):
        def scale(img, ratio):
            size = np.array(img.size)
            size = (size*ratio).astype('int')
            img = img.resize(size)
            return
        if bkg_path is None:
            bkg_path = os.path.join(path, "bkg7.png")
        self.bkg  = Image.open(bkg_path)
        agent     = Image.open(os.path.join(path, "agent.png"))
        #agent     = Image.open(os.path.join(path, "agent2.png"))
        target    = Image.open(os.path.join(path, "target.png"))
        #obstacle1 = Image.open(os.path.join(path, "obstacle1.png"))
        obstacle1 = Image.open(os.path.join(path, "obstacle1-1.png"))
        obstacle2 = Image.open(os.path.join(path, "obstacle2.png"))
        obstacle3 = Image.open(os.path.join(path, "obstacle3.png"))
        shifts = np.load(os.path.join(path, "shifts.npz"))
        # if map size is not 8 by 8, scale the object size accordingly
        ratio = 8/self.mapsize[0]
        if ratio != 1:
            agent     = scale(agent    , ratio)
            target    = scale(target   , ratio)
            obstacle1 = scale(obstacle1, ratio)
            obstacle2 = scale(obstacle2, ratio)
            obstacle3 = scale(obstacle3, ratio)
        lib = {"agent"    : (agent,  shifts['agent']),
               "target"   : (target, shifts['target']),
               "obstacles": [(obstacle1, shifts['obstacle1']),
                             #(obstacle2, shifts['obstacle2']),
                             #(obstacle3, shifts['obstacle3'])
                             ],
        }
        return lib
    def reset_bkg(self, path):
        self.bkg = Image.open(path).convert("RGB")
    def load_bkg_features(self, path):
        feature_list = glob.glob(os.path.join(path, '*.png'))
        features = []
        for file in feature_list:
            f = Image.open(file)
            features.append(f)
        return features
    def add_bkg_features(self, bkg, N):
        # randomly add features to the input image
        # N: number of random features to add
        for _ in range(N):
            f = random.choice(self.bkg_features)
            position = (random.randint(0, bkg.size[0]-f.size[0]), 
                        random.randint(0, bkg.size[1]-f.size[1]) )
            bkg.paste(f, position, f)
    def get_loc(self, position, shift):
        windowsize = self.windowsize
        position = np.multiply(position, self.windowsize) + np.array(shift)
        position = np.clip(position, 0, self.bkg.size[0])
        return tuple(position)
    def reset_map(self):
        fig = self.bkg.copy()
        # Add random features if necessary
        self.add_bkg_features(fig, self._n_random_features)
        # plot target
        target, shift = self.object_lib['target']
        position = self.get_loc(self.x0, shift)
        fig.paste(target, position, target)
        # plot obstacles
        for position in self.obstacles:
            obstacle, shift = random.choice(self.object_lib['obstacles'])
            position = self.get_loc(position, shift)
            fig.paste(obstacle, position, obstacle)
        return fig
    def get_state(self):
        # plot the high resolution image
        fig = self.map.copy()
        agent, shift = self.object_lib['agent']
        position = self.get_loc(self.x, shift)
        fig.paste(agent, position, agent)
        # Compress image, Convert to np array
        fig = fig.resize(self.output_size)
        if self.gray_img:
            fig = fig.convert("L")
            fig = np.array(fig).astype(np.float32) / 255
            fig = fig[None,...,None]
        else:
            fig = np.array(fig).astype(np.float32) / 255
            # Add batch dimension
            fig = fig[None,...]
        return fig
    def padding(self, fig, ratio=0.25, same=False):
        # pad white space around the output image
        # ratio: the width of the margin : figure size
        assert len(fig.shape) == 4, 'The input image shape should be [n_batch, width, height, n_channel]'
        width_origin = fig.shape[1]
        margin = int(ratio * width_origin)
        width = 2 * margin +  width_origin # we assume the figures before and after padding are square shaped
        canvas = np.ones([fig.shape[0], width, width, fig.shape[-1]])
        # yellow colored padding. the color is (240, 203, 78)
        #canvas[:, :, :, 0] *= 240/255
        #canvas[:, :, :, 1] *= 203/255
        #canvas[:, :, :, 2] *= 78/255
        if same:
            canvas[:, :, :, 0] *= fig[0,0,0,0]
            canvas[:, :, :, 1] *= fig[0,0,0,1]
            canvas[:, :, :, 2] *= fig[0,0,0,2]
        canvas[:, margin:width-margin, margin:width-margin, :] = fig
        return canvas

    def plot(self):
        # For rendering. Return the high resolution version of the map
        fig = self.map.copy()
        agent, shift = self.object_lib['agent']
        position = self.get_loc(self.x, shift)
        fig.paste(agent, position, agent)
        # convert to uint8 for rendering
        #fig = cv2.normalize(fig, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U )
        return np.array(fig)
    def render(self, debug=False):
        fig = self.get_state()#
        fig = np.squeeze(fig)
        #fig = self.plot()
        fig = cv2.cvtColor(fig, cv2.COLOR_BGR2RGB)
        fig = cv2.resize(fig, (300,300) )
        cv2.imshow('test',fig)
        if not debug:
            cv2.waitKey(1)
        else:
            return cv2.waitKey(0)
import time
class checker():
    '''
    Given the target, agent, obs_list. Check if the agent can reach the target.
    It is used in environment initialization to ensure the game is legal.
    '''
    def check(self, env):
        self.starttime = time.time()
        self.mapsize = env.mapsize
        target = np.array(env.x0)
        agent  = np.array(env.x)
        obs_list = env.obstacles.copy()
        flag, _ = self.DFS(target, agent, obs_list)
        return flag
    def DFS(self, target, agent, obs_list):
        '''
        obs_list -- a list of tuples. Each tuple is the location of an obstacle
        return: flag -- if true, the agent can reach target position
                action -- the action to take
        '''
        if time.time()-self.starttime>10: 
            print("checking time too long!!!!!!!!!!!!!!!!!!!!")
            print("obstacles {}, target {}, agent {}".format(obs_list, target, agent))
            return True, None
        if np.linalg.norm(target-agent, ord=1)==0:
            return True, 4
        in_range = (0<=agent[0]<self.mapsize[0]) and (0<=agent[1]<self.mapsize[1])
        if tuple(agent) in obs_list or not in_range:
            return False, None
        moves = np.array([[0,-1], [0,1], [-1,0], [1,0]])
        actions_candidate = self.priority(target - agent)
        obs_list.add(tuple(agent))
        for action in actions_candidate:
            neighbor = agent + moves[action]
            flag, _ = self.DFS(target, neighbor, obs_list)
            if flag:
                obs_list.remove(tuple(agent))
                return True, action
        # if no valid neighbors, return False
        obs_list.remove(tuple(agent))
        return False, None
    def priority(self, relative):
        # Utility function for DFS. Used to decide priory candidate actions for DFS.
        # Assuming moves = np.array([[0,-1], [0,1], [-1,0], [1,0]])
        priority = [None, None, None, None]
        if abs(relative[0]) >= abs(relative[1]): 
            # move along axis 0 first
            if relative[0] < 0: # need -1 for agent[0]
                priority[0] = 2 # highest priority
                priority[3] = 3 # lowest priority
            else:
                priority[0] = 3
                priority[3] = 2
            # Then try move along axis 1
            if relative[1] <= 0:
                priority[1] = 0 # target below agent, agent[1] need to -1
                priority[2] = 1
            else:
                priority[1] = 1
                priority[2] = 0
        else:
            # move along axis 1 first
            if relative[1] < 0: # need -1 for agent[1]
                priority[0] = 0 # highest priority
                priority[3] = 1 # lowest priority
            else:
                priority[0] = 1
                priority[3] = 0
            # Then try move along axis 1
            if relative[0] <= 0:
                priority[1] = 2 # target at left side of agent, agent[0] need to -1
                priority[2] = 3
            else:
                priority[1] = 3
                priority[2] = 2
        return priority
def test():
    env = visRL()
    s=env.reset()
    print(s.shape)
    k = env.render(True)
    while True:
        if k == ord('q'): break
        action = {
        ord('w'): 0,
        ord('s'): 1,
        ord('a'): 2,
        ord('d'): 3,
        ord(' '): 4,
        }[k]
        s, reward, done, info = env.step(action)
        if done: print('Done..')
        print(env.x, reward, s.shape)
        k = env.render(True)
    s=env.get_state()
    print(np.min(s), np.max(s), s.shape, s.dtype)
    cv2.imshow('test',s[0])
    cv2.waitKey(0)
    #import matplotlib.pyplot as plt
    #plt.imshow(s[0,:,:,0])
    #plt.show()
if __name__ == "__main__":
    test()
