from matplotlib.pyplot import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import *
from pdb import set_trace
import gym
from gym import Env
import os

from env_baseline import visRL
from optical_dqn_rgb import optical_dqn_rgb
from DQN_visrl_patchinput import DeepQNetwork
# 

# Set hyperparameters
from collections import OrderedDict
log = OrderedDict(
    FIG_PATH    = './objects/',
    BACKGROUND  = None,
    N_OBSTACLES = 2,
    N_FEATURES  = 0,
    MAP_SIZE    = 8,
    GRAY_IMG    = False,
    LEARNING_RATE   = 0.00005,
    LEARNING_RATE_MODULATOR = None,
    REWARD_DECAY    = 0.9,
    REPLACE_TARGET_ITER = 6000,
    E_GREEDY_INCREMENT  = 0.1/9e4,
    FIX_MOD    = False,
    FIX_NN     = False,
    UNITCHANGE  = 1e-4,
    MOD_RESOLUTION = 40,
    CNN_ACTIVATION = 'relu'
    )

def get_env_agent(params):
    env = visRL(
        figpath=params['FIG_PATH'],
        n_obstacles=params['N_OBSTACLES'], 
        gray_img=params['GRAY_IMG'], 
        map_size=params['MAP_SIZE'], 
        n_random_features=params['N_FEATURES'],
        bkg_path=params['BACKGROUND']
    )
    n_input = env.observation_space.low.shape
    
    #agent = DeepQNetwork(
    agent = optical_dqn_rgb(
        unitchange=params["UNITCHANGE"],
        n_actions = 5,
        n_input   = n_input,
        #[256,128],
        learning_rate=params["LEARNING_RATE"],#0.00001, #0.0005,
        learning_rate_mod = params["LEARNING_RATE_MODULATOR"],
        e_greedy_increment=params["E_GREEDY_INCREMENT"],#*5e-3,
        reward_decay=params["REWARD_DECAY"],
        replace_target_iter=params["REPLACE_TARGET_ITER"],
        batch_size=32,
        memory_max=5000,#10000
        fix_mod = params["FIX_MOD"],
        fix_nn  = params["FIX_NN"],
        mod_resolution = params["MOD_RESOLUTION"],
        cnn_activation = params["CNN_ACTIVATION"]
        )
    return env, agent

def scheme1(env, agent, args):
    run_her(env,agent, n_episodes=1,train_steps=0,test_intv=0,HER=True, log_dir=args.path) # collect data only
    run_her(env, agent,  n_episodes=500, test_intv=args.test_interval, savepath=args.path_model, log_dir=args.path, save_model_intv=args.save_model_intv) # HER

if __name__ == '__main__':
    import time
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', '--gpu', default=None, action='store', help='Specify the GPU to be used.')
    parser.add_argument('-p', '--path', default='./log/', action='store', help='Training log directory.')
    parser.add_argument('-m', '--path_model', default=None, action='store', help='Path to save models. If None, will not save the trained model.')
    parser.add_argument('-bkg', '--background', default='./objects/bkg7.png', action='store', help='Path of environment background image. If None, will use default.')
    parser.add_argument('-unit', '--unitchange', default=0.0001, type=float, action='store', help='.')
    parser.add_argument('-size', '--mapsize', default=8, type=int, action='store', help='.')
    parser.add_argument('-n_obs', '--n_obstacles', default=2, type=int, action='store', help='Number of obstacles in the game play.')
    parser.add_argument('-n_f', '--n_features', default=0, type=int, action='store', help='Number of random features on the map irrelevant to game play.')
    parser.add_argument('-gray', '--gray_img', default=False, action='store_true', help='If True, the environment outputs gray-scale image.')
    parser.add_argument('-lr', '--learning_rate', default=2e-5, type=float, action='store', help='.')
    parser.add_argument('-lr_mod', '--learning_rate_mod', default=[5e-6, 10000, 0.96],#[1e-5, 1e4, 0.5], 
        nargs='+', type=float, action='store', help='.')
    parser.add_argument('-fix_mod', '--fix_mod', default=False, action='store_true', help='If True, will fix the modulator and only train the NN part.')
    parser.add_argument('-fix_nn',  '--fix_nn',  default=False, action='store_true', help='If True, will fix the NN and only train the modulator part.')
    parser.add_argument('-gamma', '--gamma', default=0.7, type=float, action='store', help='.')
    parser.add_argument('-eps_inc', '--eps_increment', default=0.1/9e4, type=float, action='store', help='')

    parser.add_argument('-cnn_activation', '--cnn_activation', default='relu', action='store', help='Activation function of DQN network')
    parser.add_argument('-scheme', '--scheme', default='1', action='store', help='Training scheme')
    parser.add_argument('-test_intv', '--test_interval', default=0, type=int, action='store', help='If positive, will test the model along training.')
    parser.add_argument('-restore', '--restore_path', default=None, action='store', help='If provided, will restore model before training.')
    parser.add_argument('-reset_mod', '--reset_mod', default=False, action='store_true', 
        help='''
        Used in fix_nn case study. If true, will reset the modulator to flat surface.
        This parameter is not saved in note.csv
        '''
    )
    parser.add_argument('-qupdate', '--replace_target_iter', default=6000, type=int, action='store', help="The period to replace Q' network")
    parser.add_argument('-save_intv', '--save_model_intv', default=1000, type=int, action='store', help='If it is <=0 will not save models along training.')
    args = parser.parse_args()

    # set GPU
    if args.gpu: 
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # Auto-correct directory for log and model
    if os.path.basename(os.path.normpath(args.path)) != 'log': 
        args.path = os.path.join(args.path, 'log') # Auto-fill the "log"
    if args.path_model is None: 
        par_dir = os.path.normpath(os.path.join(args.path, os.pardir))
        args.path_model = os.path.join(par_dir, 'models')
    # create dirs
    os.makedirs(args.path, exist_ok=True)
    os.makedirs(args.path_model, exist_ok=True)
    # add suffix for utils function
    args.path = os.path.join(args.path, '')
    args.path_model = os.path.join(args.path_model, 'dqn')

    # Set background image if specified
    log['BACKGROUND'] = args.background
    log['GRAY_IMG'] = args.gray_img
    log["N_OBSTACLES"] = args.n_obstacles
    log['MAP_SIZE'] = args.mapsize
    log['UNITCHANGE'] = args.unitchange
    log['LEARNING_RATE'] = args.learning_rate
    log['LEARNING_RATE_MODULATOR'] = args.learning_rate_mod
    log['REPLACE_TARGET_ITER'] = args.replace_target_iter
    log['FIX_MOD'] = args.fix_mod
    log['FIX_NN'] = args.fix_nn
    log['N_FEATURES'] = args.n_features
    log['REWARD_DECAY'] = args.gamma
    log['Training_scheme'] = args.scheme
    log['test_interval'] = args.test_interval
    log['restore_path'] = args.restore_path
    log['CNN_ACTIVATION'] = args.cnn_activation
    log['E_GREEDY_INCREMENT'] = args.eps_increment
    # Write log file
    with open(args.path + "note.csv","w") as logfile:
        for key, value in log.items():
            logfile.write("{},{}\n".format(key, value))
    # record cmd line inputs
    import sys
    with open(os.path.join(args.path, 'cmd.txt'), 'a') as file:
        file.write(' '.join(sys.argv) + '\n')

    # Create env and agent
    env, agent = get_env_agent(log)
    # restore model
    if args.restore_path: 
        if args.fix_mod:
            assert os.path.splitext(args.restore_path)[1] == '.npy', "In 'fix_lens' mode, the restore_path should be a '.npy' file"
            agent.load_modulator_from_npy(args.restore_path)
        else:
            agent.restore(args.restore_path)
            import re
            step = int(re.findall("\d+", args.restore_path)[-1])
            agent.set_train_step(step)
            print('Model restored from {}. Global step {}'.format(args.restore_path, step))


    # Start timing. Print map infomation
    time_start = time.time()
    print('map size: {}. input shape: {}'.format(env.mapsize, env.observation_space.low.shape))
    # Train model
    scheme = {
        '1' : scheme1, 
        '2' : scheme2, 
        '3' : scheme3,
        'fix_mod': scheme4_fix_mod,
        'fix_nn':  scheme5_fix_nn,
        'traj_search3': scheme6_traj_search3,
        'traj_search': scheme7_traj_search,
        'fig5' : scheme8,
    }[args.scheme]
    scheme(env, agent, args)

    
    # test model
    #agent.restore("./models/dqn-9000")
    #visual_test(env, agent)
    
    runtime = time.time() - time_start
    print('Run time: {}s'.format(runtime))

