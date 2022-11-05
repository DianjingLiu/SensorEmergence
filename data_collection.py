
import numpy as np
class data_collector():
    def __init__(self,n_actions=5,memory_max=10000):
        self.n_actions = n_actions
        self.memory = Memory(memory_max)
        self.epsilon = 0
    def choose_action(self, *args, **kwargs):
        return np.random.randint(0, self.n_actions)
    def store_transition(self, s, a, r, s_, done=False):
        self.memory.store(s,a,r,s_,done)
    def save_memory(self, path):
        self.memory.save(path)
        print("Memory saved to {}. # memory {}".format(path, self.memory.size))
def getID(string):
    import re
    return int(re.findall("\d+",string)[-1])
def collect(env, agent, path):
    # check the existing memory files, append new files 
    import glob
    memory_list = glob.glob(os.path.join(path, "memory*.npz"))
    if memory_list:
        idx = getID(memory_list[-1]) + 1
    else:
        idx = 0

    while agent.memory.size < agent.memory.max_size :
        run_her(env, 
            agent, 
            HER=True if args['random'] else False, 
            ep_steps=300, 
            n_episodes=1,
            train_steps=0)
    print(agent.memory.size)
    agent.save_memory(path+'memory{}.npz'.format(idx))
    from pdb import set_trace
    set_trace()

if __name__ == "__main__":
    from run import env
    from utils import run_her
    from DQN_visrl_patchinput import Memory
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_sample', default=1000, type=int, action='store', help="Model ID to be restored. If a path is provided, this input is ignored.")
    parser.add_argument('-p', '--path', default='./memory/', action='store', help="Model path to be restored.")
    parser.add_argument('-r', '--random', default=False, type=bool, action='store', help="If True, take random actions to sample data.")
    args = vars(parser.parse_args())
    os.makedirs(args['path'], exist_ok=True)
    if args['random']:
        agent = data_collector(memory_max=args['n_sample'])
    else:
        from tutor import tutor
        agent = tutor(env,memory_max=args['n_sample'])
    collect(env, agent, args['path'])