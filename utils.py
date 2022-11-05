import numpy as np
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import cv2
import os
from pdb import set_trace
import time
plt.gray() # set colormap
#######################################################################
class modulator_monitor(object):
    def __init__(self, agent):
        self.agent = agent
        self.mod = []
    def record(self):
        modulator_params = self.agent.show_modulator_params()
        self.mod.append(modulator_params)
    def plot(self, path):
        data = np.array(self.mod)
        data = np.squeeze(data)
        for i in range(len(data[0])):
            param = data[:, i]
            if np.linalg.norm(param)<0.0001: continue
            filename = os.path.join(path, 'modulator_param_{}.png'.format(i))
            plt.close()
            plt.plot(param)
            plt.savefig(filename)

class visualizer(object):
    """docstring for visualizer"""
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
    def get_info(self, target_pos=None, action=None):
        ranges = self.env.mapsize
        values = np.zeros(ranges)
        actions= np.zeros(ranges)
        qvalue = np.zeros((ranges[0], ranges[1], 5))
        if target_pos is None:
            target_pos = (ranges/2).astype(int)
        #self.env.x0 = target_pos
        self.env.reset(x0=target_pos)
        for x in range(ranges[0]):
            for y in range(ranges[1]):
                self.env.x = np.array([x,y])
                s = self.env.get_state()
                #a = self.agent.choose_action(s)
                v = self.agent.show_q(s)
                qvalue[x, y, :] = v
                if action:
                    actions[x,y] = action
                    values[x,y]  = v[0, action]
                else:
                    actions[x,y] = np.argmax(v[0])
                    values[x,y]  = np.max(v)
        return values, actions, qvalue
    def visualize(self, target_pos=None, action=None):
        v,a, q = self.get_info(target_pos, action)
        self.plot(v,a)
        return v, a, q
    def plot(self, values, actions):
        import matplotlib.pyplot as plt
        plt.pcolor(values)
        plt.colorbar()
        #plt.show()
def visualize_value(env,agent, path, target_pos=None, action=None):
    import matplotlib.pyplot as plt
    v, a, q = visualizer(env,agent).visualize(target_pos=target_pos, action=action)
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(os.path.split(path)[0]+ '/latest_value.png', bbox_inches='tight')
    plt.close()
    np.save(os.path.splitext(path)[0], q)
    return v, a, q
#from optical_dqn_rgb import optical_dqn_rgb
def visualize_modulator(agent,path):
    #if not type(agent) == optical_dqn or not type(agent) == optical_dqn_mm: return
    modulator = agent.show_modulator()
    plt.pcolor(modulator)
    plt.colorbar()
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(os.path.split(path)[0]+ '/latest_modulator.png', bbox_inches='tight')
    plt.close()
    # save original modulator params
    params = agent.show_modulator_params()
    path, _ = os.path.splitext(path)
    np.save(path, params)
def visualize_sensor(env, agent, path=None):
    #if not type(agent) == optical_dqn or not type(agent) == optical_dqn_mm: return
    global state_for_sensor
    try:
        state_for_sensor
    except NameError:
        state_for_sensor = env.get_state()
        # show original img
        plt.imshow(np.squeeze(state_for_sensor))
        plt.axis('off')  
        dirs, _ = os.path.split(path)
        plt.savefig(os.path.join(dirs, "original_img.png"), bbox_inches='tight')
        plt.close()
        print('get a new original_img')
    img = agent.show_sensor(state_for_sensor)
    img = np.squeeze(img)
    #plt.pcolor(img[0,:,:,-1])
    #plt.colorbar()
    plt.imshow(img)
    plt.axis('off')  
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(os.path.split(path)[0]+ '/latest_sensor.png', bbox_inches='tight')
    plt.close()
    path, _ = os.path.splitext(path)
    np.save(path, img)
def visualize_perfect_img(env, agent, path=None):
    global state_for_sensor
    try:
        state_for_sensor
    except NameError:
        state_for_sensor = env.get_state()
        # show original img
        plt.imshow(np.squeeze(state_for_sensor[0]))
        plt.axis('off')  
        dirs, _ = os.path.split(path)
        plt.savefig(os.path.join(dirs, "original_img.png"), bbox_inches='tight')
        plt.close()
        print('get a new original_img')
    img = agent.show_perfect_img(state_for_sensor)
    #plt.pcolor(img[0,:,:,-1])
    #plt.colorbar()
    plt.imshow(np.squeeze(img[0]))
    plt.axis('off')  
    plt.savefig(path, bbox_inches='tight')
    plt.close()
def visual_test(env,agent,video_path="test.mp4",n_steps=200):
    import imageio
    video_writer = imageio.get_writer(video_path, fps=10)
    agent.fixed_epsilon()
    s = env.reset()
    R = 0
    for step in range(n_steps):
        a = agent.choose_action(s)
        s1, r, done, _ = env.step(a)
        fig = env.plot()
        #fig = cv2.cvtColor(fig, cv2.COLOR_BGR2RGB)
        video_writer.append_data(fig)
        s = s1
        R += r
        if done:
            print('Episode end. Total reward: {}'.format(R))
            R = 0
            s = env.reset()
    env.close()
    video_writer.close()
def test_image(env, agent, path=None):
    s = env.get_state()
    img = np.zeros(s.shape)
    img[:, 0, :] = 1
    img[:, -1, :] = 1
    img[:, :, 0] = 1
    img[:, :, -1] = 1
    img = agent.show_sensor(img)
    
    plt.imshow(np.squeeze(img[0]))
    plt.axis('off')  
    plt.savefig(path, bbox_inches='tight')
    plt.close()
######################################################################################
def run_her(
        env, 
        agent, 
        HER=True, 
        ep_steps=300, 
        n_episodes=300,
        train_steps=None, 
        savepath=None, 
        test_intv=0, 
        log_dir='./log/', 
        save_model_intv=200 # the interval to save models and test models
):
    '''
    HER: if true, will collect the data by HER.
    n_episodes: the number of iterations
    ep_steps: the trajectory length in data collection for each iteration. If HER=True, this length is not accurate
    train_steps: number of training steps for each iteration
    savepath: path to save models
    test_intv: if >0, will test the model and record the avg score every 'test_intv' steps
    '''
    if test_intv>0: 
        from evaluate import evaluate
        if log_dir: 
            log_score = open(log_dir + 'scores.csv', 'a')
    loss_recorder = open(log_dir + 'loss.csv', 'a')
    if HER:
        print('HER')
    else:
        print("Standard replay")
    for episode in range(n_episodes):
        start_time = time.time()
        # record episode history for HER
        ep_x, ep_x0, ep_a, ep_d = [], [], [], []
        ep_obs = [] # update 2021.03.31: in the HER replay, the obstacle position was changed. We use a list eps_obs to record the obstacles and retrieve in replay
        s = env.reset()
        R = 0
        # Standard experience replay
        for step in range(ep_steps):
            a = agent.choose_action(s)
            ep_x.append(env.x) # ep_x will change after env.step()
            ep_x0.append(env.x0)
            ep_a.append(a)
            ep_obs.append(env.obstacles) # updated 2021.03.31

            s1, r, done, _ = env.step(a)
            ep_d.append(done)
            R += r
            if not HER:
                agent.store_transition(s,a,r,s1,done)
                if done: 
                    print('Episode {}. Reward {:.2f}. epsilon {:.3f}. steps {}. x0={}'.format(episode, R, agent.epsilon, step, env.x0))
                    break
            s = s1
        if HER:
            # HER
            # Sample a set of additional goals for replay
            her_initx, her_x0, her_actions, her_obs = get_goals(ep_x, ep_a, ep_obs)#, ep_steps-10)
            for idx in range(len(her_initx)):
                ep_initx = her_initx[idx]
                ep_a = her_actions[idx]
                ep_x0 = her_x0[idx]
                env.obstacles = her_obs[idx] # updated 2021.03.31. Retrieve obstacles
                s = env.reset(ep_x0, ep_initx, reset_obstacles=False)
                R = 0
                for a in ep_a:
                    s1, r, done, _ = env.step(a)
                    agent.store_transition(s,a,r,s1,done)
                    s = s1
                    R += r
                    if done:
                        if ep_steps<1000: print(' HER: Reward {:.2f}. x0={}'.format(R, env.x0)) # If too many data to collect, do not print the collection process
                        break
        del ep_x, ep_x0, ep_a, ep_d, ep_obs
        time1 = time.time()
        if ep_steps>0: print("Collction finish. Run time {:.1f}s".format(time1-start_time))
        # Train
        if train_steps is None:
            train_steps = ep_steps #* 3
        
        for i in range(train_steps):
            loss = agent.learn()
            #'''
            global_steps = agent.train_step()
            if global_steps % 10 == 0:
                loss_recorder.write("{},{}\n".format(global_steps, loss))
            if log_dir and (global_steps % 200 == 0 or global_steps==1):
                visualize_value(env, agent,  log_dir + 'qvalue{:03d}.png'.format(global_steps))
                visualize_modulator(agent,   log_dir + 'modulator{:03d}.png'.format(global_steps))
                visualize_sensor(env, agent, log_dir + 'sensor{:03d}.png'.format(global_steps))
                
            if save_model_intv > 0 and global_steps % save_model_intv == 0:
                agent.save(savepath)
                print("model saved to "+savepath+"_{}.".format(global_steps))
            if test_intv>0 and global_steps % test_intv == 0:
                epsilon = agent.epsilon
                agent.epsilon = 0.8
                score = evaluate(env, agent, verbose=False, criterian='success')
                agent.epsilon = epsilon
                log_score.write("{},{}\n".format(global_steps, score))
                print("model test score {:.3f}".format(score))
        if train_steps>0: print("Training finish, step {}. Run time {:.1f}s, epsilon {}".format(global_steps, time.time()-time1, agent.epsilon ))
    if test_intv>0 and log_dir: 
        log_score.close()
    loss_recorder.close()
            #'''
            
def collect_transitions(env, agent, n=1000):
    a = 4
    for i in range(n):
        s = env.reset()
        s1, r, done, _ = env.step(a)
        agent.store_transition(s,a,r,s1,done)
    return agent
def collect_transitions_def(env, agent, *args, **kwargs):
    a = 4
    n = 5 # map size n * n
    loc = []
    for i in range(n):
        for j in range(n):
            loc.append([i,j])
    #loc.append([0,0]) # debug: boosting corner case
    #loc.append([0,0])
    for x0 in loc:
        for x in loc:
            s = env.reset(x0, x)
            s1, r, done, _ = env.step(a)
            agent.store_transition(s,a,r,s1,done)
    return agent
        
def get_goals(ep_x, ep_a, ep_obs, ep_len=None):
    '''
    Returns the initial x and x0, actions 
    '''
    #if ep_len is None: ep_len = np.random.randint(70,100)
    if ep_len is None: ep_len = np.random.randint(140,200)
    her_initx, her_x0, her_actions = [], [], []
    her_obs = []
    idx = 0
    actions = []
    # If episode is very short, no need to do HER
    if len(ep_x)-3-ep_len < 1: # To be safe, we waste a few transitions
        return [], [], [], []
    while idx < len(ep_x)-3-ep_len:
        init_x = ep_x[idx]
        actions = ep_a[idx : idx+ep_len]
        #init_x = ep_x[0]
        #actions = ep_a[0 : idx+ep_len]

        # Change the last action as mining at the target place
        # Note that the episode may end before 'ep_len' steps.
        actions.append(4)
        # Set the goal (x0) as the final place
        x0 = ep_x[idx+ep_len]
        obstacles = ep_obs[idx+ep_len]
        # Record initial x, actions and final place
        her_initx.append(init_x)
        her_actions.append(actions)
        her_x0.append(x0)
        her_obs.append(obstacles)
        # Next HER episode
        idx += ep_len
    return her_initx, her_x0, her_actions, her_obs


def pretrain(env, agent, savepath):
    print('Pretrain')
    n = 5000
    mb_img, mb_pos = [], []
    monitor = modulator_monitor(agent)
    for _ in range(n):
        #img = env.reset()
        env.reset()
        if np.random.rand()>0.5: env.mining = True
        img = env.get_state()
        pos = env.get_state_pos()
        mb_img.append(img[0])
        mb_pos.append(pos)
    mb_img = np.atleast_2d(mb_img)
    mb_pos = np.atleast_2d(mb_pos)
    for step in range(15000):
        idx = np.random.choice(n, size=32)
        batch_img = mb_img[idx]
        batch_pos = mb_pos[idx]
        cost = agent.pretrain(batch_img, batch_pos)
        if step % 100 == 0:
            print('step {}. loss: {:f}'.format(step, cost))
            visualize_sensor(env,agent, './log/pre_sensor{:03d}.png'.format(step//100))
            visualize_modulator(agent, './log/pre_modulator{:03d}.png'.format(step//100))
            monitor.record()
    monitor.plot('./log/')
    print('Test:')
    batch_img = mb_img[0:10]
    batch_pos = mb_pos[0:10]
    prediction = agent.sess.run(agent.pretrain_pos, feed_dict={agent.s:batch_img})
    print(prediction)
    print('-------------')
    print(batch_pos)
    if savepath:
        agent.save_pretrain(savepath)

def visualize_log(dirs, savepath=None):
    import pandas as pd
    try:
        df = pd.read_csv(dirs, header=None)
    except FileNotFoundError:
        return
    plt.subplot(2,1,1)
    params = [7,11,12,13,17]
    for i in params:
        plt.plot(df[i])
    plt.legend(['1','2','3','4','5'])
    plt.subplot(2,1,2)
    for i in params:
        plt.plot(df[i+25])
    plt.legend(['1','2','3','4','5'])
    if savepath: 
        plt.savefig(savepath)
    else:
        plt.show()
    plt.close()
    return np.array(df)