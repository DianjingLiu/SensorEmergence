import numpy as np
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import cv2
import os
from pdb import set_trace

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
    def get_info(self, target_pos=None):
        ranges = self.env.range
        values = np.zeros(ranges)
        actions= np.zeros(ranges)
        if target_pos is None:
            target_pos = (ranges/2).astype(int)
        self.env.x0 = target_pos
        for x in range(ranges[0]):
            for y in range(ranges[1]):
                self.env.x = np.array([x,y])
                s = self.env.get_state()
                a = self.agent.choose_action(s)
                v = self.agent.show_q(s)
                actions[x,y] = a
                values[x,y]  = np.max(v)
        return values, actions
    def visualize(self, target_pos=None):
        v,a = self.get_info(target_pos)
        self.plot(v,a)
    def plot(self, values, actions):
        import matplotlib.pyplot as plt
        plt.pcolor(values)
        plt.colorbar()
        #plt.show()
def visualize_value(env,agent, path):
    import matplotlib.pyplot as plt
    visualizer(env,agent).visualize()
    plt.savefig(path, bbox_inches='tight')
    plt.close()
from optical_cnn import optical_cnn
def visualize_modulator(agent,path):
    #if not type(agent) == optical_dqn or not type(agent) == optical_dqn_mm: return
    modulator = agent.show_modulator()
    plt.pcolor(modulator, cmap='gray')
    plt.colorbar()
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    # save original modulator params
    params = agent.show_modulator_params()
    path, _ = os.path.splitext(path)
    np.save(path, params)
    return modulator
def visualize_sensor(env, agent, path=None):
    #if not type(agent) == optical_dqn or not type(agent) == optical_dqn_mm: return
    s = env.get_state()
    img = agent.show_sensor(s)
    #plt.pcolor(img[0,:,:,-1])
    #plt.colorbar()
    plt.imshow(np.squeeze(img[0]), cmap='gray')
    plt.axis('off')  
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    # show original img
    #plt.pcolor(s[0,...,-1])
    #plt.colorbar()
    plt.imshow(np.squeeze(s[0]), cmap='gray')
    plt.axis('off')  
    plt.savefig("./log/original_img.png", bbox_inches='tight')
    plt.close()
def visual_test(env,agent,video_path="test.mp4"):
    import imageio
    video_writer = imageio.get_writer(video_path, fps=10)
    agent.fixed_epsilon()
    run(env, agent, n_steps=200, learn=False, video_writer=video_writer)
    video_writer.close()

######################################################################################
def run_her(env, agent, ep_steps=300, n_episodes=300,train_steps=None, savepath=None):
    print('HER')
    for episode in range(n_episodes):
        # Visualize training
        visualize_value(env, agent, './log/qvalue{:03d}.png'.format(episode))
        visualize_modulator(agent, './log/modulator{:03d}.png'.format(episode))
        visualize_sensor(env, agent, './log/sensor{:03d}.png'.format(episode))
        if savepath and episode % 10 == 0:
            agent.save(savepath)
        # record episode history for HER
        ep_x, ep_x0, ep_a, ep_d = [], [], [], []
        s = env.reset()
        R = 0
        # Standard experience replay
        for step in range(ep_steps):
            a = agent.choose_action(s)
            ep_x.append(env.x) # ep_x will change after env.step()
            ep_x0.append(env.x0)
            ep_a.append(a)

            s1, r, done, _ = env.step(a)
            ep_d.append(done)
            #agent.store_transition(s,a,r,s1,done)
            s = s1
            R += r
            #if done:
            #    break
        print('Episode {}. Reward {:.2f}. epsilon {:.3f}. steps {}. x0={}'.format(episode, R, agent.epsilon, step, env.x0))
        # Sample a set of additional goals for replay
        her_initx, her_x0, her_actions = get_goals(ep_x, ep_a)#, ep_steps-10)
        # Revise: if episode end with reward, skip HER
        #if done: 
        #    her_initx, her_x0, her_actions = [], [], []
        # HER
        for idx in range(len(her_initx)):
            ep_initx = her_initx[idx]
            ep_a = her_actions[idx]
            ep_x0 = her_x0[idx]
            s = env.reset(ep_x0, ep_initx)
            R = 0
            for a in ep_a:
                s1, r, done, _ = env.step(a)
                agent.store_transition(s,a,r,s1,done)
                s = s1
                R += r
                if done:
                    print(' HER: Reward {}. x0={}'.format(R, env.x0))
                    break
        # Train
        train_steps = step * 3
        for _ in range(train_steps):
            agent.learn()
        
def get_goals(ep_x, ep_a, ep_len=None):
    '''
    Returns the initial x and x0, actions 
    '''
    if ep_len is None: ep_len = np.random.randint(70,100)
    #if ep_len is None: ep_len = np.random.randint(250,295)
    her_initx, her_x0, her_actions = [], [], []
    idx = 0
    actions = []
    # If episode is very short, no need to do HER
    if len(ep_x)-3-ep_len < 1: # To be safe, we waste a few transitions
        return [], [], []
    while idx < len(ep_x)-3-ep_len:
        #init_x = ep_x[idx]
        #actions = ep_a[idx : idx+ep_len]
        init_x = ep_x[0]
        actions = ep_a[0 : idx+ep_len]

        # Change the last action as mining at the target place
        # Note that the episode may end before 'ep_len' steps.
        actions.append(4)
        # Set the goal (x0) as the final place
        x0 = ep_x[idx+ep_len]
        # Record initial x, actions and final place
        her_initx.append(init_x)
        her_actions.append(actions)
        her_x0.append(x0)
        # Next HER episode
        idx += ep_len
    return her_initx, her_x0, her_actions


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
def train(env,agent,savepath,plt_path=None,learn=True):
    print('Standard replay')
    mb_s, mb_a, mb_r, mb_done, ep_r = run(env, agent, savepath, early_stop=True)
    #np.savez('train_data.npz', s=mb_s, a=mb_a, r=mb_r, done=mb_done)
    if plt_path: plt_r(ep_r, plt_path)

def run(env, agent, savepath=None, n_steps=np.inf, n_episodes=300, learn = True, dummytrain=False, early_stop=False, video_writer=None):
    mb_s, mb_a, mb_r, mb_done = [], [], [], []
    ep_r=[]
    
    R=0
    step, episode = 0, 0 
    s = env.reset()
    while True:
        a = agent.choose_action(s)
        if not dummytrain:
            s1, r, done, _ = env.step(a)
        else:
            s1, a, r, done, _ = env.step(a)
        #env.render()
        if video_writer:
            fig = env.plot()
            fig = cv2.cvtColor(fig, cv2.COLOR_BGR2RGB)
            video_writer.append_data(fig)
        '''
        mb_s.append(s)
        mb_a.append(a)
        mb_r.append(r)
        mb_done.append(done)
        '''
        
        agent.store_transition(s,a,r,s1,done)
        s=s1
        R+=r
        if learn:
            for _ in range(3): agent.learn()
        if done:
            # For training
            if R<-20:
                agent.epsilon = max(0, agent.epsilon-0.01)
            #if report: print('Episode end. Total reward: {}'.format(R))
            print('Episode {}. Reward {:.2f}. epsilon {:.3f}. Episode steps: {}. x0={}'.format(episode, R, agent.epsilon, env.n_step, env.x0))
            # Debug: visualize values
            visualize_value(env,agent, './log/qvalue{:03d}.png'.format(episode))
            visualize_modulator(agent, './log/modulator{:03d}.png'.format(episode))
            visualize_sensor(env, agent, './log/sensor{:03d}.png'.format(episode))
            ep_r.append(R)
            R=0
            s=env.reset()
            episode+=1
            if early_stop and episode>30:
                if np.mean(ep_r[-30:])>4.8: break
        step += 1

        if done and episode%10 == 0:
            if learn and savepath: 
                #agent.save(savepath, True)
                #print("Model saved to {}".format(savepath))
                pass
        if step>=n_steps or episode>=n_episodes: break
    env.close()
    return mb_s, mb_a, mb_r, mb_done, ep_r

def test_ai(env, agent, plt_path):
    agent.fixed_epsilon()
    mb_s, mb_a, mb_r, mb_done, ep_r = run(env, agent, n_steps=1000, learn=False)
    plt_r(ep_r, plt_path)
def visualize_sensor_sup(mnist, agent, path):
    data = np.reshape(mnist.test.images[0], [-1,28,28,1])
    #set_trace()
    img = agent.show_sensor(data)
    #plt.pcolor(img[0,:,:,-1])
    #plt.colorbar()
    plt.imshow(np.squeeze(img[0]))
    plt.axis('off')  
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    np.save(os.path.splitext(path)[0]+'.npy', np.squeeze(img[0]))
    # show original img
    #plt.pcolor(s[0,...,-1])
    #plt.colorbar()
    plt.imshow(np.squeeze(data[0]), cmap='gray')
    plt.axis('off')  
    plt.savefig("./log/original_img.png", bbox_inches='tight')
    plt.close()
    np.save('./log/original_img.npy', np.squeeze(data[0]))
def train_supervised(mnist, agent, n_steps=2000, batch_size=100, savepath=None):
    log_accuracy = open('accuracy.csv','a')
    test_inputs = mnist.test.images
    test_inputs = np.reshape(test_inputs, [-1, 28, 28, 1])
    test_labels = mnist.test.labels
    test_inputs = test_inputs[:, ::-1, ::-1, :]
    for step in range(n_steps):
        train_inputs, train_labels = mnist.train.next_batch(batch_size)
        train_inputs = np.reshape(train_inputs, [-1, 28, 28, 1])
        train_inputs = train_inputs[:,::-1,::-1,:]
        agent.train(train_inputs, train_labels)
        if step %10 ==0:
            test_accu = agent.test(inputs = test_inputs, labels = test_labels)
            print('Step {}. Test accuracy {:.2f}%'.format(step, test_accu*100))
            log_accuracy.write('{},{}\n'.format(step,test_accu))
            visualize_sensor_sup(mnist, agent, './log/sensor{:04d}.png'.format(step))
            mod = visualize_modulator(agent, './log/modulator{:04d}.png'.format(step))
            np.save('./models/modulator{:04d}'.format(step), mod)
            if savepath:
                agent.save(savepath)
    log_accuracy.close()
    return agent
