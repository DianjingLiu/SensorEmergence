"""
functions include：
1. save the raw data of modulator params, sensor and Q-VALUEs as npy files.
2. if space allows, save the evaluate network and mod params in model directory
3. a function to evaluate the success rate V.S. initial distance. inputs: model, env, [params] epsilon, episode length, number of tests
4. avg Q-value visualizer: randomly initialize, calculate avg Q-value. inputs: model, env, [param] number of samples
5. calculate avg/rms/variance of modulator thicknesses -- we may check on existing data to choose one calculation method 
"""

# visualize avg thickness
import numpy as np
import os
def interp(img, res):
    from scipy.interpolate import interp2d, RectBivariateSpline
    x = np.linspace(0,1, img.shape[0])
    y = np.linspace(0,1, img.shape[1])
    #f = interp2d(x, y, img, kind='cubic')
    f = RectBivariateSpline(x, y, img)
    x_new = np.linspace(0,1,res[0])
    y_new = np.linspace(0,1,res[1])
    return f(x_new, y_new)
def mask_circle(img, r=1, erase=False, shift=None):
    # if erase=True, will set masked value to 0 -- its value will not be recovered in interpolation
    x = np.linspace(-1,1, img.shape[0])
    y = np.linspace(-1,1, img.shape[1])
    xx, yy = np.meshgrid(x, y)
    mask = np.sqrt(xx**2 + yy**2) > r
    if shift=='smooth':
        img -= img[0, int(img.shape[1]/2 * (1+r))]
    elif shift=='min':
        img -= np.min(img)
    if erase:
        return img*(1-mask)
    else:
        return np.ma.array(img, mask=mask) 
def get_thickness(mod, operation='avg'):
    mod = mod.reshape([10,10])
    mod = mask_circle(mod-np.min(mod), 1/np.sqrt(2),True)
    mod = interp(mod, [200,200]) * 50
    if operation == 'avg':
        results = np.mean(mod)
    return results


#########################################################################################
# data collection and analysis
# Q-values -- mean and variance
def sample(env, agent, n):
    # randomly sample n transitions and save to agent memory
    s = env.reset()
    while agent.memory.size < n: # For RL exp
        # step forward
        #a = agent.choose_action(s)
        a = np.random.randint(0, agent.n_actions) # random action
        s1, r, done, _ = env.step(a)
        agent.store_transition(s,a,r,s1,done)
        # update state
        if done:
            s = env.reset()
        else:
            s = s1
    ratio = np.mean(agent.memory.r[:n]!=0)
    print("Random sampling finish.")
    print("Ratio of non-zero reward transitions: {:.2f}%".format(ratio*100))
    ratio = np.mean(agent.memory.d[:n])
    print("Ratio of last step transitions: {:.2f}%".format(ratio*100))
    return agent
def get_qvalue_avg(agent, args):
    qvalues = []
    for _ in range(10):
        s, a, r, s1, d = agent.memory.sample(args.batch_size)
        q = agent.show_q(s)
        qvalues.append(q)
    qvalues = np.concatenate(qvalues, axis=0)
    q_avg = np.mean(qvalues, axis=0)
    q_var = np.var(qvalues, axis=0)
    return q_avg, q_var
def q_avg_var_prepare(args, env, agent):
    q_averages, q_variances = [], []
    log_q_avg = open(os.path.join(args.savepath, 'q_averages.csv'), 'w')
    log_q_var = open(os.path.join(args.savepath, 'q_variances.csv'), 'w')
    agent = sample(env, agent, args.memory_size)
    return args, agent, log_q_avg, log_q_var
def q_avg_var_loop(args, agent, log_q_avg, log_q_var, step):
    q_avg, q_var = get_qvalue_avg(agent, args)
    log_q_avg.write("{}, {}, {}, {}, {}, {}\n".format(step, q_avg[0], q_avg[1], q_avg[2], q_avg[3], q_avg[4]))
    log_q_var.write("{}, {}, {}, {}, {}, {}\n".format(step, q_var[0], q_var[1], q_var[2], q_var[3], q_var[4]))
def q_avg_var_close(args, agent, log_q_avg, log_q_var):
    log_q_avg.close()
    log_q_var.close()
def q_avg_var_visualize(args):
    import pandas as pd
    N = 10
    # visualize q_average
    file = os.path.join(args.path, 'q_averages.csv')
    df = pd.read_csv(file, header=None, comment='#')
    df = np.array(df)
    for i in range(1, 6):
        avg = np.convolve(df[:,i], np.ones((N,))/N, mode='valid')
        plt.plot(df[N-1:, 0]/1000, avg)
    plt.xlabel("Step ($\\times10^3$)")
    plt.ylabel("Q-value average")
    plt.legend(['Up', 'Down', 'Left', 'Right', 'Confirm'])
    plt.savefig(os.path.join(args.path, 'q_averages.png'))
    plt.close()
    # visualize q_variance
    file = os.path.join(args.path, 'q_variances.csv')
    df = pd.read_csv(file, header=None, comment='#')
    df = np.array(df)
    for i in range(1, 6):
        avg = np.convolve(df[:,i], np.ones((N,))/N, mode='valid')
        plt.plot(df[N-1:, 0]/1000, avg)
    plt.xlabel("Step ($\\times10^3$)")
    plt.ylabel("Q-value variance")
    plt.legend(['Up', 'Down', 'Left', 'Right', 'Confirm'])
    plt.savefig(os.path.join(args.path, 'q_variance.png'))
    plt.yscale('log')
    plt.savefig(os.path.join(args.path, 'q_variance_log.png'))

# general success rate
class agent_wrapper():
    """
    This wrapper helps avoid the redundant GPU calculation by using a hashmap to store the Q-network outputs.
    It also handles the randomness in the test. The randomness can have 2 forms:
    1. randomly choose action with probability P = epsilon, otherwise a=argmax(Q-values))
    2. random sampling according to the distribution softmax(Q-values)
    If the parameter deterministic=True, we use case 1. Otherwise, we use case 2.
    """
    def __init__(self,agent,epsilon=None,deterministic=True):
        self.agent = agent
        self.deterministic = deterministic
        self.action_table = {} # if deterministic=True, it saves the action; otherwise, it saves the distribution
        self.epsilon = epsilon
        assert deterministic==False or epsilon is not None, "epsilon value should be provided when deterministic=True"
    def restore(self, path):
        self.action_table = {} # reset the hashmap when loading a new model
        return self.agent.restore(path)
    def reset_table(self):
        self.action_table = {}
    def choose_action(self, s, identifier=None):
        if identifier is None:
            identifier = tuple(s.flatten())
        if self.deterministic:
            return self.choose_action_deterministic(s, identifier)
        else:
            return self.choose_action_sampling(s, identifier)
    def choose_action_deterministic(self, s, loc):
        if np.random.uniform() < self.epsilon:
            if loc in self.action_table:
                a = self.action_table[loc]
            else:
                actions_value = self.agent.show_q(s)
                a = np.argmax(actions_value)
                self.action_table[loc] = a
        else:
            a = np.random.randint(0, self.agent.n_actions)
        return a
    def choose_action_sampling(self, s, loc):
        assert len(s) == 1, "We assume the agent takes one state for each time."
        if loc in self.action_table:
            distribution = self.action_table[loc]
        else:
            actions_value = np.squeeze(self.agent.show_q(s))
            distribution = self.softmax(actions_value)
            self.action_table[loc] = distribution
            #import pdb; pdb.set_trace()
        # sampling according to the distribution
        action = np.random.choice(np.arange(self.agent.n_actions), p=distribution)
        return action
    def softmax(self,x,a=12):
        x = x * a
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
def success_rate_test(args, env, agent, pos=None, obstacles=None, target=None):
    # note: the input agent is a wrapper
    assert type(agent) == agent_wrapper
    if obstacles:
        env.obstacles = set(obstacles)
        reset_obs = False
    else:
        reset_obs = True
    s = env.reset(x0=target, x=pos, reset_obstacles=reset_obs)
    agent.reset_table()
    loc = tuple(env.x) # location of the agent. In each episode, this location uniquely determines the environment state
    success = 0
    done = False
    while not done:
        a = agent.choose_action(s)
        s1, r, done, _ = env.step(a)
        s = s1
        if r==5:
            success=1
    return success
def success_general_prepare(args, env, agent):
    #args.N = 100
    env.max_step = 100
    print('For test, the maximum step is {}'.format(env.max_step))
    agent = agent_wrapper(agent, args.epsilon, args.deterministic) # this wrapper helps handle how we do the exploration, and speed up computing by storing past results
    log_success_general = open(os.path.join(args.savepath, 'scores_general.csv'), 'a')
    log_success_general.write("# epsilon, {}\n".format(args.epsilon))
    log_success_general.write("# number of tests, {}\n".format(args.N))
    log_success_general.write("# number of steps in each test, {}\n".format(env.max_step))
    log_success_general.write("# deterministic, {}\n".format(args.deterministic))
    return args, env, agent, log_success_general
def success_general_loop(args, env, agent, log_success_general, step):
    N = args.N
    count = 0
    for episode in range(N):
        count += success_rate_test(args, env, agent, pos=None, obstacles=None, target=None)
    rate = count / N
    log_success_general.write("{}, {}\n".format(step, rate))
def success_general_close(args, env, agent, log_success_general):
    log_success_general.close()
def success_general_visualize(args):
    import pandas as pd
    figpath = os.path.join(args.savepath, 'scores_general.png')
    file = os.path.join(args.path, 'scores_general.csv')
    try:
        score = pd.read_csv(file, header=None, comment='#')
        plt.plot(score[0]/1000, score[1]*100, linewidth=2)
        plt.ylabel("Success rate (%)")
        plt.xlabel("Step ($\\times 10^3$)")
        plt.xlim([0,np.array(score)[-1,0]/1000])
        plt.ylim([0,100])
        plt.savefig(figpath)
    except FileNotFoundError:
        print("File \'{}\' not found. Skipping learning curve plot.".format(file))


# success rate map
def success_map_prepare(args, env, agent):
    #args.N = 100
    env.max_step = 100
    agent = agent_wrapper(agent, args.epsilon, args.deterministic) # this wrapper helps handle how we do the exploration, and speed up computing by storing past results
    obstacles = {(2,3), (3,2)}
    pos = np.array((4, 4))
    env.obstacles = obstacles
    #targets = np.array([[3,3], [2,1], [1,2], [2,2], [1,3], [3,1]])
    targets = []
    for x in range(8):
        for y in range(8):
            if (x, y) not in obstacles: targets.append([x, y])
    targets = np.array(targets)
    log_success_map = open(os.path.join(args.savepath, 'scores_map.csv'), 'a')
    log_success_map.write("# number of tests, {}\n".format(args.N))
    log_success_map.write("# deterministic, {}\n".format(args.deterministic))
    log_success_map.write("# epsilon, {}\n".format(args.epsilon))
    log_success_map.write("# episode length, {}\n".format(env.max_step))
    log_success_map.write("# obstacles, {}\n".format(np.array(obstacles)))
    log_success_map.write("# agent initial position, {}\n".format(pos))
    log_success_map.write("# step, targets: " + ", ".join([str(t) for t in targets]) + "\n")
    return args, env, agent, log_success_map, pos, obstacles, targets
def success_map_loop(args, env, agent, log_success_map, pos, obstacles, targets, step):
    N = args.N
    log_success_map.write("{}, ".format(step))
    for t in targets:
        count = 0
        print('Evaluating target {}'.format(t))
        for episode in range(N):
            count += success_rate_test(args, env, agent, pos=pos, obstacles=obstacles, target=t)
        rate = count / N
        log_success_map.write("{}, ".format(rate))
    log_success_map.write("\n")
def success_map_close(args, env, agent, log_success_map, pos, obstacles, targets):
    log_success_map.close()
def success_map_visualize(args):
    import pandas as pd
    import imageio
    if os.path.basename(os.path.normpath(args.savepath)) != 'success_maps': 
        args.savepath = os.path.join(args.savepath, 'success_maps') # Auto-fill the "models" to models directory
    os.makedirs(args.savepath, exist_ok=True)
    file = os.path.join(args.path, 'scores_map.csv')
    df = pd.read_csv(file, header=None, comment='#')
    print('load {} success'.format(file))
    video_writer = imageio.get_writer(os.path.join(args.savepath, 'maps.mp4'), fps=2)
    obstacles = {(2,3), (3,2)}
    targets = []
    for x in range(8):
        for y in range(8):
            if (x, y) not in obstacles: targets.append([x, y])
    targets = np.array(targets)
    #from pdb import set_trace; set_trace()
    df = np.array(df)
    for row in df:
        maps = np.zeros([8,8])
        step = row[0]
        for (x,y), rate in zip(targets, row[1:]):
            maps[x,y] = rate
        plt.imshow(maps,vmin=0,vmax=1, cmap='Blues')
        #plt.colorbar()
        #plt.title("Step {}".format(step))
        plt.tick_params(axis='both', labelsize=0, length = 0)
        #plt.axis('off')
        imgpath = os.path.join(args.savepath, 'map{}.png'.format(step))
        plt.savefig(imgpath)
        plt.close()
        img = plt.imread(imgpath)
        video_writer.append_data(img)
    video_writer.close()


# analyze agent modulator gradient
import matplotlib.pyplot as plt
def grad_prepare(args, env, agent):
    import tensorflow as tf
    grads = tf.gradients(agent.loss, agent.mod_params)
    """
    conv0 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='eval_net_params/rl/conv2d')
    conv1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='eval_net_params/rl/conv2d_1')
    conv2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='eval_net_params/rl/conv2d_2')
    dense0= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='eval_net_params/rl/dense')
    dense1= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='eval_net_params/rl/dense_1')
    """
    agent = sample(env, agent, args.memory_size)
    # make sure there is 1 positive reward sample in the batch
    idx = np.where(agent.memory.r==5)[0]
    count=-1
    while count!=0:
        s, a, r, s1, d = agent.memory.sample(args.batch_size-4)
        count = np.sum(r==5)
    s = np.concatenate((s, agent.memory.s[idx[:4]] ), axis=0)
    a = np.concatenate((a, agent.memory.a[idx[:4]] ), axis=0)
    r = np.concatenate((r, agent.memory.r[idx[:4]] ), axis=0)
    s1= np.concatenate((s1,agent.memory.s_[idx[:4]]), axis=0)
    d = np.concatenate((d, agent.memory.d[idx[:4]] ), axis=0)
    feed_dict = {
        agent.s : s , 
        agent.s_: s1, 
        agent.a : a , 
        agent.r : r , 
        agent.d : d 
    }
    log_grad = open(os.path.join(args.savepath, 'mod_grad.csv'), 'a')
    log_grad.write("# step,  \n")
    return args, agent, feed_dict, log_grad, grads
def grad_loop(args, agent, feed_dict, log_grad, grads, step):
    gradients = agent.sess.run(grads, feed_dict=feed_dict)
    gradients = np.squeeze(gradients)
    log_grad.write("{}".format(step))
    for g in gradients:
        log_grad.write(", {}".format(g))
    log_grad.write("\n")
def grad_close(args, agent, gradient_op, log_grad):
    log_grad.close()


def loss_prepare(args, env, agent):
    import tensorflow as tf
    grads = tf.gradients(agent.loss, agent.mod_params + agent.rl_params)
    """
    conv0 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='eval_net_params/rl/conv2d')
    conv1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='eval_net_params/rl/conv2d_1')
    conv2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='eval_net_params/rl/conv2d_2')
    dense0= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='eval_net_params/rl/dense')
    dense1= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='eval_net_params/rl/dense_1')
    """
    agent = sample(env, agent, args.memory_size)
    # make sure there is 1 positive reward sample in the batch
    idx = np.where(agent.memory.r==5)[0]
    count=-1
    while count!=0:
        s, a, r, s1, d = agent.memory.sample(args.batch_size-4)
        count = np.sum(r==5)
    s = np.concatenate((s, agent.memory.s[idx[:4]] ), axis=0)
    a = np.concatenate((a, agent.memory.a[idx[:4]] ), axis=0)
    r = np.concatenate((r, agent.memory.r[idx[:4]] ), axis=0)
    s1= np.concatenate((s1,agent.memory.s_[idx[:4]]), axis=0)
    d = np.concatenate((d, agent.memory.d[idx[:4]] ), axis=0)
    feed_dict = {
        agent.s : s , 
        agent.s_: s1, 
        agent.a : a , 
        agent.r : r , 
        agent.d : d 
    }
    log_loss = open(os.path.join(args.savepath, 'loss_grad_test.csv'), 'a')
    log_loss.write("# step, loss, avg_grad_mod, avg_grad_conv1, avg_grad_conv2, avg_grad_conv3, avg_grad_dense1, avg_grad_dense2 \n")
    return args, agent, feed_dict, log_loss, grads
def loss_loop(args, agent, feed_dict, log_loss, grads, step):
    loss, gradients = agent.sess.run([agent.loss, grads], feed_dict=feed_dict)
    gradients = [g.flatten() for g in gradients]
    grad_mod    = gradients[0]
    grad_conv1  = gradients[1:3]
    grad_conv2  = gradients[3:5]
    grad_conv3  = gradients[5:7]
    grad_dense1 = gradients[7:9]
    grad_dense2 = gradients[9:11]
    avg_g_mod    = np.sqrt(np.mean(np.square(grad_mod)))
    avg_g_conv1  = np.sqrt(np.mean(np.square(np.concatenate(grad_conv1))))
    avg_g_conv2  = np.sqrt(np.mean(np.square(np.concatenate(grad_conv2))))
    avg_g_conv3  = np.sqrt(np.mean(np.square(np.concatenate(grad_conv3))))
    avg_g_dense1 = np.sqrt(np.mean(np.square(np.concatenate(grad_dense1))))
    avg_g_dense2 = np.sqrt(np.mean(np.square(np.concatenate(grad_dense2))))
    log_loss.write("{}, {}, {}, {}, {}, {}, {}, {} \n".format(
        step, 
        loss, 
        avg_g_mod, 
        avg_g_conv1, 
        avg_g_conv2, 
        avg_g_conv3, 
        avg_g_dense1, 
        avg_g_dense2
        )
    )
def loss_close(args, agent, feed_dict, log_loss, grads):
    log_loss.close()
def loss_visualize(args):
    import pandas as pd
    file = os.path.join(args.path, 'loss_grad_test.csv')
    df = pd.read_csv(file, header=None, comment='#')
    df = np.array(df)
    N = 10
    # visualize loss
    plt.plot(df[:, 0]/1000, df[:, 1], '.')
    avg = np.convolve(df[:,1], np.ones((N,))/N, mode='valid')
    plt.plot(df[N-1:, 0]/1000, avg)
    plt.xlabel("Step ($\\times10^3$)")
    plt.ylabel("Loss")
    plt.ylim(bottom=0)
    plt.xlim([0,df[-1,0]/1000])
    plt.savefig(os.path.join(args.path, 'loss.png'))
    plt.close()
    # visualize gradients
    for i in range(1, 7):
        #plt.plot(df[:, 0]/1000, df[:, i])
        avg = np.convolve(df[:,i], np.ones((N,))/N, mode='valid')
        plt.plot(df[N-1:, 0]/1000, avg)
    plt.xlabel("Step ($\\times10^3$)")
    plt.ylabel("Average gradients")
    plt.legend(['Modulator', 'Conv1', 'Conv2', 'Conv3', 'Dense1', 'Dense2'])
    plt.savefig(os.path.join(args.path, 'gradients.png'))
    plt.yscale('log')
    plt.savefig(os.path.join(args.path, 'gradients_log_scale.png'))



def main(args):
    args.range[1] += 1 # the ending step becomes inclusive
    operations = {
        'q_avg_var' : {
            'prepare'   : q_avg_var_prepare,
            'loop'      : q_avg_var_loop,
            'close'     : q_avg_var_close,
            'visualize' : q_avg_var_visualize
        },
        'success_general' : {
            'prepare'   : success_general_prepare,
            'loop'      : success_general_loop,
            'close'     : success_general_close,
            'visualize' : success_general_visualize
        },
        'success_map' : {
            'prepare'   : success_map_prepare,
            'loop'      : success_map_loop,
            'close'     : success_map_close,
            'visualize' : success_map_visualize
        },
        'grad': {
            'prepare' : grad_prepare,
            'loop'    : grad_loop,
            'close'   : grad_close
        },
        'loss': {
            'prepare'  : loss_prepare,
            'loop'     : loss_loop,
            'close'    : loss_close,
            'visualize': loss_visualize
        }
    }
    # (2021.3.12) Auto-correction: 1. fill 'models' to args.path basename. 2. By default, args.savepath is the parent dir of args.path
    if os.path.basename(os.path.normpath(args.path)) != 'models': 
        args.path = os.path.join(args.path, 'models') # Auto-fill the "models" to models directory
    if args.savepath is None:
        args.savepath = os.path.normpath(os.path.join(args.path, os.pardir)) # args.savepath is the parent dir of args.path, without 'models'
    # visuzlize
    if args.mode in ['visualize', 'v']:
        args.path = args.savepath # for visualization, the log path is the same as the save path
        operations[args.data]['visualize'](args)
    elif args.mode in ['collect', 'c']:
        from run import get_env_agent, log
        log['BACKGROUND'] = args.background
        env, agent = get_env_agent(log)
        # prepare
        params = operations[args.data]['prepare'](args, env, agent)
        # major loop: load models and collect data for each model
        for s in range(*args.range):
            print("Step {}".format(s))
            # Load model
            file = os.path.join(args.path, "dqn-{:d}".format(s))
            if s!= 0: agent.restore(file)
            # Collect data
            operations[args.data]['loop'](*params, s)
        # Close
        operations[args.data]['close'](*params)
if __name__ == '__main__':
    import time
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', '--gpu', default=None, action='store', help='Specify the GPU to be used.')
    parser.add_argument('-p', '--path', default='/home/Temporary_data/Dianjing/lensnet_baseline_lrdecay/models/', action='store', help='log directory.')
    parser.add_argument('-s', '--savepath', default=None, action='store', help='log directory.')
    parser.add_argument('-mode', '--mode', default='visualize', action='store', 
        help='''
        collect: collect data
        visualize: visualize collected data
        '''
    )
    parser.add_argument('-data', '--data', default='q_avg_var', action='store', 
        help='''
        the variable to calculate and visualize.
        q_avg_var: Average and variance of Q-value
        success_general: evaluate general success rate
        success_map: evaluate success rate on specific target/agent/obstacle locations.
        grad: modulator gradient
        loss: loss and gradients for each layer
        '''
    )

    parser.add_argument('-range', '--range', default=[1000, 150000, 1000],#[1e-5, 1e4, 0.5], 
        nargs='+', type=int, action='store', help='.')
    parser.add_argument('-bkg', '--background', default='./objects/bkg7.png', action='store', help='background image.')
    parser.add_argument('-n', '--memory_size', default=5000, type=int, action='store', help='.')
    parser.add_argument('-n_b', '--batch_size', default=128, type=int, action='store', help='.')
    parser.add_argument('-n_test', '--N', default=100, type=int, action='store', help='For success rate test. The number of tests')
    parser.add_argument('-eps', '--epsilon', default=0.99, type=float, action='store', help='The probability of taking deterministic action in test.')
    parser.add_argument('-not_det', dest='deterministic', default=True, action='store_false', 
        help='''
        This parameter is used for success rate test. Including success_general and success_map.
        By default, deterministic=true. The agent explores according to Prop=epsilon. 
        otherwise if flagged by not_det, the agent explores according to random sampling on the distribution softmax(Q-values)
        '''
    )

    #parser.add_argument('-lens', '--fix_lens', default=False, action='store_true', help='If True, will train the agent with a fixed perfect lens.')
    #parser.add_argument('-gamma', '--gamma', default=0.7, type=float, action='store', help='.')

    args = parser.parse_args()
    if args.gpu: os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import time
    start_time = time.time()
    main(args)
    print("Run time: {:.1f}s".format(time.time()-start_time))

