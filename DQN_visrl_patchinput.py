"""
Using Tensorflow to build the neural network.
Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf
import cv2
from pdb import set_trace

#np.random.seed(0)
#tf.set_random_seed(1)

# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_input,
            n_hid=[200, 100, 50],
            learning_rate=0.01,
            #lr_decay_step = 15000,
            #lr_decay_rate = 0.9,
            reward_decay=0.99,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_max=50000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_input = list(np.atleast_1d(n_input)) # figure size [n1, n2, n_channel]
        self.n_hid = n_hid
        self.memory = Memory(memory_max)
        # total learning step
        self.global_step = tf.Variable(0, trainable=False)

        # Constant learning rate
        self.lr = tf.Variable(learning_rate, trainable=False)

        # exponential decay learning rate
        #self.lr = tf.train.exponential_decay(learning_rate, self.global_step, lr_decay_step, lr_decay_rate, staircase=True)

        self.gamma = tf.Variable(reward_decay, trainable=False) # set as tf variable so that we can change its value
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_max = memory_max
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # consist of [target_net, evaluate_net]
        self._build_net()
        self.saver = tf.train.Saver(var_list = tf.trainable_variables(),max_to_keep=None)
        #self.saver_dqn = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='eval_net_params'))
        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        '''
        # build network -- tuple input
        self.pos = tf.placeholder(tf.float32, [None, 2] , name='pos')  # state, input for eval net
        self.fig = tf.placeholder(tf.float32, [None] + self.n_input, name='fig')  # state, input for eval net
        self.pos_= tf.placeholder(tf.float32, [None, 2] , name='pos_')  # state, input for eval net
        self.fig_= tf.placeholder(tf.float32, [None] + self.n_input, name='fig_')  # state, input for eval netnet
        self.a  = tf.placeholder(tf.int32,   [None, 1], name='action') # action
        self.r  = tf.placeholder(tf.float32, [None, 1], name='reward') # reward
        self.d  = tf.placeholder(tf.float32, [None, 1], name='done') # if true, episode end

        self.q_eval, e_params = self.build_cnn('eval_net_params',   self.fig , self.pos )
        self.q_next, t_params = self.build_cnn('target_net_params', self.fig_, self.pos_)   # [None, n_actions]
        '''
        #'''
        # build network 
        self.s  = tf.placeholder(tf.float32, [None] + self.n_input, name='s')  # state, input for eval net
        self.s_ = tf.placeholder(tf.float32, [None] + self.n_input, name='s_') # new state, input for target net
        self.a  = tf.placeholder(tf.int32,   [None, 1], name='action') # action
        self.r  = tf.placeholder(tf.float32, [None, 1], name='reward') # reward
        self.d  = tf.placeholder(tf.float32, [None, 1], name='done') # if true, episode end
        self.training = tf.placeholder_with_default(False, shape=(), name='training')
        #self.q_eval = self.build_dense(self.s,  self.n_hid, 'eval_net_params')      
        #self.q_next = self.build_dense(self.s_, self.n_hid, 'target_net_params')   # [None, n_actions]
        #t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net_params')
        #e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net_params')
        self.q_next, t_params = self.build_cnn('target_net_params', self.s_ )   # [None, n_actions]
        self.q_eval, e_params = self.build_cnn('eval_net_params',   self.s )      
        #'''
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        self.saver_dqn = tf.train.Saver(var_list=e_params)

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, keepdims=True, name='Qmax_s_') * (1 - self.d) # shape (None, 1)
            self.q_target = tf.stop_gradient(tf.squeeze(q_target)) # shape (None, )
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), tf.squeeze(self.a)], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
        with tf.variable_scope('loss'):
            #self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a))
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, var_list=e_params, global_step=self.global_step)
    #'''
        # For pretrain
        self.pos = tf.placeholder(tf.float32, [None, 4] , name='pos')
        self.pretrain_loss = tf.reduce_mean(tf.squared_difference(self.pos, self.pretrain_pos))
        self.pretrain_op = tf.train.AdamOptimizer(self.lr).minimize(self.pretrain_loss, var_list=e_params)
        # This is to test if we can recognize states without evolve of the optics layer
        #self.pretrain_op = tf.train.AdamOptimizer(self.lr).minimize(self.pretrain_loss, var_list=self.rlparams)
    def pretrain(self, img, pos):
        feed_dict = {self.s:img, self.pos:pos}
        _,cost = self.sess.run([self.pretrain_op, self.pretrain_loss], feed_dict=feed_dict)
        return cost
    def save_pretrain(self,filename):
        self.sess.run(self.replace_target_op)
        self.save(filename)
    #'''
    def build_dense(self, input_layer, n_hid, name):
        # initialize the weights so that initial Q values are close to 0
        initializer = tf.random_uniform_initializer(-0.1, 0.1) 
        # Regularize weights and biases
        #regularizer = tf.contrib.layers.l2_regularizer(0.01, scope=None)
        regularizer = tf.contrib.layers.l1_regularizer(0.01, scope=None)
        with tf.variable_scope(name):
            hid = input_layer
            for n in n_hid:
                hid = tf.layers.dense(inputs=hid, 
                            units=n,
                            kernel_initializer=initializer, bias_initializer=initializer,
                            kernel_regularizer=regularizer, bias_regularizer=regularizer,
                            activation = tf.nn.tanh)
            out = tf.layers.dense(inputs=hid, 
                        units=self.n_actions, 
                        kernel_initializer=initializer, bias_initializer=initializer,
                        kernel_regularizer=regularizer, bias_regularizer=regularizer,
                        #activation = None
                        )
        return out



    def choose_action(self, observation, deterministic=True):
        '''
        update 2021.4.20: 
        We add a parameter deterministic. If true, the policy is not changed: action = argmax(Q-values).
        If deterministic=False, we randomly sample actions according to distribution P(actions)=softmax(Q-values).
        '''
        if np.random.uniform() < self.epsilon:
            # to have batch dimension when feed into tf placeholder
            observation = np.atleast_2d(observation)
            #pos, fig = self.memory.unpack_obs(observation)

            # forward feed the observation and get q value for every actions
            feed_dict = {self.s: observation}
            #feed_dict = {self.pos:pos, self.fig:fig}
            actions_value = self.sess.run(self.q_eval, feed_dict=feed_dict)
            if deterministic:
                action = np.argmax(actions_value)
            else:
                import scipy
                distribution = scipy.special.softmax(actions_value, axis=1)
                action = np.random.choice(numpy.arange(0, 5), p=distribution)
                return action, distribution
                import pdb;pdb.set_trace()
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.memory.size < self.batch_size: return 
        # check to replace target parameters
        if self.sess.run(self.global_step) % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            #print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        #sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        #'''
        s, a, r, s_, d = self.memory.sample(self.batch_size)
        feed_dict = {self.s : s , 
                     self.s_: s_, 
                     self.a : a , 
                     self.r : r , 
                     self.d : d , 
                     self.training: True}
        #'''
        _, self.cost = self.sess.run([self._train_op, self.loss], feed_dict=feed_dict)
        self.cost_his.append(self.cost)

        # increasing epsilon
        #if self.memory.size > 1000:
        if self.train_step() > 10000:
            self.renew_epsilon()
        return self.cost

    def store_transition(self, s, a, r, s_, done=False):
        self.memory.store(s,a,r,s_,done)
    def load_memory(self, filename):
        self.memory.load(filename)
    def save_memory(self, filename):
        self.memory.save(filename)

    def get_lr(self):
        return self.sess.run(self.lr)
    def set_lr(self, learning_rate):
        self.sess.run(self.lr.assign(learning_rate))
    def get_reward_decay(self):
        return self.sess.run(self.gamma)
    def set_reward_decay(self, reward_decay):
        self.sess.run(self.gamma.assign(reward_decay))
    def train_step(self):
        return self.sess.run(self.global_step)
    def reset_train_step(self):
        self.sess.run(self.global_step.assign(0))
    def set_train_step(self, n):
        self.sess.run(self.global_step.assign(n))
    def fixed_epsilon(self, eps=None):
        if eps is not None:
            self.epsilon_max = eps
        self.epsilon = self.epsilon_max
    def renew_epsilon(self):
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

    def plot_cost(self, filename=None):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

    def save(self, filename, save_memory=False):
        self.saver.save(self.sess, filename, global_step=self.global_step, write_meta_graph=False)
        if save_memory:
            self.memory.save('mmry.npz')

    def restore(self, filename, load_memory=False):
        self.saver.restore(self.sess, filename)
        if load_memory:
            self.memory.load('mmry.npz')
    def save_dqn(self, filename):
        self.saver_dqn.save(self.sess, filename)
    # for debug
    def show_q(self, observation):
        if observation.ndim==1:
            observation = observation[np.newaxis, :]
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        return actions_value

    def build_cnn(self, name, input1, activation=tf.nn.relu):
        # initialize the weights so that initial Q values are close to 0
        initializer = tf.random_uniform_initializer(-0.01, 0.01) 
        # Regularize weights and biases
        #regularizer = tf.contrib.layers.l2_regularizer(0.01, scope=None)
        regularizer = tf.contrib.layers.l1_regularizer(0.01, scope=None)
        with tf.variable_scope(name):
            with tf.variable_scope('rl'):
                conv0 = tf.layers.conv2d(inputs = input1,
                    filters=32,
                    kernel_size = [20,20],
                    padding = 'same',
                    #strides = 4,
                    activation=activation,
                    kernel_initializer=initializer, bias_initializer=initializer,
                    kernel_regularizer=regularizer, bias_regularizer=regularizer,
                    )
                pool0 = tf.layers.max_pooling2d(inputs=conv0, pool_size=[2, 2], strides=2)
                conv1 = tf.layers.conv2d(inputs = pool0,
                    filters=32,
                    kernel_size = [8,8],
                    padding = 'same',
                    #strides = 4,
                    activation=activation,
                    kernel_initializer=initializer, bias_initializer=initializer,
                    kernel_regularizer=regularizer, bias_regularizer=regularizer,
                    )
                pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
                conv2 = tf.layers.conv2d(
                    inputs=pool1,
                    filters=64,
                    kernel_size=[4, 4],
                    padding="same",
                    #strides = 2,
                    activation=activation,
                    kernel_initializer=initializer, bias_initializer=initializer,
                    kernel_regularizer=regularizer, bias_regularizer=regularizer,
                    )
                pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
                conv3 = tf.layers.conv2d(
                    inputs=pool2,
                    filters=64,
                    kernel_size=[3, 3],
                    padding="same",
                    activation=activation,
                    kernel_initializer=initializer, bias_initializer=initializer,
                    kernel_regularizer=regularizer, bias_regularizer=regularizer,
                    )
                flat = tf.contrib.layers.flatten(conv3)
                '''
                hid0 = tf.layers.dense(inputs=pool2_flat,
                    units=128,
                    kernel_initializer=initializer, bias_initializer=initializer,
                    kernel_regularizer=regularizer, bias_regularizer=regularizer,
                    activation = tf.nn.relu
                    )
                #'''
                hid1 = tf.layers.dense(inputs=flat,
                    units=512,
                    kernel_initializer=initializer, bias_initializer=initializer,
                    kernel_regularizer=regularizer, bias_regularizer=regularizer,
                    activation = activation
                    )
                #hid1 = tf.layers.batch_normalization(hid1, training=self.training)
                #'''
                '''
                hid2 = tf.layers.dense(inputs=hid1,
                    units=64,
                    kernel_initializer=initializer, bias_initializer=initializer,
                    kernel_regularizer=regularizer, bias_regularizer=regularizer,
                    #activation = tf.nn.relu
                    )
                #hid2 = tf.layers.batch_normalization(hid2, training=self.training)
                hid2 = tf.nn.relu(hid2)
                hid3 = tf.layers.dense(inputs=hid1,
                    units=32,
                    kernel_initializer=initializer, bias_initializer=initializer,
                    kernel_regularizer=regularizer, bias_regularizer=regularizer,
                    #activation = tf.nn.relu
                    )
                #hid3 = tf.layers.batch_normalization(hid3, training=self.training)
                hid3 = tf.nn.relu(hid3)
                #'''
                out = tf.layers.dense(inputs=hid1, 
                    units=self.n_actions, 
                    kernel_initializer=initializer, bias_initializer=initializer,
                    kernel_regularizer=regularizer, bias_regularizer=regularizer,
                    #activation = None
                    )
        #'''
        # For pretraining
        if name=='eval_net_params':
            self.pretrain_pos = tf.layers.dense(inputs=hid1,
                units=4,
                kernel_initializer=initializer, bias_initializer=initializer,
                kernel_regularizer=regularizer, bias_regularizer=regularizer,
                activation = tf.nn.tanh
                )
            self.rl_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name+'/rl')
            self.debug_conv1 = conv1
            self.debug_conv2 = conv2
            self.debug_conv3 = conv3
            self.debug_hid = hid1
            self.debug_output = out
        #'''
        #params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        return out, params
    def debug_hidden(self,inputs):
        return self.sess.run(self.debug_hid,feed_dict={self.s:inputs})
    def debug_hidden_stats(self, inputs):
        hid = self.sess.run(self.debug_hid,feed_dict={self.s:inputs})
        return np.mean(hid>0)
    def debug_stats(self,inputs):
        conv1 = self.sess.run(self.debug_conv1,feed_dict={self.s:inputs})
        conv2 = self.sess.run(self.debug_conv2,feed_dict={self.s:inputs})
        conv3 = self.sess.run(self.debug_conv3,feed_dict={self.s:inputs})
        hid   = self.sess.run(self.debug_hid,feed_dict={self.s:inputs})
        out   = self.sess.run(self.debug_output,feed_dict={self.s:inputs})

        return [
            np.mean(conv1>0),
            np.mean(conv2>0),
            np.mean(conv3>0),
            np.mean(hid  >0)
        ]
class Memory(object):
    """docstring for memory"""
    def __init__(self, max_size):
        self.max_size = max_size
        self.size = 0    # size is the number of valid transitions in the memory, this is used to sample data
        self.pointer = 0 # pointer is the index of next transition to be replaced
    def store(self,s,a,r,s_,done=False):
        a    = np.atleast_2d(a)
        r    = np.atleast_2d(r)
        s    = np.atleast_2d(s)
        s_   = np.atleast_2d(s_)
        done = np.atleast_2d(done)
        n_new = len(a) # number of new transitions to be added
        # If the memory is empty, will create space for memories
        if not hasattr(self, 'a'):
            self.a = np.zeros( [self.max_size] + list(a.shape[1:])    )
            self.r = np.zeros( [self.max_size] + list(r.shape[1:])    )
            self.s = np.zeros( [self.max_size] + list(s.shape[1:])    )
            self.s_= np.zeros( [self.max_size] + list(s_.shape[1:])   )
            self.d = np.zeros( [self.max_size] + list(done.shape[1:]) ).astype(done.dtype)

        rest = self.max_size - self.pointer # number of transitions on the right side of pointer
        if rest >= n_new: 
            self.a[self.pointer:self.pointer+n_new] = a
            self.r[self.pointer:self.pointer+n_new] = r
            self.s[self.pointer:self.pointer+n_new] = s
            self.s_[self.pointer:self.pointer+n_new]= s_
            self.d[self.pointer:self.pointer+n_new] = done
            self.pointer = (self.pointer + n_new) % self.max_size
            self.size = min(self.size+n_new, self.max_size)
        # If memory size reaches max size. Use a pointer to over write old memories.
        # if n_new is large but smaller than the max_size, we need to divide new transitions into two parts. First parts add to right side of pointer, the rest part add to the begining of memory
        elif n_new < self.max_size:
            # first part of new transitions store to right side of pointer. Number of transitions stored: rest
            self.a[self.pointer:] = a[:rest]
            self.r[self.pointer:] = r[:rest]
            self.s[self.pointer:] = s[:rest]
            self.s_[self.pointer:]= s_[:rest]
            self.d[self.pointer:] = done[:rest]
            # second part store to the begining. Number of transitions stored: n_new-rest
            self.a[:(n_new-rest)] = a[rest:]
            self.r[:(n_new-rest)] = r[rest:]
            self.s[:(n_new-rest)] = s[rest:]
            self.s_[:(n_new-rest)]= s_[rest:]
            self.d[:(n_new-rest)] = done[rest:]
            self.pointer = n_new-rest
            self.size = self.max_size
        # if n_new >= max_size, we store last "max_size" new transitions to memory
        else:
            self.a = a[-self.max_size:]
            self.r = r[-self.max_size:]
            self.s = s[-self.max_size:]
            self.s_= s_[-self.max_size:]
            self.d = done[-self.max_size:]
            self.pointer = 0
            self.size = self.max_size

    def store_slow(self,s,a,r,s_,done=False):
        '''
        slow version of storing transitions. The deleting memory operations are slow especially for large memory.
        '''
        a = np.atleast_2d(a)
        r = np.atleast_2d(r)
        s = np.atleast_2d(s)
        s_= np.atleast_2d(s_)
        done = np.atleast_2d(done)
        if not hasattr(self, 'a'):
            self.a = a  # action. shape=[None, 1]
            self.r = r  # reward. shape=[None, 1]
            self.s = s  # state.  shape=[None, n_f1, n_f2, n_channel]
            self.s_= s_ # new state. shape=[None, n_f1, n_f2, n_channel]
            self.d = done  # game end label. shape=[None, 1]. done=True means game over
        else:
            self.a = np.concatenate((self.a, a), axis=0)
            self.r = np.concatenate((self.r, r), axis=0)
            self.s = np.concatenate((self.s, s), axis=0)
            self.s_= np.concatenate((self.s_, s_), axis=0)
            self.d = np.concatenate((self.d, done), axis=0)

        n_delete = len(self.a) - self.max_size
        if n_delete>0:
            self.a = np.delete(self.a, range(n_delete), 0)
            self.r = np.delete(self.r, range(n_delete), 0)
            self.s = np.delete(self.s, range(n_delete), 0)
            self.s_= np.delete(self.s_,range(n_delete), 0)
            self.d = np.delete(self.d, range(n_delete), 0)
        self.size = len(self.a)
    def sample(self, batch_size):
        if self.size==0:
            return None,None,None,None,None
        idx = np.random.choice(self.size, size=batch_size)
        s  = self.s[idx]
        s_ = self.s_[idx]
        a  = self.a[idx].astype(int)
        r  = self.r[idx]
        d  = self.d[idx]
        return s, a, r, s_, d
    def clear(self):
        del self.s, self.a, self.r, self.s_, self.d, 
        self.pointer = 0
        self.size = 0
    def save(self,filename):
        np.savez(filename,
            s = self.s[self.size:], 
            a = self.a[self.size:], 
            r = self.r[self.size:], 
            s_= self.s_[self.size:], 
            d = self.d[self.size:] )
    def load(self,filename):
        data = np.load(filename,encoding='latin1')
        a = data['a']
        r = data['r']
        s = data['s']
        s_= data['s_']
        d = data['d']
        self.store(s,a,r,s_,d)
        #self.size = len(self.a)



def test_tuple_input():
    buildcnn = DeepQNetwork.build_cnn
    s = tf.placeholder(tf.float32, [None] + list([30,30,3]), name='s')  # state, input for eval net
    target = tf.placeholder(tf.float32, [None, 2], name='s2')  # state, input for eval net
    name = 'test'
    buildcnn(None,s,target,name)
def test_unpack():
    def rand_a(*args,**kwargs):
        return np.random.randint(0,5)
    from env_visrl_global_vision import visRL
    from run import run
    env = visRL()
    n_input = env.observation_space.low.shape
    agent = DeepQNetwork(5,n_input)
    agent.choose_action = rand_a
    mb_s, mb_a, mb_r, mb_done, ep_r = run(env, agent, n_steps=10, learn=False)
    pos, fig = agent.memory.unpack_obs(agent.memory.s)

    print(type(pos),pos.shape, fig.shape)
    mb_s = np.array(mb_s)
    print(mb_s.shape)
    print(agent.memory.s.shape,
        agent.memory.a.shape,
        agent.memory.r.shape,
        agent.memory.d.shape)
    print(pos[0:10],mb_s[:,0])
    #print(agent.memory.a[0:10])
    import cv2
    cv2.imshow('test',fig[0])
    cv2.waitKey(0)
if __name__ == "__main__":
    #test_tuple_input()
    test_unpack()