from matplotlib.pyplot import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import *
import tensorflow as tf
def get_loss_simu():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    test_inputs = mnist.test.images
    test_inputs = np.reshape(test_inputs, [-1, 28, 28, 1])
    test_labels = mnist.test.labels
    # rotate test inputs
    test_inputs = test_inputs[:,::-1,::-1,:]

    # test on one instance
    #test_inputs = test_inputs[[62]]
    #test_labels = test_labels[[62]]

    agent = optical_cnn(input_size=[28,28], output_size = 10, learning_rate=0.0001)
    #agent.restore('./models/cnn')
    agent.restore('./models/optical_cnn')
    
    mod = agent.get_mod_params()
    #mod[2] = 0.7

    agent.update_mod(mod)
    mod1 = agent.get_mod_params()
    loss1 = agent.show_loss(test_inputs, test_labels)
    #agent.show_loss(test_inputs[[62]], test_labels[[62]])
    
    mod[2] = mod[2] * 1.05
    agent.update_mod(mod)
    mod2 = agent.get_mod_params()
    loss2 = agent.show_loss(test_inputs, test_labels)
    
    mod[2] = mod[2] / 1.05 * 1.1
    agent.update_mod(mod)
    mod3 = agent.get_mod_params()
    loss3 = agent.show_loss(test_inputs, test_labels)
    print(mod1,mod2,mod3)
    print('losses in simulation: {}, {}, {}'.format(loss1,loss2,loss3))
def get_loss_exp():
    tf.reset_default_graph()
    agent = CNN(input_size=[28,28], output_size = 10, learning_rate=0.0001)
    agent.restore('./models/cnn')
    # show prediction on camera picture
    filepath = './data/processed/img_case2_norm.bmp'
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = img[None,...,None]
    filepath = './data/processed/img_case2_step2_norm.bmp'
    img1 = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.normalize(img1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img1 = img1[None,...,None]
    filepath = './data/processed/img_case2_step1_norm.bmp'
    img2 = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.normalize(img2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img2 = img2[None,...,None]

    img = img[:, ::-1, ::-1, :]
    img1= img1[:, ::-1, ::-1, :]
    img2= img2[:, ::-1, ::-1, :]
    pred = agent.sess.run(agent.pred, feed_dict={agent.input:img})
    prediction = agent.predict(img)
    print(pred, prediction)
    #print('params: '.format(agent.get_mod_params()))
    
    # show loss on camera picture
    label = np.array([[0,0,0,0,0,1,0,0,0,0]])
    loss  = agent.show_loss(img, label)
    loss1 = agent.show_loss(img1,label)
    loss2 = agent.show_loss(img2,label)
    print('losses in exp: {}, {}, {}'.format(loss, loss1, loss2))
if __name__ == '__main__':
    #savepath = './models/dqn_dummytrain.ckpt'
    #savepath = './models/dqn.ckpt'
    #plt_path = 'train.png'
    import os
    os.makedirs('./log', exist_ok=True)
    os.makedirs('./models', exist_ok=True)
    from tensorflow.examples.tutorials.mnist import input_data
    from optical_cnn import optical_cnn
    from CNN import CNN
    #mnist = input_data.read_data_sets('./data/mnist', one_hot=True)
    mnist = input_data.read_data_sets('./data/fashion', one_hot=True)  # fashion mnist dataset
    from pdb import set_trace
    #set_trace()
    import time
    time_start = time.time()
    #'''
    # Train
    agent = optical_cnn(input_size=[28,28], output_size = 10, learning_rate=0.0001, learning_rate_mod=0e-6)
    #agent = CNN(input_size=[28,28], output_size = 10, learning_rate=0.0001)
    agent = train_supervised(mnist, agent, 5000)
    agent.save('./models/cnn')
    #'''
    '''
    
    #'''

    #get_loss_simu()
    #get_loss_exp()
    runtime = time.time() - time_start
    print('Run time: {}s'.format(runtime))

    # visualize learning curve
    import pandas as pd
    df = pd.read_csv('accuracy.csv',header=None)
    df = np.array(df)
    plt.plot(df[:,0], df[:,1]*100, linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Test accuracy (%)')
    plt.xlim([0,5000])
    plt.ylim([0,100])
    plt.savefig('accuracy.png')


