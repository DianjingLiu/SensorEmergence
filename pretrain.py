from run import *
def predict(agent, imgs):
    feed_dict = {agent.s:imgs}
    pos = agent.sess.run(agent.pretrain_pos, feed_dict=feed_dict)
    return pos
def sample(env, batchsize):
    # sample data
    imgs, labels = [], []
    for _ in range(batchsize):
        imgs.append(env.reset())
        if args.obstacle:
            obstacle = sorted(env.obstacles)
            obstacle = np.reshape(obstacle, 4) /(env.mapsize[0]-1)
            labels.append(obstacle)
        else:
            labels.append(np.concatenate([env.x, env.x0])/(env.mapsize[0]-1))
    imgs = np.concatenate(imgs, axis=0) # each state is shape (1,64,64,3), concatenate the axis 0
    labels = np.atleast_2d(labels)
    return imgs, labels
def pretrain(env, agent, args):
    import numpy as np
    batchsize = agent.batch_size
    losses = []
    for step in range(args.n_steps):
        imgs, labels = sample(env, batchsize)
        # train
        cost = agent.pretrain(imgs,labels)
        losses.append(cost)
        del imgs, labels
        # Report, save records and models
        global_steps = agent.get_global_step()
        if (step+1) %100 == 0:
            print("Global step {}, loss {}".format(step, np.mean(losses[-100:])/100))
            visualize_modulator(agent,   args.path + 'modulator{:03d}.png'.format(step))
            visualize_sensor(env, agent, args.path + 'sensor{:03d}.png'.format(step))
            if args.path_model: agent.save_pretrain(args.path_model)
    # Plot learning curve
    np.save(args.path+"learning_curve.npy", losses)
    import matplotlib.pyplot as plt
    losses = np.convolve(losses, np.ones((100,))/100, mode='valid')
    plt.plot(losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.ylim(bottom=0)
    plt.savefig(args.path + "learning_curve.png")
    plt.close()
def test(env, agent):
    batchsize = 5
    imgs, labels = sample(env, batchsize)
    outputs = predict(agent, imgs)
    print('============')
    print(labels)
    print('-----------')
    print(outputs)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', default='./log_pretrain/', action='store', help='Path training log directory.')
    parser.add_argument('-m', '--path_model', default=None, action='store', help='Path to save models. If None, will not save the trained model.')
    parser.add_argument('-s', '--n_steps', default=5000, type=int, action='store', help='Training steps.')
    parser.add_argument('-obs', '--obstacle', default=False, action='store_true', help="If True, will predict obstacle positions.")
    parser.add_argument('-bkg', '--background', default=None, action='store', help='Path of environment background path. If None, will use default.')
    parser.add_argument('-gray', '--gray_img', default=False, action='store_true', help='If true, the environment outputs gray-scale images.')
    parser.add_argument('-size', '--mapsize', default=8, type=int, action='store', help='.')
    parser.add_argument('-lr', '--learning_rate', default=0.5e-4, type=float, action='store', help='.')
    args = parser.parse_args()
    # Set params
    args.path = os.path.join(args.path, '')
    args.path_model = os.path.join(args.path_model, '')
    if args.path: os.makedirs(args.path, exist_ok=True)
    if args.path_model: os.makedirs(args.path_model, exist_ok=True)
    log['BACKGROUND'] = args.background
    log['MAP_SIZE'] = args.mapsize
    if args.obstacle:
        log['N_OBSTACLES'] = 2
    log['recog_obstacle'] = args.obstacle
    log['GRAY_IMG'] = args.gray_img
    log['LEARNING_RATE'] = args.learning_rate
    # Write log file
    with open(args.path + "note.csv","w") as logfile:
        for key, value in log.items():
            logfile.write("{},{}\n".format(key, value))
    
    env, agent = get_env_agent(log)
    pretrain(env, agent, args)