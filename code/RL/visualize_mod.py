import numpy as np
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import os
from pdb import set_trace
from mpl_toolkits.mplot3d import Axes3D
def visualize_modulator_3d(agent,path):
    #if not type(agent) == optical_dqn or not type(agent) == optical_dqn_mm: return
    modulator = agent.show_modulator()
    shape = modulator.shape
    X = np.arange(-1, 1, 2/shape[0])
    Y = np.arange(-1, 1, 2/shape[1])
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, modulator, cmap=cm.coolwarm,linewidth=0.1, antialiased=True)
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.xaxis.set_major_locator(LinearLocator(5))
    ax.yaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_locator(LinearLocator())
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf,shrink=0.5,aspect=10)
    plt.savefig(path, bbox_inches='tight')
    plt.show()
    plt.close()
def load_agent(path):
    from env_baseline import visRL
    from optical_dqn import optical_dqn
    from run import log
    env = visRL()
    n_input = env.observation_space.low.shape
    agent = optical_dqn(n_input=n_input,n_actions=5,unitchange=log['UNITCHANGE'])
    agent.restore(path)
    return agent
if __name__ == "__main__":
    MODEL_PATH = "./models/dqn-90000"
    agent = load_agent(MODEL_PATH)
    visualize_modulator_3d(agent, 'modulator-90000.png')