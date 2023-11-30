import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


df = pd.read_csv('loss_grad_test.csv',comment='#',header=None)
print(df.head())
data = df.to_numpy()
steps = data[:, 0]
grads = data[:, 2:]
print(data.shape, grads.shape)
print(data.dtype)
grads = np.mean(grads, axis=1)
grads = moving_average(grads, 4)
grads = np.concatenate(([grads[0]]*3, grads)) # padding

# plot Network learning
#fig, ax1 = plt.subplots(figsize=(6, 2))
#ax1.set_xlabel('Step ($\\times10^3$)')
plt.figure(figsize=(6,3))
plt.fill_between(steps/1000, grads/2.5, color='tab:blue', alpha=0.15, interpolate=True)
plt.plot(steps/1000, grads/2.5, color='tab:blue', label='Neural Network $w$',linewidth=2)
plt.xlabel('Step ($\\times10^3$)')

df = pd.read_csv('sensor.csv',header=None)
print(df.head())
df = df[:750]

steps = df[0].to_numpy()
m = df[range(1,101)].to_numpy()
'''print(steps)
print(m.shape)
print(m[4])'''

grad = np.diff(m, axis=0)*1e4
for i in range(len(grad)):
	grad[i] = grad[i] - np.mean(grad[i])
avg_grad = np.sqrt(np.mean(grad**2, axis=1))

n=15
avg_grad = moving_average(avg_grad, n)
#plt.plot(steps[:-n],avg_grad)
#plt.ylim([0,None]); plt.xlim([0,None])


avg_m = np.zeros([m.shape[0]-n+1,m.shape[1]])
for i in range(m.shape[1]):
	avg_m[:,i] = moving_average(m[:,i],n)
avg_grad = np.diff(avg_m, axis=0)*1e4

avg_grad = np.sqrt(np.mean(avg_grad**2, axis=1))*1.2
# plot sensor learning
plt.fill_between(steps[:-n]/1000, avg_grad/2.5, color='tab:red', alpha=0.15, interpolate=True)

plt.plot(steps[:-n]/1000,avg_grad/2.5, color='tab:red', label='Sensor $\epsilon$', linewidth=2)
plt.ylim([0,1.5]); plt.xlim([0,140])
plt.locator_params(axis='y', nbins=3)

plt.rcParams["mathtext.fontset"] = "cm" # allow correct display of \epsilon
#plt.rc('text', usetex=True)
#plt.show()
plt.ylabel('Learning speed')
plt.legend(loc=(0.61,0.3), framealpha=1)
plt.gcf().subplots_adjust(bottom=0.2)
plt.savefig('fig2b.svg',transparent=True,format='svg')
plt.show()