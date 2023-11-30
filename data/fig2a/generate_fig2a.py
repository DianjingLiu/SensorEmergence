import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
plt.style.use('seaborn-paper')
def plot_avg_style(ax, x, y_raw, color, window=25):
	y = y_raw.rolling(window=window).mean()
	y[:window]=y[window]

	y_std = y_raw.rolling(window=window).std()
	y_std[:window]=y_std[window]
	ax.plot(x, y, linewidth=2, color=color)
	ax.fill_between(x, y - 2*y_std, y + 2*y_std,
                 color=color, alpha=0.2)
	


directory = "."
df1 = pd.read_csv(os.path.join(directory, "scores_general_0.99.csv"), header=None, comment='#')
df2 = pd.read_csv(os.path.join(directory, "loss_grad_test.csv"), header=None, comment='#')
color1 = 'tab:green'
color2 = 'tab:purple'
#fig, ax1 = plt.subplots(figsize=(6, 3))
fig, ax1 = plt.subplots(figsize=(6, 2))
#'''
edge = 32
rect1 = plt.Rectangle((0,0),  edge,  100, facecolor=color1, alpha=0.12)
rect2 = plt.Rectangle((edge,0), 140-edge, 100, facecolor=color2, alpha=0.12)
ax1.add_patch(rect1)
ax1.add_patch(rect2)
ax1.text(edge-25,40,
	'Stage I',
	color=color1
)
ax1.text(edge+10,40,
	'Stage II',
	color=color2
)
#'''
'''print(len(df1))
window=10
success_rate = (df1[1]*100).rolling(window=window).mean()
success_rate[:window]=success_rate[window]
success_rate_std = (df1[1]*100).rolling(window=window).std()
success_rate_std[:window]=success_rate_std[window]'''

color = 'tab:red'
ax1.set_xlabel('Step ($\\times10^3$)')
ax1.set_ylabel('Success rate (%)', color=color)
ax1.set_ylim([0,100])
#plt.plot(df1[0]/1000, df1[1]*100, linewidth=2, color=color)
#plt.plot(df1[0]/1000, success_rate, linewidth=2, color=color)
#plt.fill_between(df1[0]/1000, success_rate - success_rate_std, success_rate + success_rate_std,
#                 color=color, alpha=0.2)
plot_avg_style(ax1, df1[0]/1000, df1[1]*100, color)
ax1.tick_params(axis='y', labelcolor=color)

color = 'tab:blue'
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Loss', color=color)  # we already handled the x-label with ax1
ax2.set_ylim([0,1.1])
#plt.plot(df2[0]/1000, df2[1], linewidth=2, color=color)

plot_avg_style(ax2, df2[0]/1000, df2[1], color, 25)
ax2.tick_params(axis='y', labelcolor=color)
plt.xlim([0,np.array(df1)[-1,0]/1000])
plt.xlim([0,140]) # trim

ax1.set_zorder(1) # plot success rate on top of loss
ax1.patch.set_visible(False) # prevents ax1 from hiding ax2

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(os.path.join(directory, "fig2a.svg"), format='svg')
plt.show()


