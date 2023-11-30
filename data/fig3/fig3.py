"""
Compared with fig3a.py: success_rate ploted on top of loss. The red linewidth is increased, blue linewidth decreased.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
plt.style.use('seaborn-paper')
def plot_avg_style(ax, x, y_raw, color, window=15):
	y = y_raw.rolling(window=window).mean()
	y[:window]=y[window]

	y_std = y_raw.rolling(window=window).std()
	y_std[:window]=y_std[window]
	ax.plot(x, y, linewidth=2, color=color)
	ax.fill_between(x, y - 2*y_std, y + 2*y_std,
                 color=color, alpha=0.2)


def generate(directory, idx):
	df1 = pd.read_csv(os.path.join(directory, "scores_general.csv"), header=None, comment='#')
	df2 = pd.read_csv(os.path.join(directory, "loss_grad_test.csv"), header=None, comment='#')

	configs = {
		'test_base2': {
			'edge': 32,
			'txt1': 32-25,
			'txt2': 32+4
		},
		'test_lowlr': {
			'edge': 96,
			'txt1': 96-35,
			'txt2': 96+10
		},
		'test_highlr': {
			'edge': 21,
			'txt1': 21-20.5,
			'txt2': 21+10
		}
	}
	edge = configs[directory]['edge'] # 24
	txt1 = configs[directory]['txt1']
	txt2 = configs[directory]['txt2']
	end = 200 # 150

	df1 = df1[df1[0]%1000==0]
	df2 = df2[df2[0]%1000==0]
	df1 = df1[df1[0]<=end*1000]
	df2 = df2[df2[0]<=end*1000]
	color1 = 'tab:green'
	color2 = 'tab:purple'
	rect1 = plt.Rectangle((0,0),  edge,  100, facecolor=color1, alpha=0.12)
	rect2 = plt.Rectangle((edge,0), end-edge, 100, facecolor=color2, alpha=0.12)
	fig, ax1 = plt.subplots(figsize=(6, 3))
	ax1.add_patch(rect1)
	ax1.add_patch(rect2)
	ax1.text(txt1,40,
		'Stage I',
		color=color1,
		fontsize=9 if directory=="test_highlr" else 10 # high lr case space is narrow
	)
	ax1.text(txt2,40,
		'Stage II',
		color=color2
	)
	#'''
	color = 'tab:red'
	ax1.set_xlabel('Step ($\\times10^3$)')
	ax1.set_ylabel('Success rate (%)', color=color)
	ax1.set_ylim([0,100])
	#ax1.plot(df1[0]/1000, df1[1]*100, linewidth=2.3, color=color)
	plot_avg_style(ax1, df1[0]/1000, df1[1]*100, color)
	ax1.tick_params(axis='y', labelcolor=color)

	color = 'tab:blue'
	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
	ax2.set_ylabel('Loss', color=color)  # we already handled the x-label with ax1
	ax2.set_ylim([0,1.1])
	#ax2.plot(df2[0]/1000, df2[1], linewidth=1.5, color=color)
	plot_avg_style(ax2, df2[0]/1000, df2[1], color)
	ax2.tick_params(axis='y', labelcolor=color)


	plt.xlim([0,end])

	ax1.set_zorder(1) # plot success rate on top of loss
	ax1.patch.set_visible(False) # prevents ax1 from hiding ax2
	fig.tight_layout()  # otherwise the right y-label is slightly clipped

	#plt.savefig(os.path.join(directory, f"fig3_{idx}.svg"),format="svg")
	plt.savefig(f"fig3_{idx}.svg" ,format="svg")
	#plt.show()

if __name__ == '__main__':
	directories = [
		"test_base2", 
		"test_lowlr", 
		"test_highlr"
	]
	for idx, d in enumerate(directories):
		generate(d, idx)