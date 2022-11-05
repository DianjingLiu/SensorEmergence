import numpy as np
def run(env,agent,verbose,criterian='success'):
	reward = 0
	success = 0
	done = False
	s = env.reset()
	while not done:
		a = agent.choose_action(s)
		s1, r, done, _ = env.step(a)
		s = s1
		reward += r
		if r==5:
			success=1
	if verbose: print("episode end, reward {:.2f}. {}".format(reward, "success" if success else "fail"))
	#return reward
	if criterian == 'success':
		return success
	else:
		return reward
def evaluate(env, agent, verbose=True, criterian='success'):
	# let agent run for n times, record the average episode reward.
	N = 100
	max_step = env.max_step
	env.max_step = 100
	score=0
	#agent.fixed_epsilon()
	for episode in range(N):
		score += run(env, agent, verbose, criterian)
	score /= N
	env.max_step = max_step # restore original max_step
	return score
def interactive_test(env, agent, video_path="debug.mp4"):
	import cv2
	def visualize(fig, qvalue):
		# add text to show Q values
		bkg = np.zeros([fig.shape[0], 208, 3]).astype(fig.dtype)
		texts = [ 'Up {:.2f}'.format(qvalue[0]),   
				'Down {:.2f}'.format(qvalue[1]), 
				'Left {:.2f}'.format(qvalue[2]), 
				'Right {:.2f}'.format(qvalue[3]),
				'Pick {:.2f}'.format(qvalue[4]), ]
		texts = [ ' {:.2f}'.format(qvalue[0]),   
				  ' {:.2f}'.format(qvalue[1]), 
				  ' {:.2f}'.format(qvalue[2]), 
				  ' {:.2f}'.format(qvalue[3]),
				  ' {:.2f}'.format(qvalue[4]), ]
		dy, dx = 50, 60
		positions = [(dx,dy), (dx, 3*dy), (0,2*dy), (2*dx, 2*dy), (dx, 2*dy)]
		font      = cv2.FONT_HERSHEY_SIMPLEX
		fontScale = 0.6
		fontColor = (255,255,255)
		lineType  = 2
		for text, pos in zip(texts, positions):
			cv2.putText(bkg, text, pos, font, fontScale, fontColor, lineType)
		# plot bounding box to highlight best action
		action = np.argmax(qvalue)
		(width, height) = cv2.getTextSize(texts[action], font, fontScale=fontScale, thickness=1)[0]
		lowerleft = (positions[action][0]+5, positions[action][1]+5)
		upperright = (lowerleft[0]+width+3, lowerleft[1]-height-10)
		cv2.rectangle(bkg, lowerleft, upperright, fontColor)
		# 
		fig = cv2.cvtColor(fig, cv2.COLOR_BGR2RGB)
		compressed = cv2.resize(fig, (64,64))
		cv2.imwrite("original_img.png",compressed)

		fig = np.concatenate([fig, bkg], axis=1)
		cv2.imshow("game", fig)
		return cv2.waitKey(0), fig
	env.reset()
	if video_path:
		import imageio
		video_writer = imageio.get_writer(video_path, fps=2)
	while True:
		s = env.get_state()
		qvalue = agent.show_q(s) # q values are in the order of [up, down, left, right, pick]
		qvalue = np.squeeze(qvalue)
		#qvalue  = np.array([0.5, 0.4, 0.2145, 0.541646, 0.458687])
		fig = env.plot()
		key, fig = visualize(fig, qvalue)
		key = ord(chr(key).lower())
		if video_path: 
			fig = cv2.cvtColor(fig, cv2.COLOR_BGR2RGB)
			video_writer.append_data(fig)
		if key == ord("q"):
			break
		elif key == ord("r"):
			# reset
			env.reset()
		else:
			action = ['w', 's', 'a', 'd', ' '].index(chr(key))
			env.step(action)
	if video_path: video_writer.close()
if __name__ == "__main__":
	from run import *
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-gpu', '--gpu', default=None, action='store', help='Specify the GPU to be used.')
	parser.add_argument('-s', '--step', default=90000, action='store', help="Model ID to be restored. If a path is provided, this input is ignored.")
	parser.add_argument('-p', '--path', default=None, action='store', help="Model path to be restored.")
	parser.add_argument('-i', '--interact', default=False, action='store_true', help="If False, will test the agent for multiple episodes and output average score. If True, will do the interactive debug.")
	parser.add_argument('-v', '--visual', default=False, action='store_true', help="If True, will do the visualization of the test.")
	parser.add_argument('-e', '--epsilon', default=0.8, type=float, action='store', help="Model epsilon when doing test.")
	parser.add_argument('-c', '--criterian', default='success', action='store', help="Model evaluation criterian. If \'success\', will evaluate success rate")

	args = vars(parser.parse_args())
	if args['gpu']: os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu']
	
	env, agent = get_env_agent(log)
	# Restore model
	if args['path']:
		MODEL_PATH = args['path']
	else:
		MODEL_PATH = "./models/dqn-{}".format(args['step'])
	agent.restore(MODEL_PATH)
	print("Model restored: {}".format(MODEL_PATH))
	agent.fixed_epsilon(args['epsilon'])
	
	if args['interact']:
		interactive_test(env,agent)
	elif args['visual']:
		from utils import visual_test
		visual_test(env,agent)
	else:
		import time
		start = time.time()
		reward = evaluate(env, agent, args['criterian'])
		print("Run time {}s. Average score {:.3f}".format(time.time()-start, reward))