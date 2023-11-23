from run import env, agent
from evaluate import evaluate
import glob
import os
def getID(string):
	import re
	return int(re.findall("\d+",string)[-1])
MODEL_PATH = './models/'
model_list = glob.glob(MODEL_PATH+'*.index')
model_list.sort(key=getID)
# log score
log_score = open('./log/scores.csv', 'w')
for path in model_list:
	path,_ = os.path.splitext(path)
	step = getID(path)
	agent.restore(path)
	agent.fixed_epsilon(0.8)
	score = evaluate(env,agent, verbose=False)
	log_score.write("{},{}\n".format(step, score))
	print("Step {}. Score {:.1f}".format(step, score))
log_score.close()