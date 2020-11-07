import os
import numpy as np
import pandas as pd
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sims = ['./transfer/logs/PPO2_1/']

for i in range(0,len(sims)): 
	dpath = f'{sims[i]}'
	dname = os.listdir(dpath)
	dname.sort()
	ea = EventAccumulator(os.path.join(dpath, dname[0])).Reload()
	tags = ['episode_reward','input_info/advantage','loss/loss', ]
	labels = {'episode_reward':'episode_reward', 
			  'input_info/advantage':'advantage', 
			  'loss/loss':'loss'}
	# tags = ['input_info/advantage','loss/loss', ]
	# labels = {'input_info/advantage':'advantage', 
	# 		  'loss/loss':'loss'}

	for tag in tags:
	    tag_values=[]
	    steps=[]
	    for event in ea.Scalars(tag):
	        tag_values.append(event.value)        
	        steps.append(event.step)
	    data = np.column_stack((steps,tag_values))
	    np.save(f'{dpath}dqn_{labels[tag]}.npy', data)
	print('done')
