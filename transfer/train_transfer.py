import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
sys.path.append('./ppo2')
sys.path.append('./common')


import gym
import pybulletgym
import pickle
from common.policies import MlpPolicy
from common import make_vec_env
from ppo2.ppo2_transfer import PPO2_TRANSFER



path_TS = f'../friction_0_8/model/'
mname_TS = 'ppo2_Humanoid_friction_0_8'

path_TD = f'./model/'
mname_TD = 'ppo2_Humanoid_transfer_0_3'

pkl_filename = f'{path_TD}/svm_model_3seq_T0.pkl'
with open(pkl_filename, 'rb') as file:
    svm_TS = pickle.load(file)

# env = gym.make('HumanoidPyBulletEnv-v0')
env = make_vec_env('HumanoidPyBulletEnv-v0', n_envs=8)

model = PPO2_TRANSFER(
    policy=MlpPolicy, 
    env=env, 
    verbose=1,
    path_model_Tsource = f'{path_TS}/{mname_TS}',
    one_classifier = svm_TS,
    n_steps = 2048,
    nminibatches = 32,
    lam = 0.95,
    gamma = 0.99,
    noptepochs = 10,
    ent_coef = 0.0,
    learning_rate = 2.5e-4,
    cliprange = 0.2,
    full_tensorboard_log=True,
    tensorboard_log=f'{path_TD}/logs',
    )

# env.render()
obs = env.reset()

model.learn(total_timesteps=int(1e7))
model.save(f'{path_TD}/{mname_TD}')