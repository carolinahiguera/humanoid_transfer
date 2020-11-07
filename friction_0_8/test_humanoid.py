import gym
import pybulletgym
import time

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2



# env = pybullet_envs.make('HumanoidBulletEnv-v0')
# env = make_vec_env('HumanoidPyBulletEnv-v0', n_envs=1)
env = gym.make('HumanoidPyBulletEnv-v0')

env.render()
obs = env.reset()

# while True:
#     pass

model = PPO2.load("./old/normal_friction/ppo2_Humanoid_normalFriction")

obs = env.reset()

# Enjoy trained agent
while True:
    action, _, _ = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    time.sleep(0.05)
    if(dones==True):
        print('Un true en done')
        env.reset()
    env.render()