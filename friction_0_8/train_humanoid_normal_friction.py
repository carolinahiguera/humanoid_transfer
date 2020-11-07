import gym
import pybulletgym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2


env = gym.make('HumanoidPyBulletEnv-v0')

model = PPO2(
    policy=MlpPolicy, 
    env=env, 
    verbose=1,
    # normalize = True,
    n_steps = 2048,
    nminibatches = 32,
    lam = 0.95,
    gamma = 0.99,
    noptepochs = 10,
    ent_coef = 0.0,
    learning_rate = 2.5e-4,
    cliprange = 0.2,
    full_tensorboard_log=True,
    tensorboard_log='./normal_friction/logs',
    )

# env.render()
obs = env.reset()

model.learn(total_timesteps=int(1e7))
model.save("./normal_friction/ppo2_Humanoid_normalFriction")