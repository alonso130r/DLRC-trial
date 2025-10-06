import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

envs = make_vec_env("CartPole-v1", n_envs=12, monitor_dir="./logs/monitor") # i have too many threads lol

model = PPO("MlpPolicy", envs, verbose=1, tensorboard_log="./logs")
model.learn(total_timesteps=500000, tb_log_name="cartpole")
model.save("cartpole")