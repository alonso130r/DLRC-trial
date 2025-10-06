import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo

env = gym.make("CartPole-v1", render_mode="rgb_array")
model = PPO.load("cartpole.zip", env=env)

env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda ep: True)

obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, _ = env.reset()
env.close()
