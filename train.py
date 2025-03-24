import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO  # You can also try DQN or A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from TEWAenv import TEWAEnv  # Import your custom environment
from TERLenv import TERLEnv 
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy, plot_results
import matplotlib.pyplot as plt
import os
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import RecurrentPPO 
from tqdm import tqdm 

log_dir = "./ppo_tewa_tensorboard/"
os.makedirs(log_dir, exist_ok=True)

# ✅ Wrap environment with Monitor to log rewards
env = TEWAEnv(num_threats=4, num_weapons=1, battlefield_size=150, missiles_per_weapon=3, max_assignments=1)
env_v = Monitor(env, log_dir)

# Wrap the environment for parallel training (optional, but helps with stability)
vec_env = make_vec_env(lambda: env_v, n_envs=1)

# **2️⃣ Initialize the RL Model (Using PPO)**
model = RecurrentPPO("MlpLstmPolicy", vec_env, verbose=1, tensorboard_log="./tewa_tensorboard/")

# **3️⃣ Train the Model**
TIMESTEPS = 50000  
model.learn(total_timesteps=TIMESTEPS)

plot_results(["ppo_tewa_tensorboard/"], x_axis='timesteps', num_timesteps=TIMESTEPS, task_name="PPO TEWA")
plt.title("PPO TEWA Rewards")  # ✅ Add title separately
plt.grid() 
plt.show()


# **4️⃣ Save the Trained Model**
model.save("tewa_ppoLSTM4.1_model")
print("✅ Model saved!")

# **5️⃣ Evaluate the Model**
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"🎯 Mean reward: {mean_reward} ± {std_reward}")

results = load_results(log_dir)
episode_lengths = results["l"].values  # 'l' = episode length (steps)

plt.figure()
plt.plot(episode_lengths)
plt.xlabel("Episode")
plt.ylabel("Episode Length (steps)")
plt.title("📈 Episode Length Over Time")
plt.grid()
plt.show()




