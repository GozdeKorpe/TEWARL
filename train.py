import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, DQN, A2C # You can also try DQN or A2C
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
from stable_baselines3.common.utils import get_linear_fn
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy as maskable_evaluate_policy


log_dir = "./maskPPO_tewa_tensorboard/"
os.makedirs(log_dir, exist_ok=True)

# create factory because wrappers like ActionMasker expect a callable returning env
def make_env_fn():
    def _init():
        env = TEWAEnv(num_threats=7, num_weapons=3, battlefield_size=150,
                      missiles_per_weapon=3, max_assignments=1,
                      max_threats=10, max_weapons=10, max_missiles_per_weapon=5)
        
        # wrap with ActionMasker: mask function reads env.build_action_mask() and flattens it
        def mask_fn(inner_env):
            # build_action_mask returns (slots, max_threats) boolean array
            mask = inner_env.build_action_mask()  # bool array
            # MaskablePPO expects a 1D mask per action slot flattened to match action_space shape
            # Flatten row-major (slot0's mask then slot1's mask ..)
            return mask.reshape(-1)
        env = ActionMasker(env, action_mask_fn=mask_fn)
        # wrap with Monitor for logging
        env = Monitor(env, log_dir)
        return env
    return _init

# Wrap the environment for parallel training (optional, but helps with stability)
vec_env = make_vec_env(make_env_fn(), n_envs=1) 

# **2️⃣ Initialize the RL Model (Using PPO)**

model = MaskablePPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./tewa_tensorboard/",
                    ent_coef=0.05, gamma=0.98)

# **3️⃣ Train the Model**
TIMESTEPS = 20000
model.learn(total_timesteps=TIMESTEPS)


# **4️⃣ Save the Trained Model**
model.save("tewa_maskPPO")
print("✅ Model saved!")

# **5️⃣ Evaluate the Model**
mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)
print(f"🎯 Mean reward: {mean_reward} ± {std_reward}")

plot_results(["maskPPO_tewa_tensorboard"], x_axis='timesteps', num_timesteps=TIMESTEPS, task_name="PPO TEWA")
plt.title("maskPPO TEWA Rewards")  # ✅ Add title separately
plt.grid() 
plt.show()

results = load_results(log_dir)
episode_lengths = results["l"].values  # 'l' = episode length (steps)

plt.figure()
plt.plot(episode_lengths)
plt.xlabel("Episode")
plt.ylabel("Episode Length (steps)")
plt.title("📈 Episode Length Over Time")
plt.grid()
plt.show()




