import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, A2C
from TEWAenv import TEWAEnv  # Import your environment
from gymnasium.spaces import Box

# ✅ Load the trained model
model = A2C.load("tewa_a2c60.5.6")
print("✅ Loaded Trained Model!")

# ✅ Create the same environment
env = TEWAEnv(num_threats=60, num_weapons=5, battlefield_size=150, missiles_per_weapon=6, max_assignments=1)


# **6️⃣ Run the Trained Model on the Environment**
obs, _ = env.reset()
done = False

for _ in range(50):
    action, _states = model.predict(obs)
    if isinstance(action, np.ndarray) and action.ndim > 1:
        action = action.squeeze()
    obs, reward, done, _, _ = env.step(action)
    env.render(action)  # Visualize the assignments
    if done:
        print("\n Simulation complete.")
        break
