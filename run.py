import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, A2C
from TEWAenv import TEWAEnv  # Import your environment
from gymnasium.spaces import Box
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import imageio
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt
from EnvWoMask import EnvWoMask



# ✅ Create the same environment
# env = TEWAEnv(
#     num_threats=10, num_weapons=4, battlefield_size=150,
#     missiles_per_weapon=1, max_assignments=0,
#     max_threats=10, max_weapons=10, max_missiles_per_weapon=5
# )
env = EnvWoMask(num_threats=10, num_weapons=3, battlefield_size=150, missiles_per_weapon=4, max_assignments=1)
env_w = Monitor(env, filename=None)

def mask_fn(inner_env):
    # build_action_mask() -> (slots, max_threats)
    return inner_env.build_action_mask().reshape(-1)

env_w = ActionMasker(env_w, action_mask_fn=mask_fn)

# ✅ Load the trained model
model = A2C.load("tewa_a2c_max1")  # or path you saved
print("✅ Loaded MaskablePPO model")

# **6️⃣ Run the Trained Model on the Environment**
obs, _ = env_w.reset()
done = False

# open  (mp4)
# fig = plt.figure(figsize=(8,5), dpi=150)
# out_path = "vids/tewa_run_cv1.mp4"
# w_px = int(fig.get_figwidth() * fig.get_dpi())
# h_px = int(fig.get_figheight() * fig.get_dpi())
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')   # or 'avc1' / 'H264' if available
# writer = cv2.VideoWriter(out_path, fourcc, 15.0, (w_px, h_px))



for _ in range(50):
    action, _states = model.predict(obs,deterministic=True)
    if isinstance(action, np.ndarray) and action.shape[0] == 1:
        action = action.squeeze()
    obs, reward, done, _, _ = env.step(action)
    env.render(action)  # Visualize the assignments
    # fig.canvas.draw()
    # use buffer_rgba approach then convert to BGR for OpenCV
    # arr = np.asarray(fig.canvas.buffer_rgba())   # H x W x 4
    # rgb = arr[:, :, :3]
    # bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # writer.write(bgr)
    if done:
        print("\n Simulation complete.")
        break

# writer.release()
# plt.close(fig)
# print(f"Saved video to {out_path}")
