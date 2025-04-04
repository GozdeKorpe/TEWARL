import gymnasium as gym
import numpy as np

class ActionMaskEnv(gym.Wrapper):
    def __init__(self, env, mask_fn):
        super().__init__(env)
        self.mask_fn = mask_fn

    def action_masks(self):
        masks = []
        for discrete_action in self.action_space.nvec:
            mask = np.ones(discrete_action, dtype=bool)  # initially all True (all actions valid)
            # Set invalid actions as False, for example:
            # mask[invalid_action_index] = False
            masks.append(mask)
        return masks

    def step(self, action):
        mask = self.action_masks()
        for idx, act in enumerate(action):
            if not mask[idx][act]:
                raise ValueError(f"Action component {idx} with value {act} is invalid according to mask.")

        observation, reward, terminated, truncated, info = self.env.step(action)

        return observation, reward, terminated, truncated, info
    
    

