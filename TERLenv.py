import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from auction import auction_algorithm  

class TERLEnv(gym.Env):
    """
    TEWA Environment where RL evaluates threats and the Auction Algorithm handles assignments.
    """

    def __init__(self, num_threats=5, num_weapons=2, battlefield_size=100, missiles_per_weapon=2):
        super(TERLEnv, self).__init__()

        self.num_threats = num_threats
        self.num_weapons = num_weapons
        self.missiles_per_weapon = missiles_per_weapon
        self.battlefield_size = battlefield_size

        # New Observation Space (Only Threat Features)
        # Format: [severity, distance, heading_diff, speed] for each threat
        self.observation_space = gym.spaces.Box(
            low=0, high=100, shape=(num_threats, 4), dtype=np.float32
        )

        # New Action Space (Threat Prioritization)
        # RL outputs a ranking of threats 
        self.action_space = gym.spaces.MultiDiscrete([num_threats])

        self.reset()

    def reset(self, seed=None, options=None):
        """
        Resets the environment with random threats and missile status.
        Now using dynamic observations instead of fixed padding.
        """
        np.random.seed(seed)

        # 🔥 Initialize dynamic threats
        self.threats = np.column_stack((
            np.random.uniform(10, self.battlefield_size - 10, self.num_threats),
            np.random.uniform(10, self.battlefield_size - 10, self.num_threats),
            np.random.uniform(1, 5, self.num_threats),
            np.random.uniform(0, 360, self.num_threats),
            np.random.uniform(0.5, 1, self.num_threats)
        ))

        # 🔥 Initialize dynamic weapons
        weapon_x_positions = np.linspace(10, self.battlefield_size - 10, self.num_weapons)
        weapon_y_positions = np.full(self.num_weapons, self.battlefield_size / 2)
        self.weapons = np.column_stack((weapon_x_positions, weapon_y_positions, np.full(self.num_weapons, self.missiles_per_weapon)))

        # Reset tracking variables
        self.steps = 0
        self.weapon_assignment_duration = {}  # Track how long each weapon assigns to each threat
        self.previous_weapon_assignment = {}  # Track previous assignments
        self.tracked_assignments = []  # Store active assignments
        self.assignment_duration = {i: 0 for i in range(self.num_threats)}  # Track time assigned per threat

        # **Dynamic observation generation**
        self.state = self.get_observation()

        return self.state.astype(np.float32), {}


    def get_observation(self):
        """
        Returns the observation containing only relevant threat features.
        Format: [severity, distance, heading_diff, speed] for each threat.
        """
        base_x, base_y = self.battlefield_size / 2, self.battlefield_size / 2
        observation = []

        for i in range(self.num_threats):
            threat_x, threat_y, speed, orientation, severity = self.threats[i]

            # Distance to base
            distance = np.linalg.norm([base_x - threat_x, base_y - threat_y])

            # Heading difference
            heading_to_base = np.degrees(np.arctan2(base_y - threat_y, base_x - threat_x))
            heading_diff = min(abs(heading_to_base - orientation), 360 - abs(heading_to_base - orientation))

            # Normalize features (0-100 scale)
            distance = np.clip((100 - distance) / 100 * 100, 0, 100)
            heading_diff = np.clip((180 - heading_diff) / 180 * 100, 0, 100)
            speed = np.clip(speed / 5 * 100, 0, 100)

            observation.append([severity, distance, heading_diff, speed])

        return np.array(observation, dtype=np.float32)

    def step(self, action):
        """
        Executes an action, which is the prioritized ranking of threats.
        The environment assigns weapons to threats based on this ranking.
        """
        self.steps += 1
        reward = 0
        done = False

        # Ensure action is a NumPy array
        action = np.array(action)

        # Validate the action shape
        if action.shape != (self.num_threats,):
            raise ValueError(f"Action shape mismatch! Expected ({self.num_threats},) but got {action.shape}")

        # **Assign weapons to threats based on TEL**
        assigned_threats = action[:self.num_weapons]  # Take top threats based on weapon count

        # Track assignments
        self.tracked_assignments = []
        for weapon_idx, threat_idx in enumerate(assigned_threats):
            if 0 <= threat_idx < self.num_threats:  # Ensure valid threat index
                self.tracked_assignments.append((weapon_idx, threat_idx))
                reward += 1  # Small reward for making assignments

        # **Reward Based on Priority**
        for rank, threat_idx in enumerate(action):
            if threat_idx < self.num_threats:
                reward += (self.num_threats - rank) * 2  # Higher ranked threats give more reward

        # **Check for Eliminated Threats**
        threats_to_remove = []
        for _, threat_idx in self.tracked_assignments:
            if self.assignment_duration.get(threat_idx, 0) >= 5:  # Threshold for elimination
                threats_to_remove.append(threat_idx)

        # Remove eliminated threats
        if threats_to_remove:
            self.threats = np.delete(self.threats, threats_to_remove, axis=0)
            self.num_threats = len(self.threats)
            reward += len(threats_to_remove) * 10  # Reward for eliminating threats

        # **Threats Move**
        for i in range(self.num_threats):
            self.threats[i, 0] += self.threats[i, 2] * np.cos(np.radians(self.threats[i, 3]))
            self.threats[i, 1] += self.threats[i, 2] * np.sin(np.radians(self.threats[i, 3]))

        # **End Episode If No Threats Left**
        if self.num_threats == 0:
            done = True
            reward += 50  # Large reward for neutralizing all threats


        if self.steps >= 500:
            reward -= 10
            done = True
            
        # 🔥 Dynamically concatenate new observation without padding
        self.state = self.get_observation()

        # ✅ **Debug information**
        print(f"\n[Step {self.steps}] Active Threats: {self.num_threats}")
        for (weapon, threat), duration in self.weapon_assignment_duration.items():
            print(f"Weapon {weapon} → Threat {threat} assigned for {duration} steps")
        print(f"Reward: {reward}")


        return self.state.astype(np.float32), reward, done, False, {}


    def render(self, action=None):
        
        plt.clf()
        plt.xlim(0, self.battlefield_size)
        plt.ylim(0, self.battlefield_size)
        plt.grid(True)

        # 🔥 Plot threats with movement vectors (arrows)
        for i in range(self.num_threats):
            plt.scatter(self.threats[i, 0], self.threats[i, 1], color='red', s=100)
            plt.text(self.threats[i, 0] + 2, self.threats[i, 1] + 2, f"T{i}", fontsize=10, color='black')

            # Draw movement direction
            plt.arrow(self.threats[i, 0], self.threats[i, 1], 
                    np.cos(np.radians(self.threats[i, 3])) * 5, 
                    np.sin(np.radians(self.threats[i, 3])) * 5, 
                    head_width=2, fc='black')

        # 🔥 Plot weapons at separate locations
        for i in range(self.num_weapons):
            plt.scatter(self.weapons[i, 0], self.weapons[i, 1], color='green', s=150)
            plt.text(self.weapons[i, 0] + 2, self.weapons[i, 1] + 2, f"W{i}", fontsize=10, color='black')

        # 🔥 If we have an action, get assignments from Auction Algorithm
        if action is not None:
            reward_matrix = np.zeros((self.num_weapons, self.num_threats))
            for rank, threat_idx in enumerate(action):
                if threat_idx < self.num_threats:
                    reward_matrix[:, threat_idx] = (self.num_threats - rank) * 10  # Higher-ranked threats get higher values

            assignments = auction_algorithm(reward_matrix)  # 🔥 Assign threats using the auction algorithm

            # 🔥 Draw assignment lines
            for weapon_idx, threat_idx in assignments.items():
                if 0 <= threat_idx < len(self.threats):
                    plt.plot([self.weapons[weapon_idx, 0], self.threats[threat_idx, 0]],
                            [self.weapons[weapon_idx, 1], self.threats[threat_idx, 1]], 'k--', linewidth=1)
        
        plt.title(f"Battlefield at Step {self.steps}")
        plt.pause(0.6)

        # 🔴 **Print the Threat Evaluation List**
        print("\n🔴 **Threat Evaluation List (Most Dangerous First)** 🔴")
        print(f"{'Rank':<5} {'Threat ID':<10} {'Danger Score':<15} {'Distance':<10} {'Heading':<15} {'Severity':<10} {'Speed':<5}")
        print("-" * 80)
        
        for rank, (tid, danger_score, distance_score, heading_score, severity_score, speed_score) in enumerate(self.evaluate_threats()):
            print(f"{rank+1:<5} {tid:<10} {danger_score:<15.2f} {distance_score:<10.2f} {heading_score:<15.2f} {severity_score:<10.2f} {speed_score:<5.2f}")
