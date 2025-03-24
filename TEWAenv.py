import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class TEWAEnv(gym.Env):
    """
    TEWA Environment with Threat Evaluation, Multiple Missile Assignments, and Assignment Stability.
    """

    def __init__(self, num_threats, num_weapons, battlefield_size, missiles_per_weapon, max_assignments):
        super(TEWAEnv, self).__init__()

        self.num_threats = num_threats
        self.num_weapons = num_weapons
        self.missiles_per_weapon = missiles_per_weapon
        self.battlefield_size = battlefield_size
        self.max_assignments = max_assignments  # Maximum number of missiles per threat
        self.previous_action = None  
        self.max_threats = num_threats  # Original number of threats
        self.total_missiles = self.num_weapons * self.missiles_per_weapon
        # Observation space: [Threat positions, speeds,  severity, missile availability]
        self.observation_space = spaces.Box(
            low=0, high=battlefield_size, shape=(num_threats * 4 + num_weapons*missiles_per_weapon,), dtype=np.float32
        )
        

        # Action space: Each weapon selects multiple threats (up to available missiles)
        self.action_space = spaces.MultiDiscrete([num_threats for _ in range(num_weapons * missiles_per_weapon)])
        # Initialize tracking dictionary (threat ID → assigned time steps)
        self.assignment_duration = {i: 0 for i in range(num_threats)}

        self.reset()

    def reset(self, seed=None, options=None):
        """
        Resets the environment with random threats and missile status.
        """
        np.random.seed(seed)
        # ✅ Restore num_threats to its original value
        self.num_threats = self.max_threats  # Ensure threats are reinitialized correctly

        # Initialize threats: [X, Y, Speed, Orientation, Severity]
        self.threats = np.column_stack((
            np.random.uniform(10, self.battlefield_size - 10, self.num_threats),  
            np.random.uniform(10, self.battlefield_size - 10, self.num_threats),  
            np.random.uniform(1, 5, self.num_threats),  
            #np.random.uniform(0, 360, self.num_threats),  
            np.random.uniform(0.5, 1, self.num_threats)  
        ))

        # Initialize weapons at separate positions
        weapon_x_positions = np.linspace(10, self.battlefield_size - 10, self.num_weapons)
        weapon_y_positions = np.full(self.num_weapons, self.battlefield_size / 2)

        self.weapons = np.column_stack((
            weapon_x_positions,  
            weapon_y_positions,  
            np.full(self.num_weapons, self.missiles_per_weapon)  
        ))

        self.steps = 0
        self.previous_action = np.full((self.num_weapons, self.missiles_per_weapon), -1)  
        padded_threats = np.zeros((self.max_threats, 4))  # Create a zero matrix
        padded_threats[:len(self.threats)] = self.threats
        self.state = np.concatenate((padded_threats.flatten(), self.weapons.flatten()))
        print(f"🔄 Reset: {self.num_threats} threats initialized.")  

        return self.state.astype(np.float32), {}

    def evaluate_threats(self):
        """
        Evaluates threats based on severity, proximity, and heading toward base.
        The final danger score is normalized between 0-100.
        """
        base_x, base_y = self.battlefield_size / 2, self.battlefield_size / 2
        threat_evaluation = []

        for i in range(self.num_threats):
            threat_x, threat_y, speed, severity = self.threats[i]

            # Distance to base (normalized between 0 and 100, where closer = more dangerous)
            distance = np.linalg.norm([base_x - threat_x, base_y - threat_y])
            distance_score = max(0, 100 - (distance / self.battlefield_size) * 100)  # Closer threats get higher scores

            # Heading difference (normalized where 0° direct hit is most dangerous)
            # heading_to_base = np.degrees(np.arctan2(base_y - threat_y, base_x - threat_x))
            # heading_diff = min(abs(heading_to_base - orientation), 360 - abs(heading_to_base - orientation))
            # heading_score = max(0, 100 - (heading_diff / 180) * 100)  # Directly heading threats get 100

            # Severity is already between 0.5 and 1, scale to 0-100
    
            severity_score = ((severity - 0.5) / 0.5) * 100   

            # Speed contribution (normalize based on a maximum expected speed of 10)
            speed_score = min(100, (speed / 10) * 100)  # Speed of 10 is considered max danger

            # **Final danger score: Weighted sum of all factors**
            danger_score = (
                (severity_score * 0.4) +  # Severity is most important
                (distance_score * 0.3) +  # Distance to base is crucial
                #(heading_score * 0.2) +   # Heading toward base adds risk
                (speed_score * 0.2)       # Faster threats are slightly more dangerous
            )

            # Ensure the final score is between 0-100
            danger_score = np.clip(danger_score, 0, 100)

            threat_evaluation.append((i, danger_score, distance_score, severity_score, speed_score))

        # Sort threats by highest danger first
        threat_evaluation.sort(key=lambda x: x[1], reverse=True)

        return threat_evaluation

    def step(self, action):
        """
        Executes an action, eliminates threats after being assigned to the SAME weapon for 7+ seconds.
        Includes a stability reward to encourage consistent assignments.
        """
        self.steps += 1
        reward = 0
        done = False
        stability_reward = 0
            # Get the sorted Threat Evaluation List (Highest danger first)
        threat_evaluation = self.evaluate_threats()
        threat_ranking = {threat[0]: rank for rank, threat in enumerate(threat_evaluation)}
        step_missile_usage = {i: 0 for i in range(self.num_weapons)}
        

        # Convert action into a (num_weapons × missiles_per_weapon) matrix
        action = np.array(action).reshape(self.num_weapons, self.missiles_per_weapon)

        # Track assigned threats this step
        assigned_this_step = {}
        threat_assignments = {i: 0 for i in range(self.num_threats)}
        self.tracked_assignments = []

        # Initialize tracking if not already done
        if not hasattr(self, "weapon_assignment_duration"):
            self.weapon_assignment_duration = {}  # (weapon_idx, threat_idx) → duration

        if not hasattr(self, "previous_weapon_assignment"):
            self.previous_weapon_assignment = {}  # (threat_idx) → previous weapon_idx

        print(f"[DEBUG] Active Threats at Step {self.steps}: {list(range(self.num_threats))}")
        
        # **Process missile assignments while respecting `max_assignments_per_threat`**
        for weapon_idx in range(self.num_weapons):
            for missile_idx in range(self.missiles_per_weapon):
                if step_missile_usage[weapon_idx] < self.weapons[weapon_idx, 2]:  # Check if weapon has missiles left
                    threat_idx = action[weapon_idx, missile_idx]

                    if 0 <= threat_idx < self.num_threats and threat_assignments[threat_idx] < self.max_assignments:
                        assigned_this_step[threat_idx] = weapon_idx  # Store current assignment
                        self.tracked_assignments.append((weapon_idx, threat_idx))
                        threat_assignments[threat_idx] += 1  # Track assignments
                        reward += 2
                        # Decrease missile count for the assigned weapon
                        step_missile_usage[weapon_idx] += 1
                         # **Reward Based on Threat Priority**
                        danger_rank = threat_ranking.get(threat_idx, len(threat_evaluation))  # Get threat rank
                        max_rank = len(threat_evaluation)

                        # Assign higher reward for targeting **high-risk threats** (low rank)
                        reward += (max_rank - danger_rank) * 4 

                        # **Only increase duration if the same weapon is targeting the same threat**
                        if threat_idx in self.previous_weapon_assignment and self.previous_weapon_assignment[threat_idx] == weapon_idx:
                            self.weapon_assignment_duration[(weapon_idx, threat_idx)] = self.weapon_assignment_duration.get((weapon_idx, threat_idx), 0) + 1
                        else:
                            self.weapon_assignment_duration[(weapon_idx, threat_idx)] = 1  # Reset if new assignment
        
        for threat_idx in range(self.num_threats): #check for unassigned threats and give penality according to their rank
            could_be_assigned = sum(self.weapons[:, 2]) > 0 and threat_assignments[threat_idx] < self.max_assignments
            if threat_idx not in assigned_this_step and threat_idx in threat_ranking and could_be_assigned:
                low_priority_penalty = (1 - threat_ranking[threat_idx] / len(threat_evaluation)) * -2  # Penalize more for dangerous threats
                reward += low_priority_penalty
        
        # ✅ Count unused missiles and unassigned threats
        total_available_missiles = int(np.sum(self.weapons[:, 2]))
        total_assignments = sum(threat_assignments.values())
        available_threats = sum(1 for t in threat_assignments.values() if t < self.max_assignments)

        if available_threats > 0 and total_assignments < min(self.max_assignments * self.num_threats, total_available_missiles):
            print("🚨 Resource under-utilization detected!")
            print(f"    🔸 Available missiles: {total_available_missiles}")
            print(f"    🔸 Threats assignable: {available_threats}")
            print(f"    🔸 Assignments made:  {total_assignments}")
            reward -= 2  # Optional: penalize for not assigning when able

        # Optional: bonus reward for using missiles effectively
        if total_assignments == min(self.max_assignments * self.num_threats, total_available_missiles):
            reward += 2  # Encourage full usage

        # **Calculate Stability Reward**
        if self.previous_weapon_assignment:
            stability_reward = sum(
                1 for threat, weapon in assigned_this_step.items()
                if self.previous_weapon_assignment.get(threat) == weapon
            )
            reward += stability_reward * 2  # Adjust multiplier as needed

         # ✅ Debug print to check missiles left for each weapon
        print("\n🚀 **Missile Count After Assignments** 🚀")
        for weapon_idx in range(self.num_weapons):
            print(f"Weapon {weapon_idx}: {int(self.weapons[weapon_idx, 2])} missiles left")

        # **Apply a Penalty if a Threat Gets Too Close to a Weapon**
        danger_zone = 10  # Define how close is "too close"
        close_threat_penalty = -10  # Penalty value

        for threat_idx in range(self.num_threats):
            threat_x, threat_y, _, _, = self.threats[threat_idx]
            
            for weapon_idx in range(self.num_weapons):
                weapon_x, weapon_y, _ = self.weapons[weapon_idx]
                
                # Calculate distance between the threat and the weapon
                distance = np.linalg.norm([threat_x - weapon_x, threat_y - weapon_y])
                
                if distance < danger_zone:  # If the threat is too close, apply a penalty
                    reward += close_threat_penalty
                    print(f"⚠️ Penalty! Threat {threat_idx} is too close to Weapon {weapon_idx} (Dist: {distance:.2f})")

        # **Update assignment durations (Only for the same weapon-threat pair)**
        for threat_idx in range(self.num_threats):
            if threat_idx in assigned_this_step:  # If threat is assigned
                current_weapon = assigned_this_step[threat_idx]
                if (current_weapon, threat_idx) in self.weapon_assignment_duration:
                    self.assignment_duration[threat_idx] = self.weapon_assignment_duration[(current_weapon, threat_idx)]
            else:
                self.assignment_duration[threat_idx] = 0  # Reset if no assignment

        # **Store previous assignments for next step comparison**
        self.previous_weapon_assignment = assigned_this_step.copy()

        # **Eliminate threats assigned for 7+ seconds**
        threats_to_remove = [tid for tid, duration in self.assignment_duration.items() if duration >= 3]

        if threats_to_remove:
            threats_to_remove = [tid for tid in threats_to_remove if tid < len(self.threats)]  # Ensure valid indices
            self.threats = np.delete(self.threats, threats_to_remove, axis=0)  # Remove threats
            self.num_threats = len(self.threats)
            reward += len(threats_to_remove) * 10  # Reward for eliminating threats
            for weapon_idx, threat_idx in self.tracked_assignments:
                if threat_idx in threats_to_remove:
                    self.weapons[weapon_idx, 2] = max(0, self.weapons[weapon_idx, 2] - 1)
            

            # Remove eliminated threats from tracking
            self.weapon_assignment_duration = {
                (w, t): d for (w, t), d in self.weapon_assignment_duration.items() if t not in threats_to_remove
            }

        # **Threats continue moving**
        for i in range(self.num_threats):
            angle_rad = np.radians(self.threats[i, 3])  
            self.threats[i, 0] += self.threats[i, 2] * np.cos(angle_rad)  
            self.threats[i, 1] += self.threats[i, 2] * np.sin(angle_rad)  
            self.threats[i, 0] = np.clip(self.threats[i, 0], 0, self.battlefield_size)
            self.threats[i, 1] = np.clip(self.threats[i, 1], 0, self.battlefield_size)

        # **Check if all threats are eliminated**
        if self.num_threats == 0:
            reward += 2500 / self.steps
            print("All threats are killed")
            done = True 

        total_missiles_left = np.sum(self.weapons[:, 2])
        if total_missiles_left <= 0 and done != True:
            print("❌ All weapons are out of missiles!")
            reward += 3
            done = True

        if self.steps >= 500:
            reward -= 10
            done = True

        # **Fix Observation Shape (Padding to Keep it Constant)**
        padded_threats = np.zeros((self.max_threats, 4))  # Create a zero matrix of max size
        padded_threats[:len(self.threats)] = self.threats  # Fill existing threats
        
        self.state = np.concatenate((padded_threats.flatten(), self.weapons.flatten())) 

        # **Print debug information**)
        # Filter out removed threats from the weapon_assignment_duration dictionary
        self.weapon_assignment_duration = {
            (weapon, threat): duration
            for (weapon, threat), duration in self.weapon_assignment_duration.items()
            if threat < len(self.threats)
        }
         # Print the **Assignment Durations for Assigned Pairs**
        print(f"\n[Step {self.steps}]\n⏳ **Assignment Durations for Assigned Pairs** ⏳")
        for weapon_idx, threat_idx in self.tracked_assignments:
            if (weapon_idx, threat_idx) in self.weapon_assignment_duration:
                duration = self.weapon_assignment_duration[(weapon_idx, threat_idx)]
                print(f"Weapon {weapon_idx} → Threat {threat_idx}: {duration} steps")

        print("reward:", reward)

        
        return self.state.astype(np.float32), reward, done, False, {}


    def render(self, action):
        """
        Visualizes the battlefield, showing threats, weapons, assignment lines, and threat movement vectors.
        """
        print(f"📡 Rendering: {self.num_threats} threats, {len(self.tracked_assignments)} assignments")
        plt.clf()
        plt.xlim(0, self.battlefield_size)
        plt.ylim(0, self.battlefield_size)
        plt.grid(True)

        # Plot threats with movement vectors (arrows)
        for i in range(self.num_threats):
            plt.scatter(self.threats[i, 0], self.threats[i, 1], color='red', s=100)
            plt.text(self.threats[i, 0] + 2, self.threats[i, 1] + 2, f"T{i}", fontsize=10, color='black')

            # Draw movement direction
            # plt.arrow(self.threats[i, 0], self.threats[i, 1], 
            #           np.cos(np.radians(self.threats[i, 3])) * 5, 
            #           np.sin(np.radians(self.threats[i, 3])) * 5, head_width=2, fc='black')

        # Plot weapons at **separate locations**
        for i in range(self.num_weapons):
            plt.scatter(self.weapons[i, 0], self.weapons[i, 1], color='green', s=150)
            plt.text(self.weapons[i, 0] + 2, self.weapons[i, 1] + 2, f"W{i}", fontsize=10, color='black')

        # Draw missile assignment lines
        action = np.array(action).reshape(self.num_weapons, -1)
        for weapon_idx, threat_idx in self.tracked_assignments:
             if 0 <= threat_idx < len(self.threats):
                plt.plot([self.weapons[weapon_idx, 0], self.threats[threat_idx, 0]],
                        [self.weapons[weapon_idx, 1], self.threats[threat_idx, 1]], 'k--', linewidth=1)
       

        plt.title(f"Battlefield at Step {self.steps}")
        plt.pause(0.6)
        # Print the assignment pairs for each step
        print("\n🔗 **Assignment Pairs (Weapon → Threat)** 🔗")
        for weapon_idx, threat_idx in self.tracked_assignments:
            print(f"Threat {threat_idx} → Weapon {weapon_idx} ")

    

         # Print the **Threat Evaluation List**
        print("\n🔴 **Threat Evaluation List (Most Dangerous First)** 🔴")
        print(f"{'Rank':<5} {'Threat ID':<10} {'Danger Score':<15} {'Distance':<10} {'Severity':<10} {'Speed':<5}")
        print("-" * 80)
        for rank, (tid, danger_score, distance_score, severity_score, speed_score) in enumerate(self.evaluate_threats()):
            print(f"{rank+1:<5} {tid:<10} {danger_score:<15.2f} {distance_score:<10.2f}  {severity_score:<10.2f} {speed_score:<5.2f}")

# # Run the Simulation
# if __name__ == "__main__":
#     plt.ion()
#     env = TEWAEnv(num_threats=5, num_weapons=2, battlefield_size=150, missiles_per_weapon=2, max_assignments=1)
#     state, _ = env.reset()
#     done = False

#     for _ in range(50):  
#         action = env.action_space.sample()  
#         state, reward, done, _, _ = env.step(action)
        
#         # Check for weapons out of missiles and print a warning
#         for i, weapon in enumerate(env.weapons):
#             if weapon[2] <= 0:  # Check missile count
#                 print(f"⚠️ Warning: Weapon {i} is out of missiles!")

#         env.render(action)
#         if done:
#             print("\n🎉 All threats have been eliminated! Simulation complete.")
#             break
#     plt.ioff()
#     plt.show()
