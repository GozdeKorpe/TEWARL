import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import math

# ---------------------------
# Helper: SimpleThreat kinematics (with path history)
# ---------------------------
class SimpleThreat:
    """Threat with simple kinematic motion toward an assigned weapon (limited turn rate).
       Records path history in self.path (list of (x,y) tuples)."""
    def __init__(self, pos, speed, heading_deg, target_weapon_idx, max_turn_deg=30.0, severity=None):
        self.pos = np.array(pos, dtype=float)
        self.speed = float(speed)
        self.heading = float(heading_deg)  # degrees
        self.target_weapon_idx = int(target_weapon_idx)
        self.max_turn_deg = float(max_turn_deg)
        self.severity = severity if severity is not None else np.random.uniform(0.5, 1.0)
        # Path history for visualization
        self.path = [tuple(self.pos.copy())]

    def step_towards(self, goal_pos, dt=1.0):
        dx, dy = goal_pos[0] - self.pos[0], goal_pos[1] - self.pos[1]
        desired_rad = math.atan2(dy, dx)
        desired_deg = math.degrees(desired_rad)
        diff = ((desired_deg - self.heading + 180) % 360) - 180
        max_turn = self.max_turn_deg * dt
        turn = np.clip(diff, -max_turn, max_turn)
        self.heading = (self.heading + turn) % 360
        heading_rad = math.radians(self.heading)
        vx = math.cos(heading_rad) * self.speed * dt
        vy = math.sin(heading_rad) * self.speed * dt
        self.pos += np.array([vx, vy])
        # append new position to path history
        self.path.append((self.pos[0], self.pos[1]))

    def distance_to(self, goal_pos):
        return np.linalg.norm(self.pos - np.array(goal_pos, dtype=float))


# ---------------------------
# Helper: allocate threats by asset values
# ---------------------------
def allocate_targets_by_asset_values(asset_values, num_threats):
    assets = np.array(asset_values, dtype=float)
    if assets.size == 0:
        raise ValueError("asset_values must be non-empty")
    if assets.sum() <= 0:
        proportions = np.ones_like(assets) / len(assets)
    else:
        proportions = assets / assets.sum()
    base_counts = np.floor(proportions * num_threats).astype(int)
    remainder = int(num_threats - base_counts.sum())
    order = np.argsort(-assets)  # descending asset indices
    i = 0
    while remainder > 0:
        base_counts[order[i % len(order)]] += 1
        remainder -= 1
        i += 1
    assignment_list = []
    for w_idx, count in enumerate(base_counts):
        assignment_list += [w_idx] * int(count)
    if len(assignment_list) != num_threats:
        if len(assignment_list) > num_threats:
            assignment_list = assignment_list[:num_threats]
        else:
            assignment_list += [int(order[0])] * (num_threats - len(assignment_list))
    return base_counts.tolist(), assignment_list


# ---------------------------
# TEWAEnv (randomized asset at reset + path visualization + debug prints restored)
# ---------------------------
class TEWAEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, num_threats, num_weapons, battlefield_size, missiles_per_weapon, max_assignments):
        super(TEWAEnv, self).__init__()

        self.max_threats = num_threats
        self.num_weapons = num_weapons
        self.missiles_per_weapon = missiles_per_weapon
        self.battlefield_size = battlefield_size
        self.max_assignments = max_assignments
        self.previous_action = None
        self.total_missiles = self.num_weapons * self.missiles_per_weapon

        # Observation: pad to max_threats
        self.observation_space = spaces.Box(
            low=0, high=float(battlefield_size),
            shape=(self.max_threats * 4 + self.num_weapons * 3,),
            dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([self.max_threats for _ in range(self.num_weapons * self.missiles_per_weapon)])

        # internal containers
        self.assignment_duration = {i: 0 for i in range(self.max_threats)}
        self.simple_threat_objs = []  # list[SimpleThreat]
        self.weapon_asset_values = np.ones(self.num_weapons, dtype=float)

        self.reset()

    def reset(self, seed=None, options=None):
        np.random.seed(seed)

        # Initialize weapons (unchanged)
        weapon_x_positions = np.linspace(10, self.battlefield_size - 10, self.num_weapons)
        weapon_y_positions = np.full(self.num_weapons, self.battlefield_size / 2)
        self.weapons = np.column_stack((
            weapon_x_positions,
            weapon_y_positions,
            np.full(self.num_weapons, self.missiles_per_weapon)
        ))

        # Randomize assets once per episode
        self.weapon_asset_values = np.random.randint(1, 6, size=self.num_weapons).astype(float)
        print(f"🎯 Random weapon_asset_values at reset: {self.weapon_asset_values.tolist()}")

        # Initialize threats allocated by assets
        self._init_simple_red_team(self.max_threats)

        self.steps = 0
        self.previous_action = np.full((self.num_weapons, self.missiles_per_weapon), -1)
        padded_threats = np.zeros((self.max_threats, 4))
        padded_threats[:len(self.threats)] = self.threats
        self.state = np.concatenate((padded_threats.flatten(), self.weapons.flatten()))
        print(f"🔄 Reset: {len(self.simple_threat_objs)} threats initialized (allocated by random assets).")
        return self.state.astype(np.float32), {}

    def _init_simple_red_team(self, num_threats, speed_range=(1.0, 4.0), max_turn_deg=30.0, seed=None):
        if seed is not None:
            np.random.seed(seed)
        counts, assignments = allocate_targets_by_asset_values(self.weapon_asset_values, num_threats)
        self.simple_threat_objs = []
        margin = 5.0
        for i in range(num_threats):
            sx = np.random.uniform(margin, self.battlefield_size - margin)
            sy = np.random.uniform(margin, self.battlefield_size - margin)
            spd = np.random.uniform(speed_range[0], speed_range[1])
            target_idx = assignments[i]
            wx, wy, _ = getattr(self, "weapons", np.array([[self.battlefield_size/2, self.battlefield_size/2, 0]]))[target_idx]
            init_heading = math.degrees(math.atan2(wy - sy, wx - sx))
            thr = SimpleThreat(pos=(sx, sy), speed=spd, heading_deg=init_heading,
                               target_weapon_idx=target_idx, max_turn_deg=max_turn_deg)
            self.simple_threat_objs.append(thr)

        # Build self.threats array
        self.threats = np.zeros((len(self.simple_threat_objs), 4))
        for idx, t in enumerate(self.simple_threat_objs):
            self.threats[idx, 0:2] = t.pos
            self.threats[idx, 2] = t.speed
            self.threats[idx, 3] = t.severity

        self.assignment_duration = {i: 0 for i in range(self.max_threats)}
        self.weapon_assignment_duration = {}
        self.previous_weapon_assignment = {}

        # debug prints: counts and per-threat assignment list
        print(f"Allocated threat counts per weapon (deterministic): {counts}")
        assignments_by_threat = [t.target_weapon_idx for t in self.simple_threat_objs]
        print(f"Per-threat assigned targets (threat_idx -> weapon_idx): {assignments_by_threat}")

    def _step_simple_threats_motion(self, dt=1.0, reach_threshold=2.0):
        for thr in list(self.simple_threat_objs):
            t_idx = thr.target_weapon_idx
            if 0 <= t_idx < self.num_weapons:
                goal_pos = (self.weapons[t_idx, 0], self.weapons[t_idx, 1])
                thr.step_towards(goal_pos, dt=dt)
        # sync numpy array
        self.threats = np.zeros((len(self.simple_threat_objs), 4))
        for i, t in enumerate(self.simple_threat_objs):
            self.threats[i, 0:2] = t.pos
            self.threats[i, 2] = t.speed
            self.threats[i, 3] = t.severity
        self.num_threats = len(self.threats)

    def evaluate_threats(self):
        base_x, base_y = self.battlefield_size / 2, self.battlefield_size / 2
        threat_evaluation = []
        for i in range(len(self.threats)):
            threat_x, threat_y, speed, severity = self.threats[i]
            distance = np.linalg.norm([base_x - threat_x, base_y - threat_y])
            distance_score = max(0, 100 - (distance / self.battlefield_size) * 100)
            severity_score = ((severity - 0.5) / 0.5) * 100
            speed_score = min(100, (speed / 10) * 100)
            danger_score = (severity_score * 0.3) + (distance_score * 0.4) + (speed_score * 0.3)
            danger_score = np.clip(danger_score, 0, 100)
            threat_evaluation.append((i, danger_score, distance_score, severity_score, speed_score))
        threat_evaluation.sort(key=lambda x: x[1], reverse=True)
        return threat_evaluation

    def step(self, action):
        # Print weapon asset values each step for visibility (unchanged during episode)
        print(f"\n🎯 Weapon asset values (episode): {self.weapon_asset_values.tolist()}")

        # ===  print compact threat evaluation list ===
        if len(self.threats) > 0:
            te_list = self.evaluate_threats()  # sorted by danger desc
            print("\n🔴 Threat Evaluation List (start of step) 🔴")
            print(f"{'Rank':<5} {'ThreatID':<8} {'Assigned->':<10} {'Danger':<8} {'DistScore':<10} {'Severity':<9} {'Speed':<6}")
            print("-" * 70)
            for rank, (tid, danger_score, distance_score, severity_score, speed_score) in enumerate(te_list):
                assigned = None
                if tid < len(self.simple_threat_objs):
                    assigned = self.simple_threat_objs[tid].target_weapon_idx
                print(f"{rank+1:<5} {tid:<8} {'W'+str(assigned):<10} {danger_score:<8.2f} {distance_score:<10.2f} {severity_score:<9.2f} {speed_score:<6.2f}")
            print("-" * 70)

        print("\n🟡 Raw action input:", action)
        self.steps += 1
        reward = 0
        done = False

        # Move threats
        self._step_simple_threats_motion(dt=1.0)

        # Evaluate & ranking
        threat_evaluation = self.evaluate_threats()
        threat_ranking = {threat[0]: rank for rank, threat in enumerate(threat_evaluation)}
        step_missile_usage = {i: 0 for i in range(self.num_weapons)}
        valid_threats = np.ones(len(self.threats), dtype=bool)

        # Convert action to matrix sized by weapons x missiles_per_weapon
        action = np.array(action).reshape(self.num_weapons, self.missiles_per_weapon)

        assigned_this_step = {}
        threat_assignments = {i: 0 for i in range(len(self.threats))}
        self.tracked_assignments = []

        # initialize trackers if missing
        if not hasattr(self, "weapon_assignment_duration"):
            self.weapon_assignment_duration = {}
        if not hasattr(self, "previous_weapon_assignment"):
            self.previous_weapon_assignment = {}

        print(f"[DEBUG] Active Threats at Step {self.steps}: {list(range(len(self.threats)))}")

        # Process assignments (weapon × missile loops)
        # We print each successful assignment here for debugging
        for threat_idx in range(len(self.threats)):
            if threat_idx not in assigned_this_step and valid_threats[threat_idx] and threat_assignments[threat_idx] < self.max_assignments:
                for weapon_idx in range(self.num_weapons):
                    for missile_idx in range(self.missiles_per_weapon):
                        if step_missile_usage[weapon_idx] < int(self.weapons[weapon_idx, 2]):
                            sel_tid = int(action[weapon_idx, missile_idx])
                            if not (0 <= sel_tid < len(self.threats)):
                                continue
                            if threat_assignments[sel_tid] < self.max_assignments and valid_threats[sel_tid]:
                                assigned_this_step[sel_tid] = weapon_idx
                                self.tracked_assignments.append((weapon_idx, sel_tid))
                                threat_assignments[sel_tid] += 1
                                step_missile_usage[weapon_idx] += 1
                                if threat_assignments[sel_tid] >= self.max_assignments:
                                    valid_threats[sel_tid] = False
                                reward += 2
                                danger_rank = threat_ranking.get(sel_tid, len(threat_evaluation))
                                max_rank = len(threat_evaluation)
                                reward += (max_rank - danger_rank) * 5
                                if sel_tid in self.previous_weapon_assignment and self.previous_weapon_assignment[sel_tid] == weapon_idx:
                                    self.weapon_assignment_duration[(weapon_idx, sel_tid)] = self.weapon_assignment_duration.get((weapon_idx, sel_tid), 0) + 1
                                else:
                                    self.weapon_assignment_duration[(weapon_idx, sel_tid)] = 1

                                # DEBUG: print this assignment event
                                print(f"➡️ Assigned (Weapon {weapon_idx}, Missile {missile_idx}) -> Threat {sel_tid}")

        # Forced assignment if some threats left unassigned and missiles available (print those too)
        unassigned_threats = [i for i in range(len(self.threats)) if valid_threats[i]]
        for threat_idx in unassigned_threats:
            assigned = False
            for weapon_idx in range(self.num_weapons):
                if step_missile_usage[weapon_idx] < int(self.weapons[weapon_idx, 2]):
                    assigned_this_step[threat_idx] = weapon_idx
                    self.tracked_assignments.append((weapon_idx, threat_idx))
                    step_missile_usage[weapon_idx] += 1
                    valid_threats[threat_idx] = False
                    reward -= 5
                    assigned = True
                    print(f"🚨 Forced assignment: Weapon {weapon_idx} -> Threat {threat_idx}")
                    break
            if not assigned:
                pass

        # After processing, print compact assignment summary
        if assigned_this_step:
            print("\n🔸 Assigned this step (threat -> weapon):")
            for thr, w in sorted(assigned_this_step.items()):
                print(f"   Threat {thr} -> Weapon {w}")
            # build reverse mapping
            by_weapon = {w: [] for w in range(self.num_weapons)}
            for thr, w in assigned_this_step.items():
                by_weapon[w].append(thr)
            print("\n🔸 Assignments by weapon:")
            for w in range(self.num_weapons):
                print(f"   Weapon {w} (asset={int(self.weapon_asset_values[w])}): {by_weapon.get(w, [])}")
        else:
            print("\n🔸 No assignments this step.")

        # Penalty for not assigning dangerous threats
        for threat_idx in range(len(self.threats)):
            could_be_assigned = sum(self.weapons[:, 2]) > 0 and threat_assignments[threat_idx] < self.max_assignments
            if threat_idx not in assigned_this_step and threat_idx in threat_ranking and could_be_assigned:
                low_priority_penalty = (1 - threat_ranking[threat_idx] / len(threat_evaluation)) * -2
                reward += low_priority_penalty

        # Resource utilization check
        total_available_missiles = int(np.sum(self.weapons[:, 2]))
        total_assignments = sum(threat_assignments.values())
        available_threats = sum(1 for t in threat_assignments.values() if t < self.max_assignments)
        if available_threats > 0 and total_assignments < min(self.max_assignments * len(self.threats), total_available_missiles):
            print("🚨 Resource under-utilization detected!")
            print(f"    🔸 Available missiles: {total_available_missiles}")
            print(f"    🔸 Threats assignable: {available_threats}")
            print(f"    🔸 Assignments made:  {total_assignments}")
            reward -= 5

        if total_assignments == min(self.max_assignments * len(self.threats), total_available_missiles):
            reward += 5

        # Stability reward
        if self.previous_weapon_assignment:
            stability_reward = sum(1 for threat, weapon in assigned_this_step.items() if self.previous_weapon_assignment.get(threat) == weapon)
            reward += stability_reward * 2

        # Debug missile counts
        print("\n🚀 **Missile Count After Assignments** 🚀")
        for weapon_idx in range(self.num_weapons):
            print(f"Weapon {weapon_idx}: {int(self.weapons[weapon_idx, 2])} missiles left")

        # Penalty if a threat is too close to a weapon (threats attack weapons)
        danger_zone = 10
        close_threat_penalty = -2
        for threat_idx in range(len(self.threats)):
            threat_x, threat_y, _, _ = self.threats[threat_idx]
            for weapon_idx in range(self.num_weapons):
                weapon_x, weapon_y, _ = self.weapons[weapon_idx]
                distance = np.linalg.norm([threat_x - weapon_x, threat_y - weapon_y])
                if distance < danger_zone:
                    reward += close_threat_penalty
                    print(f"⚠️ Penalty! Threat {threat_idx} is too close to Weapon {weapon_idx} (Dist: {distance:.2f})")

        # Update assignment durations
        for threat_idx in range(len(self.threats)):
            if threat_idx in assigned_this_step:
                current_weapon = assigned_this_step[threat_idx]
                if (current_weapon, threat_idx) in self.weapon_assignment_duration:
                    self.assignment_duration[threat_idx] = self.weapon_assignment_duration[(current_weapon, threat_idx)]
            else:
                self.assignment_duration[threat_idx] = 0

        # store previous assignment
        self.previous_weapon_assignment = assigned_this_step.copy()

        # Remove threats assigned for >= 3 steps
        threats_to_remove = [tid for tid, duration in self.assignment_duration.items() if duration >= 3]
        if threats_to_remove:
            threats_to_remove = [tid for tid in threats_to_remove if tid < len(self.threats)]
            for tid in threats_to_remove:
                for threat_info in threat_evaluation:
                    if threat_info[0] == tid:
                        danger_score = threat_info[1]
                        reward += danger_score * 0.5
                        break
            # remove from simple_threat_objs (descending indices to pop safely)
            for tid in sorted(threats_to_remove, reverse=True):
                if tid < len(self.simple_threat_objs):
                    print(f"🗑️ Removing threat {tid} (assigned long enough).")
                    del self.simple_threat_objs[tid]
            # update numpy threats array
            self.threats = np.delete(self.threats, threats_to_remove, axis=0) if len(self.threats) > 0 else np.zeros((0, 4))
            self.num_threats = len(self.threats)
            # Reduce missile stock for weapons that were involved
            for weapon_idx, threat_idx in self.tracked_assignments:
                if threat_idx in threats_to_remove:
                    self.weapons[weapon_idx, 2] = max(0, self.weapons[weapon_idx, 2] - 1)
            # Filter out assignment durations for removed threats
            self.weapon_assignment_duration = {(w, t): d for (w, t), d in self.weapon_assignment_duration.items() if t not in threats_to_remove}

        # Check termination conditions
        if len(self.threats) == 0:
            reward += 2500 / max(1, self.steps)
            print("All threats are killed")
            done = True

        total_missiles_left = np.sum(self.weapons[:, 2])
        if total_missiles_left <= 0 and not done:
            print("❌ All weapons are out of missiles!")
            reward += 3
            done = True

        if self.steps >= 500:
            reward -= 10
            done = True

        # Build padded state
        padded_threats = np.zeros((self.max_threats, 4))
        if len(self.threats) > 0:
            padded_threats[:len(self.threats)] = self.threats
        self.state = np.concatenate((padded_threats.flatten(), self.weapons.flatten()))

        # Cleanup weapon_assignment_duration entries referencing out-of-range threats
        self.weapon_assignment_duration = {
            (weapon, threat): duration
            for (weapon, threat), duration in self.weapon_assignment_duration.items()
            if threat < len(self.threats)
        }

        # Print assignment durations for debugging
        print(f"\n[Step {self.steps}]\n⏳ **Assignment Durations for Assigned Pairs** ⏳")
        for weapon_idx, threat_idx in self.tracked_assignments:
            if (weapon_idx, threat_idx) in self.weapon_assignment_duration:
                duration = self.weapon_assignment_duration[(weapon_idx, threat_idx)]
                print(f"Weapon {weapon_idx} → Threat {threat_idx}: {duration} steps")

        print("reward:", reward)
        return self.state.astype(np.float32), reward, done, False, {}

    def render(self, action=None):
        plt.clf()
        plt.xlim(0, self.battlefield_size)
        plt.ylim(0, self.battlefield_size)
        plt.grid(True)

        cmap = plt.get_cmap('tab10')  # up to 10 different colors
        # plot weapons with asset labels
        for i in range(self.num_weapons):
            plt.scatter(self.weapons[i, 0], self.weapons[i, 1], color='green', s=150, zorder=5)
            asset_val = int(self.weapon_asset_values[i]) if hasattr(self, "weapon_asset_values") else None
            plt.text(self.weapons[i, 0] + 2, self.weapons[i, 1] + 2, f"W{i} (A:{asset_val})", fontsize=10, color='black', zorder=6)

        # plot threat paths and current positions
        for i, t in enumerate(self.simple_threat_objs):
            color = cmap(t.target_weapon_idx % 10)
            # path line
            xs = [p[0] for p in t.path]
            ys = [p[1] for p in t.path]
            plt.plot(xs, ys, linestyle='-', linewidth=1.5, color=color, alpha=0.8)
            # start marker
            plt.scatter(xs[0], ys[0], marker='o', s=40, color=color, edgecolor='black', zorder=7)
            # current position marker
            plt.scatter(t.pos[0], t.pos[1], marker='>', s=80, color=color, edgecolor='black', zorder=8)
            plt.text(t.pos[0] + 1, t.pos[1] + 1, f"T{i}->W{t.target_weapon_idx}", fontsize=9, color='black')

        # draw assignment lines for debugging
        if hasattr(self, "tracked_assignments") and self.tracked_assignments:
            for weapon_idx, threat_idx in self.tracked_assignments:
                if 0 <= threat_idx < len(self.threats):
                    plt.plot([self.weapons[weapon_idx, 0], self.threats[threat_idx, 0]],
                             [self.weapons[weapon_idx, 1], self.threats[threat_idx, 1]], 'k--', linewidth=1)

        plt.title(f"Battlefield at Step {getattr(self, 'steps', 0)}")
        plt.pause(0.5)



