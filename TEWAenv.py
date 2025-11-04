import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import math

# ---------------------------
# SimpleThreat kinematics (with path history)
# ---------------------------
class SimpleThreat:
    def __init__(self, pos, speed, heading_deg, target_weapon_idx, max_turn_deg=30.0, severity=None, strategy=1):
        self.pos = np.array(pos, dtype=float)
        self.speed = float(speed)
        self.heading = float(heading_deg)  # degrees
        self.target_weapon_idx = int(target_weapon_idx)
        self.max_turn_deg = float(max_turn_deg)
        self.severity = severity if severity is not None else np.random.uniform(0.5, 1.0)
        self.path = [tuple(self.pos.copy())]
        self.strategy = int(strategy)

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
        self.path.append((self.pos[0], self.pos[1]))

    def distance_to(self, goal_pos):
        return np.linalg.norm(self.pos - np.array(goal_pos, dtype=float))

    def step_forward(self, dt=1.0):
        heading_rad = math.radians(self.heading)
        vx = math.cos(heading_rad) * self.speed * dt
        vy = math.sin(heading_rad) * self.speed * dt
        self.pos += np.array([vx, vy])
        self.path.append((self.pos[0], self.pos[1]))


# ---------------------------
# Allocation helper (proportional)
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
    order = np.argsort(-assets)  # descending
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
# TEWAEnv with padding + mask support (modified/cleaned)
# ---------------------------
class TEWAEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 num_threats,
                 num_weapons,
                 battlefield_size,
                 missiles_per_weapon,
                 max_assignments,
                 max_threats=None,
                 max_weapons=None,
                 max_missiles_per_weapon=None,
                 debug=False):
        """
        num_threats, num_weapons: runtime counts
        max_threats, max_weapons: padding sizes
        """
        super(TEWAEnv, self).__init__()

        # runtime sizes
        self.num_threats = int(num_threats)
        self.initial_num_threats = int(num_threats)
        self.num_weapons = int(num_weapons)
        self.missiles_per_weapon = int(missiles_per_weapon)
        self.battlefield_size = float(battlefield_size)
        self.max_assignments = int(max_assignments)

        # padding sizes (default to runtime sizes)
        self.max_threats = int(max_threats) if max_threats is not None else int(self.num_threats)
        self.max_weapons = int(max_weapons) if max_weapons is not None else int(self.num_weapons)
        self.max_missiles_per_weapon = int(max_missiles_per_weapon) if max_missiles_per_weapon is not None else int(self.missiles_per_weapon)

        # features: threat -> [x,y,speed,severity,alive]; weapon -> [x,y,missiles_left,asset,alive]
        self.threat_feat = 5
        self.weapon_feat = 4

        # debug prints
        self.debug = bool(debug)

        # observation: flattened padded threats + padded weapons + threat_mask + weapon_alive_mask
        obs_len = (self.max_threats * self.threat_feat) + (self.max_weapons * self.weapon_feat) + self.max_threats + self.max_weapons
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)

        # action space: each action slot selects a threat index [0..max_threats-1]
        total_action_slots = self.max_weapons * self.max_missiles_per_weapon
        self.action_space = spaces.MultiDiscrete([self.max_threats for _ in range(total_action_slots)])

        # internal
        self.weapon_asset_values = np.ones(self.max_weapons, dtype=float)
        self.simple_threat_objs = []
        self.assignment_duration = {i: 0 for i in range(self.max_threats)}
        self.weapon_assignment_duration = {}
        self.previous_weapon_assignment = {}

        # initialize env state
        self.reset()

    # ---------------------------
    def reset(self, seed=None, options=None):
        """
        Robust reset: reinitialize weapons, asset values, threat team and trackers.
        Returns (state, info)
        """
        if seed is not None:
            np.random.seed(seed)

        # set desired active threat count
        self.num_threats = int(getattr(self, "initial_num_threats", self.max_threats))

        # reinit weapons (active weapons only)
        weapon_x_positions = np.linspace(10, self.battlefield_size - 10, self.num_weapons)
        weapon_y_positions = np.full(self.num_weapons, self.battlefield_size / 2)
        self.weapons = np.column_stack((weapon_x_positions, weapon_y_positions, np.full(self.num_weapons, self.missiles_per_weapon))).astype(float)

        # randomize asset values per episode (kept constant during episode)
        self.weapon_asset_values = np.random.randint(1, 6, size=self.max_weapons).astype(float)

        # create threats for this episode
        self._init_simple_red_team(self.num_threats)

        # trackers & bookkeeping
        self.steps = 0
        self.previous_action = np.full((self.num_weapons, self.missiles_per_weapon), -1)
        self.assignment_duration = {i: 0 for i in range(self.max_threats)}
        self.weapon_assignment_duration = {}
        self.previous_weapon_assignment = {}

        self.state = self._build_state()

        if self.debug:
            print(f"🔄 Reset finished: created {len(self.threats)} threats, weapons:{self.num_weapons}, missiles_per_weapon:{self.missiles_per_weapon}")
            print(f"🎯 Weapon asset values at reset: {self.weapon_asset_values.tolist()}")

        return self.state.astype(np.float32), {"action_mask": self.build_action_mask()}

    # ---------------------------
    def _init_simple_red_team(self, num_threats, speed_range=(1.0, 4.0), max_turn_deg=30.0, seed=None):
        """
        Create `num_threats` red threats and assign them to weapons.
        Guarantees creation and consistent shapes.
        """
        if seed is not None:
            np.random.seed(seed)

        if int(num_threats) <= 0:
            self.simple_threat_objs = []
            self.threats = np.zeros((0, self.threat_feat), dtype=float)
            self.num_threats = 0
            return

        num_threats = int(num_threats)
        asset_vals = np.array(self.weapon_asset_values[:self.num_weapons], dtype=float)

        # try original allocator
        try:
            counts, _ = allocate_targets_by_asset_values(asset_vals, num_threats)
            counts = np.asarray(counts, dtype=int).reshape(-1)
        except Exception as e:
            if self.debug:
                print("⚠️ allocate_targets_by_asset_values failed:", e)
            counts = np.zeros(self.num_weapons, dtype=int)

        # fallback if invalid
        if counts.size != self.num_weapons or counts.sum() != num_threats:
            if asset_vals.sum() <= 0:
                asset_vals = np.ones_like(asset_vals)
            float_alloc = (asset_vals / asset_vals.sum()) * num_threats
            base = np.floor(float_alloc).astype(int)
            remainder = num_threats - base.sum()
            frac = float_alloc - np.floor(float_alloc)
            idxs = np.argsort(-frac)
            for i in range(remainder):
                base[idxs[i % len(idxs)]] += 1
            counts = base
            if counts.sum() != num_threats:
                counts = np.zeros(self.num_weapons, dtype=int)
                for i in range(num_threats):
                    counts[i % self.num_weapons] += 1
            if self.debug:
                print("⚠️ Fallback counts used:", counts.tolist())

        # generate positions & speeds
        margin = 5.0
        xs = np.random.uniform(margin, self.battlefield_size - margin, size=num_threats)
        ys = np.random.uniform(margin, self.battlefield_size - margin, size=num_threats)
        threat_positions = np.column_stack((xs, ys)).astype(float)
        threat_speeds = np.random.uniform(speed_range[0], speed_range[1], size=num_threats).astype(float)

        # distances T x W
        W = int(self.num_weapons)
        weapon_coords = np.array([[self.weapons[w, 0], self.weapons[w, 1]] for w in range(W)], dtype=float)
        if weapon_coords.ndim == 1:
            weapon_coords = weapon_coords.reshape(1, 2)
        diffs = threat_positions[:, None, :] - weapon_coords[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)

        # greedy assignment by asset value, closest first
        asset_order = np.argsort(-asset_vals)
        assigned_weapon_for_threat = np.full(num_threats, -1, dtype=int)
        unassigned = set(range(num_threats))
        for w in asset_order:
            to_assign = int(counts[w]) if w < len(counts) else 0
            for _ in range(to_assign):
                if not unassigned:
                    break
                candidates = np.fromiter(unassigned, dtype=int)
                c_dists = dists[candidates, w]
                choice = int(candidates[np.argmin(c_dists)])
                assigned_weapon_for_threat[choice] = int(w)
                unassigned.remove(choice)

        if unassigned:
            for t in list(unassigned):
                assigned_weapon_for_threat[t] = int(np.argmin(dists[t]))
                unassigned.remove(t)

        # create SimpleThreat objects
        self.simple_threat_objs = []
        for tid in range(num_threats):
            sx, sy = float(threat_positions[tid, 0]), float(threat_positions[tid, 1])
            spd = float(threat_speeds[tid])
            target_idx = int(assigned_weapon_for_threat[tid])
            if target_idx < 0:
                target_idx = int(np.argmin(dists[tid]))
            wx, wy = float(self.weapons[target_idx, 0]), float(self.weapons[target_idx, 1])
            # Set heading perpendicular to the line toward the assigned weapon (no homing)
            base_heading = float(np.degrees(np.arctan2(wy - sy, wx - sx)))
            init_heading = (base_heading + 90.0) % 360.0
            thr = SimpleThreat(pos=(sx, sy), speed=spd, heading_deg=init_heading,
                               target_weapon_idx=target_idx, max_turn_deg=max_turn_deg,
                               strategy=int(np.random.choice([1, 2, 3])))
            self.simple_threat_objs.append(thr)

        # sync numpy threats (T x threat_feat)
        T = len(self.simple_threat_objs)
        self.threats = np.zeros((T, self.threat_feat), dtype=float)
        for i, t in enumerate(self.simple_threat_objs):
            self.threats[i, 0:2] = t.pos
            self.threats[i, 2] = t.speed
            self.threats[i, 3] = t.severity
            self.threats[i, 4] = 1.0  # alive flag

        # bookkeeping
        self.num_threats = len(self.threats)
        self.assignment_duration = {i: 0 for i in range(self.max_threats)}
        self.weapon_assignment_duration = {}
        self.previous_weapon_assignment = {}

        if self.debug:
            print(f"🔸 Created {self.num_threats} threats; counts per weapon: {counts.tolist()}")

    # ---------------------------
    def _build_state(self):
        """
        Build padded observation vector:
          - threats: max_threats x [x,y,speed,severity,alive]
          - weapons: max_weapons x [x,y,missiles_left,asset,alive]
          - threat_mask (max_threats)
          - weapon_alive_mask (max_weapons)
        """
        # ensure self.threats is in consistent shape (T, threat_feat)
        if not hasattr(self, "threats") or self.threats is None:
            self.threats = np.zeros((0, self.threat_feat), dtype=float)
        else:
            self.threats = np.asarray(self.threats, dtype=float)
            if self.threats.ndim == 1:
                if self.threats.size % self.threat_feat == 0:
                    self.threats = self.threats.reshape(-1, self.threat_feat)
                else:
                    # rebuild from simple_threat_objs if available
                    T = len(getattr(self, "simple_threat_objs", []))
                    self.threats = np.zeros((T, self.threat_feat), dtype=float)
                    for i, st in enumerate(getattr(self, "simple_threat_objs", [])):
                        self.threats[i, 0:2] = st.pos
                        self.threats[i, 2] = st.speed
                        self.threats[i, 3] = st.severity
                        self.threats[i, 4] = 1.0
            elif self.threats.ndim == 2 and self.threats.shape[1] != self.threat_feat:
                T = len(getattr(self, "simple_threat_objs", []))
                self.threats = np.zeros((T, self.threat_feat), dtype=float)
                for i, st in enumerate(getattr(self, "simple_threat_objs", [])):
                    self.threats[i, 0:2] = st.pos
                    self.threats[i, 2] = st.speed
                    self.threats[i, 3] = st.severity
                    self.threats[i, 4] = 1.0

        # threats padded
        tpad = np.zeros((self.max_threats, self.threat_feat), dtype=float)
        active_T = len(self.threats)
        for i in range(active_T):
            row = self.threats[i]
            if row.shape[0] >= self.threat_feat:
                tpad[i, 0:2] = row[0:2]
                tpad[i, 2] = row[2]
                tpad[i, 3] = row[3]
                tpad[i, 4] = row[4]
            else:
                # fallback: assume alive
                tpad[i, 0:2] = row[0:2]
                tpad[i, 2] = row[2] if row.size > 2 else 0.0
                tpad[i, 3] = row[3] if row.size > 3 else 0.5
                tpad[i, 4] = 1.0

        # weapons padded
        wpad = np.zeros((self.max_weapons, self.weapon_feat), dtype=float)
        for i in range(self.max_weapons):
            if i < self.num_weapons:
                wpad[i, 0:2] = self.weapons[i, 0:2]
                wpad[i, 2] = self.weapons[i, 2]  # missiles left
                wpad[i, 3] = float(self.weapon_asset_values[i])  # asset
                wpad[i, 4] = 1.0
            else:
                wpad[i, 2] = 0.0
                wpad[i, 3] = float(self.weapon_asset_values[i]) if i < len(self.weapon_asset_values) else 0.0
                wpad[i, 4] = 0.0

        # masks
        threat_mask = np.zeros(self.max_threats, dtype=float)
        threat_mask[:active_T] = 1.0
        weapon_alive_mask = np.zeros(self.max_weapons, dtype=float)
        for i in range(self.max_weapons):
            if i < self.num_weapons and self.weapons[i, 2] > 0:
                weapon_alive_mask[i] = 1.0

        state = np.concatenate((tpad.flatten(), wpad.flatten(), threat_mask, weapon_alive_mask))
        return state

    # ---------------------------
    def build_action_mask(self):
        """
        Returns mask with shape (max_weapons * max_missiles_per_weapon, max_threats).
        True = allowed.
        """
        slots = self.max_weapons * self.max_missiles_per_weapon
        mask = np.zeros((slots, self.max_threats), dtype=bool)
        active_T = len(self.threats)
        for w in range(self.max_weapons):
            for m in range(self.max_missiles_per_weapon):
                slot_idx = w * self.max_missiles_per_weapon + m
                if w >= self.num_weapons:
                    mask[slot_idx, :] = False
                    continue
                missiles_left = int(self.weapons[w, 2]) if w < self.num_weapons else 0
                if m < missiles_left:
                    # allow only alive threats
                    for t in range(self.max_threats):
                        alive = False
                        if t < active_T:
                            alive = (self.threats[t, 4] == 1.0)
                        mask[slot_idx, t] = alive
                else:
                    mask[slot_idx, :] = False
        return mask

    # ---------------------------
    def evaluate_threats(self):
        base_x, base_y = self.battlefield_size / 2, self.battlefield_size / 2
        threat_evaluation = []
        for i in range(len(self.threats)):
            threat_x, threat_y, speed, severity, _ = self.threats[i]
            distance = np.linalg.norm([base_x - threat_x, base_y - threat_y])
            distance_score = max(0, 100 - (distance / self.battlefield_size) * 100)
            severity_score = ((severity - 0.5) / 0.5) * 100
            speed_score = min(100, (speed / 10) * 100)
            danger_score = (severity_score * 0.3) + (distance_score * 0.4) + (speed_score * 0.2)
            danger_score = np.clip(danger_score, 0, 100)
            threat_evaluation.append((i, danger_score, distance_score, severity_score, speed_score))
        threat_evaluation.sort(key=lambda x: x[1], reverse=True)
        return threat_evaluation

    # ---------------------------
    def _step_simple_threats_motion(self, dt=1.0):
        for thr in list(self.simple_threat_objs):
            strat = getattr(thr, "strategy", 1)
            if strat == 1:
                # Random wander: small random heading jitter, then move forward
                thr.heading = (thr.heading + float(np.random.uniform(-thr.max_turn_deg, thr.max_turn_deg)) * dt) % 360.0
                thr.step_forward(dt=dt)
            elif strat == 2:
                # Go to most valuable active weapon
                if self.num_weapons > 0:
                    active_assets = np.array(self.weapon_asset_values[:self.num_weapons], dtype=float)
                    target_w = int(np.argmax(active_assets)) if active_assets.size > 0 else 0
                    goal_pos = (self.weapons[target_w, 0], self.weapons[target_w, 1])
                    thr.step_towards(goal_pos, dt=dt)
                else:
                    thr.step_forward(dt=dt)
            else:
                # strat == 3: Go to nearest weapon (dynamic each step)
                if self.num_weapons > 0:
                    diffs = self.weapons[:self.num_weapons, 0:2] - thr.pos[None, :]
                    dists = np.linalg.norm(diffs, axis=1)
                    target_w = int(np.argmin(dists))
                    goal_pos = (self.weapons[target_w, 0], self.weapons[target_w, 1])
                    thr.step_towards(goal_pos, dt=dt)
                else:
                    thr.step_forward(dt=dt)
        # sync numpy array for active threats
        T = len(self.simple_threat_objs)
        self.threats = np.zeros((T, self.threat_feat), dtype=float)
        for i, t in enumerate(self.simple_threat_objs):
            self.threats[i, 0:2] = t.pos
            self.threats[i, 2] = t.speed
            self.threats[i, 3] = t.severity
            self.threats[i, 4] = 1.0
        self.num_threats = len(self.threats)

    # ---------------------------
    def step(self, action):
        """
        Process action (shape: max_weapons x max_missiles_per_weapon).
        Returns (state, reward, done, truncated, info) with info['action_mask'].
        """
        # ensure action shape
        action = np.array(action, dtype=int).reshape(self.max_weapons, self.max_missiles_per_weapon)

        if self.debug:
            print(f"\n🎯 Weapon asset values (episode): {self.weapon_asset_values.tolist()}")

        # compact threat eval printed at start-of-step if debug
        if len(getattr(self, "threats", [])) > 0 and self.debug:
            te_list = self.evaluate_threats()
            print("\n🔴 Threat Evaluation List (start of step) 🔴")
            print(f"{'Rank':<5} {'ThreatID':<8} {'Assigned->':<10} {'Danger':<8} {'DistScore':<10} {'Severity':<9} {'Speed':<6}")
            print("-" * 70)
            for rank, (tid, danger_score, distance_score, severity_score, speed_score) in enumerate(te_list):
                assigned = None
                if tid < len(self.simple_threat_objs):
                    assigned = self.simple_threat_objs[tid].target_weapon_idx
                print(f"{rank+1:<5} {tid:<8} {'W'+str(assigned):<10} {danger_score:<8.2f} {distance_score:<10.2f} {severity_score:<9.2f} {speed_score:<6.2f}")
            print("-" * 70)

        self.steps += 1
        reward = 0.0
        done = False

        # threats move
        self._step_simple_threats_motion(dt=1.0)

        # evaluate & assign
        threat_evaluation = self.evaluate_threats()
        threat_ranking = {threat[0]: rank for rank, threat in enumerate(threat_evaluation)}
        step_missile_usage = {i: 0 for i in range(self.num_weapons)}
        valid_threats = np.ones(len(self.threats), dtype=bool)

        assigned_this_step = {}
        threat_assignments = {i: 0 for i in range(len(self.threats))}
        self.tracked_assignments = []

        if not hasattr(self, "weapon_assignment_duration"):
            self.weapon_assignment_duration = {}
        if not hasattr(self, "previous_weapon_assignment"):
            self.previous_weapon_assignment = {}

        if self.debug:
            print(f"[DEBUG] Active Threats at Step {self.steps}: {list(range(len(self.threats)))}")

        # process action slots only for active weapons
        for weapon_idx in range(self.num_weapons):
            missiles_available = min(int(self.weapons[weapon_idx, 2]), self.max_missiles_per_weapon)
            for missile_idx in range(missiles_available):
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
                    if self.debug:
                        print(f"➡️ Assigned (Weapon {weapon_idx}, Missile {missile_idx}) -> Threat {sel_tid}")

        # forced assignment if missiles remain
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
                    if self.debug:
                        print(f"🚨 Forced assignment: Weapon {weapon_idx} -> Threat {threat_idx}")
                    break
            if not assigned:
                pass

        # compact assignment summary (debug)
        if self.debug:
            if assigned_this_step:
                print("\n🔸 Assigned this step (threat -> weapon):")
                for thr, w in sorted(assigned_this_step.items()):
                    print(f"   Threat {thr} -> Weapon {w}")
                by_weapon = {w: [] for w in range(self.num_weapons)}
                for thr, w in assigned_this_step.items():
                    by_weapon[w].append(thr)
                print("\n🔸 Assignments by weapon:")
                for w in range(self.num_weapons):
                    print(f"   Weapon {w} (asset={int(self.weapon_asset_values[w])}): {by_weapon.get(w, [])}")
            else:
                print("\n🔸 No assignments this step.")

        # penalties & rewards adjustments
        for threat_idx in range(len(self.threats)):
            could_be_assigned = sum(self.weapons[:, 2]) > 0 and threat_assignments[threat_idx] < self.max_assignments
            if threat_idx not in assigned_this_step and threat_idx in threat_ranking and could_be_assigned:
                low_priority_penalty = (1 - threat_ranking[threat_idx] / len(threat_evaluation)) * -2
                reward += low_priority_penalty

        total_available_missiles = int(np.sum(self.weapons[:, 2]))
        total_assignments = sum(threat_assignments.values())
        available_threats = sum(1 for t in threat_assignments.values() if t < self.max_assignments)
        if available_threats > 0 and total_assignments < min(self.max_assignments * len(self.threats), total_available_missiles):
            if self.debug:
                print("🚨 Resource under-utilization detected!")
            reward -= 5
        if total_assignments == min(self.max_assignments * len(self.threats), total_available_missiles):
            reward += 5

        if self.previous_weapon_assignment:
            stability_reward = sum(1 for threat, weapon in assigned_this_step.items() if self.previous_weapon_assignment.get(threat) == weapon)
            reward += stability_reward * 2

        if self.debug:
            print("\n🚀 **Missile Count After Assignments** 🚀")
            for weapon_idx in range(self.num_weapons):
                print(f"Weapon {weapon_idx}: {int(self.weapons[weapon_idx, 2])} missiles left")

        # penalty for threats close to weapons
        danger_zone = 10
        close_threat_penalty = -2
        for threat_idx in range(len(self.threats)):
            threat_x, threat_y, _, _, _ = self.threats[threat_idx]
            for weapon_idx in range(self.num_weapons):
                weapon_x, weapon_y, _ = self.weapons[weapon_idx]
                distance = np.linalg.norm([threat_x - weapon_x, threat_y - weapon_y])
                if distance < danger_zone:
                    reward += close_threat_penalty
                    if self.debug:
                        print(f"⚠️ Penalty! Threat {threat_idx} is too close to Weapon {weapon_idx} (Dist: {distance:.2f})")

        # update assignment durations
        for threat_idx in range(len(self.threats)):
            if threat_idx in assigned_this_step:
                current_weapon = assigned_this_step[threat_idx]
                if (current_weapon, threat_idx) in self.weapon_assignment_duration:
                    self.assignment_duration[threat_idx] = self.weapon_assignment_duration[(current_weapon, threat_idx)]
            else:
                self.assignment_duration[threat_idx] = 0

        self.previous_weapon_assignment = assigned_this_step.copy()

        # remove threats assigned long enough (>=3)
        threats_to_remove = [tid for tid, duration in self.assignment_duration.items() if duration >= 3]
        if threats_to_remove:
            threats_to_remove = [tid for tid in threats_to_remove if tid < len(self.threats)]
            for tid in threats_to_remove:
                for threat_info in threat_evaluation:
                    if threat_info[0] == tid:
                        danger_score = threat_info[1]
                        reward += danger_score * 0.5
                        break
            # remove from simple_threat_objs safely
            for tid in sorted(threats_to_remove, reverse=True):
                if tid < len(self.simple_threat_objs):
                    if self.debug:
                        print(f"🗑️ Removing threat {tid} (assigned long enough).")
                    del self.simple_threat_objs[tid]
            # update numpy threats
            if len(self.threats) > 0:
                self.threats = np.delete(self.threats, threats_to_remove, axis=0)
            else:
                self.threats = np.zeros((0, self.threat_feat))
            self.num_threats = len(self.threats)
            # decrement missile stock for involved weapons
            for weapon_idx, threat_idx in self.tracked_assignments:
                if threat_idx in threats_to_remove:
                    self.weapons[weapon_idx, 2] = max(0, self.weapons[weapon_idx, 2] - 1)
            # filter weapon_assignment_duration
            self.weapon_assignment_duration = {(w, t): d for (w, t), d in self.weapon_assignment_duration.items() if t not in threats_to_remove}

        # termination conditions
        if len(self.threats) == 0:
            reward += 1500 / max(1, self.steps)
            if self.debug:
                print("All threats are killed")
            done = True

        total_missiles_left = np.sum(self.weapons[:, 2])
        if total_missiles_left <= 0 and not done:
            if self.debug:
                print("❌ All weapons are out of missiles!")
            reward += 3
            done = True

        if self.steps >= 500:
            reward -= 10
            done = True

        # build state, return action_mask in info
        self.state = self._build_state()
        info = {"action_mask": self.build_action_mask()}

        # cleanup assignment durations referencing removed threats
        self.weapon_assignment_duration = {
            (weapon, threat): duration
            for (weapon, threat), duration in self.weapon_assignment_duration.items()
            if threat < len(self.threats)
        }

        if self.debug:
            print(f"\n[Step {self.steps}]\n⏳ **Assignment Durations for Assigned Pairs** ⏳")
            for weapon_idx, threat_idx in self.tracked_assignments:
                if (weapon_idx, threat_idx) in self.weapon_assignment_duration:
                    duration = self.weapon_assignment_duration[(weapon_idx, threat_idx)]
                    print(f"Weapon {weapon_idx} → Threat {threat_idx}: {duration} steps")

        # small guard: if no threats early, return zeroed reward (keeps training stable)
        if len(self.threats) < 1:
            if self.debug:
                print("early return (no threats)")
            reward = 0.0

        return self.state.astype(np.float32), float(reward), bool(done), False, info

    # ---------------------------
    def render(self, action=None):
        plt.clf()
        plt.xlim(0, self.battlefield_size)
        plt.ylim(0, self.battlefield_size)
        plt.grid(True)

        cmap = plt.get_cmap('tab10')
        # plot weapons (active only)
        for i in range(self.num_weapons):
            plt.scatter(self.weapons[i, 0], self.weapons[i, 1], color='green', s=150, zorder=5)
            asset_val = int(self.weapon_asset_values[i]) if hasattr(self, "weapon_asset_values") else None
            plt.text(self.weapons[i, 0] + 2, self.weapons[i, 1] + 2, f"W{i} (A:{asset_val})", fontsize=10, color='black', zorder=6)

        # plot threat paths and positions
        for i, t in enumerate(self.simple_threat_objs):
            color = cmap(t.target_weapon_idx % 10)
            xs = [p[0] for p in t.path]
            ys = [p[1] for p in t.path]
            plt.plot(xs, ys, linestyle='-', linewidth=1.5, color=color, alpha=0.8)
            plt.scatter(xs[0], ys[0], marker='o', s=40, color=color, edgecolor='black', zorder=7)
            plt.scatter(t.pos[0], t.pos[1], marker='>', s=80, color=color, edgecolor='black', zorder=8)
            plt.text(t.pos[0] + 1, t.pos[1] + 1, f"T{i}->W{t.target_weapon_idx}", fontsize=9, color='black')

        # draw assignments
        if hasattr(self, "tracked_assignments") and self.tracked_assignments:
            for weapon_idx, threat_idx in self.tracked_assignments:
                if 0 <= threat_idx < len(self.threats):
                    plt.plot([self.weapons[weapon_idx, 0], self.threats[threat_idx, 0]],
                             [self.weapons[weapon_idx, 1], self.threats[threat_idx, 1]], 'k--', linewidth=1)

        plt.title(f"Battlefield at Step {getattr(self, 'steps', 0)}")
        plt.pause(0.01)
