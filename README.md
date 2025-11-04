# TEWARL

Threat Evaluation and Weapon Assignment (TEWA) reinforcement learning environments and training utilities.

## Files

- TEWAenv.py: padded + mask-ready environment intended for MaskablePPO (or other mask-aware algorithms). Fixed-size observation via padding
- EnvWoMask.py: unpadded environment (no action masking). Exact-size observations for the current scenario. Works well with A2C, PPO, DQN that don’t use action masking.
- train.py: Example training script (MaskablePPO by default)
- visualize_env.py: Live visualization runner to see threat motions for both envs

## Requirements

- Python 3.10+
- gymnasium
- numpy
- matplotlib
- stable-baselines3
- sb3-contrib (for MaskablePPO)

Install (example):

```bash
pip install gymnasium numpy matplotlib stable-baselines3 sb3-contrib
```

## Train 

Use `train.py` for MaskablePPO with action masking (TEWAEnv). 'a2ctrain.py' For A2C without masking(`EnvWoMask`).

## Run

Use 'run.py' to run saved models and see how your agent makes assignments.

## Core Concepts implemented in the envs

# Red-team generation
- Each episode the environment creates 'num_threats' threats.
- There are three attacking strategies that threats can follow: 
    * Random wander — small random heading changes.
    * Aim highest-asset — target the weapon with the highest asset value.
    * Aim nearest weapon — target the physically nearest weapon.
- Each threat is represented by a SimpleThreat object:
    pos (x,y), speed, heading (deg), target_weapon_idx, max_turn_deg, severity (random ∈ [0.5,1.0]).
- path history saved per step (for visualization).
- Threat kinematics:
    step_towards(goal_pos, dt) rotates heading toward the assigned weapon within max_turn_deg limit, then moves forward by speed * dt.
    This produces smooth, plausible curved trajectories toward target weapons.

# Threat evaluation
- For each active threat a final normalized danger score (0..100) is computed according to below criterias:
  * Distance score
  * Severity score 
  * Speed score
  * Final danger score (weighted sum):
- The final value is clipped to [0,100]. 
- The env sorts threats by this danger_score for prioritization.

# Reward mechanics
- Assignment reward: when a weapon assigns a missile to a threat, the env gives a base positive reward and an additional priority bonus based on the threat rank,encourages assigning higher-danger threats.
- Stability reward: if the same weapon assigns the same threat in consecutive steps, the env increments a stored (weapon, threat) duration; consistent targeting gets extra reward, encourages assignment stability.
- Elimination bonus: a threat assigned to the same weapon for >= 3 steps is considered eliminated. On elimination, a larger reward proportional to threat danger is given.
- Forced assignment penalty: if there are valid threats left but the agent did not assign them and we force assignments (to ensure resources are used), a penalty is applied to discourage poor policies.
- Under-utilization penalty: if missiles are available but the agent fails to assign, a penalty is given.
- Close-threat penalty: if a threat comes too close to a weapon (danger zone), apply a small penalty is applied.
- Final completion bonus: if all threats are eliminated, a large completion bonus scaled by episode speed is given  to encourage fast mission completion.
- Ranking penalty: small shaped penalties for assigning low-priority threats while high-priority threats remain, encourages the agent to prioritize high risk threats.
* These components are tunable

# Observations 
- TEWAEnv (padded)
  Observation vector (1D, fixed-length):
  state = [padded_threats, padded_weapons]
  padded_threats shape = (max_threats, 5) → [x, y, speed, severity, alive] (alive is implied by non-zero).
  weapons shape = (num_weapons, 4) → [x, y, missiles_left, alive]
  Observation shape = (max_threats * 5 + num_weapons * 4).

- EnvWoMask (no padding)
  Observation vector (exact runtime size):
  state = [threats, weapons]
  threats shape = (num_threats, 4) → [x, y, speed, severity]
  weapons shape = (num_weapons, 3) → [x, y, missiles_left]

# Assignment & Elimination 
Per-step, the env reads the agent's action matrix shape = (num_weapons, missiles_per_weapon) .
For each slot, it checks if the chosen threat index in the action corresponds to an active threat if so assignment is made; it honors max_assignments per threat (max missiles that can be targeting same threat per step).
If some threats remain unassigned and missiles are available, the env may forcibly assign them to ensure resource utilization for learning stability.
Threats that accumulate assignment_duration >= 3 (same weapon) are removed and considered eliminated — weapon missile stock is decremented accordingly.


## Visualize

Run the standalone visualizer to watch threats move and see threat strategies:

```bash
# Env without masking
python visualize_env.py --env nomask --steps 300 --fps 6 --debug

# Mask-ready env
python visualize_env.py --env mask --steps 200 --fps 5
```

Flags:
- --env: mask | nomask
- --steps: number of timesteps to render
- --fps: rendering frames per second
- --num_threats, --num_weapons, --battlefield, --missiles_per_weapon, --max_assignments
- --random_actions (default) | --zero_actions
- --debug: extra prints (EnvWoMask)

- Visualization draws:
  * Weapon markers labeled W{i} (A:asset_val)
  * Threat path histories T{i}->W{target}
  * Assignment dashed lines between weapon and current threat position.

- Debug feature in envs toggles printing of:
  * Weapon asset values at reset and per step.
  * Compact Threat Evaluation list (rank, threat id, assigned target, danger score, components). This version intentionally hides x,y coordinates as requested.
  * Per-weapon assignment lists.
  * Missile counts after assignments and elimination logs.






