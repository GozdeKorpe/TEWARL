import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

# Import environments
from TEWAenv import TEWAEnv
from EnvWoMask import EnvWoMask


def make_env(kind: str,
             num_threats: int = 7,
             num_weapons: int = 3,
             battlefield_size: int = 150,
             missiles_per_weapon: int = 3,
             max_assignments: int = 1,
             debug: bool = True):
	kind = kind.lower()
	if kind in ("mask", "tewa", "tewaenv"):
		# TEWAEnv (padded/mask-capable)
		return TEWAEnv(
			num_threats=num_threats,
			num_weapons=num_weapons,
			battlefield_size=battlefield_size,
			missiles_per_weapon=missiles_per_weapon,
			max_assignments=max_assignments,
		)
	elif kind in ("nomask", "envwomask", "womask"):
		# EnvWoMask (no padding/mask)
		return EnvWoMask(
			num_threats=num_threats,
			num_weapons=num_weapons,
			battlefield_size=battlefield_size,
			missiles_per_weapon=missiles_per_weapon,
			max_assignments=max_assignments,
			debug=debug,
		)
	else:
		raise ValueError(f"Unknown env kind: {kind}")


def main():
	parser = argparse.ArgumentParser(description="Visualize TEWA environments")
	parser.add_argument("--env", type=str, default="nomask",
					help="Environment kind: 'mask' (TEWAEnv) or 'nomask' (EnvWoMask)")
	parser.add_argument("--steps", type=int, default=200, help="Number of steps to visualize")
	parser.add_argument("--fps", type=float, default=5.0, help="Frames per second for rendering")
	parser.add_argument("--num_threats", type=int, default=7)
	parser.add_argument("--num_weapons", type=int, default=3)
	parser.add_argument("--battlefield", type=int, default=150)
	parser.add_argument("--missiles_per_weapon", type=int, default=3)
	parser.add_argument("--max_assignments", type=int, default=1)
	parser.add_argument("--random_actions", action="store_true", help="Use random actions (default)")
	parser.add_argument("--zero_actions", action="store_true", help="Use zero actions instead of random")
	parser.add_argument("--debug", action="store_true", help="Enable verbose debug output (EnvWoMask only)")
	args = parser.parse_args()

	plt.ion()
	env = make_env(
		kind=args.env,
		num_threats=args.num_threats,
		num_weapons=args.num_weapons,
		battlefield_size=args.battlefield,
		missiles_per_weapon=args.missiles_per_weapon,
		max_assignments=args.max_assignments,
		debug=args.debug,
	)

	obs, _ = env.reset()
	dt = 1.0 / max(1e-6, args.fps)
	for t in range(args.steps):
		if args.zero_actions:
			# choose the first threat index for all slots
			action = np.zeros(env.action_space.shape, dtype=int)
		else:
			action = env.action_space.sample()

		obs, reward, done, truncated, info = env.step(action)
		env.render(action)
		plt.pause(max(0.001, dt))
		if done or truncated:
			break

	plt.ioff()
	plt.show()


if __name__ == "__main__":
	main()
