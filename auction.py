# auction.py
import numpy as np

def auction_algorithm(reward_matrix):
    """
    Solves the weapon-threat assignment using an auction algorithm.
    
    reward_matrix[i, j] = Reward of assigning weapon i to threat j.
    """
    num_weapons, num_threats = reward_matrix.shape
    assignment = {}

    # Start with all threats unassigned
    assigned_threats = set()

    for weapon in range(num_weapons):
        # Find the best available threat for this weapon
        best_threat = np.argmax(reward_matrix[weapon])
        if best_threat not in assigned_threats:
            assignment[weapon] = best_threat
            assigned_threats.add(best_threat)

    return assignment  # {weapon_id: threat_id}
