import numpy as np
import matplotlib.pyplot as plt
import collections
import json 
import concurrent.futures 
import math 

W1_INITIAL_BIAS = 0.10  
W2_DECAY_NOISE = 0.40   
W3_CUSTOM_PEAKS = 0.35   
W4_MOD_BOOST = 0.15    

DECAY_RATE = 0.015      
MAX_CHOICE = 1000       
new_dataset_size = 100000

N_PREDICTION_PEOPLE = 115 # This should be changed for the number of people

RISK_PROFILES = {
    "Low Risk (High Confidence)": {"min_prob": 0.80, "max_prob": 0.90}, 
    "Medium Risk": {"min_prob": 0.45, "max_prob": 0.55},                
    "High Risk (Low Confidence)": {"min_prob": 0.10, "max_prob": 0.20},  
    "Near 100% Win Chance": {"min_prob": 0.95, "max_prob": 1.00},         
    "BEST CHOICE (Risk-Adjusted Value)": {"min_prob": 0.00, "max_prob": 1.00},     
}

N_SIMULATIONS = 10000
N_TOP_STATISTICAL_RESULTS = 5 
N_WORKERS = 8 

initial_data = [1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 5, 6, 6, 7, 7, 7, 8, 9] # The numbers should be from 1-9 and be recieved from a small poll
N_initial = len(initial_data)

custom_peak_numbers = [7, 13, 17, 19, 21, 23, 27, 29, 31, 33, 35, 37, 39, 41, 43, 47, 49, 53, 57, 59, 61, 63, 67, 69, 71, 73, 77, 79, 83, 87, 89, 91, 93, 97, 99, 137, 143, 149, 157, 163, 167, 173, 179, 187, 193, 197, 213, 219, 223, 227, 231, 233, 237, 239, 241, 247, 251, 257, 263, 269, 271, 277, 283, 289, 297, 317, 323, 327, 333, 337, 343, 347, 353, 357, 367, 373, 377, 383, 389, 393, 397, 413, 417, 423, 427, 437, 443, 447, 457, 467, 473, 479, 487, 491, 497, 507, 517, 523, 527, 537, 547, 557, 563, 567, 573, 577, 587, 593, 597, 613, 617, 623, 627, 637, 647, 657, 667, 673, 677, 683, 693, 697, 713, 717, 723, 737, 747, 757, 763, 767, 773, 777, 787, 793, 797, 817, 823, 827, 837, 843, 847, 853, 857, 863, 867, 873, 877, 883, 887, 897, 913, 917, 923, 937, 947, 3, 7, 8, 9, 11, 12, 13, 21, 23, 33, 42, 47, 49, 50, 66, 69, 77, 88, 99, 100, 111, 123, 222, 234, 256, 314, 333, 369, 400, 420, 444, 456, 512, 555, 600, 616, 666, 700, 717, 720, 777, 808, 818, 840, 888, 900, 911, 919, 999, 1000]

possible_choices = np.arange(1, MAX_CHOICE + 1)
N_choices = len(possible_choices)
custom_set = set(custom_peak_numbers)

initial_empirical_probabilities = np.zeros(N_choices)
counts = collections.Counter(initial_data)
for i in range(N_choices):
    choice = possible_choices[i]
    initial_empirical_probabilities[i] = counts.get(choice, 0) / N_initial
P_initial_empirical = initial_empirical_probabilities

decay_weights = np.exp(-DECAY_RATE * possible_choices)
P_decay = decay_weights / decay_weights.sum()

P_min = 0.56
P_max = 1.12
P_custom_target_unnorm = np.zeros(N_choices)
weight_range = P_max - P_min
scaling_factor = (MAX_CHOICE - 1)
for i in range(N_choices):
    choice = possible_choices[i]
    if choice in custom_set:
        Ci = P_min + weight_range * (MAX_CHOICE - choice) / scaling_factor
        P_custom_target_unnorm[i] = Ci
P_custom_target = P_custom_target_unnorm / P_custom_target_unnorm.sum()

P_min_mod = 0.10 
P_max_mod = 0.50 
P_mod_boost_unnorm = np.zeros(N_choices)
weight_range_mod = P_max_mod - P_min_mod
scaling_factor_mod = (MAX_CHOICE - 1)
for i in range(N_choices):
    choice = possible_choices[i]
    if choice % 2 == 0 or choice % 5 == 0: 
        Ci_mod = P_min_mod + weight_range_mod * (MAX_CHOICE - choice) / scaling_factor_mod
        P_mod_boost_unnorm[i] = Ci_mod
if P_mod_boost_unnorm.sum() > 0:
    P_mod_boost = P_mod_boost_unnorm / P_mod_boost_unnorm.sum()
else:
    P_mod_boost = np.zeros(N_choices)

expected_sum = W1_INITIAL_BIAS + W2_DECAY_NOISE + W3_CUSTOM_PEAKS + W4_MOD_BOOST
if not np.isclose(expected_sum, 1.0):
    total_weight = W1_INITIAL_BIAS + W2_DECAY_NOISE + W3_CUSTOM_PEAKS + W4_MOD_BOOST
    W1_INITIAL_BIAS /= total_weight
    W2_DECAY_NOISE /= total_weight
    W3_CUSTOM_PEAKS /= total_weight
    W4_MOD_BOOST /= total_weight

synthetic_probabilities = (
    W1_INITIAL_BIAS * P_initial_empirical +
    W2_DECAY_NOISE * P_decay +
    W3_CUSTOM_PEAKS * P_custom_target +
    W4_MOD_BOOST * P_mod_boost
)
synthetic_probabilities /= synthetic_probabilities.sum()


unique_custom_data = sorted(list(custom_set))
print(f"--- Distribution Analysis (Four-Component Blend) ---")
print(f"Blending Weights: W1 (Initial Bias)={W1_INITIAL_BIAS:.2f}, W2 (Decay Noise)={W2_DECAY_NOISE:.2f}, W3 (Custom Peaks)={W3_CUSTOM_PEAKS:.2f}, W4 (Mod Boost)={W4_MOD_BOOST:.2f}")
print("-" * 50)
print(f"{'Number':<8}{'Synthetic P':<15}{'Source':<20}")
print_choices = [1, 2, 3, 4, 10, 33, 100, 500, 777, 999] 
print_choices = [c for c in print_choices if c <= MAX_CHOICE]
for choice in sorted(print_choices):
    idx = choice - 1
    syn_prob = synthetic_probabilities[idx]
    source = 'Initial Bias + Decay'
    is_mod_boosted = choice % 2 == 0 or choice % 5 == 0
    is_custom = choice in custom_set
    if is_custom and is_mod_boosted:
        source = 'Custom + Mod + Blend'
    elif is_custom:
        source = 'Custom Peak + Blend'
    elif is_mod_boosted:
        source = 'Mod Boost + Blend'
    Ci_mod = 0.0
    if is_mod_boosted:
        Ci_mod = P_min_mod + weight_range_mod * (MAX_CHOICE - choice) / scaling_factor_mod
    print(f"{choice:<8}{syn_prob*100:<14.5f}% {source} (Mod Boost: {Ci_mod*100:.1f}%)")
print("-" * 50)


def get_winning_candidates(N_people, synthetic_probabilities, possible_choices, min_prob, max_prob):
    """
    Finds all numbers 'x' such that the probability P(x is NOT picked) 
    by any of N_people falls within the target range [min_prob, max_prob].
    P(unpicked) = (1 - P(x)) ^ N_people.
    """
    prob_not_x = 1 - synthetic_probabilities
    prob_unpicked_all = prob_not_x ** N_people
    full_prediction_data = list(zip(possible_choices, prob_unpicked_all))
    winning_bets = []
    for number, prob in full_prediction_data:
        if min_prob <= prob <= max_prob:
            winning_bets.append((number, prob))
    return winning_bets


print(f"\n--- Predictive Analysis (Statistical Model: N={N_PREDICTION_PEOPLE} Picks) ---")
statistical_results = {}

def risk_adjusted_score(item):
    number, prob = item
    if number == 0: return 0
    return (prob * 100) / number 

for profile_name, profile_range in RISK_PROFILES.items():
    min_p = profile_range["min_prob"]
    max_p = profile_range["max_prob"]
    
    if "BEST CHOICE (Risk-Adjusted Value)" in profile_name:
        print(f"*** Risk Profile: {profile_name} (Highest Score for P>=50% Win Chance) ***")
    else:
        print(f"*** Risk Profile: {profile_name} ({min_p*100:.0f}% - {max_p*100:.0f}% Win Chance) ***")
    
    top_bets = [] 

    if "BEST CHOICE (Risk-Adjusted Value)" in profile_name:
        prob_not_x = 1 - synthetic_probabilities
        prob_unpicked_all = prob_not_x ** N_PREDICTION_PEOPLE
        full_prediction_data = list(zip(possible_choices, prob_unpicked_all))
        
        HIGH_CONFIDENCE_THRESHOLD = 0.50
        high_confidence_candidates = [
            (number, prob) for number, prob in full_prediction_data if prob >= HIGH_CONFIDENCE_THRESHOLD
        ]
        
        if high_confidence_candidates:
            top_bet = sorted(high_confidence_candidates, key=risk_adjusted_score, reverse=True)[0]
            
            number, prob = top_bet
            top_bets = [top_bet]
            score = risk_adjusted_score(top_bet)
            print(f"  The Risk-Adjusted Choice (Confidence/Number for P>={HIGH_CONFIDENCE_THRESHOLD*100:.0f}%):")
            print(f"  #1: Number {number} (Confidence: {prob*100:.5f}%) (Score: {score:.4f})")
        else:
            print(f"  No number candidates found with Confidence >= {HIGH_CONFIDENCE_THRESHOLD*100:.0f}%.")
            
    else:
        winning_candidates = get_winning_candidates(
            N_PREDICTION_PEOPLE, synthetic_probabilities, possible_choices, min_p, max_p
        )
        if winning_candidates:
            winning_candidates.sort(key=lambda item: item[0])
            top_bets = winning_candidates[:N_TOP_STATISTICAL_RESULTS]
            print(f"  Top {N_TOP_STATISTICAL_RESULTS} Smallest Numbers in Range:")
            for i, (number, prob) in enumerate(top_bets):
                print(f"  #{i+1}: Number {number} (Confidence: {prob*100:.5f}%)")
        else:
            print(f"  No number candidates found in this risk range.")
            
    candidates_json = []
    for number, prob in top_bets:
        candidates_json.append({
            "number": int(number),
            "confidence_percent": float(prob * 100),
            "P_unpicked_range": f"{min_p*100:.0f}% - {max_p*100:.0f}%"
        })
    statistical_results[profile_name] = candidates_json

print("-" * 50)

def run_batch_simulation(batch_size, synthetic_probabilities, possible_choices, N_people, N_choices):
    unpicked_simulation_counts = np.zeros(N_choices, dtype=int)
    for _ in range(batch_size):
        picks = np.random.choice(
            possible_choices,
            size=N_people,
            p=synthetic_probabilities,
            replace=True 
        )
        
        picked_numbers = set(picks)
        for number in possible_choices:
            if number not in picked_numbers:
                unpicked_simulation_counts[number - 1] += 1
                
    return unpicked_simulation_counts

print(f"\n--- Predictive Analysis (Threaded Simulation Model: N={N_PREDICTION_PEOPLE} Picks) ---")
print(f"Running {N_SIMULATIONS:,} simulations across {N_WORKERS} workers.")
print(f"Target: Find numbers where the empirical P(unpicked by all {N_PREDICTION_PEOPLE} picks) falls into the risk bands.")
print("-" * 50)

batch_size = N_SIMULATIONS // N_WORKERS
batches = [batch_size] * N_WORKERS
remaining = N_SIMULATIONS % N_WORKERS
for i in range(remaining):
    batches[i] += 1

total_unpicked_simulation_counts = np.zeros(N_choices, dtype=int)

with concurrent.futures.ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = [
        executor.submit(
            run_batch_simulation, 
            size, 
            synthetic_probabilities, 
            possible_choices, 
            N_PREDICTION_PEOPLE, 
            N_choices
        ) for size in batches
    ]

    for future in concurrent.futures.as_completed(futures):
        try:
            batch_counts = future.result()
            total_unpicked_simulation_counts += batch_counts
            print(f"Completed a batch of {batch_size} simulations.")
        except Exception as e:
            print(f"Batch simulation generated an exception: {e}")

P_sim_unpicked = total_unpicked_simulation_counts / N_SIMULATIONS


print(f"Empirical probabilities calculated over {N_SIMULATIONS:,} simulations.")
print("-" * 50)

simulation_results = {}

for profile_name, profile_range in RISK_PROFILES.items():
    min_p = profile_range["min_prob"]
    max_p = profile_range["max_prob"]
    
    if "BEST CHOICE (Risk-Adjusted Value)" in profile_name:
        print(f"*** Risk Profile (SIMULATION): {profile_name} (Highest Score for P>=50% Win Chance) ***")
    else:
        print(f"*** Risk Profile (SIMULATION): {profile_name} ({min_p*100:.0f}% - {max_p*100:.0f}% Win Chance) ***")
    
    sim_prediction_data = list(zip(possible_choices, P_sim_unpicked))
    top_bets = []

    if "BEST CHOICE (Risk-Adjusted Value)" in profile_name:
        
        HIGH_CONFIDENCE_THRESHOLD = 0.50
        high_confidence_candidates = [
            (number, prob) for number, prob in sim_prediction_data if prob >= HIGH_CONFIDENCE_THRESHOLD
        ]
        
        if high_confidence_candidates:
            top_bet = sorted(high_confidence_candidates, key=risk_adjusted_score, reverse=True)[0]
            
            number, prob = top_bet
            top_bets = [top_bet]
            score = risk_adjusted_score(top_bet)
            print(f"  The Risk-Adjusted Choice (Confidence/Number for P>={HIGH_CONFIDENCE_THRESHOLD*100:.0f}%):")
            print(f"  #1: Number {number} (Confidence: {prob*100:.5f}%) (Score: {score:.4f})")
        else:
            print(f"  No number candidates found with Confidence >= {HIGH_CONFIDENCE_THRESHOLD*100:.0f}%.")
    
    else:
        winning_candidates = []
        for number, prob in sim_prediction_data:
            if min_p <= prob <= max_p:
                winning_candidates.append((number, prob))

        if winning_candidates:
            winning_candidates.sort(key=lambda item: item[0])
            top_bets = winning_candidates[:N_TOP_STATISTICAL_RESULTS]
            
            print(f"  Top {N_TOP_STATISTICAL_RESULTS} Smallest Numbers in Range (Empirical):")
            for i, (number, prob) in enumerate(top_bets):
                print(f"  #{i+1}: Number {number} (Confidence: {prob*100:.5f}%)")
        else:
            print(f"  No number candidates found in this risk range (Empirical).")
            
    candidates_json = []
    for number, prob in top_bets:
        candidates_json.append({
            "number": int(number),
            "confidence_percent": float(prob * 100),
            "P_unpicked_range": f"{min_p*100:.0f}% - {max_p*100:.0f}% (Empirical)"
        })
    simulation_results[profile_name] = candidates_json

print("-" * 50)


print(f"\n--- Saving Results to JSON Files ---")

statistical_file_path = 'statistical_results.json'
try:
    with open(statistical_file_path, 'w') as f:
        json.dump(statistical_results, f, indent=4)
    print(f"Statistical model results saved to '{statistical_file_path}'")
except Exception as e:
    print(f"Error saving statistical results: {e}")

simulation_file_path = 'simulation_results.json'
try:
    with open(simulation_file_path, 'w') as f:
        json.dump(simulation_results, f, indent=4)
    print(f"Simulation model results saved to '{simulation_file_path}'")
except Exception as e:
    print(f"Error saving simulation results: {e}")

print("-" * 50)


large_dataset = np.random.choice(
    possible_choices,
    size=new_dataset_size,
    p=synthetic_probabilities
)
print(f"\nSuccessfully generated a synthetic dataset with {new_dataset_size} samples (Range 1-{MAX_CHOICE}).")
print("First 25 numbers of the new dataset:")
print(large_dataset[:25])
print("-" * 50)


plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(large_dataset, bins=100, density=True, color='#4CAF50', edgecolor='darkgreen', alpha=0.7, label=f'Simulated Data P_synthetic (N={new_dataset_size})')
ax.set_yscale('log') 
ax.plot(possible_choices, synthetic_probabilities, 'r-', linewidth=2, alpha=0.9, label=f'Target P_synthetic (W1={W1_INITIAL_BIAS:.2f}, W2={W2_DECAY_NOISE:.2f}, W3={W3_CUSTOM_PEAKS:.2f}, W4={W4_MOD_BOOST:.2f})')
ax.set_title(f'Synthetic Probability Distribution with Four-Component Bias (Range 1-{MAX_CHOICE})', fontsize=16, pad=20)
ax.set_xlabel('Number Chosen', fontsize=12)
ax.set_ylabel('Probability Density (Log Scale)', fontsize=12)
active_x_ticks = list(range(1, 11)) + [100, 500, 1000]
ax.set_xticks(active_x_ticks)
ax.set_xticklabels([str(t) if t < 1000 else '1000' for t in active_x_ticks])
ax.grid(axis='both', linestyle='--', alpha=0.7)
ax.legend()
plt.tight_layout()
output_filename = 'synthetic_data_decayed_range.png'
plt.savefig(output_filename)
print(f"\nDistribution plot saved as '{output_filename}' ")
