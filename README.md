# The Smallest Unique Number (SUN) Predictor

## Project Goal

This project implements a sophisticated statistical and simulation model designed to solve a classic game theory challenge: **The Smallest Unique Number (SUN) Game.**

The core question is:

> **What is the smallest positive integer that is not picked by anyone else in a group? The person who picked this number wins.**

This statistical predictor helps identify the optimal number to choose to maximize the probability of winning based on initial human behavior data and several probabilistic biases.

## The Game Concept

The rules of the game are simple:

1. A group of players is asked to choose a single positive integer (e.g., from 1 to 1000).

2. All chosen numbers are collected and compared.

3. The winning number is the **smallest** number that was chosen by **only one** person (i.e., a unique pick).

To win, a player must balance two conflicting goals: choosing a **small** number and choosing a **unique** number.

## Model Architecture (`synthetic_data_generator.py`)

The prediction model calculates the probability of a number remaining **unpicked** by a collective group of $N$ people. The probability distribution of people's choices is modeled using a four-component blend to accurately reflect non-uniform human selection biases.

The final synthetic probability distribution $P_{synthetic}(x)$ is a weighted average of:

1. **W1: Initial Bias (`P_initial_empirical`):** Based on the frequency of choices in the initial `initial_data` (numbers placed by the user, typically 1-9).

2. **W2: Decay Noise (`P_decay`):** Reflects the general human bias towards lower numbers, with probability decaying exponentially as the number increases.

3. **W3: Custom Peaks (`P_custom_target`):** Boosts the probability of commonly non-random choices (e.g., prime numbers, favorite numbers, round numbers).

4. **W4: Mod Boost (`P_mod_boost`):** Increases the probability for numbers easily divisible by 2 or 5, representing common, slightly less random choices.

## Configuration and Usage

You must configure two primary variables in `synthetic_data_generator.py` to run the model effectively:

| **Variable** | **Description** |
| :--- | :--- |
| `initial_data` | The numbers should be from 1-9 and placed by the user. This seeds the model with the human tendency to pick single-digit numbers. |
| `N_PREDICTION_PEOPLE` | This should be changed for a number of people. This is the total number of participants (including yourself) in the game. |

### Running the Script

1. Install dependencies (if not already present): `pip install numpy matplotlib`

2. Configure `initial_data` and `N_PREDICTION_PEOPLE` in the Python file.

3. Run the script: `python synthetic_data_generator.py`

### Outputs

The script outputs two JSON files containing recommended numbers across different risk profiles:

| **File** | **Description** |
| :--- | :--- |
| `statistical_results.json` | Results based on the **pure mathematical formula** $\text{Confidence} = (1 - P_{synthetic}(x))^{N_{people}}$. |
| `simulation_results.json` | Results based on an **empirical threaded simulation** of $N_{SIMULATIONS}$ runs, which provides a more robust estimate. |

The results are grouped by risk profile, including "Low Risk," "Medium Risk," and the **"BEST CHOICE (Risk-Adjusted Value),"** which favors numbers that are both highly confident (high probability of being unpicked) and small (low number).
