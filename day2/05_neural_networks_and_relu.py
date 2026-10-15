"""
=============================================================================
DAY 2 - SCRIPT 5: NEURAL NETWORKS, RELU & GRADIENT DESCENT IN ACTION
=============================================================================
Goal: 
  1. Calculate a single neuron by hand with ReLU activation.
  2. Train a multi-layer Neural Network with ReLU.
  3. Watch the Loss decrease downhill via Gradient Descent!
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("=" * 70)
print(">>> PART 1: UNDER THE HOOD — A SINGLE NEURON WITH RELU")
print("=" * 70)

# 1. Inputs (e.g. 3 clues about an applicant)
inputs = np.array([2.5, -1.2, 4.0])

# 2. Weights (Importance of each clue)
weights = np.array([0.8, 1.5, -0.4])
bias = -1.0

# 3. Step 1: Weighted Sum
weighted_sum = np.dot(inputs, weights) + bias
print(f"Inputs:       {inputs}")
print(f"Weights:      {weights}")
print(f"Bias:         {bias}")
print(f"Weighted Sum: ({inputs[0]}*0.8) + ({inputs[1]}*1.5) + ({inputs[2]}*-0.4) + (-1.0) = {weighted_sum:.2f}")

# 4. Step 2: ReLU Activation (If negative -> 0, if positive -> keep it!)
def relu(x):
    return max(0.0, x)

relu_output = relu(weighted_sum)
print(f"\nAfter ReLU Activation: max(0.0, {weighted_sum:.2f}) = {relu_output:.2f}")
if relu_output > 0:
    print("-> The Neuron FIRED (Turned ON)!")
else:
    print("-> The Neuron stayed DORMANT (Turned OFF)!")


# ---------------------------------------------------------------------------
# PART 2: TRAINING A DEEP NEURAL NETWORK (MULTI-LAYER PERCEPTRON)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(">>> PART 2: TRAINING A NEURAL NETWORK ON CURVED DATA (MOONS)")
print("=" * 70)

# Generate curved non-linear data (like two interlocking crescent moons)
# Straight rulers (Linear models) CANNOT separate this data! Only curved networks can!
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create Neural Network:
# - 2 hidden layers: Layer 1 has 16 neurons, Layer 2 has 8 neurons
# - activation='relu' (Adds hinges so the network can bend)
# - solver='adam' (Gradient Descent optimization)
# - max_iter=300 (300 steps downhill)
nn = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    solver="adam",
    max_iter=300,
    random_state=42
)

print("Training Neural Network with Gradient Descent...")
nn.fit(X_train, y_train)

# Evaluation
y_pred = nn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Final Model Test Accuracy: {acc * 100:.2f}%")
print(f"Total training iterations (steps taken down the mountain): {nn.n_iter_}")


# ---------------------------------------------------------------------------
# PART 3: WATCHING LOSS DECREASE DOWNHILL (GRADIENT DESCENT)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(">>> PART 3: GRADIENT DESCENT LOSS CURVE")
print("=" * 70)
print(f"Starting Loss (Mistakes at Step 1):   {nn.loss_curve_[0]:.4f}")
print(f"Middle Loss (Mistakes at Step 50):    {nn.loss_curve_[50]:.4f}")
print(f"Final Loss (Mistakes at Step {nn.n_iter_}):    {nn.loss_curve_[-1]:.4f}")
print("Notice how the loss steadily dropped as the hiker stepped downhill into the valley!")

# Visualize the Loss Curve
plt.figure(figsize=(9, 5))
plt.plot(nn.loss_curve_, color="#2563eb", linewidth=2.5, label="Loss (Error Score)")
plt.title("Gradient Descent: Error Dropping Downhill with Each Step", fontsize=14, fontweight="bold")
plt.xlabel("Training Step (Epoch / Iteration)")
plt.ylabel("Loss (Amount of Error)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

plot_file = "day2/loss_curve.png"
plt.savefig(plot_file, dpi=180)
plt.close()
print(f"\n-> Saved Gradient Descent Loss Curve to: {plot_file}")
print("=" * 70)
