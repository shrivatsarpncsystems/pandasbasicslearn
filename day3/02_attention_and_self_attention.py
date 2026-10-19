"""
=============================================================================
DAY 3 - SCRIPT 2: ATTENTION & SELF-ATTENTION MECHANISMS
=============================================================================
Goal:
  1. Understand the bottleneck of RNNs/LSTMs and why Attention revolutionized NLP.
  2. Implement Scaled Dot-Product Attention from scratch in NumPy:
       Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
  3. Understand the Library Analogy: Query, Key, and Value.
  4. Understand Multi-Head Attention: Why multiple heads give multiple perspectives.
  5. Visualize an Attention Heatmap: Watch the pronoun "it" focus on "animal".
=============================================================================
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

print("=" * 75)
print(">>> PART 1: THE LIBRARY ANALOGY (QUERY, KEY, VALUE)")
print("=" * 75)

print("""
Imagine walking into a massive research library:
  1. QUERY (Q): The search phrase you type into the catalog.
                ("Who was tired and didn't cross the street?")
  2. KEY   (K): The book spine tags & titles in the library catalog.
                ("The", "animal", "didn't", "cross", "street", "tired")
  3. MATCHING:  You take the dot product (Q · K) between your Query and every Key.
                High dot product = High relevance!
  4. SOFTMAX:   Normalize match scores into percentages (Attention Weights) that sum to 100%.
  5. VALUE (V): The actual knowledge / contents inside the books.
                You multiply the Attention Weights by each Value to extract a blended summary!
""")


# ---------------------------------------------------------------------------
# PART 2: SCALED DOT-PRODUCT ATTENTION FROM SCRATCH (NUMPY)
# ---------------------------------------------------------------------------
print("=" * 75)
print(">>> PART 2: STEP-BY-STEP SCALED DOT-PRODUCT ATTENTION")
print("=" * 75)

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

# Let's consider 4 tokens: ["The", "animal", "was", "tired"]
tokens = ["The", "animal", "was", "tired"]
seq_len = len(tokens)
d_k = 8  # Dimension of Query and Key vectors
d_v = 8  # Dimension of Value vectors

np.random.seed(42)

# Simulated input token embeddings (seq_len x d_model)
X = np.random.randn(seq_len, d_k)

# Projection weight matrices W_q, W_k, W_v
W_q = np.random.randn(d_k, d_k) * 0.5
W_k = np.random.randn(d_k, d_k) * 0.5
W_v = np.random.randn(d_k, d_v) * 0.5

# Step 1: Linear projections into Q, K, V
Q = np.dot(X, W_q)  # Shape: (seq_len, d_k)
K = np.dot(X, W_k)  # Shape: (seq_len, d_k)
V = np.dot(X, W_v)  # Shape: (seq_len, d_v)

print(f"Tokens: {tokens}")
print(f"Query matrix Q shape: {Q.shape} (Each row is a query vector for that token)")
print(f"Key matrix   K shape: {K.shape}")
print(f"Value matrix V shape: {V.shape}\n")

# Step 2: Raw Attention Scores = Q * K^T
raw_scores = np.dot(Q, K.T)  # Shape: (seq_len, seq_len)
print("Step 2: Raw Attention Scores (Q · K^T):")
print(np.round(raw_scores, 2))

# Step 3: Scaling by sqrt(d_k)
# CRITICAL CONCEPT: Why scale by sqrt(d_k)?
# If d_k is large (e.g. 64 or 128), dot products grow huge in magnitude.
# Huge inputs push Softmax into saturated flat zones where gradients vanish (become 0)!
# Dividing by sqrt(d_k) maintains unit variance and keeps gradients flowing smoothly!
scaled_scores = raw_scores / np.sqrt(d_k)
print(f"\nStep 3: Scaled Scores (divided by sqrt({d_k}) = {np.sqrt(d_k):.2f}):")
print(np.round(scaled_scores, 2))

# Step 4: Softmax to get Attention Weights (rows sum to 1.0)
attention_weights = softmax(scaled_scores, axis=-1)
print("\nStep 4: Attention Weights (Softmax distribution across tokens):")
print(np.round(attention_weights, 3))
print(f"Row sums verification: {np.sum(attention_weights, axis=-1)} (All exactly 1.0!)")

# Step 5: Weighted Sum of Values
context_output = np.dot(attention_weights, V)
print(f"\nStep 5: Output Context Vectors shape: {context_output.shape}")
print("Each token's output is now a rich blend of its own meaning and all relevant neighbors!")


# ---------------------------------------------------------------------------
# PART 3: MULTI-HEAD ATTENTION (MHA) INTUITION
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print(">>> PART 3: WHY MULTI-HEAD ATTENTION?")
print("=" * 75)

print("""
Single-Head Attention has a limitation:
  It averages everything into one attention distribution.
  If a token needs to attend to BOTH:
    1. Grammatical structure ("Which verb belongs to this subject?")
    2. Semantic coreference  ("What does the pronoun 'it' refer to?")
  A single head is forced to compromise!

Multi-Head Attention (e.g. 8 or 16 heads):
  - Head 1 focuses on: Syntactic relationships (Subject <-> Verb)
  - Head 2 focuses on: Coreference resolution (Pronoun <-> Noun)
  - Head 3 focuses on: Long-distance semantic themes
  - All heads run in parallel, their outputs are concatenated, and multiplied by W_O!
""")


# ---------------------------------------------------------------------------
# PART 4: VISUALIZING SELF-ATTENTION WEIGHTS (THE CLASSIC "IT" DEMO)
# ---------------------------------------------------------------------------
print("=" * 75)
print(">>> PART 4: GENERATING ATTENTION HEATMAP VISUALIZATION")
print("=" * 75)

sentence = ["The", "animal", "didn't", "cross", "street", "because", "it", "was", "tired"]

# Construct realistic attention weights representing coreference resolution
# Notice how the token "it" pays 72% attention to "animal"!
simulated_attention = np.array([
    # The    animal  didn't   cross  street because    it     was   tired
    [0.70,   0.15,   0.02,   0.02,   0.02,   0.01,   0.02,   0.02,   0.04], # The
    [0.08,   0.65,   0.05,   0.08,   0.04,   0.02,   0.02,   0.01,   0.05], # animal
    [0.02,   0.08,   0.50,   0.25,   0.05,   0.03,   0.02,   0.02,   0.03], # didn't
    [0.02,   0.10,   0.18,   0.45,   0.18,   0.02,   0.01,   0.01,   0.03], # cross
    [0.02,   0.05,   0.04,   0.24,   0.55,   0.02,   0.02,   0.01,   0.05], # street
    [0.01,   0.03,   0.05,   0.06,   0.04,   0.60,   0.08,   0.05,   0.08], # because
    [0.01,   0.72,   0.01,   0.02,   0.03,   0.04,   0.10,   0.02,   0.05], # "it" -> ANIMAL (72%!)
    [0.01,   0.05,   0.02,   0.02,   0.01,   0.05,   0.20,   0.55,   0.09], # was
    [0.01,   0.48,   0.01,   0.02,   0.02,   0.04,   0.12,   0.05,   0.25], # "tired" -> ANIMAL (48%!)
])

plt.figure(figsize=(8, 7), dpi=150)
im = plt.imshow(simulated_attention, cmap="Blues", interpolation="nearest")

plt.xticks(range(len(sentence)), sentence, fontsize=10, rotation=45, ha="right")
plt.yticks(range(len(sentence)), sentence, fontsize=10)

plt.xlabel("Key Tokens (Attended To)", fontsize=11, fontweight="bold", labelpad=10)
plt.ylabel("Query Tokens (Attending From)", fontsize=11, fontweight="bold", labelpad=10)
plt.title("Self-Attention Heatmap: Coreference Resolution\nNotice Query 'it' attends 72% to 'animal'!", fontsize=12, fontweight="bold", pad=15)

# Annotate heatmap cells with percentages
for i in range(len(sentence)):
    for j in range(len(sentence)):
        val = simulated_attention[i, j]
        color = "white" if val > 0.4 else "black"
        fontweight = "bold" if val > 0.4 else "normal"
        plt.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8.5, fontweight=fontweight)

plt.colorbar(im, fraction=0.046, pad=0.04, label="Attention Weight")
plt.tight_layout()

output_path = os.path.join(os.path.dirname(__file__), "self_attention_heatmap.png")
plt.savefig(output_path)
plt.close()

print(f"Self-attention heatmap saved successfully to: {output_path}")
print("-> Query 'it' looked across the entire sentence and resolved its reference to 'animal' with 72% weight!")
