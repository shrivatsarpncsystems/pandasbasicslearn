"""
=============================================================================
DAY 3 - SCRIPT 3: TRANSFORMER ARCHITECTURE & POSITIONAL ENCODING
=============================================================================
Goal:
  1. Understand why Transformers need Positional Encoding:
     (Self-Attention is Permutation-Invariant: "Dog bites man" vs "Man bites dog").
  2. Implement Sinusoidal Positional Encoding from the seminal 2017 paper:
     "Attention Is All You Need" (Vaswani et al.).
  3. Walk through the Encoder and Decoder block components:
     Multi-Head Attention, Residual Connections, LayerNorm, and Feed-Forward Networks.
  4. Understand Cross-Attention (Encoder-Decoder Attention).
  5. Generate a Positional Encoding heat matrix visualization.
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
print(">>> PART 1: THE PERMUTATION INVARIANCE PROBLEM")
print("=" * 75)

print("""
Consider these two opposite sentences:
  Sentence A: 'The dog bit the man'
  Sentence B: 'The man bit the dog'

In traditional RNNs (LSTMs), words entered one tick at a time:
  t=1: 'The', t=2: 'man', t=3: 'bit'... So order was baked into time.

In Transformers:
  ALL words enter simultaneously in parallel across GPU threads!
  The dot product Q · K^T depends ONLY on the vector coordinates, NOT on word order!
  Without positional information:
    Sentence A and Sentence B produce IDENTICAL self-attention results!
  
Solution: Add a unique 'Position Signature Vector' to each word's embedding!
          Final Input = Token_Embedding + Positional_Encoding
""")


# ---------------------------------------------------------------------------
# PART 2: SINUSOIDAL POSITIONAL ENCODING FORMULA
# ---------------------------------------------------------------------------
print("=" * 75)
print(">>> PART 2: THE SINUSOIDAL ENCODING FORMULA")
print("=" * 75)

print("""
From Vaswani et al. (2017) 'Attention Is All You Need':
  PE(pos, 2i)   = sin( pos / (10000^(2i / d_model)) )   [Even dimensions]
  PE(pos, 2i+1) = cos( pos / (10000^(2i / d_model)) )   [Odd dimensions]

Why sine and cosine waves?
  1. Deterministic & bounded: Values stay safely between -1.0 and +1.0.
  2. Multi-frequency clock: Low dimensions oscillate rapidly (like the seconds hand),
     high dimensions oscillate slowly (like the hour hand).
  3. Relative Position Property: For any fixed distance k, PE(pos + k) can be
     represented as a linear transformation of PE(pos) via trigonometric identity!
""")

def get_positional_encoding(max_seq_len, d_model):
    """
    Computes sinusoidal positional encodings for a given max sequence length and embedding dimension.
    """
    pe = np.zeros((max_seq_len, d_model))
    positions = np.arange(max_seq_len)[:, np.newaxis] # Shape (max_seq_len, 1)
    
    # 10000^(2i / d_model) in log space: exp(2i * -log(10000) / d_model)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    # Apply sin to even indices (0, 2, 4...)
    pe[:, 0::2] = np.sin(positions * div_term)
    # Apply cos to odd indices (1, 3, 5...)
    pe[:, 1::2] = np.cos(positions * div_term)
    
    return pe

seq_len = 50
d_model = 64
pe_matrix = get_positional_encoding(seq_len, d_model)

print(f"Computed Positional Encoding matrix:")
print(f"Shape: {pe_matrix.shape} ({seq_len} token positions, {d_model} embedding dimensions)")
print(f"Position 0 sample (first 6 dims): {np.round(pe_matrix[0, :6], 3)}")
print(f"Position 1 sample (first 6 dims): {np.round(pe_matrix[1, :6], 3)}")
print(f"Position 2 sample (first 6 dims): {np.round(pe_matrix[2, :6], 3)}")


# ---------------------------------------------------------------------------
# PART 3: THE FULL TRANSFORMER BLOCK: RESIDUALS, LAYERNORM & FFN
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print(">>> PART 3: THE ANATOMY OF A TRANSFORMER LAYER")
print("=" * 75)

print("""
Inside EVERY Transformer Block:

1. Residual Connection (Skip Connection):
   Output = x + Sublayer(x)
   • Why? Solves vanishing gradients. In a 96-layer LLM, backprop gradients
     can flow directly backward along the highway without shrinking to zero!

2. Layer Normalization (LayerNorm):
   Normalizes the activations of a SINGLE token across its feature dimensions:
   mean = 0, variance = 1.
   Unlike BatchNorm, LayerNorm does NOT depend on other batch items or sequence length!

3. Feed-Forward Network (FFN):
   Two linear transformations with a non-linearity (GELU / ReLU) in between:
   FFN(x) = max(0, x · W_1 + b_1) · W_2 + b_2
   Typically expands the dimension 4x (e.g. from 768 to 3072 dims), then projects back!
   This is where factual knowledge and world memory are stored!

4. Cross-Attention (In Encoder-Decoder models like T5):
   • Query (Q) comes from the DECODER (e.g. French sentence being written)
   • Keys (K) & Values (V) come from the ENCODER (e.g. English source sentence)
   Allows the generator to look back at the source text at every step!
""")


# ---------------------------------------------------------------------------
# PART 4: VISUALIZING POSITIONAL ENCODINGS
# ---------------------------------------------------------------------------
print("=" * 75)
print(">>> PART 4: GENERATING POSITIONAL ENCODINGS HEATMAP")
print("=" * 75)

plt.figure(figsize=(10, 6), dpi=150)
im = plt.imshow(pe_matrix, cmap="viridis", aspect="auto")

plt.title("Sinusoidal Positional Encoding Matrix (Vaswani et al., 2017)\nEach row is a unique positional fingerprint added to token embeddings",
          fontsize=12, fontweight="bold", pad=15)
plt.xlabel("Embedding Dimension (0 to 63)\n[Low dims = Fast high-frequency waves | High dims = Slow low-frequency waves]",
           fontsize=10, labelpad=8)
plt.ylabel("Token Position in Sentence (0 to 49)", fontsize=10, labelpad=8)

plt.colorbar(im, label="Encoding Value (-1.0 to +1.0)")
plt.tight_layout()

output_path = os.path.join(os.path.dirname(__file__), "positional_encoding_matrix.png")
plt.savefig(output_path)
plt.close()

print(f"Positional Encoding matrix visualization saved to: {output_path}")
print("-> Notice the fine vertical stripes on the left (rapid changes for nearby tokens)")
print("   and the smooth horizontal gradients on the right (broad sequence-wide context)!")
