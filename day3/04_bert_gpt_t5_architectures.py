"""
=============================================================================
DAY 3 - SCRIPT 4: BERT, GPT & T5 — THE TRANSFORMER TRINITY
=============================================================================
Goal:
  1. Understand the 3 distinct families derived from the original Transformer:
     • BERT (Encoder-only, Bidirectional)
     • GPT  (Decoder-only, Autoregressive / Causal)
     • T5   (Encoder-Decoder, Text-to-Text)
  2. Master the Attention Mask:
     • Bidirectional Mask (all tokens attend to all tokens)
     • Causal Mask (lower triangular: tokens can only attend to the PAST)
  3. See why setting future attention logits to -inf forces softmax to 0.0%.
  4. Generate a side-by-side comparison visualization of Attention Masks.
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
print(">>> PART 1: THE TRANSFORMER FAMILY TREE")
print("=" * 75)

comparison_table = """
+---------------+------------------------+-------------------------+-------------------------------+
| Feature       | BERT (Encoder-Only)    | GPT (Decoder-Only)      | T5 (Encoder-Decoder)          |
+---------------+------------------------+-------------------------+-------------------------------+
| Attention     | Bidirectional (Full)   | Causal (Left-to-Right)  | Both: Bidir Enc + Causal Dec  |
| Training Goal | Masked LM (Fill-blank) | Next-Token Prediction   | Text-to-Text (Reconstruct)    |
| Can see future| YES (Past & Future)    | NO (Past only)          | Enc: YES | Dec: NO            |
| Best Use-Cases| Classification, NER,   | Story generation, Chat, | Translation, Summarization,   |
|               | Search, Embeddings     | Code, Reasoning (LLMs)  | Structured transformations    |
| Prominent Ex. | BERT, RoBERTa, DeBERTa | GPT-4, LLaMA, Mistral   | T5, FLAN-T5, BART             |
+---------------+------------------------+-------------------------+-------------------------------+
"""
print(comparison_table)


# ---------------------------------------------------------------------------
# PART 2: THE MATHEMATICS OF THE CAUSAL (AUTOREGRESSIVE) MASK
# ---------------------------------------------------------------------------
print("=" * 75)
print(">>> PART 2: HOW CAUSAL MASKING PREVENTS CHEATING IN GPT")
print("=" * 75)

print("""
When training GPT to predict the next word:
  Input: "The cat sat on the" -> Target: "mat"
  If the model could attend forward to "mat", it would just copy the answer!
  
To prevent looking ahead:
  1. Create a triangular mask where future positions = -infinity.
  2. Add mask to raw attention scores BEFORE Softmax:
     Scores_masked = Scores + Mask
  3. Because e^(-inf) = 0, Softmax assigns EXACTLY 0.0% attention to future tokens!
""")

tokens = ["The", "cat", "sat", "on", "the", "mat"]
n = len(tokens)

# 1. BERT Bidirectional Mask: All 1s (Every token can see every other token)
bert_mask = np.ones((n, n))

# 2. GPT Causal Mask: Lower triangular matrix (Tokens can only see <= current position)
gpt_mask = np.tril(np.ones((n, n)))

print(f"Sentence: {tokens}\n")
print("GPT Causal Mask (1 = Allowed to attend, 0 = Masked out / Forbidden):")
print(gpt_mask.astype(int))

# Demonstrate numerical masking with -infinity
np.random.seed(42)
raw_scores = np.random.randn(n, n) * 2.0

# Create additive mask with 0 for allowed, -inf for forbidden
additive_mask = np.where(gpt_mask == 1, 0.0, -1e9)
masked_scores = raw_scores + additive_mask

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

causal_attention_probs = softmax(masked_scores, axis=-1)

print("\nResulting GPT Attention Probabilities after Softmax:")
print("Notice how all upper-right cells are EXACTLY 0.000!")
print(np.round(causal_attention_probs, 3))


# ---------------------------------------------------------------------------
# PART 3: VISUALIZING BERT vs GPT ATTENTION MASKS SIDE-BY-SIDE
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print(">>> PART 3: GENERATING ATTENTION MASKS COMPARISON VISUALIZATION")
print("=" * 75)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

# BERT Mask Plot
im1 = axes[0].imshow(bert_mask, cmap="Blues", vmin=0, vmax=1)
axes[0].set_title("BERT (Encoder-Only)\nBidirectional Attention Mask (Full Visibility)", fontsize=11, fontweight="bold", pad=12)
axes[0].set_xticks(range(n))
axes[0].set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
axes[0].set_yticks(range(n))
axes[0].set_yticklabels(tokens, fontsize=9)
axes[0].set_xlabel("Key Tokens (Attended To)", fontsize=10, fontweight="bold")
axes[0].set_ylabel("Query Tokens (Current Token)", fontsize=10, fontweight="bold")

for i in range(n):
    for j in range(n):
        axes[0].text(j, i, "1", ha="center", va="center", color="white" if bert_mask[i, j] > 0.5 else "black", fontsize=11, fontweight="bold")

# GPT Mask Plot
im2 = axes[1].imshow(gpt_mask, cmap="Blues", vmin=0, vmax=1)
axes[1].set_title("GPT (Decoder-Only)\nCausal Autoregressive Mask (Past & Present Only)", fontsize=11, fontweight="bold", pad=12)
axes[1].set_xticks(range(n))
axes[1].set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
axes[1].set_yticks(range(n))
axes[1].set_yticklabels(tokens, fontsize=9)
axes[1].set_xlabel("Key Tokens (Attended To)", fontsize=10, fontweight="bold")
axes[1].set_ylabel("Query Tokens (Current Token)", fontsize=10, fontweight="bold")

for i in range(n):
    for j in range(n):
        val = int(gpt_mask[i, j])
        color = "white" if val == 1 else "#94a3b8"
        label = "1" if val == 1 else "0"
        axes[1].text(j, i, label, ha="center", va="center", color=color, fontsize=11, fontweight="bold")

plt.tight_layout()

output_path = os.path.join(os.path.dirname(__file__), "attention_masks_comparison.png")
plt.savefig(output_path)
plt.close()

print(f"Attention mask comparison saved to: {output_path}")
print("-> In BERT, token 1 can see token 6.")
print("-> In GPT, token 1 can ONLY see token 1; token 6 can see tokens 1 through 6.")
