"""
=============================================================================
DAY 3 - SCRIPT 1: TOKENIZATION & WORD EMBEDDINGS
=============================================================================
Goal:
  1. Understand the 3 levels of Tokenization: Character, Word, and Subword (BPE).
  2. See how text strings are mapped to Token IDs in a vocabulary.
  3. Understand Embeddings: Turning discrete token IDs into dense semantic vectors.
  4. Perform Vector Math: Cosine Similarity and semantic arithmetic (King - Man + Woman = Queen).
  5. Generate a 2D visualization of Word Embeddings in semantic space.
=============================================================================
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from transformers import AutoTokenizer

print("=" * 75)
print(">>> PART 1: THE THREE LEVELS OF TOKENIZATION")
print("=" * 75)

sample_text = "Transformers are transforming natural language processing!"
print(f"Sample Sentence: '{sample_text}'\n")

# 1. Character-level Tokenization
chars = list(sample_text)
print(f"1. Character-level ({len(chars)} tokens):")
print(f"   Sample: {chars[:20]} ...")
print("   Pros: Tiny vocabulary (all alphabet chars + punctuation). No Out-of-Vocabulary (OOV).")
print("   Cons: Sequences are very long; individual letters carry little semantic meaning.\n")

# 2. Word-level Tokenization (naive whitespace split)
words = sample_text.replace("!", "").split()
print(f"2. Word-level ({len(words)} tokens):")
print(f"   Tokens: {words}")
print("   Pros: Words carry direct meaning.")
print("   Cons: Giant vocabulary (hundreds of thousands of words); struggles with typos/new words (OOV).\n")

# 3. Subword Tokenization (Byte-Pair Encoding / WordPiece - used in modern LLMs)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokens = tokenizer.tokenize(sample_text)
token_ids = tokenizer.encode(sample_text)
readable_tokens = [t.replace("\u0120", "_") for t in tokens]

print(f"3. Modern Subword Tokenization (BPE - GPT-2) ({len(tokens)} tokens):")
print(f"   Tokens:    {readable_tokens}")
print(f"   Token IDs: {token_ids}")
print("   Note: '_' represents a leading whitespace.")
print("   Notice how 'transforming' gets split cleanly into 'transform' + 'ing'!")
print("   Pros: Best of both worlds! Compact vocabulary (~50,000) that can represent ANY word.")


# ---------------------------------------------------------------------------
# PART 2: FROM TOKEN IDS TO DENSE EMBEDDINGS
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print(">>> PART 2: DENSE EMBEDDINGS (THE GEOMETRY OF MEANING)")
print("=" * 75)

print("Why not One-Hot Encoding?")
print("• One-hot vector for 'apple' in 50k vocab: [0, 0, 0, 1, 0, ... 0] (50,000 dimensions!)")
print("• Dot product of any two one-hot vectors is 0 -> No concept of similarity!")
print("• Dense Embeddings compress words into small, continuous vectors (e.g., 768 or 1536 dims).")
print("• Words with similar meanings point in similar directions in vector space!\n")

# Let's create an intuitive embedding space (4 semantic features for demonstration)
# Features: [Is_Royal, Is_Male, Is_Tech, Is_Living]
embeddings = {
    "king":       np.array([ 0.95,  0.90, -0.80,  0.85]),
    "queen":      np.array([ 0.95, -0.90, -0.80,  0.85]),
    "man":        np.array([-0.80,  0.90, -0.70,  0.85]),
    "woman":      np.array([-0.80, -0.90, -0.70,  0.85]),
    "computer":   np.array([-0.90,  0.00,  0.98, -0.95]),
    "robot":      np.array([-0.85,  0.00,  0.95, -0.60]),
    "apple":      np.array([-0.70,  0.00,  0.50,  0.80]), # Fruit / Tech company
}

def cosine_similarity(v1, v2):
    """Measures the cosine of the angle between two vectors (-1 to +1)."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print("Cosine Similarities:")
sim_king_queen = cosine_similarity(embeddings["king"], embeddings["queen"])
sim_king_man = cosine_similarity(embeddings["king"], embeddings["man"])
sim_king_computer = cosine_similarity(embeddings["king"], embeddings["computer"])

print(f"• Similarity(king, queen):    {sim_king_queen:+.3f}  (High: both royal, living)")
print(f"• Similarity(king, man):      {sim_king_man:+.3f}  (High: both male, living)")
print(f"• Similarity(king, computer): {sim_king_computer:+.3f}  (Low / Negative: opposite concepts)")

# ---------------------------------------------------------------------------
# PART 3: VECTOR ARITHMETIC (KING - MAN + WOMAN = QUEEN)
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print(">>> PART 3: SEMANTIC VECTOR ARITHMETIC")
print("=" * 75)

# Calculate King - Man + Woman
target_vector = embeddings["king"] - embeddings["man"] + embeddings["woman"]
print(f"Formula: vector('king') - vector('man') + vector('woman')")
print(f"Resulting Vector: {np.round(target_vector, 2)}")

# Find closest word in our dictionary to this target vector
distances = {}
for word, vec in embeddings.items():
    distances[word] = cosine_similarity(target_vector, vec)

sorted_matches = sorted(distances.items(), key=lambda x: x[1], reverse=True)
print("\nRanking closest matches to [King - Man + Woman]:")
for rank, (word, score) in enumerate(sorted_matches, 1):
    print(f"  {rank}. {word.upper():<10} -> Cosine Similarity: {score:.4f}")

closest_word = [w for w, _ in sorted_matches if w != "king"][0]
print(f"\n=> Result: The closest concept is indeed '{closest_word.upper()}'!")


# ---------------------------------------------------------------------------
# PART 4: VISUALIZING WORD EMBEDDINGS (PCA / 2D PROJECTION)
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print(">>> PART 4: GENERATING 2D EMBEDDING VISUALIZATION")
print("=" * 75)

words_list = list(embeddings.keys())
vectors = np.array([embeddings[w] for w in words_list])

# Project 4D to 2D using simple SVD/PCA
U, S, Vt = np.linalg.svd(vectors - vectors.mean(axis=0))
coords_2d = U[:, :2] * S[:2]

plt.figure(figsize=(9, 6), dpi=150)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# Plot points
colors_map = {
    "king": "#1e40af", "queen": "#7c3aed",
    "man": "#0284c7", "woman": "#db2777",
    "computer": "#059669", "robot": "#10b981",
    "apple": "#ea580c"
}

for i, word in enumerate(words_list):
    x, y = coords_2d[i, 0], coords_2d[i, 1]
    color = colors_map.get(word, "#333333")
    plt.scatter(x, y, s=250, color=color, zorder=4, edgecolors="white", linewidth=1.5)
    plt.annotate(
        word.upper(),
        (x, y),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=11,
        fontweight="bold",
        color=color
    )

# Draw semantic relationship arrows (King -> Queen and Man -> Woman)
idx_king = words_list.index("king")
idx_queen = words_list.index("queen")
idx_man = words_list.index("man")
idx_woman = words_list.index("woman")

plt.arrow(
    coords_2d[idx_man, 0], coords_2d[idx_man, 1],
    coords_2d[idx_woman, 0] - coords_2d[idx_man, 0],
    coords_2d[idx_woman, 1] - coords_2d[idx_man, 1],
    color="#9333ea", linestyle="--", width=0.015, head_width=0.08, length_includes_head=True, alpha=0.7,
    label="Gender Direction Vector"
)

plt.arrow(
    coords_2d[idx_king, 0], coords_2d[idx_king, 1],
    coords_2d[idx_queen, 0] - coords_2d[idx_king, 0],
    coords_2d[idx_queen, 1] - coords_2d[idx_king, 1],
    color="#9333ea", linestyle="--", width=0.015, head_width=0.08, length_includes_head=True, alpha=0.7
)

plt.title("2D Projection of Word Embeddings: Parallel Semantic Relations", fontsize=13, fontweight="bold", pad=15)
plt.xlabel("Principal Dimension 1 (Royalty vs. Technology)", fontsize=10)
plt.ylabel("Principal Dimension 2 (Gender Axis)", fontsize=10)
plt.axhline(0, color="gray", linestyle=":", alpha=0.4)
plt.axvline(0, color="gray", linestyle=":", alpha=0.4)
plt.legend(loc="upper right", frameon=True)
plt.tight_layout()

output_path = os.path.join(os.path.dirname(__file__), "word_embeddings_2d.png")
plt.savefig(output_path)
plt.close()
print(f"Plot saved successfully to: {output_path}")
print("Notice how the vector from MAN -> WOMAN is almost parallel to KING -> QUEEN!")
