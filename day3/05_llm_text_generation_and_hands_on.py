"""
=============================================================================
DAY 3 - SCRIPT 5: HOW LLMS GENERATE TEXT & HANDS-ON INFERENCE
=============================================================================
Goal:
  1. Understand the Autoregressive Text Generation Loop (step-by-step next token prediction).
  2. Inspect Logits, Temperature, and Softmax output distribution.
  3. Compare Decoding Strategies:
     • Greedy Search (argmax)
     • Temperature Scaling (Creative vs Deterministic)
     • Top-K Sampling
     • Top-P (Nucleus) Sampling
  4. Run Hands-on Pretrained Inference with DistilGPT-2.
  5. Explore Tokenizer quirks (Code, Numbers, Emojis).
  6. Plot the Next-Token Probability Distribution bar chart.
=============================================================================
"""

import os
import sys

# Suppress Hugging Face symlink warnings on Windows and ensure UTF-8 output
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

print("=" * 75)
print(">>> PART 1: LOADING PRETRAINED TRANSFORMER (DISTILGPT-2)")
print("=" * 75)

model_name = "distilbert/distilgpt2"
print(f"Loading '{model_name}' (82 Million Parameters) and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()  # Put into inference evaluation mode
print("Model loaded successfully onto CPU!\n")


# ---------------------------------------------------------------------------
# PART 2: THE AUTOREGRESSIVE GENERATION LOOP (PREDICTING NEXT TOKEN)
# ---------------------------------------------------------------------------
print("=" * 75)
print(">>> PART 2: STEP-BY-STEP NEXT-TOKEN PROBABILITY DISTRIBUTION")
print("=" * 75)

prompt = "The quick brown fox jumps over the"
print(f"Prompt: '{prompt}'")

# Encode prompt into token IDs tensor
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]

print(f"Input Token IDs: {input_ids.tolist()[0]}")
tokens_readable = [tokenizer.decode([t]) for t in input_ids[0]]
print(f"Tokens Breakdown: {tokens_readable}\n")

# Run forward pass (no gradients needed for inference)
with torch.no_grad():
    outputs = model(**inputs)
    # Logits shape: (batch_size=1, sequence_length=7, vocab_size=50257)
    logits = outputs.logits

# We only care about the logits for the VERY LAST position (the next token!)
next_token_logits = logits[0, -1, :]  # Shape: (50257,)

# Convert raw logits to probabilities via Softmax
probs = F.softmax(next_token_logits, dim=-1)

# Get top 10 most likely candidates
top_k = 10
top_probs, top_indices = torch.topk(probs, top_k)

top_tokens = []
top_probabilities = []

print(f"Top {top_k} Next Token Candidates for '{prompt} ...':")
print(f"{'Rank':<6} | {'Token':<15} | {'Probability':<12} | {'Token ID':<8}")
print("-" * 50)
for rank, (idx, prob) in enumerate(zip(top_indices, top_probs), 1):
    token_str = tokenizer.decode([idx.item()])
    prob_val = prob.item()
    top_tokens.append(f"'{token_str.strip()}'")
    top_probabilities.append(prob_val)
    print(f"{rank:<6} | {repr(token_str):<15} | {prob_val * 100:6.2f}%      | {idx.item():<8}")

greedy_token = tokenizer.decode([top_indices[0].item()])
print(f"\n=> Greedy Choice: '{greedy_token.strip()}' (Highest probability: {top_probs[0].item()*100:.1f}%)")
print(f"Completed phrase: '{prompt} {greedy_token.strip()}'")


# ---------------------------------------------------------------------------
# PART 3: DECODING STRATEGIES COMPARED
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print(">>> PART 3: DECODING STRATEGIES IN ACTION")
print("=" * 75)

test_prompt = "In the future, artificial intelligence will"
test_inputs = tokenizer(test_prompt, return_tensors="pt")

print(f"Prompt: '{test_prompt}'\n")

# 1. Greedy Search (do_sample=False)
out_greedy = model.generate(
    **test_inputs,
    max_new_tokens=25,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)
print("1. GREEDY SEARCH (Deterministic, argmax every step):")
print(f"   \"{tokenizer.decode(out_greedy[0], skip_special_tokens=True)}\"\n")

# 2. Temperature Sampling (High Temperature = Creative, Low = Focused)
torch.manual_seed(42)
out_temp_low = model.generate(
    **test_inputs,
    max_new_tokens=25,
    do_sample=True,
    temperature=0.3,
    pad_token_id=tokenizer.eos_token_id
)
print("2. LOW TEMPERATURE (T=0.3 - Focused & Safe):")
print(f"   \"{tokenizer.decode(out_temp_low[0], skip_special_tokens=True)}\"\n")

out_temp_high = model.generate(
    **test_inputs,
    max_new_tokens=25,
    do_sample=True,
    temperature=1.2,
    pad_token_id=tokenizer.eos_token_id
)
print("3. HIGH TEMPERATURE (T=1.2 - Creative & Diverse):")
print(f"   \"{tokenizer.decode(out_temp_high[0], skip_special_tokens=True)}\"\n")

# 4. Top-P (Nucleus) Sampling (p=0.90)
out_top_p = model.generate(
    **test_inputs,
    max_new_tokens=25,
    do_sample=True,
    top_p=0.90,
    temperature=0.8,
    pad_token_id=tokenizer.eos_token_id
)
print("4. TOP-P (NUCLEUS) SAMPLING (p=0.90, T=0.8 - Industry Standard):")
print(f"   \"{tokenizer.decode(out_top_p[0], skip_special_tokens=True)}\"\n")


# ---------------------------------------------------------------------------
# PART 4: TOKENIZER QUIRKS (CODE, MATH, EMOJIS)
# ---------------------------------------------------------------------------
print("=" * 75)
print(">>> PART 4: HOW LLMS 'SEE' THE WORLD (TOKENIZER QUIRKS)")
print("=" * 75)

examples = [
    ("Python Code", "def add(x, y): return x + y"),
    ("Multi-digit Math", "9876543210"),
    ("Contraction & Words", "I didn't anticipate this!"),
    ("Emojis", "🤖 🚀 💡")
]

for title, text in examples:
    t_ids = tokenizer.encode(text)
    t_strs = [tokenizer.decode([i]) for i in t_ids]
    print(f"• {title}: '{text}'")
    print(f"  Token count: {len(t_ids)}")
    print(f"  Tokens:      {t_strs}\n")

print("Key Takeaways:")
print("1. Code is tokenized into syntax keywords ('def', 'return') and symbols.")
print("2. Big numbers are chopped into arbitrary digit chunks ('987', '654', '321', '0').")
print("   -> This is why LLMs struggle with mental arithmetic!")
print("3. Emojis take multiple bytes / tokens depending on Unicode composition.")


# ---------------------------------------------------------------------------
# PART 5: VISUALIZING NEXT-TOKEN PROBABILITIES
# ---------------------------------------------------------------------------
print("=" * 75)
print(">>> PART 5: GENERATING NEXT-TOKEN PROBABILITY BAR CHART")
print("=" * 75)

plt.figure(figsize=(9, 5), dpi=150)
y_pos = np.arange(len(top_tokens))
bars = plt.barh(y_pos, [p * 100 for p in top_probabilities], color="#2563eb", edgecolor="#1e40af", height=0.65)
plt.yticks(y_pos, top_tokens, fontsize=10, fontweight="bold")
plt.gca().invert_yaxis()  # Put highest probability on top
plt.xlabel("Probability (%)", fontsize=10, fontweight="bold", labelpad=8)
plt.title(f"Next Token Probability Distribution (DistilGPT-2)\nPrompt: '{prompt} ...'", fontsize=11, fontweight="bold", pad=12)

# Annotate bars with percentage values
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.5, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%",
             va="center", ha="left", fontsize=9, fontweight="bold", color="#1e3a8a")

plt.xlim(0, max([p * 100 for p in top_probabilities]) + 8)
plt.tight_layout()

output_path = os.path.join(os.path.dirname(__file__), "next_token_probabilities.png")
plt.savefig(output_path)
plt.close()

print(f"Next token probability plot saved successfully to: {output_path}")
print("All hands-on demonstrations complete!")
