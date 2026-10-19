"""
=============================================================================
DAY 3: PDF STUDY HANDBOOK GENERATOR (REPORTLAB)
=============================================================================
Generates a polished, professional multi-page PDF handbook for Day 3:
"NLP, Encoder-Decoder & Transformers Made Simple"
=============================================================================
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

pdf_path = os.path.join(os.path.dirname(__file__), "Day_3_Complete_Notes.pdf")
doc = SimpleDocTemplate(
    pdf_path,
    pagesize=letter,
    rightMargin=34,
    leftMargin=34,
    topMargin=34,
    bottomMargin=34
)

styles = getSampleStyleSheet()

# Typography Styles
title_style = ParagraphStyle(
    "DocTitle",
    parent=styles["Heading1"],
    fontSize=18,
    leading=22,
    textColor=colors.HexColor("#1e3a8a"),
    spaceAfter=3
)

subtitle_style = ParagraphStyle(
    "DocSubTitle",
    parent=styles["Normal"],
    fontSize=9,
    leading=12,
    textColor=colors.HexColor("#4b5563"),
    spaceAfter=6
)

h1_style = ParagraphStyle(
    "SectionH1",
    parent=styles["Heading2"],
    fontSize=11,
    leading=15,
    textColor=colors.HexColor("#1e40af"),
    spaceBefore=7,
    spaceAfter=3,
    keepWithNext=True
)

body_style = ParagraphStyle(
    "BodyDark",
    parent=styles["Normal"],
    fontSize=8,
    leading=11.5,
    textColor=colors.HexColor("#1f2937"),
    spaceAfter=3
)

callout_style = ParagraphStyle(
    "CalloutText",
    parent=styles["Normal"],
    fontSize=7.5,
    leading=10.5,
    textColor=colors.HexColor("#1e293b")
)

code_style = ParagraphStyle(
    "CodeText",
    parent=styles["Code"],
    fontSize=7.5,
    leading=10,
    textColor=colors.HexColor("#0f172a")
)

story = []

# =========================================================================
# PAGE 1: NLP FOUNDATIONS, TOKENIZATION & WORD EMBEDDINGS
# =========================================================================
story.append(Paragraph("Day 3: NLP, Encoder-Decoder & Transformers", title_style))
story.append(Paragraph("<b>Comprehensive Study Handbook: Intuitive Concepts, Mathematics & Architectures</b>", subtitle_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=6))

# Section 1: NLP Foundations
story.append(Paragraph("1. Natural Language Processing (NLP) & The Evolution of Language Models", h1_style))
story.append(Paragraph(
    "• <b>What is NLP?</b> Teaching computers to understand, interpret, and generate messy, contextual human language.<br/>"
    "• <b>Why is it Hard?</b> Ambiguity (<i>'river bank'</i> vs <i>'money bank'</i>), sarcasm, syntax vs. semantics (<i>'Colorless green ideas sleep furiously'</i> is grammatically valid but semantically meaningless).<br/>"
    "• <b>The 4 Eras:</b> 1. Rule-Based (Regex/N-grams) &rarr; 2. Statistical (TF-IDF/Naive Bayes) &rarr; 3. Recurrent (Word2Vec, RNN/LSTM) &rarr; 4. Transformer Era (2017&ndash;Present: Attention, BERT, GPT, Gemini).",
    body_style
))

# Section 2: Tokenization
story.append(Paragraph("2. Text Tokenization: How Computers Chop Text into IDs", h1_style))
token_table_data = [
    [Paragraph("<b>Level</b>", callout_style), Paragraph("<b>How It Works</b>", callout_style), Paragraph("<b>Example: 'unhappy'</b>", callout_style), Paragraph("<b>Pros & Cons</b>", callout_style)],
    [Paragraph("<b>Character</b>", callout_style), Paragraph("Every letter/symbol is a token", callout_style), Paragraph("['u','n','h','a','p','p','y']", code_style), Paragraph("No OOV words; but long sequences and letters lack meaning.", callout_style)],
    [Paragraph("<b>Word</b>", callout_style), Paragraph("Split by whitespace/punctuation", callout_style), Paragraph("['unhappy']", code_style), Paragraph("Direct meaning; but giant vocabulary (&gt;500k) and fails on typos.", callout_style)],
    [Paragraph("<b>Subword (BPE)</b>", callout_style), Paragraph("Frequent words whole; rare split into morphemes", callout_style), Paragraph("['un', 'happy']", code_style), Paragraph("<b>Modern Standard (GPT, BERT)!</b> Compact vocab (~50k), handles any word.", callout_style)],
]
t_token = Table(token_table_data, colWidths=[70, 160, 110, 200])
t_token.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#e0e7ff")),
    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
    ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ('TOPPADDING', (0,0), (-1,-1), 3),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#c7d2fe")),
]))
story.append(t_token)
story.append(Spacer(1, 4))

# Section 3: Word Embeddings
story.append(Paragraph("3. Dense Word Embeddings: The Geometry of Meaning", h1_style))
story.append(Paragraph(
    "• <b>Why Not One-Hot Encoding?</b> One-hot vectors are huge (50,000 dims) and completely sparse. More critically, the dot product between any two one-hot vectors is <b>0.0</b> (no concept of semantic similarity!).<br/>"
    "• <b>Dense Embeddings:</b> Map words into compact continuous vectors (e.g., 768 dims). Words with similar meanings cluster together in vector space.<br/>"
    "• <b>Semantic Vector Arithmetic:</b> <i>vector('King') &minus; vector('Man') + vector('Woman') &approx; vector('Queen')</i>.<br/>"
    "• <b>Cosine Similarity:</b> Measures angle between vectors: <i>cos(&theta;) = (u &middot; v) / (||u|| &middot; ||v||)</i>. Values range from &minus;1.0 (opposite) to +1.0 (identical direction).",
    body_style
))

story.append(PageBreak())

# =========================================================================
# PAGE 2: THE ATTENTION REVOLUTION & SELF-ATTENTION (Q, K, V)
# =========================================================================
story.append(Paragraph("Day 3: Scaled Dot-Product & Self-Attention", title_style))
story.append(Paragraph("<b>Breaking the Sequential Bottleneck: From RNNs to Attention</b>", subtitle_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=6))

story.append(Paragraph("4. The RNN/LSTM Bottleneck: Why Transformers Won", h1_style))
story.append(Paragraph(
    "• <b>Sequential Execution:</b> RNNs process tokens one by one (word 1 &rarr; word 2 &rarr; word 3). Modern GPUs have thousands of cores, but RNNs forced them to wait sequentially.<br/>"
    "• <b>Information Forgetting:</b> RNNs cram the entire sentence history into a single hidden vector <i>h<sub>t</sub></i>. By token 50, early context is largely lost to vanishing gradients.<br/>"
    "• <b>The Attention Revelation (Vaswani et al., 2017):</b> <i>'Attention Is All You Need'</i>. Remove recurrence completely! Ingest all words in parallel, and let every word directly attend to every other word.",
    body_style
))

story.append(Paragraph("5. The Library Analogy: Query, Key, and Value", h1_style))
qkv_table_data = [
    [Paragraph("<b>Component</b>", callout_style), Paragraph("<b>Library Analogy</b>", callout_style), Paragraph("<b>Mathematical Role in Self-Attention</b>", callout_style)],
    [Paragraph("<b>Query (Q)</b>", callout_style), Paragraph("The search question you bring to the librarian", callout_style), Paragraph("Current token asking: <i>'What information do I need from others?'</i>", callout_style)],
    [Paragraph("<b>Key (K)</b>", callout_style), Paragraph("The title/index tag on book spines", callout_style), Paragraph("Other tokens advertising: <i>'Here is what I represent!'</i>", callout_style)],
    [Paragraph("<b>Value (V)</b>", callout_style), Paragraph("The actual knowledge inside the book pages", callout_style), Paragraph("The content vector that gets extracted and blended into the output.", callout_style)],
]
t_qkv = Table(qkv_table_data, colWidths=[90, 190, 260])
t_qkv.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#e0f2fe")),
    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ('TOPPADDING', (0,0), (-1,-1), 4),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#bae6fd")),
]))
story.append(t_qkv)
story.append(Spacer(1, 4))

story.append(Paragraph("6. Scaled Dot-Product Attention: Step-by-Step Math", h1_style))
story.append(Paragraph(
    "<b>The Golden Formula:</b><br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;<b>Attention(Q, K, V) = softmax( (Q &middot; K<sup>T</sup>) / &radic;d<sub>k</sub> ) &middot; V</b><br/>"
    "• <b>Step 1 (Raw Scores):</b> Compute dot product <i>Q &middot; K<sup>T</sup></i>. High dot product = high relevance between tokens.<br/>"
    "• <b>Step 2 (Scaling by &radic;d<sub>k</sub>):</b> If <i>d<sub>k</sub></i> is large (e.g. 64), dot products explode in magnitude. Large values drive Softmax into flat saturated zones where gradients vanish (become 0)! Dividing by &radic;<i>d<sub>k</sub></i> maintains unit variance and healthy gradients.<br/>"
    "• <b>Step 3 (Softmax):</b> Normalizes scaled scores into probabilities summing to exactly 1.0 (100%).<br/>"
    "• <b>Step 4 (Weighted Sum):</b> Multiply attention probabilities by <i>V</i> to produce a context-rich blended output vector for each word.<br/>"
    "• <b>Multi-Head Attention (MHA):</b> Splits the embedding into <i>h</i> heads (e.g., 8 or 16). Head 1 tracks grammar (subject-verb), Head 2 tracks pronoun resolution ('it' &rarr; 'animal'), Head 3 tracks themes.",
    body_style
))

story.append(PageBreak())

# =========================================================================
# PAGE 3: TRANSFORMER ARCHITECTURE & POSITIONAL ENCODINGS
# =========================================================================
story.append(Paragraph("Day 3: Transformer Architecture & Positional Encodings", title_style))
story.append(Paragraph("<b>Building Deep, Scalable Language Models</b>", subtitle_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=6))

story.append(Paragraph("7. The Permutation Invariance Problem & Positional Encoding", h1_style))
story.append(Paragraph(
    "• <b>The Problem:</b> Self-attention computes pairwise dot products simultaneously across all tokens. Without order information, <i>'Dog bites man'</i> and <i>'Man bites dog'</i> produce identical attention scores! Self-attention is order-agnostic.<br/>"
    "• <b>The Sinusoidal Solution (Vaswani et al.):</b> Add a unique positional wave vector to each token's embedding:<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;<i>PE<sub>(pos, 2i)</sub> = sin( pos / 10000<sup>2i/d<sub>model</sub></sup> )</i> &nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp; <i>PE<sub>(pos, 2i+1)</sub> = cos( pos / 10000<sup>2i/d<sub>model</sub></sup> )</i><br/>"
    "• <b>Why Sine & Cosine?</b> Low dimensions oscillate rapidly (like a seconds hand, tracking immediate neighbors); high dimensions oscillate slowly (like an hour hand, tracking global sentence position). Enables models to easily learn <b>relative positions</b> via linear transformations.",
    body_style
))

story.append(Paragraph("8. The 4 Essential Components Inside Every Transformer Block", h1_style))
arch_table_data = [
    [Paragraph("<b>Component</b>", callout_style), Paragraph("<b>Mechanism & Formula</b>", callout_style), Paragraph("<b>Why It Is Crucial</b>", callout_style)],
    [Paragraph("<b>Multi-Head Attention</b>", callout_style), Paragraph("<i>Concat(head<sub>1</sub>, ..., head<sub>h</sub>) &middot; W<sub>O</sub></i>", callout_style), Paragraph("Allows tokens to gather context across multiple linguistic relationships simultaneously.", callout_style)],
    [Paragraph("<b>Residual Connection</b>", callout_style), Paragraph("<i>Output = x + Sublayer(x)</i>", callout_style), Paragraph("Solves vanishing gradients: creates an uninterrupted gradient highway through 100+ layers.", callout_style)],
    [Paragraph("<b>Layer Normalization</b>", callout_style), Paragraph("Normalize features for a single token: <i>&mu;=0, &sigma;<sup>2</sup>=1</i>", callout_style), Paragraph("Unlike BatchNorm, LayerNorm does not depend on batch size or sequence length.", callout_style)],
    [Paragraph("<b>Feed-Forward Net (FFN)</b>", callout_style), Paragraph("<i>Linear &rarr; GELU &rarr; Linear</i> (expands 4&times; dim)", callout_style), Paragraph("Acts as the permanent key-value memory store of factual world knowledge in LLMs.", callout_style)],
]
t_arch = Table(arch_table_data, colWidths=[110, 180, 250])
t_arch.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#fef3c7")),
    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ('TOPPADDING', (0,0), (-1,-1), 4),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#fde68a")),
]))
story.append(t_arch)
story.append(Spacer(1, 4))

story.append(Paragraph("9. Cross-Attention: How the Decoder Listens to the Encoder", h1_style))
story.append(Paragraph(
    "In Encoder-Decoder models (like T5):<br/>"
    "• <b>Queries (Q):</b> Generated by the <b>Decoder</b> (the target sentence being written, e.g. German).<br/>"
    "• <b>Keys (K) & Values (V):</b> Provided by the <b>Encoder</b> output (the source sentence, e.g. English).<br/>"
    "• This allows the decoder to reference the source text at every generation step!",
    body_style
))

story.append(PageBreak())

# =========================================================================
# PAGE 4: THE GREAT TRINITY & HOW LLMS GENERATE TEXT
# =========================================================================
story.append(Paragraph("Day 3: BERT, GPT, T5 & LLM Text Generation", title_style))
story.append(Paragraph("<b>The Transformer Trinity & Autoregressive Decoding Strategies</b>", subtitle_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=6))

story.append(Paragraph("10. The Great Trinity: BERT vs. GPT vs. T5", h1_style))
trinity_table_data = [
    [Paragraph("<b>Architecture</b>", callout_style), Paragraph("<b>Attention Mask</b>", callout_style), Paragraph("<b>Pre-Training Task</b>", callout_style), Paragraph("<b>Best For & Examples</b>", callout_style)],
    [Paragraph("<b>Encoder-Only (BERT)</b>", callout_style), Paragraph("<b>Bidirectional</b> (Full 1s matrix: sees left & right)", callout_style), Paragraph("<b>Masked LM (MLM):</b><br/><i>'The [MASK] sat on mat'</i>", callout_style), Paragraph("Classification, NER, Search, Sentiment (BERT, RoBERTa, DeBERTa)", callout_style)],
    [Paragraph("<b>Decoder-Only (GPT)</b>", callout_style), Paragraph("<b>Causal Mask</b> (Lower triangular: sees past only)", callout_style), Paragraph("<b>Next-Token Prediction:</b><br/><i>P(token<sub>t+1</sub> | token<sub>1...t</sub>)</i>", callout_style), Paragraph("Text generation, Chat, Coding, Reasoning (GPT-4, LLaMA, Gemini)", callout_style)],
    [Paragraph("<b>Encoder-Decoder (T5)</b>", callout_style), Paragraph("Enc: Bidirectional<br/>Dec: Causal + Cross-Attn", callout_style), Paragraph("<b>Text-to-Text:</b><br/>Span corruption & reconstruction", callout_style), Paragraph("Translation, Abstractive Summarization (T5, FLAN-T5, BART)", callout_style)],
]
t_trinity = Table(trinity_table_data, colWidths=[100, 130, 150, 160])
t_trinity.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#ede9fe")),
    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ('TOPPADDING', (0,0), (-1,-1), 4),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#ddd6fe")),
]))
story.append(t_trinity)
story.append(Spacer(1, 4))

story.append(Paragraph("11. How LLMs Generate Text & Decoding Strategies", h1_style))
story.append(Paragraph(
    "• <b>Autoregressive Loop:</b> 1. Encode prompt &rarr; 2. Forward pass &rarr; 3. Extract logits of final token &rarr; 4. Scale by Temperature (<i>z<sub>i</sub>/T</i>) &rarr; 5. Softmax to probabilities &rarr; 6. Sample next token &rarr; 7. Append and repeat!<br/>"
    "• <b>Greedy Search (argmax):</b> Always picks top-1 token. Fast, deterministic, but repetitive.<br/>"
    "• <b>Temperature (T):</b> <i>T &lt; 1.0</i> sharpens distribution (logical, coding, math); <i>T &gt; 1.0</i> flattens it (creative, diverse).<br/>"
    "• <b>Top-K Sampling:</b> Limits choices to top <i>K</i> tokens (e.g. 50), eliminating strange low-probability words.<br/>"
    "• <b>Top-P (Nucleus) Sampling:</b> <b>Industry Standard!</b> Dynamically keeps the smallest set of tokens whose cumulative probability &ge; <i>P</i> (e.g. 90%). Adaptively expands when uncertain and contracts when confident.",
    body_style
))

story.append(Paragraph("12. High-Yield Interview Flashcards", h1_style))
story.append(Paragraph(
    "• <b>Q: What is the computational complexity of standard self-attention?</b><br/>"
    "&nbsp;&nbsp;<b>A:</b> <i>O(N<sup>2</sup>)</i> in time and memory, where <i>N</i> is sequence length, because every token attends to all <i>N</i> tokens.<br/>"
    "• <b>Q: Why do LLMs make silly math errors (e.g., 98765 + 4321)?</b><br/>"
    "&nbsp;&nbsp;<b>A:</b> Tokenizer artifacts! Multi-digit numbers are chopped into arbitrary subword chunks (e.g. ['987', '65']), obscuring place value alignment without step-by-step scratchpads.<br/>"
    "• <b>Q: Why does GPT need a Causal Mask?</b><br/>"
    "&nbsp;&nbsp;<b>A:</b> To prevent the model from looking ahead at future tokens during training, which would allow it to cheat instead of learning to predict.",
    body_style
))

doc.build(story)
print(f"Day 3 Complete Notes PDF generated successfully at: {pdf_path}")
