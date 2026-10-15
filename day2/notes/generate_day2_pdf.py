import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

pdf_path = "day2/notes/Day_2_Complete_Notes.pdf"
doc = SimpleDocTemplate(
    pdf_path,
    pagesize=letter,
    rightMargin=34,
    leftMargin=34,
    topMargin=34,
    bottomMargin=34
)

styles = getSampleStyleSheet()

# Typography styling
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

code_snippet_style = ParagraphStyle(
    "CodeSnippet",
    parent=styles["Code"],
    fontSize=7,
    leading=9.5,
    textColor=colors.HexColor("#0f766e"),
    fontName="Courier"
)

story = []

# =========================================================================
# PAGE 1: AI/ML/DL & THE 5 CORE ALGORITHMS DEEP DIVE
# =========================================================================
story.append(Paragraph("Day 2 Master Handbook: Machine Learning & Neural Networks", title_style))
story.append(Paragraph("<b>Comprehensive Theoretical Reference & Technical Study Guide</b>", subtitle_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=6))

# 1. AI vs ML vs DL
story.append(Paragraph("1. AI vs. Machine Learning vs. Deep Learning", h1_style))
story.append(Paragraph(
    "• <b>Artificial Intelligence (AI):</b> Any system mimicking human decision-making. Does NOT require learning (e.g. 1990s chess engines with hard-coded if/else rules).<br/>"
    "• <b>Machine Learning (ML):</b> Algorithms discovering predictive rules from past data: <i>Data + Past Answers &rarr; Machine Learns Rules</i>. Requires human feature engineering.<br/>"
    "• <b>Deep Learning (DL):</b> Multi-layered neural networks performing <b>Automatic Feature Extraction</b> on raw unstructured sensory inputs (images, audio, text).",
    body_style
))

# 2. Supervised vs Unsupervised & Regression vs Classification
story.append(Paragraph("2. Learning Paradigms & Task Taxonomy", h1_style))
story.append(Paragraph(
    "• <b>Supervised Learning:</b> Ground truth target labels (y) exist. Learns mapping f(X) &rarr; y. Tasks: <b>Regression</b> (continuous numbers like Price, Salary) and <b>Classification</b> (discrete categories like Pass/Fail, Cat/Dog).<br/>"
    "• <b>Unsupervised Learning:</b> NO target labels exist. Discovers latent geometry, groupings, and distributions: P(X). Tasks: <b>Clustering</b> (K-Means) and <b>Dimensionality Reduction</b> (PCA).",
    body_style
))

# 3. The 5 Algorithms Table
story.append(Paragraph("3. The 5 Core Algorithms Master Comparison", h1_style))

algo_table_data = [
    ["Algorithm", "Type", "Mathematical Mechanism", "Strengths", "Weaknesses"],
    ["Linear Regression", "Supervised (Regression)", "Ordinary Least Squares (OLS) minimizing RSS: y = wX + b", "Highly interpretable, fast", "Assumes strictly linear relations"],
    ["Logistic Regression", "Supervised (Classification)", "Bends line via Sigmoid into probability: P = 1/(1+e^-z)", "Outputs calibrated probability", "Linear decision boundary only"],
    ["Decision Tree", "Supervised (Both)", "Partitions feature space via Gini Impurity or Entropy", "Zero scaling needed, interpretable", "Severe Overfitting if unpruned"],
    ["Random Forest", "Supervised (Both)", "Bagging ensemble of 100+ de-correlated Trees + feature sub-sampling", "State-of-the-art on tabular data", "Slow on massive real-time data"],
    ["K-Means", "Unsupervised (Clustering)", "Iteratively minimizes WCSS distance to K geometric centroids", "Scalable, discovers groups", "Assumes spherical clusters"]
]

t1 = Table(algo_table_data, colWidths=[80, 95, 155, 105, 105])
t1.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2563eb")),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 6.5),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ('TOPPADDING', (0, 0), (-1, -1), 2.5),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")])
]))
story.append(t1)

# =========================================================================
# PAGE 2: EVALUATION METRICS, OVERFITTING & BIAS-VARIANCE
# =========================================================================
story.append(PageBreak())

story.append(Paragraph("4. Model Diagnostics & Evaluation Metrics", title_style))
story.append(Paragraph("<b>The Bias-Variance Tradeoff, K-Fold Validation, and Mathematical Scoring</b>", subtitle_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=6))

story.append(Paragraph("The Bias-Variance Tradeoff & Overfitting", h1_style))
overfit_box = [
    [Paragraph("• <b>High Bias (Underfitting):</b> Model is too simple. Fails on both training and test data. Fix: Increase model capacity, add features.<br/>"
               "• <b>High Variance (Overfitting):</b> Model memorizes training noise (e.g. 99% train score, 55% test score). Fix: Regularization, more data, Random Forests, feature reduction.<br/>"
               "• <b>K-Fold Cross-Validation:</b> Splits data into K folds (e.g. 5 folds). Rotates test fold 5 times and averages scores to prevent split luck.", callout_style)]
]
t_box1 = Table(overfit_box, colWidths=[540])
t_box1.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f0fdf4")),
    ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#86efac")),
    ('PADDING', (0, 0), (-1, -1), 4)
]))
story.append(t_box1)
story.append(Spacer(1, 4))

story.append(Paragraph("Regression Metrics (Measuring Numerical Error)", h1_style))
reg_metrics = [
    ["Metric", "Formula / Concept", "Interpretation", "Outlier Sensitivity"],
    ["MAE", "Mean(|y_true - y_pred|)", "Average mistake magnitude in actual dollars/units", "Robust to outliers"],
    ["MSE", "Mean((y_true - y_pred)^2)", "Squares errors before averaging", "Highly sensitive"],
    ["RMSE", "Sqrt(MSE)", "Converts squared penalty back into original units", "Penalizes large errors"],
    ["R² Score", "1 - (RSS / TSS)", "Proportion of target variance explained (0.0 to 1.0)", "Relative benchmark"]
]
t_reg = Table(reg_metrics, colWidths=[65, 155, 210, 110])
t_reg.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e40af")),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 6.5),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")])
]))
story.append(t_reg)
story.append(Spacer(1, 4))

story.append(Paragraph("Classification Metrics (Confusion Matrix & Tradeoffs)", h1_style))
clf_metrics = [
    ["Metric", "Formula", "Real-World Meaning", "When to Prioritize"],
    ["Accuracy", "(TP + TN) / Total", "Percentage of overall correct guesses", "Balanced classes only"],
    ["Precision", "TP / (TP + FP)", "Of all positive alarms called, how many were real?", "When False Alarms are costly (Spam)"],
    ["Recall", "TP / (TP + FN)", "Of all actual positive cases, how many did we catch?", "When Missed Cases are fatal (Cancer, Fraud)"],
    ["F1-Score", "2*(Prec*Rec)/(Prec+Rec)", "Harmonic mean balancing Precision and Recall", "Imbalanced datasets"],
    ["ROC-AUC", "Area Under TPR vs FPR", "Model ability to separate classes across all thresholds", "Overall diagnostic evaluation"]
]
t_clf = Table(clf_metrics, colWidths=[65, 125, 205, 145])
t_clf.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e40af")),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 6.5),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")])
]))
story.append(t_clf)

# =========================================================================
# PAGE 3: NEURAL NETWORKS, ACTIVATION FUNCTIONS & OPTIMIZATION
# =========================================================================
story.append(PageBreak())

story.append(Paragraph("5. Neural Networks, Activation Functions & Optimization", title_style))
story.append(Paragraph("<b>The Architecture of Modern Deep Learning: Neurons, ReLU, Loss, and Gradient Descent</b>", subtitle_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=6))

story.append(Paragraph("The Artificial Neuron (Perceptron)", h1_style))
neuron_box = [
    [Paragraph("An artificial neuron mimics biological synapses via a linear weighted combination followed by an activation:<br/>"
               "&nbsp;&nbsp;&nbsp;&nbsp;<b>1. Weighted Sum:</b> <code>z = (w1*x1) + (w2*x2) + ... + (wd*xd) + b</code><br/>"
               "&nbsp;&nbsp;&nbsp;&nbsp;<b>2. Activation:</b> <code>output = f(z)</code><br/>"
               "<b>Why Non-Linear Activations are Mandatory:</b> Stacking 50 purely linear layers collapses mathematically into a single linear matrix multiplication: <code>W2 * (W1 * x) = W_combined * x</code>. Without non-linear activation functions, a deep neural network is just Linear Regression and can never learn curved boundaries, circles, or image patterns!", callout_style)]
]
t_box2 = Table(neuron_box, colWidths=[540])
t_box2.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#eff6ff")),
    ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#93c5fd")),
    ('PADDING', (0, 0), (-1, -1), 4)
]))
story.append(t_box2)
story.append(Spacer(1, 4))

story.append(Paragraph("Activation Functions Deep Dive", h1_style))
act_table_data = [
    ["Function", "Formula", "Range", "Mechanisms & When to Use"],
    ["ReLU", "f(z) = max(0, z)", "[0, +inf)", "The king of hidden layers. Turns negatives to 0, passes positives. Solves vanishing gradient, ultra-fast."],
    ["Sigmoid", "f(z) = 1 / (1 + e^-z)", "[0, 1]", "Smooth S-curve. Standard output layer for Binary Classification (outputs probability). Causes vanishing gradient in hidden layers."],
    ["Tanh", "f(z) = (e^z - e^-z)/(e^z + e^-z)", "[-1, 1]", "Zero-centered S-curve. Preferred over Sigmoid in recurrent networks."],
    ["Softmax", "e^zi / Sum(e^zj)", "[0, 1] (Sum=1)", "Multi-class classification output layer. Normalizes raw logits into mutually exclusive percentage distribution."]
]
t_act = Table(act_table_data, colWidths=[55, 125, 60, 300])
t_act.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e40af")),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 6.5),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")])
]))
story.append(t_act)
story.append(Spacer(1, 4))

story.append(Paragraph("Loss Functions & Gradient Descent Optimization", h1_style))
gd_box = [
    [Paragraph("• <b>Loss Function:</b> Mathematical penalty score. Regression uses <b>MSE</b>; Binary classification uses <b>Binary Cross-Entropy (Log Loss)</b>: <code>Loss = -[y*log(p) + (1-y)*log(1-p)]</code>.<br/>"
               "• <b>Gradient Descent:</b> Iterative optimization algorithm finding the lowest valley of loss. Updates weights in the opposite direction of the gradient:<br/>"
               "&nbsp;&nbsp;&nbsp;&nbsp;<code>New_Weight = Old_Weight - (Learning_Rate * Gradient)</code><br/>"
               "• <b>Learning Rate (Step Size):</b><br/>"
               "&nbsp;&nbsp;&nbsp;&nbsp;• <i>Too Large:</i> Steps overshoot the valley bottom; loss diverges erratically (Loss &rarr; NaN).<br/>"
               "&nbsp;&nbsp;&nbsp;&nbsp;• <i>Too Small:</i> Takes glacial baby steps; network takes days to converge or gets trapped.<br/>"
               "• <b>Backpropagation:</b> Applies the Chain Rule of calculus from output backwards through every layer to compute exact weight gradients.", callout_style)]
]
t_box3 = Table(gd_box, colWidths=[540])
t_box3.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#fefce8")),
    ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#fde047")),
    ('PADDING', (0, 0), (-1, -1), 4)
]))
story.append(t_box3)

doc.build(story)
print(f"-> Successfully compiled exhaustive 3-page Day 2 PDF at: {pdf_path}")
