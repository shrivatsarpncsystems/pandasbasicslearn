import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

pdf_path = "day2/notes/Day_2_Complete_Notes.pdf"
doc = SimpleDocTemplate(
    pdf_path,
    pagesize=letter,
    rightMargin=36,
    leftMargin=36,
    topMargin=36,
    bottomMargin=36
)

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "DocTitle",
    parent=styles["Heading1"],
    fontSize=19,
    leading=23,
    textColor=colors.HexColor("#1e3a8a"),
    spaceAfter=4
)

subtitle_style = ParagraphStyle(
    "DocSubTitle",
    parent=styles["Normal"],
    fontSize=9.5,
    leading=13,
    textColor=colors.HexColor("#4b5563"),
    spaceAfter=8
)

h1_style = ParagraphStyle(
    "SectionH1",
    parent=styles["Heading2"],
    fontSize=12,
    leading=16,
    textColor=colors.HexColor("#1e40af"),
    spaceBefore=8,
    spaceAfter=4,
    keepWithNext=True
)

body_style = ParagraphStyle(
    "BodyDark",
    parent=styles["Normal"],
    fontSize=8.5,
    leading=12,
    textColor=colors.HexColor("#1f2937"),
    spaceAfter=4
)

callout_style = ParagraphStyle(
    "CalloutText",
    parent=styles["Normal"],
    fontSize=8,
    leading=11.5,
    textColor=colors.HexColor("#1e293b")
)

code_snippet_style = ParagraphStyle(
    "CodeSnippet",
    parent=styles["Code"],
    fontSize=7.5,
    leading=10,
    textColor=colors.HexColor("#0f766e"),
    fontName="Courier"
)

story = []

# =========================================================================
# PAGE 1: CORE CONCEPTS & THE 5 ALGORITHMS
# =========================================================================
story.append(Paragraph("Day 2: Machine Learning & Neural Networks", title_style))
story.append(Paragraph("<b>Complete High-Yield Reference Handbook & Study Notes</b>", subtitle_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=8))

# Section 1: AI vs ML vs DL
story.append(Paragraph("1. AI vs. Machine Learning vs. Deep Learning", h1_style))
story.append(Paragraph(
    "• <b>Artificial Intelligence (AI):</b> Any system that mimics human decisions. Does NOT have to learn (can be 50 hard-coded if/else rules).<br/>"
    "• <b>Machine Learning (ML):</b> Systems that discover patterns directly from past data without hard-coded rules.<br/>"
    "• <b>Deep Learning (DL):</b> Machine learning powered by multi-layer Neural Networks. Automatically extracts features from raw images, audio, and text.",
    body_style
))

# Section 2: Supervised vs Unsupervised & Regression vs Classification
story.append(Paragraph("2. Learning Paradigms & Task Types", h1_style))
story.append(Paragraph(
    "• <b>Supervised Learning:</b> Computer has an <b>Answer Key (y)</b>. Learns to predict future answers (e.g. house price, cancer detection).<br/>"
    "• <b>Unsupervised Learning:</b> Computer has <b>NO Answer Key</b>. Finds natural clusters/patterns on its own (e.g. customer segmentation).<br/>"
    "• <b>Regression:</b> Supervised task predicting a continuous <b>number</b> (e.g. Salary, Temperature, Used Car Price).<br/>"
    "• <b>Classification:</b> Supervised task predicting a discrete <b>bucket/category</b> (e.g. Pass/Fail, Spam/Not Spam, Cat/Dog).",
    body_style
))

# Section 3: The 5 Algorithms Table
story.append(Paragraph("3. The 5 Core Algorithms Comparison Table", h1_style))

algo_table_data = [
    ["Algorithm", "Type", "Core Intuition", "Best For"],
    ["Linear Regression", "Supervised (Regression)", "Draws best-fit straight line: y = mx + b", "Continuous numbers (Salaries, Prices)"],
    ["Logistic Regression", "Supervised (Classification)", "S-curve estimating 0% to 100% probability", "Yes/No decisions (Defaults, Churn)"],
    ["Decision Tree", "Supervised (Both)", "Flowchart of Yes/No threshold questions", "Easy business explanations"],
    ["Random Forest", "Supervised (Both)", "Ensemble of 100+ Trees voting together", "Tabular data, avoids overfitting"],
    ["K-Means", "Unsupervised (Clustering)", "Groups unlabelled points into K center pins", "Customer segmentation, groupings"]
]

t1 = Table(algo_table_data, colWidths=[95, 115, 175, 145])
t1.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2563eb")),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 7.5),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ('TOPPADDING', (0, 0), (-1, -1), 3),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")])
]))
story.append(t1)

# =========================================================================
# PAGE 2: EVALUATION, OVERFITTING, AND NEURAL NETWORKS
# =========================================================================
story.append(PageBreak())

story.append(Paragraph("4. Training, Testing, Overfitting & Metrics", title_style))
story.append(Paragraph("<b>Ensuring your model generalizes to unseen test data and grading its performance accurately.</b>", subtitle_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=8))

story.append(Paragraph("Overfitting vs. Underfitting", h1_style))
overfit_box = [
    [Paragraph("• <b>Underfitting (Under-prepared):</b> Model is too simple. Performs poorly on both training and test sets.<br/>"
               "• <b>Overfitting (Over-memorized):</b> Model memorized training noise. Scores 99% on training, drops to 55% on test exam! Fix with more data or Random Forests.<br/>"
               "• <b>Regression Metrics:</b> <b>MAE</b> (average mistake size in real $), <b>RMSE</b> (penalizes massive mistakes), <b>R-Squared</b> (% variance explained, 0 to 1).<br/>"
               "• <b>Classification Metrics:</b> <b>Accuracy</b> (% correct), <b>Precision</b> (few false alarms), <b>Recall</b> (catches every real positive case, vital in medicine), <b>F1</b> (harmonic balance).", callout_style)]
]
t_box1 = Table(overfit_box, colWidths=[530])
t_box1.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f0fdf4")),
    ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#86efac")),
    ('PADDING', (0, 0), (-1, -1), 5)
]))
story.append(t_box1)
story.append(Spacer(1, 6))

story.append(Paragraph("5. Neural Networks & Activation Functions", h1_style))
nn_box = [
    [Paragraph("<b>How a Neuron Thinks:</b> Multiplies inputs by weights and adds a bias: <code>Sum = (X1*W1) + (X2*W2) + Bias</code>.<br/>"
               "<b>Why Activation Functions?</b> Without them, stacking 100 layers is still just a flat ruler. Activation functions add hinges so the network can bend to fit complex curves (cat faces, speech).<br/>"
               "• <b>ReLU (Rectified Linear Unit):</b> <code>If x < 0 -> 0; If x >= 0 -> x</code>. Fast, acts like an automatic threshold switch (e.g., AC turns ON only above 24°C).<br/>"
               "• <b>Sigmoid:</b> Squashes values between 0 and 1 (ideal for final Yes/No probability).<br/>"
               "• <b>Softmax:</b> Converts multi-class outputs into percentages that sum to 100%.", callout_style)]
]
t_box2 = Table(nn_box, colWidths=[530])
t_box2.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#eff6ff")),
    ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#93c5fd")),
    ('PADDING', (0, 0), (-1, -1), 5)
]))
story.append(t_box2)
story.append(Spacer(1, 6))

story.append(Paragraph("6. Loss Function & Gradient Descent", h1_style))
gd_box = [
    [Paragraph("<b>The Foggy Mountain Analogy:</b><br/>"
               "• <b>Loss Function:</b> The error scoreboard measuring how wrong the model is. Goal = reach 0 error at the valley bottom.<br/>"
               "• <b>Gradient Descent:</b> Stepping downhill along the steepest slope to reduce loss step-by-step.<br/>"
               "• <b>Learning Rate (Step Size):</b><br/>"
               "&nbsp;&nbsp;&nbsp;&nbsp;• <i>Too Large:</i> Giant leap &rarr; overshoots the valley completely & loss jumps erratically!<br/>"
               "&nbsp;&nbsp;&nbsp;&nbsp;• <i>Too Small:</i> Tiny baby steps &rarr; takes years to reach the valley bottom.<br/>"
               "• <b>Backpropagation:</b> Calculating the gradient backwards through the network to tweak every neuron's weight.", callout_style)]
]
t_box3 = Table(gd_box, colWidths=[530])
t_box3.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#fefce8")),
    ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#fde047")),
    ('PADDING', (0, 0), (-1, -1), 5)
]))
story.append(t_box3)

doc.build(story)
print(f"-> Successfully generated Day 2 PDF at: {pdf_path}")
