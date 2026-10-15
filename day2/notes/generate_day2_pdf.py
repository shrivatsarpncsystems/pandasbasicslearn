import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak
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

# Clean Typography Styles
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

story = []

# =========================================================================
# PAGE 1: THE FOUNDATIONS (AI vs ML vs DL, Paradigms & Algorithm Table)
# =========================================================================
story.append(Paragraph("Day 2: Machine Learning & Neural Networks Made Simple", title_style))
story.append(Paragraph("<b>Beginner-Friendly Study Handbook: Real-Life Analogies & Intuitive Concepts</b>", subtitle_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=6))

# Section 1: The Russian Dolls
story.append(Paragraph("1. AI vs. Machine Learning vs. Deep Learning (The 3 Russian Dolls)", h1_style))
story.append(Paragraph(
    "• <b>Artificial Intelligence (AI) - The Big Umbrella:</b> Any computer acting smart. <b>AI does NOT have to learn!</b> An old 1990s chess game with 100 hard-coded if/else rules is still AI.<br/>"
    "• <b>Machine Learning (ML) - Finding Patterns:</b> The computer discovers the rules itself from past data tables: <i>Data + Past Answers &rarr; Computer Learns the Rules</i>.<br/>"
    "• <b>Deep Learning (DL) - Brain-Like Networks:</b> Handles raw sensory data (images, voice, text, ChatGPT). Automatically extracts shapes and features without humans measuring columns.",
    body_style
))

# Section 2: Supervised vs Unsupervised
story.append(Paragraph("2. Supervised vs. Unsupervised Learning (The Answer Key Question)", h1_style))
story.append(Paragraph(
    "• <b>Supervised Learning (Learning with a Teacher):</b> The computer gets the clues (X) <b>AND</b> the answer key (y). Goal: Learn the pattern to predict future answers (e.g. house price, spam, cancer).<br/>"
    "• <b>Unsupervised Learning (Learning on Your Own):</b> Data has <b>NO answer key / NO labels</b>. Like giving a child a mixed box of LEGO bricks—they naturally group similar bricks into piles! Goal: Discover customer groups, clusters, or patterns.",
    body_style
))

# Section 3: Regression vs Classification
story.append(Paragraph("3. Regression vs. Classification (What Type of Answer?)", h1_style))
story.append(Paragraph(
    "• <b>Regression (Measuring Tape):</b> Predicting a continuous <b>number</b> on a scale (Salary: $85,000, Temperature: 28.5 C, Uber fare: $24.50).<br/>"
    "• <b>Classification (Sorting Mailboxes):</b> Predicting a discrete <b>bucket or category</b> (Will they quit? Yes/No, Email: Spam/Inbox, Photo: Cat/Dog).",
    body_style
))

# Section 4: Algorithm Cheat Sheet Table
story.append(Paragraph("4. The 5 Core Algorithms at a Glance", h1_style))
algo_table_data = [
    ["Algorithm", "Type", "Real-Life Analogy", "What it does in 1 Sentence"],
    ["Linear Regression", "Supervised (Regression)", "The Trend Line", "Draws best-fit straight line to predict a continuous number."],
    ["Logistic Regression", "Supervised (Classification)", "The S-Curve Probability", "Bends the line into an S-curve to calculate a 0% to 100% Yes/No probability."],
    ["Decision Tree", "Supervised (Both)", "The 20 Questions Game", "Splits data into a simple flowchart of Yes/No questions."],
    ["Random Forest", "Supervised (Both)", "Wisdom of 100 Doctors", "Combines 100+ Decision Trees voting together to prevent mistakes."],
    ["K-Means", "Unsupervised (Clustering)", "The Party Groups", "Drops K center pins into unlabelled data to group similar customers."]
]

t1 = Table(algo_table_data, colWidths=[90, 110, 115, 225])
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
# PAGE 2: THE 5 ALGORITHMS IN PLAIN ENGLISH
# =========================================================================
story.append(PageBreak())

story.append(Paragraph("5. The 5 Core Algorithms Explained in Plain English", title_style))
story.append(Paragraph("<b>How each algorithm thinks, makes decisions, and solves problems.</b>", subtitle_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=6))

# Algorithm 1: Linear Regression
box_lin = [
    [Paragraph("<b>1. Linear Regression — 'The Trend Line'</b><br/>"
               "• <b>Concept:</b> Plots your data points and draws the single best-fitting straight line through them.<br/>"
               "• <b>Equation:</b> <code>Salary = (Slope * Experience) + Starting_Salary</code>.<br/>"
               "• <b>What it means:</b> If someone has 0 years experience, they get the Starting Salary ($45k). For every 1 year of experience, their pay goes up by the Slope ($7.2k).", callout_style)]
]
t_lin = Table(box_lin, colWidths=[540])
t_lin.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#eff6ff")),
    ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#93c5fd")),
    ('PADDING', (0, 0), (-1, -1), 4)
]))
story.append(t_lin)
story.append(Spacer(1, 4))

# Algorithm 2: Logistic Regression
box_log = [
    [Paragraph("<b>2. Logistic Regression — 'The Probability S-Curve'</b><br/>"
               "• <b>Concept:</b> Don't let the name fool you—it is used for <b>Classification (Yes/No)</b>, not predicting numbers!<br/>"
               "• <b>Mechanism:</b> Bends the straight line into an <b>S-shaped curve</b> that stays strictly between <b>0% and 100%</b>.<br/>"
               "• <b>Decision Rule:</b> If probability > 50% &rarr; Predict Yes (1). If < 50% &rarr; Predict No (0).<br/>"
               "• <b>Best For:</b> Will the customer buy? Will the patient get sick? Will the employee leave?", callout_style)]
]
t_log = Table(box_log, colWidths=[540])
t_log.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f0fdf4")),
    ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#86efac")),
    ('PADDING', (0, 0), (-1, -1), 4)
]))
story.append(t_log)
story.append(Spacer(1, 4))

# Algorithm 3: Decision Trees
box_dt = [
    [Paragraph("<b>3. Decision Tree — 'The 20 Questions Game'</b><br/>"
               "• <b>Concept:</b> A flowchart of simple Yes/No questions (e.g., <i>Salary < $60k? &rarr; Yes &rarr; Rating < 3.0? &rarr; Quit</i>).<br/>"
               "• <b>The Big Strength:</b> Super transparent! You can print it on paper and explain it to any non-technical manager.<br/>"
               "• <b>The Big Flaw (Overfitting):</b> A single tree gets greedy and memorizes every quirk of the training data, then fails in real life.", callout_style)]
]
t_dt = Table(box_dt, colWidths=[540])
t_dt.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#fefce8")),
    ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#fde047")),
    ('PADDING', (0, 0), (-1, -1), 4)
]))
story.append(t_dt)
story.append(Spacer(1, 4))

# Algorithm 4: Random Forest
box_rf = [
    [Paragraph("<b>4. Random Forest — 'The Wisdom of 100 Doctors'</b><br/>"
               "• <b>Concept:</b> Instead of trusting 1 tree, it builds <b>100 different trees</b> and takes a majority vote.<br/>"
               "• <b>Analogy:</b> If 1 doctor makes a mistake, 100 independent doctors voting together will easily agree on the right diagnosis!<br/>"
               "• <b>Why it's King:</b> The #1 most reliable algorithm for spreadsheet / tabular data. It almost never overfits.", callout_style)]
]
t_rf = Table(box_rf, colWidths=[540])
t_rf.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#fdf2f8")),
    ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#f472b6")),
    ('PADDING', (0, 0), (-1, -1), 4)
]))
story.append(t_rf)
story.append(Spacer(1, 4))

# Algorithm 5: K-Means
box_km = [
    [Paragraph("<b>5. K-Means — 'The Party Groups' (Clustering)</b><br/>"
               "• <b>Concept:</b> Unsupervised learning with <b>no labels</b>. Automatically groups points into K clusters based on distance.<br/>"
               "• <b>How it works:</b> Drops K center pins (Centroids) into the data and groups every point to its closest pin until clusters stabilize.<br/>"
               "• <b>Best For:</b> Grouping 100,000 customers into marketing personas (Bargain Hunters, High Rollers, Casuals).", callout_style)]
]
t_km = Table(box_km, colWidths=[540])
t_km.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f5f3ff")),
    ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#c4b5fd")),
    ('PADDING', (0, 0), (-1, -1), 4)
]))
story.append(t_km)

# =========================================================================
# PAGE 3: OVERFITTING, EVALUATION & NEURAL NETWORKS
# =========================================================================
story.append(PageBreak())

story.append(Paragraph("6. Overfitting, Evaluation Metrics & Neural Networks", title_style))
story.append(Paragraph("<b>The 3 Students, Grading Your Models, and The Mechanics of Brain-Like AI.</b>", subtitle_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=6))

# Overfitting (The 3 Students)
story.append(Paragraph("Overfitting vs. Underfitting (The Story of 3 Students)", h1_style))
story.append(Paragraph(
    "• <b>Underfitting ('The Lazy Student'):</b> Didn't study concepts. Model is too simple. Scores 50% on homework, 50% on real exam.<br/>"
    "• <b>Overfitting ('The Memorizer Student'):</b> <b>OVER-memorized</b> the homework! Scores 99% on practice homework, but flunks the real exam (55%)! Fix: Use Random Forest or more data.<br/>"
    "• <b>Just Right ('The Smart Student'):</b> Learned actual patterns. Scores 90% on homework and 88% on the real exam!",
    body_style
))

# Evaluation Metrics
story.append(Paragraph("How We Grade Models (Evaluation Metrics)", h1_style))
eval_table = [
    ["Type", "Metric", "What it Means in Plain English"],
    ["Regression (Numbers)", "MAE", "Average mistake size in real dollars (e.g. 'off by $3,700 on average')."],
    ["Regression (Numbers)", "RMSE", "The harsh teacher. Squares errors to penalize huge embarrassing mistakes."],
    ["Regression (Numbers)", "R² Score", "Percentage of perfection (0% to 100%). 96% means 96% of variance is explained."],
    ["Classification (Yes/No)", "Accuracy", "Overall % correct. Can be dangerous if 99% of people are healthy!"],
    ["Classification (Yes/No)", "Precision", "When alarm rings, is there a real fire? High Precision = Few False Alarms."],
    ["Classification (Yes/No)", "Recall", "Did we catch every sick patient? High Recall = Zero Missed Cases (vital in medicine!)."],
    ["Classification (Yes/No)", "F1-Score", "The fair referee balancing Precision and Recall into one final score."]
]
t_ev = Table(eval_table, colWidths=[105, 75, 360])
t_ev.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e40af")),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 6.5),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")])
]))
story.append(t_ev)
story.append(Spacer(1, 4))

# Neural Networks, ReLU, Gradient Descent
story.append(Paragraph("Neural Networks, ReLU & Gradient Descent", h1_style))
nn_story_box = [
    [Paragraph("• <b>The Neuron:</b> Multiplies clues by importance dials (Weights) and adds them up: <code>Sum = (X1*W1) + (X2*W2) + Bias</code>.<br/>"
               "• <b>What is ReLU? (The Automatic AC Analogy):</b><br/>"
               "&nbsp;&nbsp;&nbsp;&nbsp;• Rule: <i>If negative &rarr; 0; If positive &rarr; Keep as-is.</i> (e.g. -10 &rarr; 0, +5 &rarr; 5).<br/>"
               "&nbsp;&nbsp;&nbsp;&nbsp;• Like an AC set to 24°C: If room is cold (-4°C), AC stays OFF (0). If room heats up (+4°C), AC turns ON at power 4.<br/>"
               "&nbsp;&nbsp;&nbsp;&nbsp;• <b>Why we need it:</b> Without ReLU, a neural network is a flat straight ruler. ReLU adds hinges/corners so the network can bend to learn complex shapes (cat faces, speech, text)!<br/>"
               "• <b>Loss Function:</b> The scoreboard of mistakes. Goal: Drive loss down to ZERO.<br/>"
               "• <b>Gradient Descent (The Foggy Mountain Hiker):</b> Stepping downhill into the valley where mistakes are 0.<br/>"
               "&nbsp;&nbsp;&nbsp;&nbsp;• <i>Learning Rate Too Large:</i> Hiker takes 50-foot giant leaps &rarr; overshoots the valley completely & loss jumps erratically!<br/>"
               "&nbsp;&nbsp;&nbsp;&nbsp;• <i>Learning Rate Too Small:</i> Hiker moves 1mm per day &rarr; takes 10 years to reach the bottom.", callout_style)]
]
t_nn = Table(nn_story_box, colWidths=[540])
t_nn.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f0fdf4")),
    ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#86efac")),
    ('PADDING', (0, 0), (-1, -1), 4)
]))
story.append(t_nn)

doc.build(story)
print(f"-> Successfully compiled beginner-friendly Day 2 PDF at: {pdf_path}")
