import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

pdf_path = "day1/notes/Day_1_Complete_Notes.pdf"
doc = SimpleDocTemplate(
    pdf_path,
    pagesize=letter,
    rightMargin=36,
    leftMargin=36,
    topMargin=36,
    bottomMargin=36
)

styles = getSampleStyleSheet()

# Custom typography styles
title_style = ParagraphStyle(
    "DocTitle",
    parent=styles["Heading1"],
    fontSize=20,
    leading=24,
    textColor=colors.HexColor("#1e3a8a"),
    spaceAfter=4
)

subtitle_style = ParagraphStyle(
    "DocSubTitle",
    parent=styles["Normal"],
    fontSize=10,
    leading=14,
    textColor=colors.HexColor("#4b5563"),
    spaceAfter=10
)

h1_style = ParagraphStyle(
    "SectionH1",
    parent=styles["Heading2"],
    fontSize=13,
    leading=17,
    textColor=colors.HexColor("#1e40af"),
    spaceBefore=10,
    spaceAfter=5,
    keepWithNext=True
)

h2_style = ParagraphStyle(
    "SectionH2",
    parent=styles["Heading3"],
    fontSize=11,
    leading=15,
    textColor=colors.HexColor("#2563eb"),
    spaceBefore=7,
    spaceAfter=3,
    keepWithNext=True
)

body_style = ParagraphStyle(
    "BodyDark",
    parent=styles["Normal"],
    fontSize=9,
    leading=13,
    textColor=colors.HexColor("#1f2937"),
    spaceAfter=5
)

callout_style = ParagraphStyle(
    "CalloutText",
    parent=styles["Normal"],
    fontSize=8.5,
    leading=12,
    textColor=colors.HexColor("#1e293b")
)

code_snippet_style = ParagraphStyle(
    "CodeSnippet",
    parent=styles["Code"],
    fontSize=8,
    leading=11,
    textColor=colors.HexColor("#0f766e"),
    fontName="Courier"
)

story = []

# =========================================================================
# PAGE 1: PANDAS FOUNDATIONS, CLEANING & EDA
# =========================================================================
story.append(Paragraph("Day 1: Python, Pandas & Machine Learning Preprocessing", title_style))
story.append(Paragraph("<b>Complete Master Handbook & Beginner-Friendly Concept Guide</b>", subtitle_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=10))

story.append(Paragraph("1. Core Pandas Foundations", h1_style))
story.append(Paragraph(
    "• <b>Series:</b> A 1-Dimensional list with labels (index). Think of it as a single spreadsheet column.<br/>"
    "• <b>DataFrame:</b> A 2-Dimensional table made of multiple Series sharing the same row index.<br/>"
    "• <b>.loc vs .iloc:</b> <code>.loc</code> accesses data by <b>Label / Name</b> (e.g., <code>df.loc[df['Age'] > 30]</code>). "
    "<code>.iloc</code> accesses data by <b>Integer Position</b> (e.g., <code>df.iloc[0:5, 0:2]</code> for first 5 rows and 2 columns).",
    body_style
))

story.append(Paragraph("2. Data Cleaning in a Nutshell", h1_style))
story.append(Paragraph(
    "• <b>Duplicates:</b> <code>df.drop_duplicates(subset=['id'])</code> removes repeat rows that falsely bias models.<br/>"
    "• <b>Missing Values (NaN):</b> Don't blindly drop! Impute numerical columns with <b>Median</b> (resistant to outliers) and categorical columns with <b>Mode</b> or 'Unknown'.<br/>"
    "• <b>Dirty Text & Numbers:</b> Strip spaces (<code>.str.strip()</code>) and remove currency/commas (<code>.str.replace(r'[\\$,]', '', regex=True)</code>) before converting with <code>pd.to_numeric(..., errors='coerce')</code>.",
    body_style
))

story.append(Paragraph("3. Data Manipulation & Aggregations", h1_style))
story.append(Paragraph(
    "• <b>Filtering:</b> Wrap conditions in parentheses: <code>df[(df['Salary'] > 75000) & (df['Age'] <= 35)]</code>.<br/>"
    "• <b>GroupBy:</b> Summarize categories: <code>df.groupby('Department')['Salary'].agg(['mean', 'median', 'count'])</code>.<br/>"
    "• <b>Merging:</b> Combine tables like SQL joins: <code>pd.merge(df, df_dept, on='Department', how='left')</code>.",
    body_style
))

story.append(Paragraph("4. Exploratory Data Analysis (EDA)", h1_style))
story.append(Paragraph(
    "• <b>5-Number Summary:</b> <code>df.describe()</code> reveals count, mean, min, 25%, 50% (median), 75%, and max.<br/>"
    "• <b>Correlation:</b> <code>df.corr(numeric_only=True)</code> measures linear dependence between -1 and +1.<br/>"
    "• <b>Visuals:</b> Histograms (distribution shape), Boxplots (outliers), Countplots (class imbalance), Heatmaps (relationships).",
    body_style
))

story.append(Spacer(1, 10))

# Quick Reference Table on Page 1
quick_table_data = [
    ["Goal", "Pandas Code"],
    ["Inspect Shape & Nulls", "df.shape;  df.isnull().sum()"],
    ["Drop Duplicates", "df.drop_duplicates(subset=['Emp_ID'])"],
    ["Impute Missing Median", "df['Salary'] = df['Salary'].fillna(df['Salary'].median())"],
    ["Filter Multiple Conditions", "df[(df['Age'] > 30) & (df['Dept'] == 'IT')]"],
    ["GroupBy Summary", "df.groupby('Dept')['Salary'].mean()"],
    ["Left Join Tables", "pd.merge(df, budget_df, on='Dept', how='left')"]
]

t1 = Table(quick_table_data, colWidths=[140, 390])
t1.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2563eb")),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 8),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ('TOPPADDING', (0, 0), (-1, -1), 3),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")])
]))
story.append(t1)

# =========================================================================
# PAGE 2: MACHINE LEARNING PREPROCESSING EXPLAINED IN SIMPLE TERMS
# =========================================================================
story.append(PageBreak())

story.append(Paragraph("5. Machine Learning Preprocessing Explained in Simple Terms", title_style))
story.append(Paragraph("<b>Why do we need this? A Machine Learning model is just a math formula (Prediction = W1*X1 + W2*X2 + b). Because it is pure math, it cannot understand words, gets confused by big numbers, and cheats if you give it exam answers in advance.</b>", subtitle_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=10))

# Concept 1: Target (y) vs Features (X)
story.append(Paragraph("Concept 1: The Secret Answer (y) vs. The Clues (X)", h1_style))
concept1_box = [
    [Paragraph("<b>Real-Life Analogy: The Detective & The Crime</b><br/>"
               "• <b>Target (y):</b> The Mystery you want to solve (e.g., <code>y = df['Left_Company']</code>: Did the employee quit? 1 = Yes, 0 = No).<br/>"
               "• <b>Features (X):</b> The Clues the detective uses to guess (Age, Department, Salary, Rating, Experience).<br/>"
               "• <b>Why drop 'Emp_ID' and 'Name'?</b> If a detective says <i>'Employee #104 named David quit, so every future person named David will quit!'</i>, that is terrible logic! Names and IDs are random unique labels. Keeping them causes the model to memorize badges rather than learning real human patterns.", callout_style)],
    [Paragraph("<b>Code:</b> <code>y = df['Left_Company']</code><br/>"
               "<code>X = df.drop(columns=['Emp_ID', 'Name', 'Dept_Head', 'Left_Company'])</code>", code_snippet_style)]
]
t_box1 = Table(concept1_box, colWidths=[530])
t_box1.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f0fdf4")),
    ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#86efac")),
    ('PADDING', (0, 0), (-1, -1), 6)
]))
story.append(t_box1)
story.append(Spacer(1, 8))

# Concept 2: One-Hot Encoding
story.append(Paragraph("Concept 2: Turning Words into Numbers (One-Hot Encoding)", h1_style))
concept2_box = [
    [Paragraph("<b>Why can't we just write IT = 1, HR = 2, Sales = 3?</b><br/>"
               "Because computers do math! If IT=1 and Sales=3, the computer thinks: <i>'Sales is 3 times bigger than IT'</i> and <i>'IT (1) + HR (2) = Sales (3)'</i>! That is completely absurd.<br/>"
               "<b>The Solution (Light Switches):</b> We create a separate 1 or 0 column for each department:<br/>"
               "&nbsp;&nbsp;• If Alice is in IT &rarr; <code>Dept_IT = 1</code>, <code>Dept_HR = 0</code>, <code>Dept_Sales = 0</code>.<br/>"
               "&nbsp;&nbsp;• <code>drop_first=True</code>: If both Dept_IT=0 and Dept_HR=0, the model already knows they are in Sales! Dropping 1 redundant column prevents mathematical confusion (the dummy variable trap).", callout_style)],
    [Paragraph("<b>Code:</b> <code>X_encoded = pd.get_dummies(X, columns=['Department'], drop_first=True)</code>", code_snippet_style)]
]
t_box2 = Table(concept2_box, colWidths=[530])
t_box2.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#eff6ff")),
    ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#93c5fd")),
    ('PADDING', (0, 0), (-1, -1), 6)
]))
story.append(t_box2)
story.append(Spacer(1, 8))

# Concept 3: Train-Test Split
story.append(Paragraph("Concept 3: Train vs. Test Split (Homework vs. Surprise Exam)", h1_style))
concept3_box = [
    [Paragraph("<b>Real-Life Analogy: The Student & The Final Exam</b><br/>"
               "If a teacher gives you the exact exam questions to practice at home yesterday, and you score 100% today, you didn't learn—you just memorized answers!<br/>"
               "• <b>Training Set (75%):</b> The practice textbook exercises the model is allowed to study.<br/>"
               "• <b>Testing Set (25%):</b> The surprise exam. The model has never seen these rows before. This is how we prove the model actually works in the real world.<br/>"
               "• <b>stratify=y:</b> Ensures the test exam has the exact same proportion of quitters (30%) and stayers (70%) as real life.", callout_style)],
    [Paragraph("<b>Code:</b> <code>X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.25, random_state=42, stratify=y)</code>", code_snippet_style)]
]
t_box3 = Table(concept3_box, colWidths=[530])
t_box3.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#fefce8")),
    ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#fde047")),
    ('PADDING', (0, 0), (-1, -1), 6)
]))
story.append(t_box3)
story.append(Spacer(1, 8))

# Concept 4: Feature Scaling
story.append(Paragraph("Concept 4: Feature Scaling (The Giant vs. The Ant)", h1_style))
concept4_box = [
    [Paragraph("<b>The Problem:</b> Look at two features for an employee:<br/>"
               "&nbsp;&nbsp;• Salary = <b>$90,000</b> (A massive giant)<br/>"
               "&nbsp;&nbsp;• Rating = <b>4.5</b> (A tiny ant)<br/>"
               "When the model calculates <code>W1 * 90000 + W2 * 4.5</code>, the $90,000 completely drowns out the 4.5! The model will ignore Rating entirely, even though performance rating is often the #1 reason people leave!<br/>"
               "<b>The Fix (StandardScaler):</b> We convert all numbers into how many steps they are away from their average:<br/>"
               "&nbsp;&nbsp;• Average Salary becomes <b>0</b>. A high salary becomes <b>+1.5</b>. A low salary becomes <b>-1.2</b>.<br/>"
               "&nbsp;&nbsp;• Average Rating becomes <b>0</b>. A high rating becomes <b>+1.4</b>.<br/>"
               "Now both features compete on a level, fair playing field!<br/>"
               "<b>Golden Rule:</b> <code>fit_transform()</code> ONLY on train data. <code>transform()</code> on test data. Never let the scaler look at the test exam (prevents data leakage)!", callout_style)],
    [Paragraph("<b>Code:</b><br/>"
               "<code>scaler = StandardScaler()</code><br/>"
               "<code>X_train[num_cols] = scaler.fit_transform(X_train[num_cols])  # Learn average from homework & scale</code><br/>"
               "<code>X_test[num_cols] = scaler.transform(X_test[num_cols])        # Use homework average on exam (NO PEEKING!)</code>", code_snippet_style)]
]
t_box4 = Table(concept4_box, colWidths=[530])
t_box4.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#fdf2f8")),
    ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#f472b6")),
    ('PADDING', (0, 0), (-1, -1), 6)
]))
story.append(t_box4)

doc.build(story)
print(f"-> Successfully generated enhanced PDF notes at: {pdf_path}")
