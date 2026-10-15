# Day 2 Master Handbook: Machine Learning and Neural Networks

---

# Table of Contents
1. [AI vs. Machine Learning vs. Deep Learning](#1-ai-vs-machine-learning-vs-deep-learning)
2. [Learning Paradigms: Supervised vs. Unsupervised](#2-learning-paradigms-supervised-vs-unsupervised)
3. [Task Types: Regression vs. Classification](#3-task-types-regression-vs-classification)
4. [Deep Dive into the 5 Core Algorithms](#4-deep-dive-into-the-5-core-algorithms)
   - [4.1 Linear Regression](#41-linear-regression)
   - [4.2 Logistic Regression](#42-logistic-regression)
   - [4.3 Decision Trees](#43-decision-trees)
   - [4.4 Random Forest](#44-random-forest)
   - [4.5 K-Means Clustering](#45-k-means-clustering)
5. [Training, Testing, Overfitting & Evaluation Metrics](#5-training-testing-overfitting--evaluation-metrics)
   - [5.1 The Bias-Variance Tradeoff](#51-the-bias-variance-tradeoff)
   - [5.2 K-Fold Cross-Validation](#52-k-fold-cross-validation)
   - [5.3 Regression Metrics (MAE, MSE, RMSE, R-Squared)](#53-regression-metrics)
   - [5.4 Classification Metrics (Confusion Matrix, Precision, Recall, F1, ROC-AUC)](#54-classification-metrics)
6. [Neural Networks, Activation Functions & Optimization](#6-neural-networks-activation-functions--optimization)
   - [6.1 The Biological vs. Artificial Neuron](#61-the-biological-vs-artificial-neuron)
   - [6.2 Activation Functions Deep Dive (ReLU, Sigmoid, Tanh, Softmax)](#62-activation-functions-deep-dive)
   - [6.3 Loss Functions (Cost Functions)](#63-loss-functions)
   - [6.4 Gradient Descent & Backpropagation](#64-gradient-descent--backpropagation)
   - [6.5 Learning Rate Dynamics](#65-learning-rate-dynamics)
7. [Day 2 Summary Comparison & Algorithm Decision Guide](#7-day-2-summary-comparison--algorithm-decision-guide)

---

# 1. AI vs. Machine Learning vs. Deep Learning

Modern data science rests upon three concentric fields. Understanding their boundaries is essential:

```
┌─────────────────────────────────────────────────────────────────┐
│ ARTIFICIAL INTELLIGENCE (AI)                                     │
│ Broadest concept: Any computer system mimicking human cognition  │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ MACHINE LEARNING (ML)                                   │   │
│   │ Statistical methods discovering patterns from data      │   │
│   │                                                         │   │
│   │   ┌─────────────────────────────────────────────────┐   │   │
│   │   │ DEEP LEARNING (DL)                              │   │   │
│   │   │ Multi-layered neural networks for raw sensory   │   │   │
│   │   │ data (Images, Speech, Text, Video)              │   │   │
│   │   └─────────────────────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 1.1 Artificial Intelligence (The Broad Dream)
* **Definition:** The science of engineering machines that perform tasks that typically require human intelligence (problem solving, visual perception, decision making).
* **Key Characteristic:** **AI does NOT require learning!**
* **Symbolic / Rule-Based AI:** In the 1970s–1990s, "Expert Systems" ruled AI. Humans manually programmed thousands of deterministic `IF / THEN` rules.
  * *Example:* IBM's Deep Blue (1997) beat Garry Kasparov in chess not by learning, but by using high-speed tree search algorithms evaluating 200 million positions per second against hard-coded valuation rules.

### 1.2 Machine Learning (Learning from Experience)
* **Definition:** Coined by Arthur Samuel in 1959: *"Field of study that gives computers the ability to learn without being explicitly programmed."*
* **The Paradigm Inversion:**
  * **Classical Programming:** `Input Data + Human Written Rules = Output Answers`
  * **Machine Learning:** `Input Data + Output Answers = Computer Learns The Rules`
* **Requirement:** Tabular data with **manual feature engineering**. A human must create and clean columns (e.g. `Height`, `Weight`, `Income`, `Credit_Score`).

### 1.3 Deep Learning (Automatic Representation Learning)
* **Definition:** A subset of ML based on artificial neural networks with multiple "deep" hidden layers (often dozens or hundreds of layers).
* **The Breakthrough: Automatic Feature Extraction:**
  * In classical ML, if you want to classify dog vs. cat photos, a human data scientist must extract features: number of whiskers, ear aspect ratio, snout length.
  * In Deep Learning, you provide **raw pixel arrays**. Early layers detect edges; middle layers combine edges into contours and textures; deep layers combine contours into full eyes, snouts, and ears.
* **Cost:** Requires enormous datasets (millions of samples) and specialized matrix hardware (GPUs / TPUs).

---

# 2. Learning Paradigms: Supervised vs. Unsupervised

Every machine learning task falls into one of two fundamental categories depending on the presence of ground truth answers:

```
                          MACHINE LEARNING
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
SUPERVISED LEARNING                             UNSUPERVISED LEARNING
(Target labels 'y' exist)                       (No target labels 'y')
         │                                               │
   ┌─────┴─────┐                                   ┌─────┴─────┐
   ▼           ▼                                   ▼           ▼
Regression  Classification                     Clustering   Dimensionality
(Numbers)   (Categories)                       (Grouping)     Reduction
```

### 2.1 Supervised Learning (Learning with a Teacher)
* **Mathematical Form:** Given a dataset `D = {(X_1, y_1), (X_2, y_2), ... (X_n, y_n)}`, find a function `f(X) ≈ y` that minimizes prediction error across unseen future inputs.
* **Components:**
  * `X` (Features / Independent Variables): The input signals (e.g. square footage, bedrooms).
  * `y` (Target / Dependent Variable): The answer key we want to predict (e.g. sale price).
* **Applications:** Spam filtering, credit scoring, medical disease detection, price forecasting.

### 2.2 Unsupervised Learning (Self-Discovery)
* **Mathematical Form:** Given a dataset `D = {X_1, X_2, ... X_n}` with **no labels `y`**, model the underlying probability distribution `P(X)` or discover hidden structural partitions.
* **Goal:** Grouping, association, or compressing data.
* **Applications:**
  * **Clustering:** Grouping consumers into behavioral personas (VIPs, Bargain Seekers).
  * **Anomaly Detection:** Flagging credit card transactions that fall outside all normal spending clusters.
  * **Dimensionality Reduction (e.g. PCA):** Compressing 500 columns down to 10 principal components while retaining 95% of information.

---

# 3. Task Types: Regression vs. Classification

Within Supervised Learning, tasks are divided strictly by the mathematical nature of the target variable `y`:

### 3.1 Regression (Continuous Quantitative Outputs)
* **Nature of Output:** Any real number on a continuous scale: `y in Real Numbers (-infinity to +infinity)`.
* **Questions Answered:** *"How much?"*, *"How many?"*, *"At what rate?"*
* **Interpolation:** If the model predicts 24.5°C today and 26.5°C tomorrow, predicting 25.5°C is naturally meaningful.
* **Examples:** House prices ($425,500), stock index returns, flight delay duration (42.5 minutes), crop yield (metric tons).

### 3.2 Classification (Discrete Qualitative Outputs)
* **Nature of Output:** A member of a finite set of unordered categories: `y in {Class_0, Class_1, ... Class_K}`.
* **Questions Answered:** *"Which bucket?"*, *"Is it A or B?"*
* **No Middle Ground:** An email cannot be "35% Spam and 65% Ham" as a final delivery state; it must be classified into one box.
* **Sub-types:**
  * **Binary Classification:** Exactly 2 possible outcomes (0 or 1). E.g., `Defaulted` vs `Repaid`, `Benign` vs `Malignant`.
  * **Multi-Class Classification:** More than 2 mutually exclusive classes. E.g., Handwritten digit recognition (0 through 9), Pokémon Primary Type (18 possible types).
  * **Multi-Label Classification:** A sample can belong to multiple classes simultaneously (e.g., A movie tagged as both `[Action, Sci-Fi]`).

---

# 4. Deep Dive into the 5 Core Algorithms

---

## 4.1 Linear Regression

### The Core Concept
Linear Regression fits the single hyper-plane (a line in 2D, a flat plane in 3D) that minimizes the vertical distances between actual data points and the line.

```
Salary
  ^
  |               * (Actual Point)
  |             / :  <-- Error / Residual (e)
  |           / * :
  |         /     :
  |       / *
  |     /
  |   / *
  | /
  +-------------------------> Experience
```

### The Mathematical Equation
For a single input feature:
`y_hat = (w * x) + b`
* `w` (Weight / Slope / Coefficient): For every 1-unit increase in `x`, `y` increases by `w`.
* `b` (Bias / Intercept): The baseline value of `y` when `x = 0`.

For multiple features (Multiple Linear Regression):
`y_hat = b + (w_1 * x_1) + (w_2 * x_2) + ... + (w_d * x_d)`

### How It Learns: Ordinary Least Squares (OLS)
The model calculates the sum of all squared vertical residuals:
`Residual Sum of Squares (RSS) = Sum of (y_actual - y_predicted)^2`
Using linear algebra (the Normal Equation) or Gradient Descent, it calculates the exact weights that make `RSS` as small as theoretically possible.

### Key Assumptions of Linear Regression
1. **Linearity:** The true relationship between features and target is linear.
2. **Homoscedasticity:** The variance of residuals remains constant across all values of `x`.
3. **Independence:** Observations are independent (no autocorrelation, as in time series).
4. **Normality:** Residuals follow a normal distribution centered at zero.

---

## 4.2 Logistic Regression

### The Core Concept
Despite having "Regression" in its name, Logistic Regression is a **Classification algorithm**. 
Instead of predicting an unbounded straight line (which could predict absurd values like a `-200%` or `1,500%` probability of disease), it bends the straight line through the **Sigmoid Link Function**.

```
Probability (y)
 1.0 +--------------------******** (Pass)
     |                 ***
     |               **
 0.5 + - - - - - - ** - - - - - - - (Decision Threshold = 0.5)
     |           **
     |        ***
 0.0 ********--------------------+ (Fail)
     0          Hours Studied   10
```

### The Sigmoid Function
The Sigmoid function maps any real number from negative infinity to positive infinity into a strict probability interval `[0.0, 1.0]`:
`P(y = 1 | x) = 1 / (1 + e^(-z))`
where `z = (w * x) + b`.

### Decision Threshold
By default, the classification rule is:
* If `P(y = 1 | x) >= 0.50`, predict **Class 1** (e.g. Pass).
* If `P(y = 1 | x) < 0.50`, predict **Class 0** (e.g. Fail).
* *Note:* In medical diagnosis or fraud, engineers frequently adjust the threshold down to `0.20` or `0.30` to maximize Recall!

### Why NOT use MSE for Logistic Regression?
In Linear Regression, the error curve is bowl-shaped (convex), so gradient descent slides straight to the bottom. If you use Mean Squared Error with Sigmoid, the math creates a wavy surface with multiple false bottoms (local minima). Therefore, Logistic Regression uses **Log Loss (Binary Cross-Entropy)**:
`Loss = - [ y * log(p) + (1 - y) * log(1 - p) ]`

---

## 4.3 Decision Trees

### The Core Concept
A Decision Tree splits the dataset by asking a sequential hierarchy of threshold questions. It partitions feature space into rectangular decision regions.

```
                    [ Income <= $50,000? ]
                         /          \
                      Yes            No
                      /                \
           [ Credit < 600? ]      [ Debt Ratio > 0.5? ]
               /       \               /          \
             Yes        No           Yes           No
             /            \          /               \
         DEFAULT         PAY      DEFAULT            PAY
```

### Anatomy of a Tree
* **Root Node:** The initial top split containing the entire dataset.
* **Internal / Decision Nodes:** Intermediate questions testing a specific feature.
* **Leaf Nodes:** Terminal nodes providing the final prediction (class label or average numeric value).

### How Does the Tree Decide Where to Split?
At every node, the tree evaluates every possible threshold for every feature and chooses the split that maximizes **Purity** (makes the subsets as homogeneous as possible).

1. **Gini Impurity (Default in Scikit-Learn):**
   Measures the probability of misclassifying a randomly chosen element:
   `Gini = 1 - Sum of (p_i)^2`
   * A pure node (100% yes, 0% no) has `Gini = 0.0`.
   * A completely mixed node (50% yes, 50% no) has `Gini = 0.5`.
2. **Entropy & Information Gain:**
   Measures disorder in information theory:
   `Entropy = - Sum of p_i * log2(p_i)`
   The algorithm selects the split with the highest **Information Gain** (reduction in entropy).

### Pros and Cons:
* **Pros:** Highly interpretable, handles non-linear relationships, requires zero feature scaling.
* **Cons (Major):** Extremely prone to **Overfitting**. A tree with unrestricted depth will split until every single training sample has its own private leaf!

---

## 4.4 Random Forest

### The Core Concept: Ensemble Learning
A Random Forest solves the fatal flaw of single Decision Trees (overfitting) by combining the predictions of **hundreds of distinct trees**. This is based on **Ensemble Learning** and the **Wisdom of the Crowd**.

```
                      [ Raw Input Data ]
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
   [ Bootstrap 1 ]      [ Bootstrap 2 ]      [ Bootstrap 3 ]
         │                    │                    │
         ▼                    ▼                    ▼
   ( Tree #1 )          ( Tree #2 )          ( Tree #3 )
         │                    │                    │
         ▼                    ▼                    ▼
    Votes: Class 1       Votes: Class 0       Votes: Class 1
         └────────────────────┬────────────────────┘
                              ▼
                     MAJORITY VOTE: CLASS 1
```

### The Two Pillars of Randomness (Bagging)
If you train 100 identical trees on the same data, they will make identical mistakes. Random Forests ensure diversity via two tricks:
1. **Bootstrap Aggregation (Bagging):** Each tree is trained on a random sample of rows chosen *with replacement*. ~63% of rows are sampled; the remaining ~37% are left out (**Out-Of-Bag / OOB data** used for internal validation).
2. **Feature Sub-sampling:** At every single split inside a tree, the algorithm only considers a random subset of features (typically `sqrt(Total_Features)`). This prevents one ultra-dominant feature from dictating every tree!

### Aggregation Mechanism:
* **Classification:** Majority Voting (e.g. 78 trees vote Yes, 22 vote No &rarr; Output Yes).
* **Regression:** Average of predictions across all 100 trees.

---

## 4.5 K-Means Clustering

### The Core Concept
K-Means is an **Unsupervised algorithm** that partitions unlabelled data points into `K` distinct, non-overlapping clusters based on geometric distance.

### The 4-Step Iterative Algorithm:
1. **Initialize:** Randomly place `K` starting centroid pins in feature space.
2. **Assign:** Measure the Euclidean distance from every data point to all `K` centroids. Assign each point to its closest centroid.
3. **Update:** Calculate the mean coordinate of all points assigned to each cluster. Move the centroid pin to that new mean position.
4. **Repeat:** Repeat steps 2 and 3 until the centroids stop moving (**Convergence**).

```
Step 1: Random Centroids   Step 2: Assign Nearest    Step 3: Update Centroids
       * (C1)                   x1  x2 (C1)                (C1 moves to center)
  x1       x2                       
                         --------------------->     
       * (C2)                   x3  x4 (C2)                (C2 moves to center)
  x3       x4
```

### How to Choose the Value of K: The Elbow Method
Because K is unlabelled, how do you know whether to pick K=2, 3, or 5?
* We calculate the **Within-Cluster Sum of Squares (WCSS)** (the total distance of points to their cluster center).
* As K increases, WCSS naturally drops.
* We plot WCSS vs. K. The point where the drop dramatically slows down and forms a sharp bend is the **"Elbow"**—the optimal number of clusters!

---

# 5. Training, Testing, Overfitting & Evaluation Metrics

---

## 5.1 The Bias-Variance Tradeoff

The ultimate challenge in machine learning is balancing **Bias** (Underfitting) and **Variance** (Overfitting):

```
Error
  ^
  |  \                 Total Error                 /
  |   \                 (U-Shaped)                /
  |    \                 /-----\                 /
  |     \               /       \               /
  |      \             /         \             /
  |       \           /           \           /
  |  High  \         /  Optimal    \         /  High
  |  Bias   \_______/     Spot      \_______/   Variance
  | (Underfitting)                             (Overfitting)
  +----------------------------------------------------------> Model Complexity
```

1. **Bias (Error from erroneous assumptions):**
   * The model is overly simplistic (e.g. forcing a straight line through a curved dataset).
   * **Result:** Underfitting. Performs poorly on both Training and Testing data.
2. **Variance (Error from sensitivity to small fluctuations):**
   * The model is overly complex (e.g. a 50-level Decision Tree memorizing random noise).
   * **Result:** Overfitting. Scores 99% on Training, drops to 55% on Testing.
3. **The Sweet Spot:** Minimizes total error by capturing true underlying patterns without memorizing noise.

---

## 5.2 K-Fold Cross-Validation

A single train-test split (e.g. 75/25) can be lucky or unlucky depending on which rows ended up in the test set. 
**K-Fold Cross-Validation** provides a much more robust evaluation:

```
Full Dataset (Split into K = 5 Folds)
Fold 1: [ TEST ] [ Train ] [ Train ] [ Train ] [ Train ] -> Score 1
Fold 2: [ Train ] [ TEST ] [ Train ] [ Train ] [ Train ] -> Score 2
Fold 3: [ Train ] [ Train ] [ TEST ] [ Train ] [ Train ] -> Score 3
Fold 4: [ Train ] [ Train ] [ Train ] [ TEST ] [ Train ] -> Score 4
Fold 5: [ Train ] [ Train ] [ Train ] [ Train ] [ TEST ] -> Score 5
Final Performance = Average of (Score 1 + Score 2 + Score 3 + Score 4 + Score 5)
```
* **Stratified K-Fold:** Crucial for imbalanced classification. Guarantees that every fold contains the exact same percentage of positive/negative classes as the original dataset.

---

## 5.3 Regression Metrics

When evaluating continuous numbers, you cannot use "Accuracy". You use distance errors:

| Metric | Formula / Concept | What it Tells You | Sensitivity |
| :--- | :--- | :--- | :--- |
| **MAE** (Mean Absolute Error) | `Average of |y_actual - y_pred|` | The average mistake in real units ($). Super intuitive for business reports. | Robust to extreme outliers. |
| **MSE** (Mean Squared Error) | `Average of (y_actual - y_pred)^2` | Squares errors before averaging. | Extremely sensitive to outliers. |
| **RMSE** (Root Mean Squared Error) | `Square Root of MSE` | Brings squared units back to real units ($) while maintaining heavy penalties for catastrophic errors. | Preferred for risk-sensitive modeling. |
| **R² Score** (Coefficient of Determination) | `1 - (RSS / Total_Sum_of_Squares)` | Measures the proportion of variance explained by the model (0.0 = baseline average guess, 1.0 = perfect model). | Relative performance benchmark. |

---

## 5.4 Classification Metrics

### The Confusion Matrix:
For binary classification, all predictions fall into a 2x2 grid:

```
                    ACTUAL POSITIVE (1)       ACTUAL NEGATIVE (0)
PREDICTED POSITIVE  True Positive (TP)        False Positive (FP) [Type I Error]
PREDICTED NEGATIVE  False Negative (FN)       True Negative (TN)
                    [Type II Error]
```

### The 4 Essential Classification Scores:

1. **Accuracy:**
   `Accuracy = (TP + TN) / (TP + TN + FP + FN)`
   * *The Critical Flaw:* In credit fraud where 99.8% of transactions are legitimate, a model predicting "Never Fraud" achieves 99.8% accuracy while being completely useless!

2. **Precision (Exactness / Reliability):**
   `Precision = TP / (TP + FP)`
   * *"When the model predicts Positive, how often is it right?"*
   * **Use Precision when False Positives (False Alarms) are costly:** E.g., Spam filter (you don't want a vital work email marked as spam and deleted).

3. **Recall / Sensitivity (Completeness):**
   `Recall = TP / (TP + FN)`
   * *"Of all the actual positives out there in the world, what percentage did the model catch?"*
   * **Use Recall when False Negatives (Missed Cases) are fatal:** E.g., Cancer detection, pandemic screening, financial fraud.

4. **F1-Score (Harmonic Mean):**
   `F1 = 2 * (Precision * Recall) / (Precision + Recall)`
   * Combines Precision and Recall into a single balanced metric. It punishes extreme imbalances (if Precision is 99% but Recall is 2%, F1 drops dramatically).

5. **ROC-AUC (Receiver Operating Characteristic - Area Under Curve):**
   * Plots True Positive Rate vs. False Positive Rate across all possible probability thresholds (from 0.0 to 1.0).
   * **AUC = 1.0:** Perfect separation.
   * **AUC = 0.5:** Random coin flip guessing.

---

# 6. Neural Networks, Activation Functions & Optimization

---

## 6.1 The Biological vs. Artificial Neuron

An Artificial Neuron (Perceptron) mathematically mimics biological dendrites and synapses:

```
Inputs (x)      Weights (w)
  x_1  ───────>   w_1  ───┐
                          │
  x_2  ───────>   w_2  ───┼──>  [ SUM + Bias ]  ──> [ Activation f(z) ] ──> Output (a)
                          │         z = Σ w*x + b          a = f(z)
  x_3  ───────>   w_3  ───┘
```

1. **Inputs (`x`):** The incoming feature values.
2. **Weights (`w`):** Learnable connection strengths (synapses). Positive weights excite; negative weights inhibit.
3. **Bias (`b`):** An offset shifting the activation threshold up or down regardless of inputs.
4. **Weighted Sum (`z`):** `z = (x_1 * w_1) + (x_2 * w_2) + ... + (x_d * w_d) + b`
5. **Activation Function (`f`):** Transforms `z` into final activation `a`.

---

## 6.2 Activation Functions Deep Dive

### Why are Activation Functions Mandatory?
If every neuron simply computed `z = w*x + b`, then stacking 50 layers would collapse into one single linear equation:
`Layer_2(Layer_1(x)) = W_2 * (W_1 * x) = (W_2 * W_1) * x = W_combined * x`
A network without non-linear activation functions is nothing more than Linear Regression! It can never learn curved boundaries, circles, or complex computer vision representations.

### The 4 Major Activation Functions:

```
1. ReLU:                   2. Sigmoid:                3. Tanh:
        |    /                    |                          |       /
        |   /                 1.0 +     .---.                |   .--'
        |  /                      |    /                     |  /
  ------+--/---             0.5 + - / - - - -          0.0 --+-/-----
        | 0                       |  /                       |/
                               0.0 +-'                    -1.0|
```

1. **ReLU (Rectified Linear Unit):**
   * **Equation:** `f(z) = max(0.0, z)`
   * **Behavior:** Negative inputs output 0; positive inputs pass through with an unchanged gradient of 1.
   * **Why it's King:** Solves the **Vanishing Gradient Problem**, computationally ultra-fast, and induces *sparse activations* (only a subset of neurons fire simultaneously, mimicking biological brains).
   * **Caveat:** Can suffer from "Dying ReLU" if weights update such that a neuron permanently outputs 0 for all samples.

2. **Sigmoid:**
   * **Equation:** `f(z) = 1 / (1 + e^(-z))`
   * **Behavior:** Smooth S-curve mapping `(-infinity, +infinity)` into `[0.0, 1.0]`.
   * **Usage:** Primarily used in the **final output layer** of binary classification networks.
   * **Flaw in Hidden Layers:** Gradients become nearly flat at extreme values (saturation), causing deep networks to stop learning.

3. **Tanh (Hyperbolic Tangent):**
   * **Equation:** `f(z) = (e^z - e^(-z)) / (e^z + e^(-z))`
   * **Behavior:** S-curve scaled between `[-1.0, +1.0]`.
   * **Advantage over Sigmoid:** Zero-centered (average output is near zero), which speeds up training in recurrent models.

4. **Softmax:**
   * **Equation:** `Softmax(z_i) = e^(z_i) / Sum of (e^(z_j))`
   * **Usage:** The standard **output layer for multi-class classification** (e.g. recognizing digits 0 to 9). Converts unnormalized raw scores into normalized probabilities summing strictly to 1.0 (100%).

---

## 6.3 Loss Functions (Cost Functions)

The Loss Function quantifies the penalty for imperfect predictions.

1. **Mean Squared Error (MSE):** Used for **Regression**.
   `Loss = (1 / N) * Sum of (y_actual - y_pred)^2`
2. **Binary Cross-Entropy (Log Loss):** Used for **Binary Classification** (with Sigmoid output).
   `Loss = - (1 / N) * Sum of [ y * log(p) + (1 - y) * log(1 - p) ]`
3. **Categorical Cross-Entropy:** Used for **Multi-Class Classification** (with Softmax output).
   `Loss = - Sum of y_i * log(p_i)`

---

## 6.4 Gradient Descent & Backpropagation

Training a neural network is an iterative optimization problem: finding the combination of millions of weights `W` that minimizes the loss surface `L(W)`.

### Gradient Descent:
1. Initialize weights randomly.
2. Feed data forward through layers to compute predictions and current Loss.
3. Compute the **Gradient** (vector of partial derivatives `dL / dW` pointing in the direction of steepest loss increase).
4. Update weights in the **opposite (negative) direction** of the gradient:
   `New_Weight = Old_Weight - (Learning_Rate * Gradient)`
5. Repeat for hundreds of epochs until loss converges at a global minimum.

### Variants of Gradient Descent:
* **Batch Gradient Descent:** Calculates loss on the **entire dataset** before making 1 weight update. (Extremely stable, but very slow on massive data).
* **Stochastic Gradient Descent (SGD):** Updates weights after **every single sample**. (Very fast, but updates fluctuate erratically).
* **Mini-Batch Gradient Descent (Standard Practice):** Updates weights on small batches (e.g. 32, 64, or 128 samples). Combines stability with GPU vectorization.
* **Adam Optimizer:** Adaptive Moment Estimation. The gold-standard modern optimizer that dynamically tunes individual learning rates for every single weight based on historical momentum.

---

## 6.5 Learning Rate Dynamics

The **Learning Rate (`eta`)** controls step size:

```
Too Large Learning Rate              Too Small Learning Rate             Optimal Learning Rate
      \    ^                                                                   
       \  / \                               \                                   \
        \/   \                               \                                   \
     (Overshoots Valley,                      \ (Takes 10,000 steps,              \__ (Smooth Descent
      Loss Explodes!)                            Glacially Slow)                       to Minimum)
```

* **Too Large:** Steps overshoot the valley bottom, bounce erratically between walls, and diverge (`Loss -> NaN`).
* **Too Small:** Network gets trapped in shallow local plateaus or takes weeks of compute time.
* **Modern Solution:** Learning Rate Schedulers (warm-up followed by cosine decay).

---

# 7. Day 2 Summary Comparison & Algorithm Decision Guide

| Problem Scenario | Best Algorithm | Why? |
| :--- | :--- | :--- |
| Predicting apartment rent based on size and location | **Linear Regression** | Simple continuous number, linear relationship, easily explainable coefficients. |
| Predicting if a bank customer will default on a credit card | **Logistic Regression** or **Random Forest** | Binary classification; outputs clear calibrated probability. |
| Explaining loan approval logic to non-technical regulators | **Decision Tree** | Produces simple `IF/THEN` rule trees that can be audited on paper. |
| High-stakes tabular tabular prediction (e.g. Kaggle competition) | **Random Forest** (or XGBoost) | State-of-the-art accuracy on tabular data; immune to overfitting. |
| Segmenting 100,000 shoppers into marketing groups without labels | **K-Means Clustering** | Unsupervised distance grouping into K distinct personas. |
| Facial recognition, voice synthesis, or large language models | **Deep Neural Network** | Massive multi-layer representation learning; automatic feature extraction. |
