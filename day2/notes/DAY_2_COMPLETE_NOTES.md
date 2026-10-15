# Day 2: Machine Learning and Neural Networks

---

## 1. AI vs. Machine Learning vs. Deep Learning

The relationship between these three terms is like nested Russian dolls:

* **Artificial Intelligence (AI) — The Outer Circle:**
  * Any computer technique that mimics human decisions or intelligence.
  * **Crucial Rule:** AI does **not** have to learn! A 1990s chess bot or a calculator with 50 hard-coded `if/else` rules is still AI.
* **Machine Learning (ML) — The Middle Circle:**
  * A branch of AI where the machine **learns the rules from past data** instead of humans typing the rules.
  * *Traditional Programming:* Data + Human Rules -> Answers.
  * *Machine Learning:* Data + Past Answers -> **Machine Learns The Rules!**
* **Deep Learning (DL) — The Inner Circle:**
  * A specialized subset of Machine Learning powered by **Multi-Layer Artificial Neural Networks**.
  * Its superpower is **Automatic Feature Extraction**: you feed it raw images, audio, or text, and it discovers shapes, sounds, and grammar on its own without human feature engineering.

---

## 2. Supervised vs. Unsupervised Learning

The core difference is: **Does the computer have an Answer Key ($y$)?**

| Feature | Supervised Learning | Unsupervised Learning |
| :--- | :--- | :--- |
| **Answer Key ($y$)** | **YES** (Data has labels/answers) | **NO** (Only raw input clues) |
| **Main Goal** | **Predict** the answer for new data | **Discover** natural groups / patterns |
| **Analogy** | Studying with an answer key | Sorting a messy toy box by similarity |
| **Real Examples** | House prices, Spam filter, Cancer detection | Customer segmentation, Song clustering |

---

## 3. Regression vs. Classification

Both belong to **Supervised Learning** (both have an Answer Key). The difference is the **type of answer**:

* **Regression — "How much?" (Continuous Number):**
  * Predicting a number on a continuous scale (e.g., House Price: $345,000, Temperature: 28.5 C, Car Price: $18,200).
* **Classification — "Which bucket?" (Discrete Category):**
  * Predicting a group or category.
  * *Binary (2 choices):* Pass/Fail, Spam/Not Spam, Quit/Stayed.
  * *Multi-class (3+ choices):* Cat/Dog/Bird, Blood Type (A/B/AB/O).

---

## 4. The 5 Core Algorithms

### 1. Linear Regression
* **Task:** Supervised Regression (predicts numbers).
* **Concept:** Finds the single **best-fit straight line** through data points.
* **Formula:** `Prediction = (Slope * Input) + Starting_Intercept`

### 2. Logistic Regression
* **Task:** Supervised Classification (Yes/No categories).
* **Concept:** Bends the line into an **S-shaped curve** that calculates probabilities between **0% and 100%**.
* **Decision Rule:** If probability > 50% -> Class 1; otherwise Class 0.

### 3. Decision Tree
* **Task:** Supervised (both Classification & Regression).
* **Concept:** A flowchart of **Yes/No questions** (like playing 20 Questions).
* **Weakness:** A deep tree easily **overfits** by memorizing every detail of the training data.

### 4. Random Forest
* **Task:** Supervised (both Classification & Regression).
* **Concept:** The "Wisdom of the Crowd." Creates **100+ different Decision Trees** and lets them vote.
* **Why it's King:** Averaging 100 trees cancels out random errors and prevents overfitting on spreadsheet data.

### 5. K-Means Clustering
* **Task:** Unsupervised Learning (Clustering).
* **Concept:** Drops $K$ center pins (Centroids) into unlabelled data and groups every point to its nearest pin until natural clusters form.
* **Use Case:** Customer segmentation, recommendation engines.

---

## 5. Training, Testing, Overfitting & Evaluation Metrics

### Overfitting vs. Underfitting (The 3 Students)
* **Underfitting (Under-prepared):** The model is too simple. Performs terribly on both homework and final exam.
* **Overfitting (Over-memorized):** The model memorized the practice homework (99%), but fails the real test exam (55%).
* **Just Right:** Learns the underlying concepts. Performs well on both.

### How to Grade Models

#### Grading Regression (Numbers):
1. **MAE (Mean Absolute Error):** The average size of your mistake in real units (e.g. "off by $3,500 on average").
2. **RMSE (Root Mean Squared Error):** Heavily penalizes large, catastrophic mistakes.
3. **R-Squared ($R^2$):** Percentage of variation explained by the model (0.0 to 1.0, where 1.0 is perfection).

#### Grading Classification (Yes/No):
1. **Accuracy:** Total correct guesses / Total samples. (Can be misleading if data is heavily imbalanced).
2. **Precision:** When the model shouts "Positive!", how often is it actually right? (Low false alarms).
3. **Recall:** Of all the actual positive cases in real life, how many did the model catch? (Crucial for medical diagnosis to avoid missed cases).
4. **F1-Score:** The harmonic balance between Precision and Recall.

---

## 6. Neural Networks & Activation Functions

### The Artificial Neuron
Computes a weighted sum of its inputs:
`Sum = (Input_1 * Weight_1) + (Input_2 * Weight_2) + Bias`

### Why Activation Functions? (The "Origami / Bend" Principle)
Without activation functions, stacking 100 layers is just one long flat line. Real-world patterns (faces, audio) are curved and complex. Activation functions give the network **hinges to bend into any shape**.

* **ReLU (Rectified Linear Unit):**
  * `Rule: If negative -> 0. If positive -> Keep as-is.`
  * The workhorse of modern AI. Simple, lightning-fast, and acts like a biological threshold switch.
* **Sigmoid:**
  * Squashes numbers between 0 and 1. Perfect for final Yes/No probability outputs.
* **Softmax:**
  * Squashes multi-class outputs so they add up to 100% probability.

### Loss Function & Gradient Descent
* **Loss Function:** The scoreboard of mistakes. Lower loss = smarter model.
* **Gradient Descent:** The "Foggy Mountain Hiker." Steps downhill in the direction of steepest slope to reach minimum loss.
* **Learning Rate (Step Size):**
  * Too large: Giant leap, overshoots the valley, loss jumps erratically.
  * Too small: Baby steps, takes years to converge.
  * Just right: Smoothly descends into the valley.
