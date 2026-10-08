# WIX3002 Group 3 S2 25/26 - Sentiment Analysis

## A Sentiment Prediction System and Analytics Dashboard for Customer Reviews

This project develops an end-to-end **customer review sentiment analysis system** using the Flipkart Ratings and Reviews dataset. The system combines Natural Language Processing (NLP), machine learning, exploratory data analysis, and Streamlit deployment to classify customer reviews into **positive, negative, and neutral** sentiments.

The final system consists of a trained **LightGBM sentiment classifier**, a **TF-IDF text vectorizer**, and an interactive **Streamlit analytics dashboard** for real-time sentiment prediction and dataset analysis.

---

## 👥 Group Members

**Group 3**

1. **IMRAN HAZIQ BIN KHAIRUL ANUAR** - 23001784
2. **IRDINA NAFISHA BINTI MOHD SHAHAR** - 23002573
3. **CHIAM HUAI REN** - 23110201
4. **THAM WING SHAN** - 24059824
5. **YAP YU HANG** - 23102016
6. **TAN WEI REN** - 23112262

---

## 📌 Project Overview

The project aims to build a sentiment prediction system capable of automatically identifying the sentiment expressed in Flipkart customer reviews.

The project follows a complete data science workflow:

```text
Dataset Acquisition
        ↓
Data Exploration
        ↓
Data Cleaning & Preprocessing
        ↓
Text Deduplication
        ↓
Exploratory Data Analysis
        ↓
TF-IDF Feature Engineering
        ↓
Logistic Regression Baseline
        ↓
LightGBM Model
        ↓
Model Evaluation
        ↓
Synthetic Data Testing
        ↓
Streamlit Deployment
```

The system provides both:

* **Sentiment prediction** for new customer reviews
* **Analytics and visualizations** for understanding the review dataset

---

## 📊 Dataset

The project uses the **Flipkart Ratings and Reviews Dataset** hosted on Kaggle.

The original dataset contains **100,000 customer reviews** with four main columns:

| Column      | Description                                            |
| ----------- | ------------------------------------------------------ |
| `Rate`      | Customer rating from 1 to 5 stars                      |
| `Review`    | Short review label or summary                          |
| `Summary`   | Customer-written review text                           |
| `Sentiment` | Target sentiment: `positive`, `neutral`, or `negative` |

The dataset is retrieved programmatically using the `kagglehub` Python package to support a reproducible data acquisition workflow.

### Dataset Source

**Flipkart Ratings and Reviews Dataset - Kaggle**

---

## 🔍 Exploratory Data Analysis

Initial analysis was performed before preprocessing to understand the characteristics of the dataset.

The original dataset contained a highly imbalanced sentiment distribution:

| Sentiment | Records | Percentage |
| --------- | ------: | ---------: |
| Positive  |  81,691 |      81.7% |
| Negative  |  13,491 |      13.5% |
| Neutral   |   4,815 |       4.8% |

Three invalid values were identified in the `Rate` column. These values were converted to `NaN` and removed, resulting in **99,997 valid reviews**.

The rating distribution also showed a strong concentration toward higher ratings, with **58.5% of reviews receiving 5 stars**.

---

## 🧹 Data Preprocessing

Several preprocessing techniques were applied before model training.

### 1. Safe Stopword Removal

Standard English stopwords were removed to reduce unnecessary textual noise.

However, sentiment-related negation words were intentionally preserved because removing words such as:

```text
not
no
never
don't
isn't
wasn't
but
```

could change the meaning of a review.

For example:

```text
not good
```

should not be reduced to:

```text
good
```

Therefore, a protected set of **28 negation-related words** was excluded from the stopword removal process.

### 2. Text Combination

The `Review` and `Summary` columns were combined into a single text feature:

```text
Cleaned_Text
```

This provides the model with both the short review description and the detailed customer feedback.

### 3. Duplicate Removal

Duplicate reviews were removed to reduce data leakage between the training and testing sets.

The analysis identified:

* **42,830 duplicate records**
* **42.83%** of the original valid dataset

After deduplication, the dataset was reduced from:

```text
99,997 → 57,167 unique reviews
```

The resulting dataset was used as the main modeling dataset.

### Final Sentiment Distribution

| Sentiment |    Records | Percentage |
| --------- | ---------: | ---------: |
| Positive  |     42,962 |     75.15% |
| Negative  |     10,815 |     18.92% |
| Neutral   |      3,390 |      5.93% |
| **Total** | **57,167** |   **100%** |

---

## 📈 Exploratory Visualizations

Several visualizations were generated to understand the processed dataset.

### Sentiment Distribution

The sentiment distribution was visualized after preprocessing to show the remaining class imbalance.

### Word Clouds

Separate word clouds were generated for positive and negative reviews.

Common positive terms include:

```text
good
great
excellent
awesome
nice
love
quality
best
```

Common negative terms include:

```text
bad
poor
worst
waste
disappointed
damage
return
money
```

The neutral class was excluded from the word cloud visualization because its vocabulary overlaps considerably with positive and negative language.

---

## 🤖 Machine Learning Models

Two classification models were developed and compared.

### 1. Logistic Regression

Logistic Regression was used as the baseline model.

The dataset was divided using an **80/20 stratified train-test split**:

* Training set: **45,733 reviews**
* Testing set: **11,434 reviews**

Stratification preserved the original sentiment proportions across the training and testing sets.

#### TF-IDF Configuration

The baseline model used:

```text
TF-IDF Vectorizer
N-gram range: (1, 2)
Maximum features: 10,000
```

Both unigrams and bigrams were used to capture individual words and short contextual phrases.

The Logistic Regression classifier used:

```text
class_weight = balanced
max_iter = 1000
random_state = 42
```

### Logistic Regression Results

| Class        | Precision | Recall | F1-Score |
| ------------ | --------: | -----: | -------: |
| Negative     |      0.83 |   0.86 |     0.84 |
| Neutral      |      0.27 |   0.53 |     0.35 |
| Positive     |      0.97 |   0.89 |     0.93 |
| **Accuracy** |           |        | **0.86** |

The baseline achieved an overall accuracy of **86%**.

---

# 🚀 Final Model: LightGBM

LightGBM was selected as the improved model because it can capture more complex patterns from the high-dimensional text features.

### TF-IDF Configuration

The final text representation was optimized using:

```text
Maximum features: 15,000
N-gram range: (1, 2)
sublinear_tf = True
min_df = 2
```

### LightGBM Configuration

The classifier used:

```text
objective = multiclass
num_class = 3
is_unbalance = True
n_estimators = 500
learning_rate = 0.05
num_leaves = 31
```

The three target classes are:

```text
positive
negative
neutral
```

---

## 📊 LightGBM Performance

The final LightGBM model achieved **91% overall accuracy**, improving the baseline by **5 percentage points**.

| Class        | Precision | Recall | F1-Score |
| ------------ | --------: | -----: | -------: |
| Negative     |      0.84 |   0.88 |     0.86 |
| Neutral      |      0.52 |   0.22 |     0.31 |
| Positive     |      0.94 |   0.97 |     0.96 |
| **Accuracy** |           |        | **0.91** |

### Model Comparison

| Model               | Accuracy |
| ------------------- | -------: |
| Logistic Regression |  **86%** |
| LightGBM            |  **91%** |

LightGBM was therefore selected as the final deployment model.

### Important Observation

The neutral class remains the most challenging category.

Compared with Logistic Regression:

* Neutral precision increased from **0.27 → 0.52**
* Neutral recall decreased from **0.53 → 0.22**

This indicates that the LightGBM model is more conservative when assigning the neutral label. It produces fewer false neutral predictions but also misses more actual neutral reviews.

---

## 🧪 Synthetic Data Testing

A synthetic validation dataset was created to stress-test the trained sentiment classification pipeline.

The synthetic dataset contained **57,170 records** designed to reflect the sentiment distribution of the processed dataset:

| Sentiment | Samples | Percentage |
| --------- | ------: | ---------: |
| Positive  |  42,964 |     75.15% |
| Negative  |  10,816 |     18.92% |
| Neutral   |   3,390 |      5.93% |

The trained LightGBM model achieved approximately **97% accuracy** on this synthetic validation set.

The test demonstrated particularly strong performance on clearly polarized positive and negative examples, while neutral examples remained comparatively more difficult.

---

# 🖥️ Streamlit Application

The trained model was integrated into an interactive Streamlit application called:

**Flipkart Review Sentiment Hub**

The application combines real-time prediction with analytics dashboards.

### Main Features

#### 1. Real-Time Sentiment Prediction

Users can enter or paste a customer review into the application.

The system processes the text through:

```text
User Review
    ↓
TF-IDF Vectorization
    ↓
LightGBM Classifier
    ↓
Sentiment Prediction
    ↓
Confidence Score
```

The prediction is returned as:

```text
Positive
Neutral
Negative
```

along with the model's confidence percentage.

#### 2. Try An Example

The dashboard provides example buttons for:

* Positive
* Neutral
* Negative

These buttons automatically populate the review input field with example customer reviews.

#### 3. Sentiment Analytics

The dashboard displays:

* Sentiment distribution
* Sentiment percentages
* Number of reviews per sentiment
* Positive and negative word clouds

#### 4. Review Length Analysis

The application analyzes the length of customer reviews using:

* Average word count by sentiment
* Review word-count distribution

#### 5. N-Gram Analysis

The dashboard extracts common:

* Bigrams
* Trigrams

from the review dataset.

This provides additional insight into common phrases and contextual language used by customers.

#### 6. Model Performance Dashboard

The application displays the trained LightGBM model's:

* Precision
* Recall
* F1-score
* Classification performance
---

## ⚙️ Technologies Used

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Natural Language Processing

* NLTK
* TF-IDF
* CountVectorizer

### Machine Learning

* Scikit-learn
* Logistic Regression
* LightGBM

### Visualization

* Matplotlib
* Seaborn
* Altair
* WordCloud

### Deployment

* Streamlit
* Joblib

### Dataset Acquisition

* Kaggle
* KaggleHub

---

## ▶️ Running the Notebook

The notebook is designed to be executed in a Google Colab environment.

The dataset is downloaded programmatically using `kagglehub`, after which the notebook performs:

1. Dataset loading
2. Initial EDA
3. Data cleaning
4. Text preprocessing
5. Duplicate removal
6. Post-cleaning EDA
7. Word cloud generation
8. Logistic Regression training
9. LightGBM training
10. Model evaluation
11. Synthetic data testing
12. Model serialization
13. Streamlit application generation

---

## 🔄 Reproducibility

The project uses fixed random states where applicable, including:

```text
random_state = 42
```

The notebook also retrieves the dataset programmatically rather than relying on a manually uploaded CSV file.

However, **minor numerical differences may occur between executions** due to stochastic behavior in data shuffling and multi-threaded LightGBM optimization.

These differences are not expected to significantly change the overall model behavior or conclusions.

---

## 🏆 Final Outcome

The project successfully develops a complete sentiment analysis pipeline from raw customer reviews to an interactive machine learning application.

The final system:

* Processes **100,000 original reviews**
* Removes invalid and duplicate records
* Produces **57,167 unique modeling records**
* Uses context-preserving NLP preprocessing
* Applies TF-IDF unigram and bigram feature extraction
* Compares Logistic Regression against LightGBM
* Achieves **91% test accuracy** with LightGBM
* Performs additional synthetic consistency testing
* Deploys the final model through Streamlit
* Provides both real-time prediction and analytical visualizations

The final deployed architecture is:

```text
Flipkart Reviews
       ↓
Data Cleaning
       ↓
Text Preprocessing
       ↓
Deduplication
       ↓
TF-IDF
       ↓
LightGBM
       ↓
Sentiment Prediction
       ↓
Streamlit Analytics Dashboard
```

---

## 📚 Academic Context

**Course:** WIX3002
**Project:** Group 3 Sentiment Analysis Assignment
**Semester:** S2 2025/2026

This project demonstrates the application of data science concepts including data preprocessing, exploratory data analysis, natural language processing, supervised machine learning, model evaluation and interactive deployment.
