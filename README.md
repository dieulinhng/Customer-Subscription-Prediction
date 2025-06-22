# Customer Subscription Prediction

- Content:
  - [Introduction](#introduction)  
  - [Data Dictionary](#data-dictionary)  
  - [Implementation Steps](#implementation-steps)  
    - [Import Libraries](#import-libraries)  
    - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)    
    - [Split Data](#split-data)
    - [Pipeline](#pipeline)
    - [Model](#model)
    - [Overfitting/Underfitting](#overfittingunderfitting)
    - [Population Stability Index (PSI)](#population-stability-index-psi)
    - [Cross Validation](#cross-validation)

**Disclaimer:** This is a demo project. I will continue to update it until the final product brings the best possible results.

![shopping_behavior_clf](https://github.com/user-attachments/assets/93696038-459f-4fe9-9565-a83268e284b7)

# Introduction

Based on the [Consumer Behavior and Shopping Habits](https://www.kaggle.com/datasets/zeesolver/consumer-behavior-and-shopping-habits-dataset/data) dataset from Kaggle, this project analyzes customer behavior to classify and predict whether a customer has opted in for a subscription service. The goal is to assess customer loyalty and estimate the potential for recurring revenue.

# Data Dictionary

| Variable Name               | Role     | Type        | Description                                                                                                                                             |
|-----------------------------|----------|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| Customer_ID                 | Feature  | Integer     | Unique identifier for each customer, enabling tracking and analysis of their shopping behavior over time.                                               |
| Age                         | Feature  | Integer     | The customer’s age, providing demographic information for segmentation and targeted marketing.                                                         |
| Gender                      | Feature  | Categorical | The customer’s gender, a key demographic factor influencing product preferences and purchasing patterns.                                                |
| Item_Purchased              | Feature  | Categorical | The specific item chosen by the customer during the transaction.                                                                                        |
| Category                    | Feature  | Categorical | The broad group to which the purchased item belongs (e.g., clothing, electronics, groceries).                                                           |
| Purchase_Amount_USD         | Feature  | Integer     | The transaction’s monetary value in U.S. dollars, indicating the cost of the purchased item(s).                                                         |
| Location                    | Feature  | Categorical | The geographical region where the purchase was made, offering insights into regional preferences and market trends.                                      |
| Size                        | Feature  | Categorical | The size specification (if applicable) of the purchased item, relevant for apparel, footwear, and certain consumer goods.                                |
| Color                       | Feature  | Categorical | The color variant of the purchased item, influencing customer preferences and stock availability.                                                        |
| Season                      | Feature  | Categorical | Seasonal relevance of the purchased item (spring, summer, fall, winter), impacting inventory management and marketing strategies.                        |
| Review_Rating               | Feature  | Float       | Numerical assessment provided by the customer regarding their satisfaction with the purchased item.                                                     |
| Shipping_Type               | Feature  | Categorical | Delivery method used (e.g., standard shipping, express delivery), influencing delivery time and cost.                                                     |
| Discount_Applied            | Feature  | Binary      | Indicates whether a promotional discount was used, shedding light on price sensitivity and promotion effectiveness.                                     |
| Promo_Code_Used             | Feature  | Binary      | Indicates whether a coupon or promo code was applied, aiding evaluation of marketing campaign success.                                                  |
| Previous_Purchases          | Feature  | Integer     | Number of prior purchases made by the customer, contributing to customer segmentation and retention strategies.                                         |
| Payment_Method              | Feature  | Categorical | Mode of payment used by the customer (e.g., credit card, cash), offering insights into preferred payment options.                                        |
| Frequency_of_Purchases      | Feature  | Categorical | How often the customer makes purchases (e.g., daily, weekly, monthly), a critical metric for assessing loyalty and lifetime value.                      |
| **Subscription_Status**     | **Target** | **Binary** | Indicates whether the customer has opted for a subscription service, offering insights into loyalty and potential for recurring revenue (Yes/No).       |

# Implementation Steps

## Import Libraries

The following Python libraries are used in this project:

<p align="center">
  <a href="https://numpy.org/">
    <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
  </a>
  <a href="https://pandas.pydata.org/">
    <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  </a>
  <a href="https://matplotlib.org/">
    <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib"/>
  </a>
  <a href="https://seaborn.pydata.org/">
    <img src="https://img.shields.io/badge/Seaborn-1A7F9D?style=for-the-badge&logo=seaborn&logoColor=white" alt="Seaborn"/>
  </a>
  <a href="https://scipy.org/">
    <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy"/>
  </a>
  <a href="https://scikit-learn.org/">
    <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
  </a>
  <a href="https://imbalanced-learn.org/">
    <img src="https://img.shields.io/badge/Imbalanced--Learn-0E9AA7?style=for-the-badge&logo=python&logoColor=white" alt="imbalanced-learn"/>
  </a>
</p>

## Exploratory Data Analysis (EDA)

- **Dataset:** `shopping_behavior_updated.csv` (3,900 rows × 18 columns)  
- **Missing values:** None  
- **Duplicates:** None  

### Key Findings

#### 1. Subscription Information
- **73%** of customers do **not** subscribe; **27%** subscribe  
- No female customer has subscribed; only male customers do  
- Subscription rates rise when discounts are applied; with no discount, subscription = 0%  

#### 2. Customer Characteristics
- Majority are middle-aged or older  
  - These groups spend more and make repeat purchases more often  
- Purchases tend to be either mid-range (20–40 USD) or high-end (90–100 USD)  
- Clothing is the top category  
- Free shipping is the most chosen delivery option  

#### 3. Gender Differences
- **68%** of customers are male  
- Males shop about once every three months; females about twice per week 
- Males only buy on discount; females buy at full price more often  
- Males prefer spring; females prefer autumn
- Although females have a higher average order value, total male order value is nearly double due to larger customer base and more frequent repeat purchases  

#### 4. Customer Satisfaction
- Ratings indicate high satisfaction; every customer makes at least one repeat purchase  

## Split Data

Use **StratifiedShuffleSplit** to split the dataset into:  
- **Train set:** 70%  
- **Test set:** 30%

## Pipeline

Build a Pipeline:
- Applies a ColumnTransformer to:
  - One-Hot encode categorical features
  - Standard scale numerical features
  - Optionally applies SMOTE on the training data only
- The pipeline fits only the train set.

## Model

Train and evaluate two classifiers: **Logistic Regression** and **RandomForestClassifier**, using a variety of metrics:
1. **ROC AUC** (with Gini coefficient)  
2. **Accuracy**  
3. **Precision**  
4. **Recall**  
5. **F1-score**

### Logistic Regression
![image](lr_auc.png)
![image](lr_score.png)

### RandomForestClassifier
![image](rf_auc.png)
![image](rf_score.png)

| Metric             | Logistic Regression | Random Forest Classifier |
|--------------------|---------------------|--------------------------|
| ROC AUC            | 0.890               | **0.901**                |
| Gini Coefficient   | 0.781               | **0.802**                |
| Accuracy           | 0.85                | 0.85                     |
| Precision          | 0.64                | 0.65                     |
| Recall             | 0.99                | 0.96                     |
| F1-score           | 0.78                | 0.77                     |

## Overfitting/Underfitting
| Dataset | Score  |
|---------|--------|
| Train   | 0.838  |
| Test    | 0.848  |
for both models.
Neither model exhibits overfitting or underfitting.

## Population Stability Index (PSI)
Interpreting PSI Values
- Low PSI (e.g., < 0.1): Indicates little or no shift in the feature’s distribution between the datasets.
- Moderate PSI (e.g., 0.1 to 0.25): Indicates a moderate shift; you may need to monitor the variable.
- High PSI (e.g., > 0.25): Indicates a significant shift in the distribution, which might warrant further investigation or model recalibration.

1. Logistic Regression
- PSI score: 0.0336 - LOW
- AUC: 0.892
- Gini Coefficient: 0.783
2. Random Forest
- PSI score: 2.9781 - HIGH
- AUC: 0.897
- Gini Coefficient: 0.794

## Cross Validation
Use **Stratified K-Fold** (5 splits) to compare two models:
| Model                 | CV Mean   | Std      | Test Scores                                                                                                           |
|-----------------------|-----------|----------|-----------------------------------------------------------------------------------------------------------------------|
| LogisticRegression    | 0.883644  | 0.007064 | 0.8837430395219341, 0.8700937118022545, 0.8891446302831909, 0.8890423337254703, 0.8861950795355736                    |
| RandomForest          | 0.895573  | 0.008571 | 0.8866885101181584, 0.8859245552084747, 0.9076603072306617, 0.8950266823521389, 0.9025625287709069                    |

- Random Forest has a **slightly higher CV mean** than Logistic Regression.  
- Both models show **consistent performance** across folds (Std Dev ≈ 0.007).

> Although the Random Forest model achieves a higher ROC AUC and Gini coefficient than Logistic Regression and outperforms it on every fold, Logistic Regression demonstrates greater stability with a lower standard deviation and a low PSI score. Since our priority is deployment stability, I choose the **Logistic Regression model**.

# Note
I will continue to update and refine the model to achieve even better performance.
