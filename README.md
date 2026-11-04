# 💳 Credit Card Fraud Detection

### Machine Learning-Based Transaction Classification

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white">
  <img src="https://img.shields.io/badge/Machine_Learning-Classification-blue?style=for-the-badge">
</p>

---

## Overview

Credit Card Fraud Detection is a machine learning application designed to identify potentially fraudulent credit card transactions using transaction data and predictive classification techniques.

The application uses a Logistic Regression model with class weighting to address the highly imbalanced nature of financial transaction data. An interactive Streamlit dashboard provides dataset analytics, model evaluation, transaction monitoring and individual transaction risk analysis.

The project focuses on combining practical machine learning techniques with an accessible financial security dashboard.

## Features

- Dataset overview and transaction analysis
- Legitimate vs. fraudulent transaction visualization
- Imbalanced dataset handling using class-weighted learning
- Logistic Regression classification model
- Feature standardization using StandardScaler
- Model evaluation using Accuracy, Precision, Recall, F1-score and ROC-AUC
- Confusion matrix visualization
- Detailed classification report
- Transaction monitoring with risk-level filtering
- Search transactions using transaction IDs
- Individual transaction fraud prediction
- Adjustable fraud decision threshold
- Fraud probability and risk-level assessment
- CSV export of filtered transaction reports
- Recent manual prediction history
- Deep Ocean and Cyber Teal dashboard interface

## Dashboard Preview

The application provides an interactive interface for exploring transaction data, reviewing model performance and analyzing potentially fraudulent transactions.

**Dashboard screenshots:** Add your application screenshots here after capturing them.

## Machine Learning Approach

The project uses **Logistic Regression** as its classification algorithm.

### Workflow

1. Load and validate the credit card transaction dataset.
2. Separate input features and target labels.
3. Split the dataset into training and testing sets using stratified sampling.
4. Standardize numerical features using StandardScaler.
5. Train Logistic Regression with `class_weight="balanced"`.
6. Generate transaction predictions and fraud probabilities.
7. Evaluate model performance on the held-out test dataset.

The dataset is split before feature scaling to avoid data leakage.

### Evaluation Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Precision | Proportion of predicted fraud cases that are actually fraud |
| Recall | Proportion of actual fraud cases detected |
| F1-score | Balance between precision and recall |
| ROC-AUC | Ability to distinguish fraudulent from legitimate transactions |

## Dataset

The project uses the Credit Card Fraud Detection dataset.

**Dataset source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Dataset overview:

| Category | Count |
|---|---:|
| Total transactions | 284,807 |
| Legitimate transactions | 284,315 |
| Fraudulent transactions | 492 |
| Input features | 30 |

The dataset is highly imbalanced, making fraud detection a challenging classification problem.

The dataset is not included in this repository. Download `creditcard.csv` from Kaggle and place it in the project root directory.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Dashboard | Streamlit |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Styling | HTML, CSS |
| Version Control | Git, GitHub |

## Project Structure

```text
Credit-Card-Det-AI/
│
├── app.py
├── style.css
├── index.html
├── requirements.txt
├── README.md
├── .gitignore
│
├── Legitimate transaction.txt
└── fraud test.txt
```

The `creditcard.csv` dataset is downloaded separately and is not committed to this repository.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/DVinay19/Credit-Card-Det-AI.git
cd Credit-Card-Det-AI
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

For Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Download the dataset

Download the Kaggle dataset and place `creditcard.csv` inside the project folder.

### 6. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

## Future Enhancements

- Bulk CSV transaction prediction
- Downloadable PDF fraud analysis reports
- Explainable AI for transaction predictions
- Advanced anomaly detection algorithms
- Cloud deployment
- Enhanced transaction monitoring

## Disclaimer

This project is developed for educational and demonstration purposes. Model predictions represent estimated fraud probabilities and should not be treated as definitive financial decisions or used as a production banking security system.

## Author

**Vinay Kumar**

GitHub: [@DVinay19](https://github.com/DVinay19)

---

<p align="center">
  Built with Python, Machine Learning and Streamlit.
</p>
