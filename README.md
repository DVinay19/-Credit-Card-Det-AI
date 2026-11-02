# 💳 Credit Card Fraud Detection

**Machine Learning-Based Transaction Classification**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Classification-2563EB?style=flat-square" />
</p>

---

## Overview

Credit Card Fraud Detection is a machine learning project that explores the identification of potentially fraudulent credit card transactions using anonymized transaction data.

The application uses Logistic Regression to classify transactions and provides an interactive interface for exploring the dataset, evaluating model performance and testing individual transactions.

## Features

- Dataset overview and transaction analysis
- Fraudulent vs. legitimate transaction visualization
- Balanced dataset preparation through undersampling
- Logistic Regression classification
- Feature standardization
- Training and testing accuracy evaluation
- Confusion matrix visualization
- Sample transaction examples
- Interactive transaction prediction
- Fraud probability output

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core development |
| Streamlit | Interactive application |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |

## Project Structure

```text
Credit-Card-Det-AI/
│
├── app.py
├── creditcard.csv
├── fraud test.txt
├── Legitimate transaction.txt
├── index.html
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Credit-Card-Det-AI
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Environment

Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Prepare the Dataset

Download the dataset from [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

Place `creditcard.csv` in the project directory.

### 6. Run the Application

```bash
streamlit run app.py
```

## Machine Learning Approach

The project uses **Logistic Regression**, a supervised machine learning algorithm for binary classification.

The workflow includes:

1. Loading and exploring transaction data.
2. Separating legitimate and fraudulent transactions.
3. Creating a balanced dataset using undersampling.
4. Standardizing transaction features.
5. Splitting data into training and testing sets.
6. Training the Logistic Regression model.
7. Evaluating predictions using accuracy and a confusion matrix.

## Classification Labels

| Label | Meaning |
|---|---|
| `0` | Legitimate transaction |
| `1` | Fraudulent transaction |

## Future Improvements

- [ ] Improve model evaluation and validation
- [ ] Compare multiple classification algorithms
- [ ] Add precision, recall and F1-score visualizations
- [ ] Introduce ROC curve analysis
- [ ] Integrate explainable AI techniques
- [ ] Enhance transaction analytics and dashboard design

## Disclaimer

This project is intended for educational and research purposes. Model predictions are not guaranteed to identify every fraudulent transaction and should not be used as the sole basis for real-world financial decisions.

---

## Author

**DVinay19**

[GitHub Profile](https://github.com/DVinay19)

<p align="center">
  <sub>Exploring machine learning for smarter financial security.</sub>
</p>