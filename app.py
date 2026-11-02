import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# App configuration
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("Credit Card Fraud Detection Model")
st.markdown("<h2 style='text-align: center; font-family: sans-serif;'>Methodist college of engineering</h2>", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('creditcard.csv')

data = load_data()

# Show data overview
with st.expander("📂 Dataset Overview"):
    st.write(data.head())
    st.write("Shape:", data.shape)
    st.write("Columns:", data.columns.tolist())

# EDA: Fraud vs Legit
st.subheader("⚖️ Transaction Distribution")
fraud_count = data['Class'].value_counts()
fig, ax = plt.subplots()
sns.barplot(x=fraud_count.index, y=fraud_count.values, palette="viridis", ax=ax)
ax.set_xticklabels(["Legit", "Fraud"])
ax.set_ylabel("Count")
st.pyplot(fig)

# Prepare balanced dataset for training
legit = data[data.Class == 0]
fraud = data[data.Class == 1]
legit_sample = legit.sample(n=len(fraud), random_state=2)
balanced_data = pd.concat([legit_sample, fraud], axis=0)

X = balanced_data.drop(columns="Class", axis=1)
y = balanced_data["Class"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=2)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Display model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)
st.markdown("### 🎯 Model Accuracy")
st.write(f"**Training Accuracy:** {train_acc:.2f}")
st.write(f"**Testing Accuracy:** {test_acc:.2f}")

# Show confusion matrix
with st.expander("📊 Confusion Matrix"):
    cm = confusion_matrix(y_test, model.predict(X_test))
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

# Function to load example transactions
def load_examples(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
    cleaned = []
    for line in lines:
        parts = [p.strip() for p in line.strip().split(',')]
        if parts[0] in ['0', '1']:
            cleaned.append(', '.join(parts[1:]))
        else:
            cleaned.append(', '.join(parts))
    return [ex for ex in cleaned if ex]

legit_examples = load_examples("Legitimate transaction.txt")
fraud_examples = load_examples("fraud test.txt")

st.markdown("### ✅ Example Legitimate Transactions ")
for example in legit_examples[:2]:
    st.code(example, language='text')

st.markdown("### 🚨 Example Fraudulent Transactions ")
for example in fraud_examples[:2]:
    st.code(example, language='text')

# Input for prediction
st.markdown("### ✍️ Test a Transaction")
st.write("👉 **Enter 30 comma-separated feature values below (excluding 0/1 at start):**")

user_input = st.text_area('Example: -1.3598, -0.0728, 2.5363, ...', height=100)
submit = st.button("Submit")

if submit:
    try:
        input_list = [float(i.strip()) for i in user_input.split(',')]
        if len(input_list) != 30:
            st.error("❗ Please enter exactly **30** feature values separated by commas.")
        else:
            scaled_input = scaler.transform([input_list])
            prediction = model.predict(scaled_input)
            prediction_prob = model.predict_proba(scaled_input)[0][1]

            if prediction[0] == 0:
                st.success("✅ Prediction: Legitimate Transaction")
            else:
                st.error("🚨 Prediction: Fraudulent Transaction")

            st.write(f"**Fraud Probability:** {prediction_prob:.4f}")
    except Exception as e:
        st.error("⚠️ Invalid input format. Please enter **30 numbers** separated by commas.")