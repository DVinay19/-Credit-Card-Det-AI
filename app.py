
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# Deep Ocean + Cyber Teal palette
OCEAN = "#102337"
DEEP_OCEAN = "#0B1928"
TEAL = "#20C9B0"
TEAL_LIGHT = "#76E5D4"
TEXT = "#E8F4F5"
MUTED = "#9BB2C2"
BORDER = "#244257"
BLUE = "#397FA3"
DANGER = "#F16B75"


# ==================================================
# LOAD PROFESSIONAL CSS
# ==================================================

def load_css():
    css_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "style.css"
    )

    try:
        with open(css_path, "r", encoding="utf-8") as file:
            st.markdown(
                f"<style>{file.read()}</style>",
                unsafe_allow_html=True
            )
    except FileNotFoundError:
        st.warning("style.css was not found in the project folder.")


load_css()


# ==================================================
# MATPLOTLIB THEME HELPER
# ==================================================

def apply_chart_theme(fig, ax):
    """Apply Deep Ocean + Cyber Teal styling to charts."""

    fig.patch.set_facecolor(OCEAN)
    ax.set_facecolor(OCEAN)

    ax.tick_params(
        axis="both",
        colors=MUTED,
        labelsize=9
    )

    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEAL_LIGHT)

    ax.grid(
        color=BORDER,
        alpha=0.65,
        linewidth=0.7
    )

    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color(BORDER)



# ==================================================
# HEADER
# ==================================================

st.markdown(
    f"""<div style="padding:12px 0 24px 0;">
    <div style="color:{TEAL};font-size:13px;
    font-weight:700;letter-spacing:2px;">
    FINANCIAL SECURITY PLATFORM
    </div>
    <h1 style="color:{TEXT};font-size:38px;margin-bottom:5px;">
    🛡️ Credit Card Fraud Detection
    </h1>
    <p style="color:{MUTED};font-size:16px;">
    Intelligent transaction analysis and fraud monitoring
    </p>
    </div>""",
    unsafe_allow_html=True
)



# ==================================================
# LOAD DATASET
# ==================================================

@st.cache_data
def load_data():

    dataset_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "creditcard.csv"
    )

    return pd.read_csv(dataset_path)


try:
    data = load_data()

except FileNotFoundError:

    st.error(
        "Dataset not found. Please place creditcard.csv "
        "inside the project folder."
    )

    st.stop()


if "Class" not in data.columns:
    st.error("Dataset must contain a Class column.")
    st.stop()

if data.empty:
    st.error("The dataset is empty.")
    st.stop()

if data["Class"].isna().any():
    st.error("The Class column contains missing values.")
    st.stop()

if not set(data["Class"].unique()).issubset({0, 1}):
    st.error("Class must contain only 0 and 1.")
    st.stop()

if data["Class"].nunique() != 2:
    st.error(
        "Dataset must contain both legitimate and fraudulent records."
    )
    st.stop()


feature_columns = [
    column for column in data.columns
    if column != "Class"
]


# ==================================================
# DATASET OVERVIEW
# ==================================================

total_transactions = len(data)

fraud_count = int((data["Class"] == 1).sum())

legit_count = int((data["Class"] == 0).sum())

fraud_percentage = (
    fraud_count / total_transactions
) * 100


st.subheader("Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Transactions",
    f"{total_transactions:,}"
)

c2.metric(
    "Legitimate Transactions",
    f"{legit_count:,}"
)

c3.metric(
    "Fraudulent Transactions",
    f"{fraud_count:,}"
)

c4.metric(
    "Fraud Rate",
    f"{fraud_percentage:.3f}%"
)


with st.expander("View Dataset Information"):

    st.write("Dataset shape:", data.shape)

    st.write(
        "Available columns:",
        feature_columns + ["Class"]
    )

    st.dataframe(
        data.head(),
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# TRANSACTION ANALYTICS
# ==================================================

st.divider()

st.subheader("Transaction Analytics")

chart_col, insight_col = st.columns([1.7, 1])

with chart_col:

    fig, ax = plt.subplots(
        figsize=(7, 3.2),
        dpi=110
    )

    apply_chart_theme(fig, ax)

    labels = [
        "Legitimate",
        "Fraudulent"
    ]

    values = [
        legit_count,
        fraud_count
    ]

    bar_colors = [
        TEAL,
        DANGER
    ]

    bars = ax.bar(
        labels,
        values,
        color=bar_colors,
        width=0.48,
        edgecolor="none"
    )

    ax.set_title(
        "Transaction Distribution",
        fontsize=13,
        color=TEAL_LIGHT,
        pad=14
    )

    ax.set_ylabel(
        "Number of Transactions",
        fontsize=9,
        color=TEXT
    )

    ax.set_yscale("log")

    for bar, value in zip(bars, values):

        ax.annotate(
            f"{value:,}",
            (
                bar.get_x() + bar.get_width() / 2,
                bar.get_height()
            ),
            ha="center",
            va="bottom",
            color=TEXT,
            fontsize=9,
            fontweight="bold",
            xytext=(0, 5),
            textcoords="offset points"
        )

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


with insight_col:

    st.markdown("#### Dataset Insights")

    st.metric(
        "Total Records",
        f"{total_transactions:,}"
    )

    st.metric(
        "Fraud Cases",
        f"{fraud_count:,}"
    )

    st.metric(
        "Legitimate Cases",
        f"{legit_count:,}"
    )

    st.info(
        "Fraud cases represent a small portion of this dataset. "
        "Class imbalance is an important consideration."
    )


# ==================================================
# MODEL TRAINING
# ==================================================

@st.cache_resource
def train_model(df):

    X = df.drop(columns=["Class"])

    y = df["Class"]

    # Split data before scaling to prevent data leakage

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Standardize features

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    # Train Logistic Regression

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    # Predictions

    train_predictions = model.predict(
        X_train_scaled
    )

    test_predictions = model.predict(
        X_test_scaled
    )

    test_probabilities = model.predict_proba(
        X_test_scaled
    )[:, 1]

    # Evaluation metrics

    metrics = {

        "train_accuracy": accuracy_score(
            y_train,
            train_predictions
        ),

        "accuracy": accuracy_score(
            y_test,
            test_predictions
        ),

        "precision": precision_score(
            y_test,
            test_predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            test_predictions,
            zero_division=0
        ),

        "f1": f1_score(
            y_test,
            test_predictions,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_test,
            test_probabilities
        )
    }

    # Confusion matrix

    cm = confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1]
    )

    # Classification report

    report = classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Fraud"],
        output_dict=True,
        zero_division=0
    )

    return (
        scaler,
        model,
        metrics,
        cm,
        report,
        X_test,
        y_test,
        test_predictions,
        test_probabilities
    )


# ==================================================
# TRAINING STATUS
# ==================================================

with st.spinner("Training fraud detection model..."):

    (
        scaler,
        model,
        metrics,
        cm,
        report,
        X_test,
        y_test,
        test_predictions,
        test_probabilities
    ) = train_model(data)


# ==================================================
# MACHINE LEARNING PERFORMANCE
# ==================================================

st.divider()

st.subheader("Machine Learning Performance")

st.caption(
    "Evaluation results from the held-out test dataset."
)

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Accuracy",
    f"{metrics['accuracy'] * 100:.2f}%"
)

m2.metric(
    "Precision",
    f"{metrics['precision'] * 100:.2f}%"
)

m3.metric(
    "Recall",
    f"{metrics['recall'] * 100:.2f}%"
)

m4.metric(
    "F1 Score",
    f"{metrics['f1'] * 100:.2f}%"
)

roc_col, training_col = st.columns(2)

roc_col.metric(
    "ROC-AUC Score",
    f"{metrics['roc_auc']:.4f}"
)

training_col.metric(
    "Training Accuracy",
    f"{metrics['train_accuracy'] * 100:.2f}%"
)


# ==================================================
# MODEL EVALUATION
# ==================================================

st.divider()

st.subheader("Model Evaluation")

matrix_col, summary_col = st.columns([1.2, 1])

with matrix_col:

    fig, ax = plt.subplots(
        figsize=(4.5, 3.2),
        dpi=110
    )

    fig.patch.set_facecolor(OCEAN)

    ax.set_facecolor(OCEAN)

    ax.imshow(
        cm,
        cmap="PuBuGn",
        interpolation="nearest"
    )

    ax.set_xticks([0, 1])

    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["Legitimate", "Fraud"],
        fontsize=9,
        color=TEXT
    )

    ax.set_yticklabels(
        ["Legitimate", "Fraud"],
        fontsize=9,
        color=TEXT
    )

    ax.set_xlabel(
        "Predicted",
        fontsize=9,
        color=TEXT
    )

    ax.set_ylabel(
        "Actual",
        fontsize=9,
        color=TEXT
    )

    ax.set_title(
        "Confusion Matrix",
        fontsize=12,
        color=TEAL_LIGHT,
        pad=12
    )

    threshold_value = cm.max() / 2

    for i in range(2):

        for j in range(2):

            text_color = (
                "#FFFFFF"
                if cm[i, j] > threshold_value
                else TEXT
            )

            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
                fontweight="bold"
            )

    ax.tick_params(
        length=0,
        colors=MUTED
    )

    for spine in ax.spines.values():
        spine.set_color(BORDER)

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


with summary_col:

    st.markdown("#### Evaluation Summary")

    st.write(
        "The confusion matrix compares actual transaction "
        "classes against model predictions."
    )

    st.metric(
        "Correct Legitimate Predictions",
        f"{cm[0, 0]:,}"
    )

    st.metric(
        "False Fraud Alerts",
        f"{cm[0, 1]:,}"
    )

    st.metric(
        "Missed Fraud Cases",
        f"{cm[1, 0]:,}"
    )

    st.metric(
        "Detected Fraud Cases",
        f"{cm[1, 1]:,}"
    )


with st.expander("Detailed Classification Report"):

    report_df = pd.DataFrame(report).T

    st.dataframe(
        report_df.style.format("{:.3f}"),
        use_container_width=True
    )


# ==================================================
# TRANSACTION MONITORING
# ==================================================

st.divider()

st.subheader("Transaction Monitoring")

st.caption(
    "Review model predictions on the held-out test transactions."
)

monitor_data = pd.DataFrame({

    "Transaction ID": X_test.index.astype(str),

    "Actual Status": [
        "Fraud" if value == 1 else "Legitimate"
        for value in y_test
    ],

    "Predicted Status": [
        "Fraud" if value == 1 else "Legitimate"
        for value in test_predictions
    ],

    "Fraud Probability": test_probabilities * 100

})

monitor_data["Risk Level"] = pd.cut(
    monitor_data["Fraud Probability"],
    bins=[-0.01, 40, 75, 100],
    labels=["Low", "Medium", "High"]
)

search_col, filter_col = st.columns([1.5, 1])

with search_col:

    search_id = st.text_input(
        "Search Transaction ID",
        placeholder="Enter transaction ID..."
    )

with filter_col:

    risk_filter = st.selectbox(
        "Filter by Risk Level",
        ["All", "High", "Medium", "Low"]
    )

filtered_data = monitor_data.copy()

if search_id:

    filtered_data = filtered_data[
        filtered_data["Transaction ID"].str.contains(
            search_id,
            case=False,
            na=False
        )
    ]

if risk_filter != "All":

    filtered_data = filtered_data[
        filtered_data["Risk Level"].astype(str) == risk_filter
    ]

st.dataframe(
    filtered_data.round({
        "Fraud Probability": 2
    }),
    use_container_width=True,
    hide_index=True
)

st.caption(
    f"Showing {len(filtered_data):,} of "
    f"{len(monitor_data):,} test transactions."
)

st.download_button(
    label="Download Transaction Report",
    data=filtered_data.to_csv(index=False),
    file_name="fraud_transaction_report.csv",
    mime="text/csv"
)


# ==================================================
# EXAMPLE TRANSACTIONS
# ==================================================

def load_examples(filename):

    file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        filename
    )

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            lines = file.readlines()

        cleaned = []

        for line in lines:

            parts = [
                part.strip()
                for part in line.strip().split(",")
            ]

            if parts and parts[0] in ["0", "1"]:

                parts = parts[1:]

            if parts:

                cleaned.append(", ".join(parts))

        return cleaned

    except FileNotFoundError:

        return []


st.divider()

st.subheader("Example Transactions")

with st.expander("View Sample Transactions"):

    legitimate_examples = load_examples(
        "Legitimate transaction.txt"
    )

    fraudulent_examples = load_examples(
        "fraud test.txt"
    )

    left, right = st.columns(2)

    with left:

        st.markdown("**Legitimate Examples**")

        if legitimate_examples:

            for example in legitimate_examples[:2]:

                st.code(example, language="text")

        else:

            st.info("No legitimate examples found.")

    with right:

        st.markdown("**Fraudulent Examples**")

        if fraudulent_examples:

            for example in fraudulent_examples[:2]:

                st.code(example, language="text")

        else:

            st.info("No fraudulent examples found.")


# ==================================================
# MANUAL TRANSACTION PREDICTION
# ==================================================

st.divider()

st.subheader("Test a Transaction")

st.caption(
    f"Enter exactly {len(feature_columns)} comma-separated "
    "feature values, excluding the Class label."
)

threshold = st.slider(
    "Fraud Decision Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05
)

user_input = st.text_area(
    "Transaction Features",
    placeholder="Enter transaction values separated by commas...",
    height=120
)

if st.button(
    "Analyze Transaction",
    type="primary"
):

    try:

        input_list = [
            float(value.strip())
            for value in user_input.split(",")
        ]

        if len(input_list) != len(feature_columns):

            st.error(
                f"Please enter exactly {len(feature_columns)} "
                "feature values."
            )

        elif not np.all(np.isfinite(input_list)):

            st.error(
                "Please enter only valid finite numbers."
            )

        else:

            input_df = pd.DataFrame(
                [input_list],
                columns=feature_columns
            )

            scaled_input = scaler.transform(input_df)

            fraud_probability = model.predict_proba(
                scaled_input
            )[0][1]

            prediction = int(
                fraud_probability >= threshold
            )

            risk_score = fraud_probability * 100

            if risk_score >= 75:

                risk_level = "High"

            elif risk_score >= 40:

                risk_level = "Medium"

            else:

                risk_level = "Low"

            st.divider()

            st.markdown("### Analysis Result")

            r1, r2, r3 = st.columns(3)

            r1.metric(
                "Fraud Risk Score",
                f"{risk_score:.2f}%"
            )

            r2.metric(
                "Risk Level",
                risk_level
            )

            r3.metric(
                "Decision Threshold",
                f"{threshold:.0%}"
            )

            if prediction == 1:

                st.error(
                    "Potential Fraudulent Transaction Detected"
                )

            else:

                st.success(
                    "Transaction Classified as Legitimate"
                )

            st.progress(
                float(fraud_probability)
            )

            st.caption(
                "The fraud probability is a model estimate, "
                "not a guarantee of actual fraud."
            )

            # Save prediction history

            if "history" not in st.session_state:

                st.session_state.history = []

            st.session_state.history.insert(
                0,
                {
                    "Fraud Probability": round(
                        risk_score, 2
                    ),

                    "Risk Level": risk_level,

                    "Prediction": (
                        "Fraud"
                        if prediction == 1
                        else "Legitimate"
                    )
                }
            )

    except ValueError:

        st.error(
            "Invalid input. Enter comma-separated numeric values."
        )


# ==================================================
# PREDICTION HISTORY
# ==================================================

if (
    "history" in st.session_state
    and st.session_state.history
):

    st.divider()

    st.subheader("Recent Transaction Analysis")

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    if st.button("Clear Transaction History"):

        st.session_state.history = []

        st.rerun()



# ==================================================
# FOOTER
# ==================================================

st.divider()

st.markdown(
    f"""<div style="text-align:center;padding:18px 0;">
    <div style="color:{TEAL};font-weight:700;font-size:15px;">
    Credit Card Fraud Detection
    </div>
    <div style="color:{MUTED};font-size:12px;margin-top:5px;">
    Intelligent Transaction Analysis & Fraud Monitoring
    </div>
    <div style="color:{MUTED};font-size:11px;margin-top:12px;">
    Machine Learning Based Financial Security
    </div>
    </div>""",
    unsafe_allow_html=True
)

