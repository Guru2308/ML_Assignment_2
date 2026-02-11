import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

from model import (
    train_logistic_regression,
    train_decision_tree,
    train_knn,
    train_naive_bayes,
    train_random_forest,
    train_xgboost,
    evaluate_model
)

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("Machine Learning Classification App")

st.markdown("Upload a test dataset and select a model to evaluate.")

# -------------------------
# Upload CSV
# -------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Mapping diagnosis if present (for consistency with notebook)
    if 'diagnosis' in df.columns and df['diagnosis'].dtype == 'object':
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    # Assume last column is target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -------------------------
    # Model Selection
    # -------------------------
    model_choice = st.selectbox(
        "Select Model",
        (
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        )
    )

    # -------------------------
    # Model Training
    # -------------------------
    with st.spinner(f"Training {model_choice}..."):
        if model_choice == "Logistic Regression":
            model = train_logistic_regression(X_train, y_train)
        elif model_choice == "Decision Tree":
            model = train_decision_tree(X_train, y_train)
        elif model_choice == "KNN":
            model = train_knn(X_train, y_train)
        elif model_choice == "Naive Bayes":
            model = train_naive_bayes(X_train, y_train)
        elif model_choice == "Random Forest":
            model = train_random_forest(X_train, y_train)
        elif model_choice == "XGBoost":
            model = train_xgboost(X_train, y_train)

    # -------------------------
    # Evaluation
    # -------------------------
    metrics = evaluate_model(model, X_test, y_test)
    y_pred = model.predict(X_test)

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", round(metrics["Accuracy"], 4))
    col2.metric("Precision", round(metrics["Precision"], 4))
    col3.metric("Recall", round(metrics["Recall"], 4))

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", round(metrics["F1 Score"], 4))
    col5.metric("MCC", round(metrics["MCC"], 4))
    col6.metric("AUC", round(metrics["AUC"], 4) if isinstance(metrics["AUC"], (int, float, np.float64)) else metrics["AUC"])

    # -------------------------
    # Confusion Matrix
    # -------------------------
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    # -------------------------
    # Classification Report
    # -------------------------
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred)
    st.text(report)

else:
    st.info("Please upload a CSV file to begin. You can find a sample dataset in the 'data' directory.")