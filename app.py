import streamlit as st
import pandas as pd
import numpy as np
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
st.markdown("### Name: Chatheriyan T")
st.markdown("### ID: 2025aa05339")

st.markdown("Upload a test dataset and select a model to evaluate.")

# -------------------------
# Upload CSV
# -------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Dataset Preview")
    st.write(df.head())
    
    # Drop ID column if present (usually first column with name containing 'id')
    if 'id' in df.columns.str.lower():
        id_col = [col for col in df.columns if 'id' in col.lower()][0]
        df = df.drop(columns=[id_col])

    # Mapping diagnosis if present (for breast cancer dataset compatibility)
    target_col = None
    if 'diagnosis' in df.columns:
        target_col = 'diagnosis'
        if df['diagnosis'].dtype == 'object':
            df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
            st.success("Mapped diagnosis: M → 1 (Malignant), B → 0 (Benign)")
    
    # Separate features and target
    if target_col:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        # Assume last column is target if no diagnosis column
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        st.warning("No 'diagnosis' column found - assuming last column is target")

    # Check if stratification is appropriate (classification with reasonable class distribution)
    n_unique_classes = y.nunique()
    is_classification = n_unique_classes < 20  # Heuristic: less than 20 unique values suggests classification
    
    # Check if stratification is possible (each class has at least 2 samples)
    can_stratify = False
    if is_classification:
        min_class_count = y.value_counts().min()
        can_stratify = min_class_count >= 2

    # Train-Test Split
    if can_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        st.warning(f"Using random split (stratification not possible, {n_unique_classes} unique values)")

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
    
    # Get classification report as dictionary
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    
    # Convert to DataFrame for better visualization
    report_df = pd.DataFrame(report_dict).transpose()
    
    # Format the DataFrame
    report_df = report_df.round(2)
    
    # Display as table with styling
    st.dataframe(
        report_df.style.format("{:.2f}"),
        use_container_width=True
    )

else:
    st.info("Please upload a CSV file to begin. You can find a sample dataset in the 'data' directory.")