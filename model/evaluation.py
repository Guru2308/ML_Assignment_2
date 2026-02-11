"""
Common evaluation utilities for ML models.
"""

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a machine learning model using various metrics.
    
    Args:
        model: Trained model with predict and predict_proba methods
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary containing evaluation metrics:
        - Accuracy: Overall accuracy score
        - AUC: Area Under the ROC Curve
        - F1 Score: F1 score (harmonic mean of precision and recall)
        - Precision: Precision score
        - Recall: Recall score
        - MCC: Matthews Correlation Coefficient
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "F1 Score": f1_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }
