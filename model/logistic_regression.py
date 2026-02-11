from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef
)


def train_logistic_regression(X_train, y_train, random_state=42):
    """
    Train a Logistic Regression model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random state for reproducibility
        
    Returns:
        Trained Logistic Regression model
    """
    lr = LogisticRegression(random_state=random_state)
    lr.fit(X_train, y_train)
    return lr


def predict_logistic_regression(model, X_test):
    """
    Make predictions using a trained Logistic Regression model.
    
    Args:
        model: Trained Logistic Regression model
        X_test: Test features
        
    Returns:
        Predictions
    """
    return model.predict(X_test)


def evaluate_logistic_regression(model, X_test, y_test):
    """
    Evaluate a Logistic Regression model.
    
    Args:
        model: Trained Logistic Regression model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary containing evaluation metrics
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
