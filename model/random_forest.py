from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef
)


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of trees in the forest
        random_state: Random state for reproducibility
        
    Returns:
        Trained Random Forest model
    """
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)
    return rf


def predict_random_forest(model, X_test):
    """
    Make predictions using a trained Random Forest model.
    
    Args:
        model: Trained Random Forest model
        X_test: Test features
        
    Returns:
        Predictions
    """
    return model.predict(X_test)


def evaluate_random_forest(model, X_test, y_test):
    """
    Evaluate a Random Forest model.
    
    Args:
        model: Trained Random Forest model
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
