from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef
)


def train_decision_tree(X_train, y_train, random_state=42):
    """
    Train a Decision Tree model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random state for reproducibility
        
    Returns:
        Trained Decision Tree model
    """
    dt = DecisionTreeClassifier(random_state=random_state)
    dt.fit(X_train, y_train)
    return dt


def predict_decision_tree(model, X_test):
    """
    Make predictions using a trained Decision Tree model.
    
    Args:
        model: Trained Decision Tree model
        X_test: Test features
        
    Returns:
        Predictions
    """
    return model.predict(X_test)


def evaluate_decision_tree(model, X_test, y_test):
    """
    Evaluate a Decision Tree model.
    
    Args:
        model: Trained Decision Tree model
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
