from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef
)


def train_naive_bayes(X_train, y_train):
    """
    Train a Naive Bayes model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained Naive Bayes model
    """
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    return nb


def predict_naive_bayes(model, X_test):
    """
    Make predictions using a trained Naive Bayes model.
    
    Args:
        model: Trained Naive Bayes model
        X_test: Test features
        
    Returns:
        Predictions
    """
    return model.predict(X_test)


def evaluate_naive_bayes(model, X_test, y_test):
    """
    Evaluate a Naive Bayes model.
    
    Args:
        model: Trained Naive Bayes model
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
