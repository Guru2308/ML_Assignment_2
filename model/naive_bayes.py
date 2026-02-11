from sklearn.naive_bayes import GaussianNB
from .evaluation import evaluate_model


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
    return evaluate_model(model, X_test, y_test)
