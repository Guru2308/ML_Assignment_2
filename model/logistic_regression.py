from sklearn.linear_model import LogisticRegression
from .evaluation import evaluate_model


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
    return evaluate_model(model, X_test, y_test)
