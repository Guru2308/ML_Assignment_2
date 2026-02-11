from sklearn.tree import DecisionTreeClassifier
from .evaluation import evaluate_model


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
    return evaluate_model(model, X_test, y_test)
