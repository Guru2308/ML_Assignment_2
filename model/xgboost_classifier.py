from xgboost import XGBClassifier
from .evaluation import evaluate_model


def train_xgboost(X_train, y_train, random_state=42, eval_metric="logloss"):
    """
    Train an XGBoost model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random state for reproducibility
        eval_metric: Evaluation metric to use
        
    Returns:
        Trained XGBoost model
    """
    xgb = XGBClassifier(
        random_state=random_state,
        eval_metric=eval_metric
    )
    xgb.fit(X_train, y_train)
    return xgb


def predict_xgboost(model, X_test):
    """
    Make predictions using a trained XGBoost model.
    
    Args:
        model: Trained XGBoost model
        X_test: Test features
        
    Returns:
        Predictions
    """
    return model.predict(X_test)


def evaluate_xgboost(model, X_test, y_test):
    """
    Evaluate an XGBoost model.
    
    Args:
        model: Trained XGBoost model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary containing evaluation metrics
    """
    return evaluate_model(model, X_test, y_test)
