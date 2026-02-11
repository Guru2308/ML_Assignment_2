from sklearn.neighbors import KNeighborsClassifier
from .evaluation import evaluate_model


def train_knn(X_train, y_train, n_neighbors=5):
    """
    Train a K-Nearest Neighbors model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_neighbors: Number of neighbors to use
        
    Returns:
        Trained KNN model
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn


def predict_knn(model, X_test):
    """
    Make predictions using a trained KNN model.
    
    Args:
        model: Trained KNN model
        X_test: Test features
        
    Returns:
        Predictions
    """
    return model.predict(X_test)


def evaluate_knn(model, X_test, y_test):
    """
    Evaluate a KNN model.
    
    Args:
        model: Trained KNN model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary containing evaluation metrics
    """
    return evaluate_model(model, X_test, y_test)
