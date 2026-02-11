"""
ML Models Package
Contains implementations of various machine learning models for breast cancer classification.
"""

from .logistic_regression import (
    train_logistic_regression,
    predict_logistic_regression,
    evaluate_logistic_regression
)
from .decision_tree import (
    train_decision_tree,
    predict_decision_tree,
    evaluate_decision_tree
)
from .knn import (
    train_knn,
    predict_knn,
    evaluate_knn
)
from .naive_bayes import (
    train_naive_bayes,
    predict_naive_bayes,
    evaluate_naive_bayes
)
from .random_forest import (
    train_random_forest,
    predict_random_forest,
    evaluate_random_forest
)
from .xgboost import (
    train_xgboost,
    predict_xgboost,
    evaluate_xgboost
)

__all__ = [
    # Logistic Regression
    'train_logistic_regression',
    'predict_logistic_regression',
    'evaluate_logistic_regression',
    # Decision Tree
    'train_decision_tree',
    'predict_decision_tree',
    'evaluate_decision_tree',
    # KNN
    'train_knn',
    'predict_knn',
    'evaluate_knn',
    # Naive Bayes
    'train_naive_bayes',
    'predict_naive_bayes',
    'evaluate_naive_bayes',
    # Random Forest
    'train_random_forest',
    'predict_random_forest',
    'evaluate_random_forest',
    # XGBoost
    'train_xgboost',
    'predict_xgboost',
    'evaluate_xgboost',
]
