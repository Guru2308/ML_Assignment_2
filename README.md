# Machine Learning Classification Project

## Problem Statement

Breast cancer is one of the most common cancers among women worldwide. Early detection and accurate diagnosis are crucial for successful treatment and improving patient outcomes. This project aims to develop and compare multiple machine learning classification models to predict whether a breast tumor is malignant (cancerous) or benign (non-cancerous) based on various features extracted from digitized images of fine needle aspirate (FNA) of breast mass.

The goal is to identify the most effective machine learning algorithm for this binary classification task, enabling automated and reliable diagnostic support for medical professionals.

---

## Dataset Description

**Dataset:** Breast Cancer Dataset

**Source:** Kaggle

**Description:**
The dataset contains features computed from digitized images of fine needle aspirate (FNA) of breast mass. These features describe characteristics of the cell nuclei present in the image.

**Dataset Statistics:**
- **Total Samples:** 569 instances
- **Features:** 30 numerical features (all real-valued)
- **Target Variable:** Diagnosis (M = Malignant, B = Benign)
  - Malignant (1): 212 instances (37.3%)
  - Benign (0): 357 instances (62.7%)
- **Missing Values:** None

**Feature Categories:**
The 30 features are computed for each cell nucleus and include:

1. **Mean values** (10 features): radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension
2. **Standard error** (10 features): Same measurements as above
3. **"Worst" or largest values** (10 features): Same measurements as above

**Key Characteristics:**
- All features are continuous numerical values
- Features are standardized before model training
- Dataset is slightly imbalanced (37.3% malignant, 62.7% benign)
- Train-test split: 80% training, 20% testing with stratification

---

## Models Used

### Comparison Table: Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|----------|-----|-----------|--------|----------|-----|
| **Logistic Regression** | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| **Decision Tree** | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| **K-Nearest Neighbors (KNN)** | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| **Naive Bayes** | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| **Random Forest (Ensemble)** | **0.9737** | 0.9929 | **1.0000** | 0.9286 | 0.9630 | 0.9442 |
| **XGBoost (Ensemble)** | **0.9737** | **0.9940** | **1.0000** | 0.9286 | 0.9630 | 0.9442 |


---

### Model Performance Observations

| ML Model Name | Observation about Model Performance |
|--------------|-------------------------------------|
| **Logistic Regression** | Achieved excellent performance with 96.49% accuracy and nearly perfect AUC (0.9960). The model shows strong precision (97.5%) with good recall (92.86%), making it highly reliable for clinical applications. Its interpretability and fast training make it a strong baseline choice for medical diagnosis. |
| **Decision Tree** | Demonstrated moderate performance with 92.98% accuracy. While it achieved balanced precision and recall (both 90.48%), it had the lowest AUC (0.9246) among all models, indicating some difficulty in distinguishing between classes. The model is prone to overfitting and showed the weakest overall performance, though still acceptable for basic classification tasks. |
| **K-Nearest Neighbors (KNN)** | Performed well with 95.61% accuracy and strong AUC (0.9823). The model exhibited excellent precision (97.44%) but slightly lower recall (90.48%), suggesting it's conservative in predicting malignancy. The simplicity of KNN makes it easy to implement, though it can be computationally expensive for large datasets. |
| **Naive Bayes** | Showed the lowest overall accuracy at 92.11% but maintained a high AUC (0.9891), indicating good separability despite misclassifications. With the lowest recall (85.71%), the model tends to miss some malignant cases, which is concerning in medical contexts where false negatives are critical. However, its speed and simplicity make it suitable for quick preliminary screening. |
| **Random Forest (Ensemble)** | Achieved top-tier performance with 97.37% accuracy and perfect precision (100%), meaning zero false positives. The high recall (92.86%) and excellent MCC (0.9442) demonstrate robust and balanced performance. As an ensemble method, it provides feature importance insights and handles non-linear relationships well, making it highly suitable for complex medical data. |
| **XGBoost (Ensemble)** | Tied with Random Forest for the highest accuracy (97.37%) and achieved the best AUC (0.9940) with perfect precision (100%). The model's gradient boosting approach provides excellent generalization and handles class imbalance effectively. Its superior AUC indicates the strongest discriminative ability among all models, making it the most reliable choice for clinical deployment. The trade-off is increased training time and complexity. |

---

## Key Findings

1. **Best Overall Model:** XGBoost and Random Forest tied for accuracy but XGBoost slightly edges out with the highest AUC (0.9940)
2. **Perfect Precision:** Both ensemble methods achieved 100% precision, eliminating false positives
3. **Clinical Reliability:** All models except Naive Bayes exceeded 93% accuracy, demonstrating strong diagnostic potential
4. **Ensemble Advantage:** Ensemble methods (Random Forest and XGBoost) significantly outperformed single models
5. **Trade-off Consideration:** While Decision Tree offers interpretability, its lower performance suggests ensemble methods are more suitable for critical medical applications

---

## Project Structure

```
ML_Assignment_2/
├── app.py                          # Streamlit web application
├── model/
│   ├── __init__.py                 # Package initialization
│   ├── evaluation.py               # Common evaluation utilities
│   ├── logistic_regression.py      # Logistic Regression model
│   ├── decision_tree.py            # Decision Tree model
│   ├── knn.py                      # K-Nearest Neighbors model
│   ├── naive_bayes.py              # Naive Bayes model
│   ├── random_forest.py            # Random Forest model
│   └── xgboost_classifier.py       # XGBoost model
├── data/
│   └── dataset.csv                 # Breast cancer dataset
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## How to Run

### Prerequisites
- Python 3.12 or higher
- Virtual environment (recommended)

### Installation

1. **Clone the repository:**
   ```bash
   cd ML_Assignment_2
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Streamlit App

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Features
- Upload CSV dataset for analysis
- Select from 6 different machine learning models
- View comprehensive evaluation metrics
- Visualize confusion matrix as heatmap
- Examine detailed classification report in table format

---

## Technologies Used

- **Python 3.12**
- **Streamlit** - Web application framework
- **scikit-learn** - Machine learning models and evaluation
- **XGBoost** - Gradient boosting framework
- **pandas** - Data manipulation
- **NumPy** - Numerical computations
- **Matplotlib & Seaborn** - Data visualization

---

## Conclusion

This project successfully demonstrates the application of multiple machine learning algorithms for breast cancer diagnosis. The ensemble methods (Random Forest and XGBoost) proved to be the most effective, achieving 97.37% accuracy with perfect precision. These models can serve as reliable diagnostic support tools for medical professionals, potentially improving early detection and patient outcomes.
