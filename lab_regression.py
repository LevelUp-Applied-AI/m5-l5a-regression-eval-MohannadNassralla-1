import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             mean_absolute_error, r2_score)


def load_data(filepath="data/telecom_churn.csv"):
    """Load the telecom churn dataset."""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return None


def split_data(df, target_col, test_size=0.2, random_state=42):
    """Split data into train and test sets with stratification if applicable."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    stratify_param = y if y.nunique() < 10 else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=stratify_param
    )
    return X_train, X_test, y_train, y_test


def build_logistic_pipeline():
    """Build a Pipeline with StandardScaler and LogisticRegression."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced"))
    ])
    return pipeline


def build_ridge_pipeline():
    """Build a Pipeline with StandardScaler and Ridge regression."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=1.0))
    ])
    return pipeline


def evaluate_classifier(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return classification metrics."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred))
    }
    return metrics


def evaluate_regressor(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return regression metrics."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred))
    }
    return metrics


def run_cross_validation(pipeline, X_train, y_train, cv=5):
    """Run stratified cross-validation on the pipeline."""
   
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    scores = cross_val_score(
        pipeline, X_train, y_train, 
        cv=cv_splitter, 
        scoring="accuracy"
    )
    return scores


if __name__ == "__main__":
    df = load_data()
    if df is not None:
        print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

        # Select numeric features for classification
        numeric_features = ["tenure", "monthly_charges", "total_charges",
                           "num_support_calls", "senior_citizen",
                           "has_partner", "has_dependents"]

        # Classification: predict churn
        df_cls = df[numeric_features + ["churned"]].dropna()
        split = split_data(df_cls, "churned")
        if split:
            X_train, X_test, y_train, y_test = split
            pipe = build_logistic_pipeline()
            if pipe:
                metrics = evaluate_classifier(pipe, X_train, X_test, y_train, y_test)
                print(f"Logistic Regression: {metrics}")

                scores = run_cross_validation(pipe, X_train, y_train)
                if scores is not None:
                    print(f"CV Accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

        # Regression: predict monthly_charges
        df_reg = df[["tenure", "total_charges", "num_support_calls",
                     "senior_citizen", "has_partner", "has_dependents",
                     "monthly_charges"]].dropna()
        
      
        split_reg = split_data(df_reg, "monthly_charges")
        if split_reg:
            X_tr, X_te, y_tr, y_te = split_reg
            ridge_pipe = build_ridge_pipeline()
            if ridge_pipe:
                reg_metrics = evaluate_regressor(ridge_pipe, X_tr, X_te, y_tr, y_te)
                print(f"Ridge Regression: {reg_metrics}")


"""
--- Task 7: Summary of Findings ---

1. Important Features: 'tenure' and 'total_charges' typically show the strongest influence 
   on churn. 'num_support_calls' often correlates positively with churn.
   
2. Performance: Logistic Regression with balanced weights usually increases recall (catching 
   more churners) but decreases precision (more false alarms). In churn, high recall is 
   usually preferred because the cost of losing a customer outweighs the cost of a promo.
   
3. Next Steps: 
   - Feature engineering (creating ratios between charges and tenure).
   - Hyperparameter tuning for 'C' in LogisticRegression.
   - Testing non-linear models like Random Forest for better boundary capture.
"""