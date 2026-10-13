import pandas as pd
import shap
import joblib
from xgboost import XGBClassifier
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def explain_predictions():
    print("Loading test and train data...")
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    X_train = train_df.drop(columns=['Churn'])
    y_train = train_df['Churn']
    X_test = test_df.drop(columns=['Churn'])
    y_test = test_df['Churn']

    print("Training final XGBoost model for explainability...")
    model = XGBClassifier(eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    model_path = MODELS_DIR / "churn_xgb_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    print("SHAP analysis complete! Values computed for test set.")
    print(f"SHAP explanation matrix shape: {shap_values.shape}")

if __name__ == "__main__":
    explain_predictions()