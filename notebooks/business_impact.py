import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import confusion_matrix

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

def calculate_financial_impact():
    print("Loading test data and model...")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    model = joblib.load(MODELS_DIR / "churn_xgb_model.pkl")

    X_test = test_df.drop(columns=['Churn'])
    y_test = test_df['Churn']

    y_proba = model.predict_proba(X_test)[:, 1]
    threshold = 0.5
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    avg_customer_value = 500.0   
    retention_cost = 50.0        
    retention_success_rate = 0.4 

    retained_customers = tp * retention_success_rate
    gross_savings = retained_customers * avg_customer_value
    total_targeted = tp + fp
    total_campaign_cost = total_targeted * retention_cost
    net_savings = gross_savings - total_campaign_cost

    print("\n==============================================")
    print("💰 CHURNGUARD: FINANCIAL IMPACT SIMULATION")
    print("==============================================")
    print(f"Total Test Customers Evaluated : {len(y_test)}")
    print(f"True Churners Identified (TP)  : {tp}")
    print(f"False Alarms (FP)              : {fp}")
    print(f"Customers Successfully Retained: {retained_customers:.0f}")
    print("----------------------------------------------")
    print(f"Gross Revenue Saved            : ${gross_savings:,.2f}")
    print(f"Total Retention Campaign Cost  : ${total_campaign_cost:,.2f}")
    print(f"Net Financial Savings          : ${net_savings:,.2f}")
    print("==============================================\n")

if __name__ == "__main__":
    calculate_financial_impact()