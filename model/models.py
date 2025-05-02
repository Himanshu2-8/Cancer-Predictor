import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_cleaning.main import get_clean_data

def model_training():
    df=get_clean_data()
    X = df.drop(['diagnosis'], axis=1)
    y = df['diagnosis']
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models={
        "logistic Regression":LogisticRegression(),
        "SVC":SVC(),
        "Adaboost":AdaBoostClassifier(),
        "XGBoost":XGBClassifier()
    }

    for name,model in models.items():
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        print(f"\n{name} Results:")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

def main():
    model_training()

if __name__=="__main__":
    main()