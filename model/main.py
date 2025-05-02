import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_cleaning.main import get_clean_data

import pickle as pickle

def create_model():
    df=get_clean_data()
    X = df.drop(['diagnosis'], axis=1)
    y = df['diagnosis']
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model=SVC()
    model.fit(X_train,y_train)
    return model, scaler


def main():
    model, scaler = create_model()

    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

if __name__=="__main__":
    main()