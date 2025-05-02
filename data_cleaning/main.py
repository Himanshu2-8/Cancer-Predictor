import pandas as pd
import os

def get_clean_data():
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, "data.csv")
    
    df = pd.read_csv(data_path)
    df = df.drop(['Unnamed: 32', 'id'], axis=1)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df

def main():
    df=get_clean_data()

if __name__=="__main__":
    main()