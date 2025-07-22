import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(csv_path):
    df = pd.read_csv(csv_path)

    
    if 'Employee_ID' in df.columns:
        df = df.drop(columns=['Employee_ID'])
    if 'Name' in df.columns:
        df = df.drop(columns=['Name'])

    
    label_encoders = {}
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    
    if 'Salary' not in df.columns:
        raise ValueError("Missing 'Salary' column. Please adjust the target column name.")

    X = df.drop(columns=['Salary'])
    y = df['Salary']

    return X, y, label_encoders
