import numpy as np
import pandas as pd

def clean_data(train, test):
    """
    Cleans and preprocesses train and test datasets.
    Returns:
        X (features), y (target), test (processed test set)
    """
    # Fill missing values in train
    train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
    train['Married'].fillna(train['Married'].mode()[0], inplace=True)
    train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
    train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
    train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
    train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
    train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

    # Fill missing values in test
    test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
    test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
    test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
    test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
    test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
    test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

    # Separate target variable BEFORE encoding
    y = train['Loan_Status'].map({'Y': 1, 'N': 0})  # Convert to 1/0
    X = train.drop(['Loan_Status', 'Loan_ID'], axis=1)
    test = test.drop('Loan_ID', axis=1)

    # Feature Engineering
    X['LoanAmount_log'] = np.log(X['LoanAmount'])
    test['LoanAmount_log'] = np.log(test['LoanAmount'])

    # One-hot encoding (only on features)
    X = pd.get_dummies(X)
    test = pd.get_dummies(test)

    # Align columns of test with X (sometimes missing dummies)
    test = test.reindex(columns=X.columns, fill_value=0)

    return X, y, test
