import pandas as pd

def create_submission(model, test, test_original):
    pred_test = model.predict(test)

    sample = pd.read_csv("data/sample.csv")
    sample['Loan_Status'] = pred_test
    sample['Loan_ID'] = test_original['Loan_ID']
    sample['Loan_Status'].replace(0, 'N', inplace=True)
    sample['Loan_Status'].replace(1, 'Y', inplace=True)

    sample[['Loan_ID', 'Loan_Status']].to_csv("submission/logistic.csv", index=False)
    print("📄 Submission file 'logistic.csv' created.")
