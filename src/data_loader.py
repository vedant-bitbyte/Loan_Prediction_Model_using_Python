import pandas as pd

def load_data():
    train = pd.read_csv("data/train_ctrUa4K.csv")
    test = pd.read_csv("data/test_lAUu6dG.csv")
    train_original = train.copy()
    test_original = test.copy()
    return train, test, train_original, test_original
