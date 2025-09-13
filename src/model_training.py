import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
import numpy as np

def train_and_validate(X, y):
    """
    Train Logistic Regression with Stratified K-Fold CV and plot a single averaged ROC curve.
    """
    kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
    model = LogisticRegression(random_state=1, max_iter=1000)

    all_fpr, all_tpr, aucs = [], [], []

    for i, (train_index, test_index) in enumerate(kf.split(X, y), start=1):
        print(f"\n🔁 Fold {i} of {kf.n_splits}")
        xtr, xvl = X.iloc[train_index], X.iloc[test_index]
        ytr, yvl = y.iloc[train_index], y.iloc[test_index]

        model.fit(xtr, ytr)
        pred = model.predict(xvl)
        score = accuracy_score(yvl, pred)
        print(f"✅ Accuracy Score: {score:.4f}")

        pred_proba = model.predict_proba(xvl)[:, 1]
        fpr, tpr, _ = roc_curve(yvl, pred_proba)
        auc = roc_auc_score(yvl, pred_proba)

        all_fpr.append(fpr)
        all_tpr.append(tpr)
        aucs.append(auc)

    # Plot single combined ROC curve
    plt.figure(figsize=(8, 6))
    for i in range(len(all_fpr)):
        plt.plot(all_fpr[i], all_tpr[i], alpha=0.3, label=f"Fold {i+1} (AUC={aucs[i]:.3f})")

    # Average AUC
    avg_auc = np.mean(aucs)
    plt.title(f"ROC Curve (Average AUC={avg_auc:.3f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc=4)
    plt.show()

    return model
