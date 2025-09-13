# Loan_Prediction_Model_using_Python

This project predicts loan approval status using **Logistic Regression** on the provided dataset.  
It follows a modular structure for data cleaning, model training, evaluation (including ROC curve), and submission file generation.

---

## 🛠 Tech Stack
- Python
- Pandas / NumPy / Matplotlib / Seaborn
- scikit-learn (Logistic Regression, Stratified K-Fold, ROC-AUC)

---

## ⚙️ Setup & Run

```bash
# 1️⃣ Create virtual environment
python -m venv venv
venv\Scripts\activate     # (Windows)

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Run the project
python main.py
```

---

## 📊 Output
- Figures are saved in figures/ (ROC curve, distributions, etc.) 
- Final predictions are stored in submission/logistic.csv

---

## 📌 Notes
- figures/ is ignored in git — regenerate by running the code.