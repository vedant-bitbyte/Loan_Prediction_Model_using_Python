import os
import matplotlib.pyplot as plt
import seaborn as sns

def save_plot(fig, filename, folder="figures"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    fig.savefig(os.path.join(folder, filename), bbox_inches="tight")
    plt.close(fig)

def visualize_dataset(train):
    print("📊 Generating dataset visualizations...")

    # Loan Status
    fig, ax = plt.subplots()
    train['Loan_Status'].value_counts().plot.bar(ax=ax, color="skyblue")
    ax.set_title("Loan Status Distribution")
    save_plot(fig, "loan_status_distribution.png")

    # Categorical
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    train['Gender'].value_counts(normalize=True).plot.bar(ax=axes[0, 0], title="Gender")
    train['Married'].value_counts(normalize=True).plot.bar(ax=axes[0, 1], title="Married")
    train['Self_Employed'].value_counts(normalize=True).plot.bar(ax=axes[1, 0], title="Self Employed")
    train['Credit_History'].value_counts(normalize=True).plot.bar(ax=axes[1, 1], title="Credit History")
    fig.tight_layout()
    save_plot(fig, "categorical_features.png")

    # Ordinal
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    train['Dependents'].value_counts(normalize=True).plot.bar(ax=axes[0], title="Dependents")
    train['Education'].value_counts(normalize=True).plot.bar(ax=axes[1], title="Education")
    train['Property_Area'].value_counts(normalize=True).plot.bar(ax=axes[2], title="Property Area")
    fig.tight_layout()
    save_plot(fig, "ordinal_features.png")

    # Income distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(train['ApplicantIncome'], kde=True, ax=axes[0])
    axes[0].set_title("Applicant Income Distribution")
    train['ApplicantIncome'].plot.box(ax=axes[1])
    axes[1].set_title("Applicant Income Boxplot")
    fig.tight_layout()
    save_plot(fig, "applicant_income.png")

    # Loan Amount
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(train['LoanAmount'], kde=True, ax=axes[0])
    axes[0].set_title("Loan Amount Distribution")
    train['LoanAmount'].plot.box(ax=axes[1])
    axes[1].set_title("Loan Amount Boxplot")
    fig.tight_layout()
    save_plot(fig, "loan_amount.png")

    print("✅ Visualizations saved in 'figures/' folder.")
