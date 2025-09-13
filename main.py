from src.data_loader import load_data
from src.data_cleaning import clean_data
from src.visualization import visualize_dataset
from src.model_training import train_and_validate
from src.submission import create_submission

def main():
    # Load data
    train, test, train_original, test_original = load_data()

    # Visualize and save figures
    visualize_dataset(train_original)

    # Clean and preprocess
    X, y, test = clean_data(train, test)

    # Train and validate
    model = train_and_validate(X, y)

    # Create submission file
    create_submission(model, test, test_original)

if __name__ == "__main__":
    main()
