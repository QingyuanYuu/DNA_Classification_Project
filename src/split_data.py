import numpy as np
import os
from sklearn.model_selection import train_test_split

# Define the data path
INPUT_X = "/Users/jaaasnyu/Desktop/DNA_Classification_Project/data/processed/X.npy"
INPUT_Y = "/Users/jaaasnyu/Desktop/DNA_Classification_Project/data/processed/y.npy"
OUTPUT_TRAIN_X = "/Users/jaaasnyu/Desktop/DNA_Classification_Project/data/processed/X_train.npy"
OUTPUT_TEST_X = "/Users/jaaasnyu/Desktop/DNA_Classification_Project/data/processed/X_test.npy"
OUTPUT_TRAIN_Y = "/Users/jaaasnyu/Desktop/DNA_Classification_Project/data/processed/y_train.npy"
OUTPUT_TEST_Y = "/Users/jaaasnyu/Desktop/DNA_Classification_Project/data/processed/y_test.npy"

def split_data():
    """
    Load One-hot encoded data and divide training sets & test sets
    """
    # make sure the dirs exits
    os.makedirs("/Users/jaaasnyu/Desktop/DNA_Classification_Project/data/processed", exist_ok=True)

    # loading dataset
    print("📥 Loading data...")
    X = np.load(INPUT_X)
    y = np.load(INPUT_Y)

    # Dividing the dataset (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # save the dataset
    np.save(OUTPUT_TRAIN_X, X_train)
    np.save(OUTPUT_TEST_X, X_test)
    np.save(OUTPUT_TRAIN_Y, y_train)
    np.save(OUTPUT_TEST_Y, y_test)

    print(f"✅ Data split complete:")
    print(f"   - Training samples: {X_train.shape[0]}")
    print(f"   - Testing samples: {X_test.shape[0]}")
    print(f"   - Data saved in ../data/processed/")

if __name__ == "__main__":
    split_data()
