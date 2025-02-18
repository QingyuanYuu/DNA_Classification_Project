import numpy as np
import os
import matplotlib.pyplot as plt

# self define the root
TRAIN_X = "/Users/jaaasnyu/Desktop/DNA_Classification_Project/data/processed/X_train.npy"
TEST_X = "/Users/jaaasnyu/Desktop/DNA_Classification_Project/data/processed/X_test.npy"
TRAIN_Y = "/Users/jaaasnyu/Desktop/DNA_Classification_Project/data/processed/y_train.npy"
TEST_Y = "/Users/jaaasnyu/Desktop/DNA_Classification_Project/data/processed/y_test.npy"

def load_data():
    """ loading NumPy data """
    if not all(os.path.exists(f) for f in [TRAIN_X, TEST_X, TRAIN_Y, TEST_Y]):
        print("❌ need to run split_data.py first")
        return
    
    X_train = np.load("/Users/jaaasnyu/Desktop/DNA_Classification_Project/data/processed/X_train.npy")
    X_test = np.load(TEST_X)
    y_train = np.load(TRAIN_Y)
    y_test = np.load(TEST_Y)

    # print the shape of the data
    print("✅ successfully loaded the data")
    print(f"🔹 X_train shape: {X_train.shape}")  # (sample shape, sequence length, 4)
    print(f"🔹 X_test shape: {X_test.shape}")    # (sample shape, sequence length, 4)
    print(f"🔹 y_train shape: {y_train.shape}")  # (sample shape,)
    print(f"🔹 y_test shape: {y_test.shape}")    # (sample shape,)

    # check the type discribution 
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"📊 Category distribution of the training set: {dict(zip(unique, counts))}")
    unique, counts = np.unique(y_test, return_counts=True)
    print(f"📊 Category distribution of the training set: {dict(zip(unique, counts))}")

    # Visualize One-hot encoding of a DNA sequence
    plt.imshow(X_train[0].T, cmap="viridis", aspect="auto")
    plt.colorbar(label="One-hot Encoding")
    plt.xlabel("DNA Sequence Position")
    plt.ylabel("A, T, C, G")
    plt.title("Example One-hot Encoding of DNA Sequence")
    plt.show()

if __name__ == "__main__":
    load_data()
