import numpy as np
import pandas as pd
import os

# define the root
DATA_FILE = "/Users/jaaasnyu/Desktop/DNA_Classification_Project/data/processed/E_coli_DNA_sequences_augmented.csv"
OUTPUT_X = "/Users/jaaasnyu/Desktop/DNA_Classification_Project/data/processed/X.npy"
OUTPUT_Y = "/Users/jaaasnyu/Desktop/DNA_Classification_Project/data/processed/y.npy"

# define One-hot Mapping dictionaries
one_hot_map = {
    "A": [1, 0, 0, 0],
    "T": [0, 1, 0, 0],
    "C": [0, 0, 1, 0],
    "G": [0, 0, 0, 1]
}

def one_hot_encode_sequence(sequence, max_length):
    """
    transer DNA sequence as One-hot matrix, enhence to max_length
    """
    one_hot_seq = np.zeros((max_length, 4))  # initialize the matrix
    for i, base in enumerate(sequence[:max_length]):  # give the maximum limit
        if base in one_hot_map:
            one_hot_seq[i] = one_hot_map[base]
    return one_hot_seq

def process_data():
    """
    read CSV, transfer DNA sequence as One-hot matrix, and save as NumPy
    """
    os.makedirs("/Users/jaaasnyu/Desktop/DNA_Classification_Project/data/processed", exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    sequences = df["sequence"].values
    labels = df["label"].values

    # Set the maximum sequence length (take the 95th percentile)
    max_length = int(np.percentile([len(seq) for seq in sequences], 95))
    print(f"📏 Max sequence length (95th percentile): {max_length}")

    # One-hot
    X = np.array([one_hot_encode_sequence(seq, max_length) for seq in sequences])
    y = np.array(labels)

    # save NumPy 
    np.save(OUTPUT_X, X)
    np.save(OUTPUT_Y, y)
    print(f"✅ One-hot encoded data saved: {OUTPUT_X}, {OUTPUT_Y}")

if __name__ == "__main__":
    process_data()

