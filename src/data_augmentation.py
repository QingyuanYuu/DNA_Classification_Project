import pandas as pd
import random
import numpy as np

# Read the origial Data
DATA_FILE = "/Users/jaaasnyu/Desktop/DNA_Classification_Project/data/processed/E_coli_DNA_sequences.csv"
OUTPUT_FILE = "/Users/jaaasnyu/Desktop/DNA_Classification_Project/data/processed/E_coli_DNA_sequences_augmented.csv"

# Complementary base mapping
complement_map = str.maketrans("ATCG", "TAGC")

def generate_complementary_sequence(seq):
    """ Generation of DNA complementary strands """
    return seq.translate(complement_map)[::-1] 

def mutate_sequence(seq, mutation_rate=0.02):
    """ Random mutations are performed """
    seq_list = list(seq)
    bases = ['A', 'T', 'C', 'G']
    
    for i in range(len(seq_list)):
        if random.random() < mutation_rate:  # 以 `mutation_rate` 概率突变
            seq_list[i] = random.choice([b for b in bases if b != seq_list[i]])

    return "".join(seq_list)

def augment_data():
    """ Random mutations are performed """
    df = pd.read_csv(DATA_FILE)
    augmented_sequences = []
    augmented_labels = []

    for seq, label in zip(df["sequence"], df["label"]):
        # Original Sequence
        augmented_sequences.append(seq)
        augmented_labels.append(label)

        # Complementary Chains
        comp_seq = generate_complementary_sequence(seq)
        augmented_sequences.append(comp_seq)
        augmented_labels.append(label)

        # Mutation Sequence
        mutated_seq = mutate_sequence(seq, mutation_rate=0.02)
        augmented_sequences.append(mutated_seq)
        augmented_labels.append(label)

    # Save new dataset
    df_aug = pd.DataFrame({"sequence": augmented_sequences, "label": augmented_labels})
    df_aug.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Augmented data saved to {OUTPUT_FILE} with {len(df_aug)} samples")

if __name__ == "__main__":
    augment_data()
