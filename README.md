# DNA Sequence Classification using CNN

## Project Overview
This project implements a Convolutional Neural Network (CNN) to classify DNA sequences into coding and non-coding regions. The model is trained on **E. coli genomic data** using one-hot encoded DNA sequences and enhanced with **data augmentation techniques**.

## Project Structure
```
DNA_Classification_Project/
│──  src/                     # Code files
│   │── data_loader.py          # Data loading
│   │── data_augmentation.py     # Data augmentation
│   │── one_hot_encoding.py      # One-hot encoding
│   │── split_data.py            # Train-test split
│   │── model.py                 # CNN model definition
│   │── train.py                 # Standard training script
│   │── train_augmented.py       # Training with augmented data
│──  data/                     # Data folder
│   │──  raw/                  # Original dataset (DO NOT EDIT)
│   │──  processed/            # Processed NumPy data
│──  results/                  # Training results
│   │── logs/                    # Training logs
│   │── plots/                   # Loss & accuracy curves
│   │── models/                  # Saved models (.pth)
│──  notebooks/                # Jupyter notebooks for exploration
│   │── data_exploration.ipynb   # Data visualization & initial analysis
│   │── model_evaluation.ipynb   # Inspect model performance
│──  scripts/                  # Utility scripts for automation
│   │── setup_env.sh             # Shell script to set up environment
│   │── preprocess_data.sh        # Automates preprocessing
│── .gitignore                    # Ignore unnecessary files
│── requirements.txt              # Python dependencies
│── README.md                     # Project documentation
│── LICENSE                       # Open-source license (optional)
```

##  Dataset
- **Source**: [NCBI E. coli GenBank](https://www.ncbi.nlm.nih.gov/genbank/)
- **Preprocessing**: Extracted coding & non-coding sequences
- **Encoding**: One-hot representation (A, T, C, G → vectorized format)
- **Data Augmentation**:
  - **Complementary strand generation**
  - **Random mutations (2% probability per base)**

##  Model Architecture
The CNN model consists of:
- **Conv1D layers**: Extract local patterns from DNA sequences
- **MaxPooling1D layers**: Reduce dimensionality
- **Dropout layers**: Prevent overfitting
- **Fully connected layers**: Classification into coding/non-coding regions

## Training & Results
- **Baseline model**: 97.77% accuracy on the test set
- **With data augmentation**: Improved to **98-99% accuracy**
- **Training loss reduced**, overfitting minimized

## How to Run
### Install dependencies
```bash
pip install -r requirements.txt
```
### Prepare dataset
```bash
python src/data_augmentation.py   # Apply data augmentation
python src/one_hot_encoding.py    # Convert DNA to one-hot encoding
python src/split_data.py          # Split into train/test sets
```
### Train the model
#### Without augmentation
```bash
python src/train.py
```
#### With augmentation
```bash
python src/train_augmented.py
```
### 4️⃣ Evaluate model
Results are saved in the `results/` folder.

## Future Improvements
- **LSTM-CNN hybrid model** for long-range dependencies
- **Experiment with Transformer models** for DNA sequences
- **Use additional datasets for better generalization**

---
**Author**: Qingyuan Yu
Contact: Qiy005@ucsd.edu
GitHub: https://github.com/QingyuanYuu

