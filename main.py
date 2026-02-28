# Files in which we are joining DataCleaning and DataPreprocessing

import pandas as pd
import os
from components.data_cleaning import DataCleaning
from components.data_preprocessing import DataPreprocessing

# Load raw data
df = pd.read_csv("data/raw_data.csv")

# Cleaning
cleaner = DataCleaning(df)
df_clean = cleaner.clean_data()

# Preprocessing
preprocessor_obj = DataPreprocessing(df_clean)
X_train, X_test, y_train, y_test, preprocessor = preprocessor_obj.preprocess()

# Create data folder if not exists
os.makedirs("data", exist_ok=True)

# Save datasets
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("Train/Test data saved successfully")