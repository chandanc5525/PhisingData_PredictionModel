import pandas as pd
import logging
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class DataIngestion:

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.label_encoder = LabelEncoder()

    def load_data(self):

        logging.info("Starting Data Ingestion")

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found at path: {self.file_path}")

        df = pd.read_csv(self.file_path)

        logging.info("Data Loaded Successfully. Shape: %s", df.shape)

        return df

    def split_data(self, df, target="status"):

        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataset")

        X = df.drop(columns=[target])
        y = df[target]

        # Encode target to numeric (0/1)
        y_encoded = self.label_encoder.fit_transform(y)

        # Save encoder for deployment
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(self.label_encoder, "artifacts/label_encoder.pkl")

        logging.info("Target Encoding Completed")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=0.3,
            random_state=42,
            stratify=y_encoded
        )

        logging.info("Train Test Split Completed")

        return X_train, X_test, y_train, y_test