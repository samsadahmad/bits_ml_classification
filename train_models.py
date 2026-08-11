import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class MLClassificationPipeline:
    def __init__(self, random_state=42):
        """Initialize empty container for Data, model and results"""
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    #Load data (EDA)
    def load_dataset(self):
        """Load data from Skitlearn for Breast Cancer dataset and 
          Returns the input measurements (X) and the correct answers (y)
        """
        print("Loading Breast Cancer dataset from sklearn...")
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name='target')

        print(f"Dataset shape: {X.shape}")
        print(f"Features: {X.shape[1]}, Instances: {X.shape[0]}")
        print(f"Classes: {np.unique(y)}")

        # Save the data to csv files for later use
        dataset = X.copy()
        dataset['target'] = y
        dataset.to_csv('dataset.csv', index=False)

        return X, y

    def preprocess_data(self, X, y):
        """Preprocess the data by splitting into training and testing sets"""
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        # Standarize features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print(f"Training set shape: {self.X_train.shape}, Testing set shape: {self.X_test.shape}")


def main():
    pipeline = MLClassificationPipeline()
    X, y = pipeline.load_data()
    print("Data loaded successfully.")
        
