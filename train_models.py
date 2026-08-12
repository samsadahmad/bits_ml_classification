import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

class MLClassificationPipeline:
    """Run the complete process of preparing data, training models, and saving results."""

    def __init__(self, random_state=42):
        """Set up empty containers for the data, models, and results."""
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        
    def load_dataset(self):
        """Load the breast cancer data and save a copy as a CSV file.

        Returns the input measurements (X) and the correct answers (y).
        """
        print("Loading Breast Cancer dataset...")
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name='target')
        
        print(f"Dataset shape: {X.shape}")
        print(f"Features: {X.shape[1]}, Instances: {X.shape[0]}")
        print(f"Classes: {np.unique(y)}")
        
        # Save raw dataset
        dataset = X.copy()
        dataset['target'] = y
        dataset.to_csv('dataset.csv', index=False)
        
        return X, y
    
    def preprocess_data(self, X, y, test_size=0.2):
        """Split the data into practice and test sets, then put features on a similar scale."""
        print("\nPreprocessing data...")
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Standardize features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"Train set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")

    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Compare the model's guesses with the correct answers using several scores."""
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'AUC': roc_auc_score(y_true, y_pred_proba[:, 1]) if y_pred_proba is not None else None,
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1': f1_score(y_true, y_pred, zero_division=0),
            'MCC': matthews_corrcoef(y_true, y_pred)
        }
        return metrics
    
    def train_logistic_regression(self):
        """Teach a Logistic Regression model and record how well it performs."""
        print("\nTraining Logistic Regression...")
        model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)
        
        metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
        self.models['Logistic Regression'] = model
        self.results['Logistic Regression'] = {
            'metrics': metrics,
            'y_pred': y_pred,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'classification_report': classification_report(self.y_test, y_pred)
        }
        print(f"Logistic Regression - Accuracy: {metrics['Accuracy']:.4f}, AUC: {metrics['AUC']:.4f}")
    
    def train_decision_tree(self):
        """Teach a Decision Tree model and record how well it performs."""
        print("\nTraining Decision Tree Classifier...")
        model = DecisionTreeClassifier(random_state=self.random_state)
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)
        
        metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
        self.models['Decision Tree'] = model
        self.results['Decision Tree'] = {
            'metrics': metrics,
            'y_pred': y_pred,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'classification_report': classification_report(self.y_test, y_pred)
        }
        print(f"Decision Tree - Accuracy: {metrics['Accuracy']:.4f}, AUC: {metrics['AUC']:.4f}")
    
    def train_knn(self):
        """Teach a K-Nearest Neighbor model and record how well it performs."""
        print("\nTraining K-Nearest Neighbor Classifier...")
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)
        
        metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
        self.models['K-Nearest Neighbor'] = model
        self.results['K-Nearest Neighbor'] = {
            'metrics': metrics,
            'y_pred': y_pred,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'classification_report': classification_report(self.y_test, y_pred)
        }
        print(f"K-Nearest Neighbor - Accuracy: {metrics['Accuracy']:.4f}, AUC: {metrics['AUC']:.4f}")
    
    def train_naive_bayes(self):
        """Teach a Naive Bayes model and record how well it performs."""
        print("\nTraining Naive Bayes Classifier...")
        model = GaussianNB()
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)
        
        metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
        self.models['Naive Bayes'] = model
        self.results['Naive Bayes'] = {
            'metrics': metrics,
            'y_pred': y_pred,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'classification_report': classification_report(self.y_test, y_pred)
        }
        print(f"Naive Bayes - Accuracy: {metrics['Accuracy']:.4f}, AUC: {metrics['AUC']:.4f}")
    
    def train_random_forest(self):
        """Teach a Random Forest made of many decision trees and record its performance."""
        print("\nTraining Random Forest Ensemble...")
        model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)
        
        metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
        self.models['Random Forest'] = model
        self.results['Random Forest'] = {
            'metrics': metrics,
            'y_pred': y_pred,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'classification_report': classification_report(self.y_test, y_pred)
        }
        print(f"Random Forest - Accuracy: {metrics['Accuracy']:.4f}, AUC: {metrics['AUC']:.4f}")
    
    def train_all_models(self):
        """Train each available model so their results can be compared."""
        print("="*60)
        print("Starting Model Training Pipeline")
        print("="*60)
        
        self.train_logistic_regression()
        self.train_decision_tree()
        self.train_knn()
        self.train_naive_bayes()
        self.train_random_forest()
        
        print("\n" + "="*60)
        print("All models trained successfully!")
        print("="*60)
    
    def save_models(self, model_dir='model'):
        """Save each trained model and the feature scaler as files for later use."""
        print("\nSaving models...")
        os.makedirs(model_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            filename = f"{model_dir}/{model_name.replace(' ', '_').lower()}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved: {filename}")
        
        # Save scaler
        with open(f"{model_dir}/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Saved: {model_dir}/scaler.pkl")
    
    def save_results_csv(self):
        """Save every model's scores in a CSV file for the Streamlit app.

        Returns a table containing the performance scores.
        """
        print("\nSaving results to CSV...")
        
        results_data = []
        for model_name, result in self.results.items():
            metrics = result['metrics']
            results_data.append({
                'Model': model_name,
                'Accuracy': metrics['Accuracy'],
                'AUC': metrics['AUC'],
                'Precision': metrics['Precision'],
                'Recall': metrics['Recall'],
                'F1': metrics['F1'],
                'MCC': metrics['MCC']
            })
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv('model_results.csv', index=False)
        print("Saved: model_results.csv")
        print("\nModel Performance Summary:")
        print(results_df.to_string(index=False))
        
        return results_df
    
    def save_test_data(self):
        """Save the test examples, their correct answers, and each model's guesses."""
        print("\nSaving test data...")
        
        # Create test data CSV with predictions from all models
        test_data = self.X_test.copy()
        test_data = pd.DataFrame(test_data)
        test_data['Actual'] = self.y_test.values
        
        # Add predictions from each model
        for model_name, result in self.results.items():
            test_data[f'{model_name}_Prediction'] = result['y_pred']
        
        test_data.to_csv('test_data.csv', index=False)
        print("Saved: test_data.csv")
    
    def generate_report(self):
        """Print detailed results showing correct and incorrect predictions for each model."""
        print("\n" + "="*60)
        print("DETAILED CLASSIFICATION REPORTS")
        print("="*60)
        
        for model_name, result in self.results.items():
            print(f"\n{model_name}:")
            print(f"Confusion Matrix:\n{result['confusion_matrix']}\n")
            print(f"Classification Report:\n{result['classification_report']}")


def main():
    """Run the complete machine-learning workflow from start to finish."""
    # Change to project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Prepare the data, teach several models, and save their results for the app.
    # Initialize pipeline
    pipeline = MLClassificationPipeline()
    
    # Load dataset
    X, y = pipeline.load_dataset()
    
    # Preprocess data
    pipeline.preprocess_data(X, y)
    
    # Train all models
    pipeline.train_all_models()
    
    # Save models
    pipeline.save_models()
    
    # Save results
    pipeline.save_results_csv()
    
    # Save test data
    pipeline.save_test_data()
    
    # Generate reports
    pipeline.generate_report()
    
    print("\n" + "="*60)
    print("Pipeline Completed Successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
