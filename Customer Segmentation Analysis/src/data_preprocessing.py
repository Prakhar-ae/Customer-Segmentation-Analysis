"""
Data Preprocessing Module
Handles data cleaning, feature engineering, and normalization
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pathlib import Path

class DataPreprocessor:
    """Handle all data preprocessing operations"""
    
    def __init__(self, config):
        """
        Initialize preprocessor with configuration
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.scaler = None
        self.feature_names = None
        
    def load_data(self, filepath):
        """
        Load data from CSV
        
        Parameters:
        -----------
        filepath : str
            Path to CSV file
            
        Returns:
        --------
        pd.DataFrame
            Loaded dataframe
        """
        print(f"\n📂 Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"   ✓ Loaded {len(df)} records with {len(df.columns)} features")
        return df
    
    def explore_data(self, df):
        """Print data exploration statistics"""
        print("\n🔍 DATA EXPLORATION")
        print("-" * 60)
        print(f"\nDataset Shape: {df.shape}")
        print(f"\nData Types:\n{df.dtypes}")
        print(f"\nMissing Values:\n{df.isnull().sum()}")
        print(f"\nBasic Statistics:\n{df.describe()}")
        
    def select_clustering_features(self, df):
        """
        Select numerical features for clustering
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Features for clustering
        """
        # Select numerical features for clustering
        clustering_features = ['Age', 'Annual_Income_K', 'Spending_Score', 
                              'Purchase_Frequency', 'Avg_Order_Value', 
                              'Customer_Lifetime_Value', 'Years_Customer']
        
        X = df[clustering_features].copy()
        self.feature_names = clustering_features
        
        print(f"\n✓ Selected {len(clustering_features)} features for clustering:")
        for feat in clustering_features:
            print(f"  - {feat}")
        
        return X
    
    def handle_missing_values(self, X):
        """
        Handle missing values using appropriate strategies
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with handled missing values
        """
        if X.isnull().sum().sum() == 0:
            print("\n✓ No missing values detected")
            return X
        
        print(f"\n⚠️  Found missing values:\n{X.isnull().sum()}")
        
        # Fill missing values with median
        X = X.fillna(X.median())
        print("✓ Missing values filled with median")
        
        return X
    
    def remove_outliers(self, X, method='iqr'):
        """
        Detect and remove outliers
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature dataframe
        method : str
            'iqr' for IQR method or 'zscore' for Z-score
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with outliers removed
        int
            Number of outliers removed
        """
        original_size = len(X)
        
        if method == 'iqr':
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            mask = ~((X < lower_bound) | (X > upper_bound)).any(axis=1)
            X = X[mask]
            
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(X))
            mask = (z_scores < 3).all(axis=1)
            X = X[mask]
        
        removed = original_size - len(X)
        if removed > 0:
            print(f"\n⚠️  Removed {removed} outliers using {method} method")
        else:
            print(f"\n✓ No outliers detected using {method} method")
        
        return X
    
    def normalize_features(self, X, fit=True):
        """
        Normalize features using specified method
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature dataframe
        fit : bool
            Whether to fit the scaler (True for training, False for test)
            
        Returns:
        --------
        np.ndarray
            Normalized features
        """
        norm_method = self.config['preprocessing']['normalization_method']
        
        if norm_method == 'standardscaler':
            if fit:
                self.scaler = StandardScaler()
                X_normalized = self.scaler.fit_transform(X)
                print("\n✓ Features normalized using StandardScaler")
            else:
                X_normalized = self.scaler.transform(X)
        
        elif norm_method == 'minmax':
            if fit:
                self.scaler = MinMaxScaler()
                X_normalized = self.scaler.fit_transform(X)
                print("✓ Features normalized using MinMaxScaler")
            else:
                X_normalized = self.scaler.transform(X)
        
        else:
            raise ValueError(f"Unknown normalization method: {norm_method}")
        
        return X_normalized
    
    def preprocess(self, df, handle_outliers=False):
        """
        Complete preprocessing pipeline
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        handle_outliers : bool
            Whether to remove outliers
            
        Returns:
        --------
        np.ndarray
            Normalized features
        pd.DataFrame
            Original dataframe with selected features
        """
        print("\n" + "=" * 60)
        print("DATA PREPROCESSING PIPELINE")
        print("=" * 60)
        
        # Explore data
        self.explore_data(df)
        
        # Select features
        X = self.select_clustering_features(df)
        
        # Handle missing values
        X = self.handle_missing_values(X)
        
        # Remove outliers if configured
        if handle_outliers or self.config['preprocessing']['outlier_detection']:
            X = self.remove_outliers(X, method='iqr')
        
        # Normalize features
        X_normalized = self.normalize_features(X, fit=True)
        
        print("\n✓ Preprocessing complete!")
        print(f"  Final shape: {X_normalized.shape}")
        print(f"  Features: {self.feature_names}")
        
        return X_normalized, X

def main():
    """Test preprocessing pipeline"""
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    preprocessor = DataPreprocessor(config)
    
    # Load and preprocess data
    df = preprocessor.load_data('./data/customers.csv')
    X_normalized, X = preprocessor.preprocess(df)
    
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"\nOriginal data shape: {df.shape}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Normalized data shape: {X_normalized.shape}")
    print(f"\nNormalized data statistics:")
    print(f"  Mean (should be ~0): {X_normalized.mean(axis=0).round(3)}")
    print(f"  Std (should be ~1): {X_normalized.std(axis=0).round(3)}")
    
    return X_normalized, X, df

if __name__ == "__main__":
    X_normalized, X, df = main()
