"""
Clustering Module
Implements K-Means clustering for customer segmentation
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pickle
from pathlib import Path

class CustomerSegmentationModel:
    """K-Means clustering model for customer segmentation"""
    
    def __init__(self, config):
        """
        Initialize the clustering model
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.model = None
        self.optimal_k = None
        self.inertias = []
        self.silhouette_scores = []
        self.cluster_labels = None
        
    def find_optimal_k(self, X):
        """
        Find optimal number of clusters using Elbow Method and Silhouette Score
        
        Parameters:
        -----------
        X : np.ndarray
            Normalized feature matrix
            
        Returns:
        --------
        int
            Optimal number of clusters
        """
        print("\n" + "=" * 60)
        print("FINDING OPTIMAL NUMBER OF CLUSTERS")
        print("=" * 60)
        
        k_range = range(self.config['clustering']['optimal_k_range'][0],
                       self.config['clustering']['optimal_k_range'][1] + 1)
        
        self.inertias = []
        self.silhouette_scores = []
        
        print("\nTesting different K values...")
        print(f"{'K':<5} {'Inertia':<15} {'Silhouette':<15} {'Davies-Bouldin':<15}")
        print("-" * 50)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k,
                          random_state=self.config['clustering']['random_state'],
                          n_init=self.config['clustering']['n_init'],
                          max_iter=self.config['clustering']['max_iterations'])
            
            labels = kmeans.fit_predict(X)
            
            inertia = kmeans.inertia_
            silhouette = silhouette_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)
            
            self.inertias.append(inertia)
            self.silhouette_scores.append(silhouette)
            
            print(f"{k:<5} {inertia:<15.2f} {silhouette:<15.4f} {davies_bouldin:<15.4f}")
        
        # Find optimal k based on silhouette score (highest is best)
        optimal_k = k_range[np.argmax(self.silhouette_scores)]
        
        print(f"\n📊 Analysis Results:")
        print(f"   Optimal K (Silhouette): {optimal_k}")
        print(f"   Max Silhouette Score: {max(self.silhouette_scores):.4f}")
        print(f"   Using K: {self.config['clustering']['final_k']}")
        
        return optimal_k
    
    def train(self, X):
        """
        Train K-Means clustering model
        
        Parameters:
        -----------
        X : np.ndarray
            Normalized feature matrix
            
        Returns:
        --------
        np.ndarray
            Cluster labels for each sample
        """
        print("\n" + "=" * 60)
        print("TRAINING K-MEANS CLUSTERING MODEL")
        print("=" * 60)
        
        k = self.config['clustering']['final_k']
        
        print(f"\nTraining K-Means with K={k}...")
        
        self.model = KMeans(
            n_clusters=k,
            random_state=self.config['clustering']['random_state'],
            n_init=self.config['clustering']['n_init'],
            max_iter=self.config['clustering']['max_iterations'],
            verbose=0
        )
        
        self.cluster_labels = self.model.fit_predict(X)
        
        # Calculate evaluation metrics
        inertia = self.model.inertia_
        silhouette = silhouette_score(X, self.cluster_labels)
        davies_bouldin = davies_bouldin_score(X, self.cluster_labels)
        calinski_harabasz = calinski_harabasz_score(X, self.cluster_labels)
        
        print(f"\n✓ Model trained successfully!")
        print(f"\nEvaluation Metrics:")
        print(f"   Inertia (WCSS): {inertia:.2f}")
        print(f"   Silhouette Score: {silhouette:.4f}")
        print(f"   Davies-Bouldin Index: {davies_bouldin:.4f}")
        print(f"   Calinski-Harabasz Score: {calinski_harabasz:.2f}")
        print(f"\n   Cluster Distribution:")
        
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            percentage = (count / len(self.cluster_labels)) * 100
            print(f"      Cluster {cluster_id}: {count:4d} customers ({percentage:5.1f}%)")
        
        return self.cluster_labels
    
    def get_cluster_centers(self):
        """
        Get cluster center coordinates
        
        Returns:
        --------
        np.ndarray
            Cluster centers in original feature space
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.model.cluster_centers_
    
    def predict(self, X):
        """
        Predict cluster labels for new data
        
        Parameters:
        -----------
        X : np.ndarray
            Normalized feature matrix
            
        Returns:
        --------
        np.ndarray
            Cluster labels
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.model.predict(X)
    
    def save_model(self, filepath):
        """
        Save trained model to disk
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"\n✓ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained model from disk
        
        Parameters:
        -----------
        filepath : str
            Path to load the model from
        """
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"\n✓ Model loaded from {filepath}")

class SegmentAnalyzer:
    """Analyze customer segments"""
    
    def __init__(self, df, X, cluster_labels, feature_names):
        """
        Initialize segment analyzer
        
        Parameters:
        -----------
        df : pd.DataFrame
            Original dataframe
        X : pd.DataFrame
            Unnormalized features used for clustering
        cluster_labels : np.ndarray
            Cluster assignment for each customer
        feature_names : list
            Names of clustering features
        """
        self.df = df.copy()
        self.X = X.copy()
        self.cluster_labels = cluster_labels
        self.feature_names = feature_names
        self.df['Segment'] = cluster_labels
        
    def get_segment_profiles(self):
        """
        Generate statistical profiles for each segment
        
        Returns:
        --------
        pd.DataFrame
            Segment profiles
        """
        print("\n" + "=" * 60)
        print("CUSTOMER SEGMENT PROFILES")
        print("=" * 60)
        
        profiles = []
        
        for segment in sorted(self.df['Segment'].unique()):
            segment_data = self.df[self.df['Segment'] == segment]
            
            profile = {
                'Segment': segment,
                'Size': len(segment_data),
                'Percentage': f"{(len(segment_data)/len(self.df)*100):.1f}%",
                'Avg_Age': f"{segment_data['Age'].mean():.1f}",
                'Avg_Income': f"{segment_data['Annual_Income_K'].mean():.1f}K",
                'Avg_Spending_Score': f"{segment_data['Spending_Score'].mean():.1f}",
                'Avg_Purchase_Freq': f"{segment_data['Purchase_Frequency'].mean():.1f}",
                'Avg_CLV': f"${segment_data['Customer_Lifetime_Value'].mean():.0f}",
            }
            profiles.append(profile)
        
        profile_df = pd.DataFrame(profiles)
        print("\n" + profile_df.to_string(index=False))
        
        return profile_df
    
    def get_segment_characteristics(self):
        """
        Detailed characteristics of each segment
        
        Returns:
        --------
        dict
            Detailed segment characteristics
        """
        characteristics = {}
        
        for segment in sorted(self.df['Segment'].unique()):
            segment_data = self.df[self.df['Segment'] == segment]
            
            characteristics[segment] = {
                'size': len(segment_data),
                'age_range': (segment_data['Age'].min(), segment_data['Age'].max()),
                'income_range': (segment_data['Annual_Income_K'].min(), 
                               segment_data['Annual_Income_K'].max()),
                'top_category': segment_data['Primary_Category'].mode()[0],
                'gender_split': segment_data['Gender'].value_counts().to_dict(),
                'avg_tenure': segment_data['Years_Customer'].mean(),
            }
        
        return characteristics
    
    def generate_business_insights(self):
        """
        Generate actionable business insights for each segment
        
        Returns:
        --------
        dict
            Business insights per segment
        """
        print("\n" + "=" * 60)
        print("BUSINESS INSIGHTS & RECOMMENDATIONS")
        print("=" * 60)
        
        insights = {}
        profiles = self.get_segment_profiles()
        
        for segment in sorted(self.df['Segment'].unique()):
            segment_data = self.df[self.df['Segment'] == segment]
            profile_row = profiles[profiles['Segment'] == segment].iloc[0]
            
            size = len(segment_data)
            avg_income = segment_data['Annual_Income_K'].mean()
            avg_spending = segment_data['Spending_Score'].mean()
            avg_age = segment_data['Age'].mean()
            clv = segment_data['Customer_Lifetime_Value'].mean()
            
            # Generate insight based on characteristics
            if avg_income > 100 and avg_spending > 70:
                segment_name = "Premium High-Value"
                strategy = "VIP treatment, exclusive offers, personalized service"
            elif avg_income > 80 and avg_age < 40:
                segment_name = "Young Professionals"
                strategy = "Digital marketing, trendy products, convenience focus"
            elif avg_spending > 75 and avg_income < 60:
                segment_name = "Budget-Conscious Spenders"
                strategy = "Value deals, bundled offers, loyalty rewards"
            elif avg_age > 50:
                segment_name = "Established Customers"
                strategy = "Trust-building, quality focus, long-term relationships"
            else:
                segment_name = "Growth Potential"
                strategy = "Engagement campaigns, product discovery, conversion focus"
            
            insights[segment] = {
                'name': segment_name,
                'size': size,
                'percentage': (size / len(self.df)) * 100,
                'characteristics': f"Age: {avg_age:.0f}yr, Income: ${avg_income*1000:.0f}, CLV: ${clv:.0f}",
                'strategy': strategy,
                'priority': 'High' if clv > self.df['Customer_Lifetime_Value'].mean() else 'Medium'
            }
            
            print(f"\n📊 Segment {segment}: {segment_name}")
            print(f"   Size: {size} customers ({(size/len(self.df)*100):.1f}%)")
            print(f"   Profile: {insights[segment]['characteristics']}")
            print(f"   Strategy: {strategy}")
            print(f"   Priority: {insights[segment]['priority']}")
        
        return insights

def main():
    """Main execution"""
    import yaml
    from data_generation import generate_customer_data
    from data_preprocessing import DataPreprocessor
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate and preprocess data
    df = generate_customer_data(config)
    preprocessor = DataPreprocessor(config)
    X_normalized, X = preprocessor.preprocess(df)
    
    # Train clustering model
    model = CustomerSegmentationModel(config)
    optimal_k = model.find_optimal_k(X_normalized)
    cluster_labels = model.train(X_normalized)
    
    # Analyze segments
    analyzer = SegmentAnalyzer(df, X, cluster_labels, preprocessor.feature_names)
    profiles = analyzer.get_segment_profiles()
    insights = analyzer.generate_business_insights()
    
    # Save model
    if config['output']['save_model']:
        model.save_model(config['output']['model_file'])
    
    return model, analyzer, df

if __name__ == "__main__":
    model, analyzer, df = main()
