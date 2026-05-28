"""
Visualization Module
Creates visualizations for clustering analysis and segment profiles
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

class ClusterVisualizer:
    """Create visualizations for cluster analysis"""
    
    def __init__(self, config):
        """
        Initialize visualizer
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.style = config['visualization']['style']
        self.dpi = config['visualization']['dpi']
        self.output_dir = Path(config['visualization']['plot_directory'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use(self.style)
    
    def plot_elbow_curve(self, inertias, k_range):
        """
        Plot elbow curve for optimal K selection
        
        Parameters:
        -----------
        inertias : list
            Inertia values for different K
        k_range : range
            Range of K values tested
        """
        print("\n📊 Creating elbow curve plot...")
        
        fig, ax = plt.subplots(figsize=self.config['visualization']['figsize_2d'])
        
        ax.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12, fontweight='bold')
        ax.set_title('Elbow Method For Optimal K', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        # Add annotations
        for i, inertia in enumerate(inertias):
            ax.annotate(f'{inertia:.0f}', 
                       xy=(k_range[i], inertia),
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', fontsize=9)
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'elbow_curve.png'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        print(f"   ✓ Saved to {filepath}")
        plt.close()
    
    def plot_clusters_2d(self, X, cluster_labels, centers=None, features=None):
        """
        Plot clusters in 2D using first two principal components
        
        Parameters:
        -----------
        X : np.ndarray
            Normalized feature matrix
        cluster_labels : np.ndarray
            Cluster assignments
        centers : np.ndarray, optional
            Cluster centers
        features : list, optional
            Feature names
        """
        print("\n📊 Creating 2D cluster visualization...")
        
        from sklearn.decomposition import PCA
        
        # Reduce to 2D using PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        if centers is not None:
            centers_pca = pca.transform(centers)
        
        fig, ax = plt.subplots(figsize=self.config['visualization']['figsize_2d'])
        
        # Define colors
        colors = sns.color_palette('husl', len(np.unique(cluster_labels)))
        
        # Plot clusters
        for cluster in np.unique(cluster_labels):
            mask = cluster_labels == cluster
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      c=[colors[cluster]], label=f'Cluster {cluster}',
                      s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Plot centers
        if centers is not None:
            ax.scatter(centers_pca[:, 0], centers_pca[:, 1],
                      c='red', marker='X', s=400, edgecolors='black', linewidth=2,
                      label='Centroids', zorder=5)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                     fontsize=11, fontweight='bold')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
                     fontsize=11, fontweight='bold')
        ax.set_title('Customer Clusters (2D PCA Projection)', fontsize=13, fontweight='bold', pad=15)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'clusters_2d.png'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        print(f"   ✓ Saved to {filepath}")
        plt.close()
    
    def plot_clusters_3d(self, X, cluster_labels, centers=None):
        """
        Plot clusters in 3D using first three principal components
        
        Parameters:
        -----------
        X : np.ndarray
            Normalized feature matrix
        cluster_labels : np.ndarray
            Cluster assignments
        centers : np.ndarray, optional
            Cluster centers
        """
        print("\n📊 Creating 3D cluster visualization...")
        
        from sklearn.decomposition import PCA
        
        # Reduce to 3D using PCA
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X)
        
        if centers is not None:
            centers_pca = pca.transform(centers)
        
        fig = plt.figure(figsize=self.config['visualization']['figsize_3d'])
        ax = fig.add_subplot(111, projection='3d')
        
        # Define colors
        colors = sns.color_palette('husl', len(np.unique(cluster_labels)))
        
        # Plot clusters
        for cluster in np.unique(cluster_labels):
            mask = cluster_labels == cluster
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                      c=[colors[cluster]], label=f'Cluster {cluster}',
                      s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Plot centers
        if centers is not None:
            ax.scatter(centers_pca[:, 0], centers_pca[:, 1], centers_pca[:, 2],
                      c='red', marker='X', s=400, edgecolors='black', linewidth=2,
                      label='Centroids', zorder=5)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=10, fontweight='bold')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=10, fontweight='bold')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})', fontsize=10, fontweight='bold')
        ax.set_title('Customer Clusters (3D PCA Projection)', fontsize=13, fontweight='bold', pad=20)
        ax.legend(fontsize=9, loc='best')
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'clusters_3d.png'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        print(f"   ✓ Saved to {filepath}")
        plt.close()
    
    def plot_cluster_distribution(self, cluster_labels):
        """
        Plot cluster size distribution
        
        Parameters:
        -----------
        cluster_labels : np.ndarray
            Cluster assignments
        """
        print("\n📊 Creating cluster distribution plot...")
        
        unique, counts = np.unique(cluster_labels, return_counts=True)
        percentages = (counts / len(cluster_labels)) * 100
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(unique, counts, color=sns.color_palette('husl', len(unique)),
                     edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # Add value labels on bars
        for bar, count, pct in zip(bars, counts, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}\n({pct:.1f}%)',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
        ax.set_title('Customer Distribution Across Clusters', fontsize=13, fontweight='bold', pad=15)
        ax.set_xticks(unique)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'cluster_distribution.png'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        print(f"   ✓ Saved to {filepath}")
        plt.close()
    
    def plot_segment_profiles(self, segment_profiles):
        """
        Plot segment profile comparison
        
        Parameters:
        -----------
        segment_profiles : pd.DataFrame
            Segment profile data
        """
        print("\n📊 Creating segment profiles heatmap...")
        
        # Select numeric columns
        profile_data = segment_profiles.copy()
        
        # Convert string columns to numeric
        for col in profile_data.columns:
            if col not in ['Segment', 'Percentage']:
                profile_data[col] = pd.to_numeric(profile_data[col], errors='coerce')
        
        profile_data = profile_data.set_index('Segment')
        profile_data = profile_data.drop('Percentage', axis=1, errors='ignore')
        
        # Normalize for heatmap
        profile_normalized = (profile_data - profile_data.min()) / (profile_data.max() - profile_data.min())
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.heatmap(profile_normalized.T, annot=profile_data.T.values, fmt='.1f',
                   cmap='RdYlGn', cbar_kws={'label': 'Normalized Value'},
                   linewidths=0.5, linecolor='gray', ax=ax)
        
        ax.set_title('Segment Profile Comparison (Normalized)', fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel('Segment', fontsize=12, fontweight='bold')
        ax.set_ylabel('Characteristics', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'segment_profiles_heatmap.png'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        print(f"   ✓ Saved to {filepath}")
        plt.close()
    
    def plot_feature_importance_by_cluster(self, X, cluster_labels, feature_names):
        """
        Plot feature importance/variation by cluster
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features dataframe
        cluster_labels : np.ndarray
            Cluster assignments
        feature_names : list
            Feature names
        """
        print("\n📊 Creating feature importance plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        for idx, feature in enumerate(feature_names[:4]):
            ax = axes[idx]
            
            data_to_plot = [X[X.index.isin(np.where(cluster_labels == c)[0])][feature].values 
                           for c in np.unique(cluster_labels)]
            
            bp = ax.boxplot(data_to_plot, labels=np.unique(cluster_labels),
                           patch_artist=True, widths=0.6)
            
            # Color the boxes
            colors = sns.color_palette('husl', len(np.unique(cluster_labels)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_xlabel('Cluster', fontsize=10, fontweight='bold')
            ax.set_ylabel(feature, fontsize=10, fontweight='bold')
            ax.set_title(f'{feature} Distribution by Cluster', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Feature Distribution Across Clusters', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        filepath = self.output_dir / 'feature_distributions.png'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        print(f"   ✓ Saved to {filepath}")
        plt.close()

def main():
    """Test visualization"""
    pass

if __name__ == "__main__":
    main()
