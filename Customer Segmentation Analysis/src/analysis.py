import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from data_generation import (generate_customer_data,create_data_dictionary)
from data_preprocessing import DataPreprocessor
from clustering import (CustomerSegmentationModel,SegmentAnalyzer)
from visualization import ClusterVisualizer
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def generate_analysis_report(analyzer, model, config, output_file):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = []
    report.append("# Customer Segmentation Analysis Report\n")
    report.append(f"Generated On: {timestamp}\n")
    report.append("## Executive Summary\n")
    report.append(f"""This report presents customer segmentation analysis using K-Means clustering.
A total of {config['clustering']['final_k']} customer
segments were identified using behavioral,
demographic, and transactional features.
""")
    report.append("## Dataset Overview\n")
    report.append(f"""
- Total Customers: {len(analyzer.df)}
- Features Used: {len(analyzer.feature_names)}
- Clustering Algorithm: K-Means
- Number of Clusters: {config['clustering']['final_k']}
"""
)
    report.append("\n### Features Used\n")
    for feature in analyzer.feature_names:
        report.append(f"- {feature}")
    report.append("\n## Preprocessing Details\n")
    report.append(f"""
- Normalization Method:
  {config['preprocessing']['normalization_method']}
- Outlier Detection:
  {config['preprocessing']['outlier_detection']}
- Missing Value Handling:
  Median Imputation
"""
    )
    report.append("\n## Cluster Distribution\n")
    unique_clusters, cluster_counts = np.unique(
        analyzer.cluster_labels,
        return_counts=True
    )
    total_customers = len(analyzer.df)
    for cluster_id, count in zip(unique_clusters, cluster_counts):
        percentage = (count / total_customers) * 100
        report.append(
            f"- Cluster {cluster_id}: "
            f"{count} customers ({percentage:.1f}%)"
        )
    report.append("\n## Segment Profiles\n")
    profiles = analyzer.get_segment_profiles()
    try:
        report.append(profiles.to_markdown(index=False))
    except ImportError:
        report.append(profiles.to_string(index=False))
    report.append("\n## Business Insights\n")
    characteristics = analyzer.get_segment_characteristics()
    insights = analyzer.generate_business_insights()
    for segment_id, insight in insights.items():
        report.append(
            f"""
### Segment {segment_id}: {insight['name']}
Customer Count: {insight['size']}
Segment Share: {insight['percentage']:.1f}%
Priority Level: {insight['priority']}
Profile:
{insight['characteristics']}
Recommended Strategy:
{insight['strategy']}
Key Characteristics:
- Age Range:
  {characteristics[segment_id]['age_range'][0]} -
  {characteristics[segment_id]['age_range'][1]} years

- Income Range:
  ${characteristics[segment_id]['income_range'][0]:.1f}K -
  ${characteristics[segment_id]['income_range'][1]:.1f}K

- Preferred Category:
  {characteristics[segment_id]['top_category']}

- Average Tenure:
  {characteristics[segment_id]['avg_tenure']:.1f} years

- Gender Split:
  {characteristics[segment_id]['gender_split']}
"""
        )
    report.append("\n## Technical Details\n")
    report.append(
        f"""
- Algorithm: K-Means Clustering
- Final K Value: {config['clustering']['final_k']}
- Random State: {config['clustering']['random_state']}
- Max Iterations: {config['clustering']['max_iterations']}
- Convergence Status: Success
"""
    )
    report.append("\n## Conclusion\n")
    report.append(
        f"""
The clustering analysis successfully identified
{config['clustering']['final_k']} unique customer groups.

These customer segments can help improve:
- Marketing personalization
- Product recommendations
- Customer retention
- Revenue optimization
"""
    )
    final_report = "\n".join(report)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as report_file:
        report_file.write(final_report)
    print(f"\n Report saved: {output_file}")
    return final_report

def main():
    print("\n" + "=" * 70)
    print("CUSTOMER SEGMENTATION ANALYSIS".center(70))
    print("=" * 70)
    print("\n Loading configuration...")
    config = load_config()
    print("\n[ STEP 1 ] DATA GENERATION")
    print("-" * 70)
    df = generate_customer_data(config)
    create_data_dictionary()
    dataset_path = Path(config["output"]["csv_file"])
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dataset_path, index=False)
    print(f"Dataset saved: {dataset_path}")
    print("\n[ STEP 2 ] DATA PREPROCESSING")
    print("-" * 70)
    preprocessor = DataPreprocessor(config)
    X_normalized, X = preprocessor.preprocess(df)
    print(" Preprocessing completed")
    print("\n[ STEP 3 ] OPTIMAL CLUSTER SEARCH")
    print("-" * 70)
    model = CustomerSegmentationModel(config)
    k_range = range(
        config["clustering"]["optimal_k_range"][0],
        config["clustering"]["optimal_k_range"][1] + 1
    )
    optimal_k = model.find_optimal_k(X_normalized)
    print(f" Optimal K found: {optimal_k}")
    print("\n[ STEP 4 ] MODEL TRAINING")
    print("-" * 70)
    cluster_labels = model.train(X_normalized)
    print(" Model training completed")
    print("\n[ STEP 5 ] SEGMENT ANALYSIS")
    print("-" * 70)
    analyzer = SegmentAnalyzer(
        df,
        X,
        cluster_labels,
        preprocessor.feature_names
    )
    profiles = analyzer.get_segment_profiles()
    print(" Segment analysis completed")
    print("\n[ STEP 6 ] VISUALIZATION")
    print("-" * 70)
    visualizer = ClusterVisualizer(config)
    visualizer.plot_elbow_curve(model.inertias, k_range)
    visualizer.plot_clusters_2d(
        X_normalized,
        cluster_labels,
        model.get_cluster_centers()
    )
    visualizer.plot_clusters_3d(
        X_normalized,
        cluster_labels,
        model.get_cluster_centers()
    )
    visualizer.plot_cluster_distribution(cluster_labels)
    visualizer.plot_segment_profiles(profiles)
    visualizer.plot_feature_importance_by_cluster(
        X,
        cluster_labels,
        preprocessor.feature_names
    )
    print(" Visualizations generated")
    if config["output"]["save_model"]:
        print("\n Saving model...")
        model.save_model(config["output"]["model_file"])
        print(" Model saved")
    print("\n[ STEP 7 ] REPORT GENERATION")
    print("-" * 70)
    generate_analysis_report(
        analyzer,
        model,
        config,
        config["output"]["report_file"]
    )
    print("\n" + "=" * 70)
    print(" ANALYSIS COMPLETED")
    print("=" * 70)
    print(
        f"""
SUMMARY

Customers Analyzed : {len(df)}
Segments Found     : {config['clustering']['final_k']}
Features Used      : {len(preprocessor.feature_names)}

OUTPUT FILES

Dataset      : {config['output']['csv_file']}
Report       : {config['output']['report_file']}
Visuals Path : {config['visualization']['plot_directory']}
"""
    )

    print("=" * 70)

    return df, model, analyzer, preprocessor


if __name__ == "__main__":

    df, model, analyzer, preprocessor = main()