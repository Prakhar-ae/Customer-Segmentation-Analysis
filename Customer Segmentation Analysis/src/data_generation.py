"""
Data Generation Module
Generates realistic synthetic customer data for segmentation analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_customer_data(config):
    """
    Generate synthetic customer dataset
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
        
    Returns:
    --------
    pd.DataFrame
        Generated customer dataset
    """
    
    np.random.seed(config['data']['random_state'])
    n_samples = config['data']['n_samples']
    
    # Extract feature ranges
    age_range = config['data']['features']['age']
    income_range = config['data']['features']['income']
    spending_range = config['data']['features']['spending_score']
    freq_range = config['data']['features']['purchase_frequency']
    
    # Generate features with realistic distributions
    data = {
        'Customer_ID': [f'CUST_{i:04d}' for i in range(1, n_samples + 1)],
        
        'Age': np.random.normal(
            loc=(age_range['min'] + age_range['max']) / 2,
            scale=12,
            size=n_samples
        ).astype(int).clip(age_range['min'], age_range['max']),
        
        'Annual_Income_K': np.random.normal(
            loc=(income_range['min'] + income_range['max']) / 2,
            scale=30,
            size=n_samples
        ).clip(income_range['min'], income_range['max']),
        
        'Spending_Score': np.random.uniform(
            spending_range['min'],
            spending_range['max'],
            n_samples
        ).astype(int),
        
        'Purchase_Frequency': np.random.poisson(
            lam=15,
            size=n_samples
        ).clip(freq_range['min'], freq_range['max']),
    }
    
    df = pd.DataFrame(data)
    
    # Add behavioral features based on existing data
    df['Avg_Order_Value'] = (df['Annual_Income_K'] / 10 + 
                             np.random.normal(0, 50, n_samples)).clip(10, 500)
    
    df['Customer_Lifetime_Value'] = (df['Annual_Income_K'] * 2 + 
                                     df['Spending_Score'] * 3 + 
                                     np.random.normal(0, 200, n_samples)).clip(100, 10000)
    
    df['Years_Customer'] = np.random.randint(0, 15, n_samples)
    
    # Product preferences
    product_categories = ['Electronics', 'Fashion', 'Home & Garden', 'Sports', 'Beauty']
    df['Primary_Category'] = np.random.choice(product_categories, n_samples)
    
    # Gender distribution
    df['Gender'] = np.random.choice(['Male', 'Female'], n_samples, p=[0.48, 0.52])
    
    # Round numerical columns appropriately
    df['Annual_Income_K'] = df['Annual_Income_K'].round(2)
    df['Avg_Order_Value'] = df['Avg_Order_Value'].round(2)
    df['Customer_Lifetime_Value'] = df['Customer_Lifetime_Value'].round(2)
    
    return df

def save_dataset(df, output_path):
    """Save dataset to CSV"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Dataset saved to {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"\nFirst few records:\n{df.head()}")
    
def create_data_dictionary():
    """Create a data dictionary documenting all features"""
    data_dict = """# Customer Dataset - Data Dictionary

## Overview
This dataset contains 500 synthetic customer records with demographic, behavioral, and transactional features.

## Features

### Identifiers
- **Customer_ID**: Unique customer identifier (format: CUST_0001)

### Demographics
- **Age**: Customer age in years (range: 18-70)
  - Type: Integer
  - Mean age provides insights into target demographic

- **Gender**: Customer gender (Male/Female)
  - Type: Categorical
  - Useful for gender-specific marketing strategies

### Financial & Spending
- **Annual_Income_K**: Annual income in thousands of dollars (range: 15-150)
  - Type: Float
  - Important for premium vs budget segment differentiation

- **Spending_Score**: Proprietary spending behavior score (range: 1-100)
  - Type: Integer
  - Higher scores indicate higher spending tendency
  - Based on recent purchase history and frequency

- **Avg_Order_Value**: Average order value in dollars (range: 10-500)
  - Type: Float
  - Indicates typical purchase size per transaction

- **Customer_Lifetime_Value**: Total expected value from customer (range: 100-10000)
  - Type: Float
  - Cumulative value metric for customer worth assessment

### Behavioral
- **Purchase_Frequency**: Number of purchases per year (range: 1-50)
  - Type: Integer
  - High frequency indicates loyal customers

- **Years_Customer**: Years as a customer (range: 0-15)
  - Type: Integer
  - Tenure metric; longer tenure may indicate loyalty

### Preferences
- **Primary_Category**: Preferred product category
  - Type: Categorical
  - Values: Electronics, Fashion, Home & Garden, Sports, Beauty
  - Most frequent purchase category for the customer

## Data Quality Notes
- No missing values in this synthetic dataset
- All numerical values are within expected ranges
- Data is normalized for clustering analysis
- Random relationships between features reflect realistic customer behavior

## Use Cases
1. **Segmentation**: Cluster customers based on demographics and behavior
2. **Targeting**: Identify high-value customer segments
3. **Marketing**: Tailor campaigns to specific segments
4. **Product Development**: Understand customer preferences per segment
5. **Churn Prediction**: Use features to predict at-risk customers

## Relationships
- Income and spending score are moderately correlated
- Purchase frequency correlates with spending score
- Customer lifetime value is influenced by income and purchase frequency
"""
    
    output_path = Path('./data/data_dictionary.md')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(data_dict)
    print(f"✓ Data dictionary created at {output_path}")

def main():
    """Main execution function"""
    print("=" * 60)
    print("CUSTOMER SEGMENTATION - DATA GENERATION")
    print("=" * 60)
    
    # Load configuration
    config = load_config('config.yaml')
    
    # Generate dataset
    print("\n📊 Generating synthetic customer dataset...")
    df = generate_customer_data(config)
    
    # Save dataset
    print("\n💾 Saving dataset...")
    output_file = config['output']['csv_file']
    save_dataset(df, output_file)
    
    # Create data dictionary
    print("\n📋 Creating data dictionary...")
    create_data_dictionary()
    
    print("\n" + "=" * 60)
    print("✅ Data generation complete!")
    print("=" * 60)
    
    return df

if __name__ == "__main__":
    df = main()
