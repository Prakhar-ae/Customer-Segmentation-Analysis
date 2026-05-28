# Customer Dataset - Data Dictionary

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
