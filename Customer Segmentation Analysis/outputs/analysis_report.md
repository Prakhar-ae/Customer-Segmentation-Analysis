# Customer Segmentation Analysis Report

Generated On: 2026-05-27 17:43:18

## Executive Summary

This report presents customer segmentation analysis using K-Means clustering.
A total of 5 customer
segments were identified using behavioral,
demographic, and transactional features.

## Dataset Overview


- Total Customers: 500
- Features Used: 7
- Clustering Algorithm: K-Means
- Number of Clusters: 5


### Features Used

- Age
- Annual_Income_K
- Spending_Score
- Purchase_Frequency
- Avg_Order_Value
- Customer_Lifetime_Value
- Years_Customer

## Preprocessing Details


- Normalization Method:
  standardscaler
- Outlier Detection:
  False
- Missing Value Handling:
  Median Imputation


## Cluster Distribution

- Cluster 0: 116 customers (23.2%)
- Cluster 1: 60 customers (12.0%)
- Cluster 2: 100 customers (20.0%)
- Cluster 3: 127 customers (25.4%)
- Cluster 4: 97 customers (19.4%)

## Segment Profiles

|   Segment |   Size | Percentage   |   Avg_Age | Avg_Income   |   Avg_Spending_Score |   Avg_Purchase_Freq | Avg_CLV   |
|----------:|-------:|:-------------|----------:|:-------------|---------------------:|--------------------:|:----------|
|         0 |    116 | 23.2%        |      46.4 | 78.9K        |                 41.9 |                14.6 | $234      |
|         1 |     60 | 12.0%        |      45.8 | 74.0K        |                 51.7 |                13.8 | $293      |
|         2 |    100 | 20.0%        |      36.1 | 110.7K       |                 64.8 |                13.4 | $547      |
|         3 |    127 | 25.4%        |      39.3 | 77.7K        |                 25.8 |                15.5 | $224      |
|         4 |     97 | 19.4%        |      52   | 73.9K        |                 76.1 |                16.9 | $428      |

## Business Insights


### Segment 0: Growth Potential
Customer Count: 116
Segment Share: 23.2%
Priority Level: Medium
Profile:
Age: 46yr, Income: $78918, CLV: $234
Recommended Strategy:
Engagement campaigns, product discovery, conversion focus
Key Characteristics:
- Age Range:
  18 -
  70 years

- Income Range:
  $28.4K -
  $135.1K

- Preferred Category:
  Electronics

- Average Tenure:
  2.7 years

- Gender Split:
  {'Male': 63, 'Female': 53}


### Segment 1: Growth Potential
Customer Count: 60
Segment Share: 12.0%
Priority Level: Medium
Profile:
Age: 46yr, Income: $74031, CLV: $293
Recommended Strategy:
Engagement campaigns, product discovery, conversion focus
Key Characteristics:
- Age Range:
  22 -
  70 years

- Income Range:
  $15.0K -
  $139.0K

- Preferred Category:
  Fashion

- Average Tenure:
  6.9 years

- Gender Split:
  {'Female': 34, 'Male': 26}


### Segment 2: Young Professionals
Customer Count: 100
Segment Share: 20.0%
Priority Level: High
Profile:
Age: 36yr, Income: $110730, CLV: $547
Recommended Strategy:
Digital marketing, trendy products, convenience focus
Key Characteristics:
- Age Range:
  18 -
  59 years

- Income Range:
  $55.3K -
  $150.0K

- Preferred Category:
  Beauty

- Average Tenure:
  7.6 years

- Gender Split:
  {'Male': 54, 'Female': 46}


### Segment 3: Growth Potential
Customer Count: 127
Segment Share: 25.4%
Priority Level: Medium
Profile:
Age: 39yr, Income: $77722, CLV: $224
Recommended Strategy:
Engagement campaigns, product discovery, conversion focus
Key Characteristics:
- Age Range:
  18 -
  69 years

- Income Range:
  $15.0K -
  $150.0K

- Preferred Category:
  Fashion

- Average Tenure:
  10.1 years

- Gender Split:
  {'Male': 66, 'Female': 61}


### Segment 4: Established Customers
Customer Count: 97
Segment Share: 19.4%
Priority Level: High
Profile:
Age: 52yr, Income: $73935, CLV: $428
Recommended Strategy:
Trust-building, quality focus, long-term relationships
Key Characteristics:
- Age Range:
  21 -
  70 years

- Income Range:
  $15.0K -
  $133.4K

- Preferred Category:
  Electronics

- Average Tenure:
  8.6 years

- Gender Split:
  {'Female': 52, 'Male': 45}


## Technical Details


- Algorithm: K-Means Clustering
- Final K Value: 5
- Random State: 42
- Max Iterations: 300
- Convergence Status: Success


## Conclusion


The clustering analysis successfully identified
5 unique customer groups.

These customer segments can help improve:
- Marketing personalization
- Product recommendations
- Customer retention
- Revenue optimization
