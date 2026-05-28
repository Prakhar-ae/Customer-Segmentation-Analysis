# Customer Segmentation Analysis

## Overview

Customer segmentation project using K-Means clustering to group customers based on income, spending, and purchasing behavior.

## Features

- Customer data generation
- Data preprocessing
- K-Means clustering
- Data visualization
- Segment analysis

## Dataset

- Age
- Annual Income
- Spending Score
- Purchase Frequency
- Product Category

## Technologies

- Python
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Project Structure

customer-segmentation/
│
├── data/
├── outputs/
├── src/
├── tests/
├── config.yaml
├── requirements.txt
└── README.md

# installation 
cd customer-segmentation

python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt

## Output Files
- customers.csv
- elbow_curve.png
- clusters_2d.png
- cluster_distribution.png
- analysis_report.md

## Customer Segments
- High Value Customers
- Young Customers
- Budget Customers
- Regular Customers

## Configuration

Edit config.yaml to change:

- sample size
- cluster count
- preprocessing settings
- output paths