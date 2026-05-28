# Quick Start
## Setup
cd customer-segmentation

python -m venv venv

### Windows
venv\Scripts\activate

### Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
## Run Project
python src/analysis.py

### This will:
- Generate data
- Train clustering model
- Create graphs
- Save results

## Output Files
After running, check:
- outputs/

### Main files:
- elbow_curve.png
- clusters_2d.png
- analysis_report.md
- customers_with_segments.csv

## Project Structure
customer-segmentation/
│
├── data/
├── outputs/
├── notebooks/
├── src/
├── tests/
├── requirements.txt
└── config.yaml
