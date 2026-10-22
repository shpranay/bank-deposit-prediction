# Bank Deposit Subscription Prediction Using PySpark and Spark MLlib

A Streamlit demo for a bank term-deposit subscription prediction project using PySpark MLlib.

## Models
- Logistic Regression
- Decision Tree
- Saved Spark preprocessing pipeline

## Important
This app uses Spark in local mode (`local[*]`) on the hosting machine. It is a Spark MLlib application, not a multi-node cluster deployment. Predictions are educational estimates based on historical data.

## Run locally
1. Install Java compatible with PySpark 3.5.x (Java 8/11/17 are commonly supported).
2. Install requirements: `pip install -r requirements.txt`
3. Ensure the `models/` folder contains the three exported Spark model directories.
4. Run: `streamlit run app.py`
