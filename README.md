# Retail Sales Prediction System

A simple end-to-end machine learning project that predicts retail sales using Python, Pandas, SQL, Scikit-learn, and Flask. The project is designed to be easy to explain in a fresher data science interview while still showing a complete workflow from data loading to model deployment.

## Project Overview

This system uses a retail sales dataset with columns such as order date, region, category, sub-category, sales, profit, and quantity. It cleans the data, performs exploratory data analysis, trains regression models, saves the best model, and serves predictions through a Flask web app.

## Features

- Loads retail sales data from `sales_data.csv`
- Handles missing values and duplicate rows
- Creates date-based features such as order year and order month
- Generates EDA charts for trends and business insights
- Trains Linear Regression and Random Forest Regressor
- Compares models using RMSE, MAE, and R2 score
- Saves the best trained model using Joblib
- Provides a Flask prediction form
- Includes SQL queries for business analysis
- Shows dashboard-style insights on the homepage

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Flask
- MySQL
- Joblib

## Project Structure

```text
retail-sales-predictor/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── model.pkl
├── sales_data.csv
├── model_metrics.csv
├── templates/
│   ├── index.html
│   └── result.html
├── static/
│   ├── style.css
│   └── graphs/
├── notebooks/
│   └── eda.ipynb
└── sql/
    └── queries.sql
```

## Project Workflow

1. Load the retail sales CSV dataset.
2. Remove duplicate rows and handle missing values.
3. Convert order dates into useful features like month and year.
4. Create EDA charts for monthly sales, region sales, category sales, profit, and correlations.
5. Train Linear Regression and Random Forest models.
6. Evaluate both models using RMSE, MAE, and R2 score.
7. Save the best model as `model.pkl`.
8. Use Flask to collect user inputs and display predicted sales.
9. Use SQL queries to answer business questions from MySQL.

## How To Run Locally

1. Create and activate a virtual environment.

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Train the models and generate charts.

```bash
python train_model.py
```

4. Start the Flask app.

```bash
python app.py
```

5. Open the app in your browser.

```text
http://127.0.0.1:5000
```

## MySQL Setup

Create a database in MySQL:

```sql
CREATE DATABASE retail_sales_db;
```

Set the MySQL connection string before loading data:

```bash
export MYSQL_URL="mysql+pymysql://root:your_password@localhost:3306/retail_sales_db"
```

On Windows PowerShell:

```powershell
$env:MYSQL_URL="mysql+pymysql://root:your_password@localhost:3306/retail_sales_db"
```

Load cleaned data into MySQL:

```bash
python train_model.py --load-mysql
```

Business analysis queries are available in `sql/queries.sql`.

## Generated Visualizations

The training script saves charts inside `static/graphs/`:

- Monthly sales trend
- Sales by region
- Sales by category
- Profit analysis
- Correlation heatmap
- Feature importance

## Screenshots

Add screenshots after running the app:

- Home page dashboard: `screenshots/home.png`
- Prediction form: `screenshots/form.png`
- Prediction result: `screenshots/result.png`
- EDA charts: `screenshots/charts.png`

## Machine Learning Models

The project trains two regression models:

- Linear Regression
- Random Forest Regressor

The best model is selected using the lowest RMSE value and saved as `model.pkl`.

## Possible Interview Questions

### Why Random Forest?

Random Forest works well because it combines many decision trees and can capture non-linear relationships between features such as quantity, profit, region, category, and sales. It is also less likely to overfit than a single decision tree.

### Difference between MAE and RMSE

MAE gives the average absolute prediction error. RMSE also measures prediction error, but it gives more penalty to large errors because the errors are squared before taking the square root.

### Why preprocessing matters

Preprocessing improves data quality before training. Missing values, duplicate rows, incorrect data types, and categorical values can reduce model performance or even cause model training to fail.

### Why Flask was used

Flask is lightweight and easy to understand. It is a good choice for creating a simple web app where users enter values and get predictions from a trained machine learning model.

### How feature engineering helps

Feature engineering creates useful input variables from raw data. For example, extracting month from order date helps the model learn seasonal sales patterns.

## Future Improvements

- Use a larger real-world Superstore dataset
- Add more features such as discounts, customer segment, and shipping mode
- Add user login for business users
- Store predictions in a database
- Deploy the Flask app on Render, Railway, or AWS
- Add automated tests for preprocessing and prediction logic
