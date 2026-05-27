"""
Train machine learning models for the Retail Sales Prediction System.

This script keeps the workflow beginner-friendly:
1. Create/load a retail CSV dataset
2. Clean and preprocess the data
3. Generate EDA charts
4. Train Linear Regression and Random Forest models
5. Compare metrics and save the best model with Joblib
6. Optionally load the cleaned data into MySQL
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib_cache"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = BASE_DIR / "sales_data.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.csv"
GRAPHS_DIR = BASE_DIR / "static" / "graphs"

FEATURES = ["Region", "Category", "Quantity", "Profit", "Order_Month"]
TARGET = "Sales"


def create_one_hot_encoder() -> OneHotEncoder:
    """Create an encoder that works across recent Scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def generate_sample_dataset(file_path: Path = DATA_PATH, rows: int = 320) -> None:
    """
    Generate a realistic retail-style dataset.

    The values are synthetic, but the columns are similar to common Superstore
    datasets, which makes the project easy to run without downloading anything.
    """
    rng = np.random.default_rng(42)

    regions = ["East", "West", "Central", "South"]
    categories = {
        "Furniture": ["Chairs", "Tables", "Bookcases", "Furnishings"],
        "Office Supplies": ["Binders", "Paper", "Storage", "Appliances"],
        "Technology": ["Phones", "Accessories", "Copiers", "Machines"],
    }
    product_names = {
        "Chairs": ["Ergo Chair", "Task Chair", "Executive Chair"],
        "Tables": ["Meeting Table", "Coffee Table", "Study Table"],
        "Bookcases": ["Wood Bookcase", "Metal Bookcase", "Corner Bookcase"],
        "Furnishings": ["Desk Lamp", "Wall Clock", "Floor Mat"],
        "Binders": ["Basic Binder", "Premium Binder", "Report Binder"],
        "Paper": ["Copy Paper", "Legal Pad", "Photo Paper"],
        "Storage": ["File Box", "Storage Bin", "Drawer Unit"],
        "Appliances": ["Label Maker", "Desk Fan", "Paper Shredder"],
        "Phones": ["Office Phone", "Smartphone", "Conference Phone"],
        "Accessories": ["Keyboard", "Mouse", "USB Hub"],
        "Copiers": ["Compact Copier", "Laser Copier", "Office Copier"],
        "Machines": ["Printer", "Scanner", "Fax Machine"],
    }
    category_base_price = {
        "Furniture": 120,
        "Office Supplies": 35,
        "Technology": 210,
    }
    region_multiplier = {
        "East": 1.08,
        "West": 1.12,
        "Central": 0.94,
        "South": 1.02,
    }

    dates = pd.date_range("2022-01-01", "2025-12-31", freq="D")
    records = []

    for order_number in range(1, rows + 1):
        order_date = rng.choice(dates)
        region = rng.choice(regions)
        category = rng.choice(list(categories.keys()), p=[0.30, 0.42, 0.28])
        sub_category = rng.choice(categories[category])
        product_name = rng.choice(product_names[sub_category])
        quantity = int(rng.integers(1, 11))

        month = pd.Timestamp(order_date).month
        seasonal_multiplier = 1 + (0.18 if month in [10, 11, 12] else 0)
        price_noise = rng.normal(0, 18)
        sales = (
            category_base_price[category]
            * quantity
            * region_multiplier[region]
            * seasonal_multiplier
            + price_noise
        )
        sales = round(max(sales, 12), 2)

        profit_margin = {
            "Furniture": rng.uniform(0.08, 0.18),
            "Office Supplies": rng.uniform(0.10, 0.25),
            "Technology": rng.uniform(0.12, 0.30),
        }[category]
        profit = round(sales * profit_margin + rng.normal(0, 8), 2)

        records.append(
            {
                "Order ID": f"ORD-{order_number:04d}",
                "Order Date": pd.Timestamp(order_date).strftime("%Y-%m-%d"),
                "Region": region,
                "Category": category,
                "Sub-Category": sub_category,
                "Product Name": product_name,
                "Sales": sales,
                "Profit": profit,
                "Quantity": quantity,
            }
        )

    df = pd.DataFrame(records)

    # Add a few missing values and duplicates so preprocessing has a visible role.
    df.loc[[8, 41, 97], "Profit"] = np.nan
    df.loc[[23, 144], "Region"] = np.nan
    df.loc[[65], "Category"] = np.nan
    df = pd.concat([df, df.iloc[[5, 20, 55]]], ignore_index=True)

    df.to_csv(file_path, index=False)
    print(f"Sample dataset created at: {file_path}")


def load_data(file_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the CSV dataset. Generate one first if it does not exist."""
    if not file_path.exists():
        generate_sample_dataset(file_path)

    return pd.read_csv(file_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean missing values, remove duplicates, and add date features."""
    cleaned_df = df.copy()

    cleaned_df = cleaned_df.drop_duplicates()

    cleaned_df["Order Date"] = pd.to_datetime(cleaned_df["Order Date"], errors="coerce")
    cleaned_df["Sales"] = pd.to_numeric(cleaned_df["Sales"], errors="coerce")
    cleaned_df["Profit"] = pd.to_numeric(cleaned_df["Profit"], errors="coerce")
    cleaned_df["Quantity"] = pd.to_numeric(cleaned_df["Quantity"], errors="coerce")

    cleaned_df = cleaned_df.dropna(subset=["Order Date", "Sales"])

    for column in ["Region", "Category", "Sub-Category", "Product Name"]:
        cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode()[0])

    cleaned_df["Profit"] = cleaned_df["Profit"].fillna(cleaned_df["Profit"].median())
    cleaned_df["Quantity"] = cleaned_df["Quantity"].fillna(cleaned_df["Quantity"].median())

    cleaned_df = cleaned_df[(cleaned_df["Sales"] > 0) & (cleaned_df["Quantity"] > 0)]

    cleaned_df["Order_Year"] = cleaned_df["Order Date"].dt.year
    cleaned_df["Order_Month"] = cleaned_df["Order Date"].dt.month

    return cleaned_df


def generate_eda_charts(df: pd.DataFrame) -> None:
    """Create and save beginner-friendly EDA visualizations."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    monthly_sales = (
        df.set_index("Order Date")
        .resample("MS")["Sales"]
        .sum()
        .reset_index()
    )
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=monthly_sales, x="Order Date", y="Sales", marker="o", color="#2563eb")
    plt.title("Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "monthly_sales_trend.png", dpi=150)
    plt.close()

    sales_by_region = df.groupby("Region", as_index=False)["Sales"].sum()
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=sales_by_region,
        x="Region",
        y="Sales",
        hue="Region",
        palette="Set2",
        legend=False,
    )
    plt.title("Sales by Region")
    plt.xlabel("Region")
    plt.ylabel("Total Sales")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "sales_by_region.png", dpi=150)
    plt.close()

    sales_by_category = df.groupby("Category", as_index=False)["Sales"].sum()
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=sales_by_category,
        x="Category",
        y="Sales",
        hue="Category",
        palette="Set3",
        legend=False,
    )
    plt.title("Sales by Category")
    plt.xlabel("Category")
    plt.ylabel("Total Sales")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "sales_by_category.png", dpi=150)
    plt.close()

    profit_by_category = df.groupby("Category", as_index=False)["Profit"].sum()
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=profit_by_category,
        x="Category",
        y="Profit",
        hue="Category",
        palette="viridis",
        legend=False,
    )
    plt.title("Profit Analysis by Category")
    plt.xlabel("Category")
    plt.ylabel("Total Profit")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "profit_analysis.png", dpi=150)
    plt.close()

    numeric_columns = ["Sales", "Profit", "Quantity", "Order_Month", "Order_Year"]
    plt.figure(figsize=(7, 5))
    sns.heatmap(df[numeric_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()


def build_preprocessor() -> ColumnTransformer:
    """Prepare categorical and numerical columns for modeling."""
    categorical_features = ["Region", "Category"]
    numeric_features = ["Quantity", "Profit", "Order_Month"]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", create_one_hot_encoder()),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_features),
            ("numeric", numeric_pipeline, numeric_features),
        ]
    )


def evaluate_models(df: pd.DataFrame) -> tuple[Pipeline, str, list[dict[str, float | str]]]:
    """Train Linear Regression and Random Forest, then return the best model."""
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            random_state=42,
        ),
    }

    results = []
    trained_pipelines = {}

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        mae = float(mean_absolute_error(y_test, predictions))
        r2 = float(r2_score(y_test, predictions))

        results.append(
            {
                "Model": model_name,
                "RMSE": round(rmse, 2),
                "MAE": round(mae, 2),
                "R2 Score": round(r2, 4),
            }
        )
        trained_pipelines[model_name] = pipeline

    best_result = min(results, key=lambda row: row["RMSE"])
    best_model_name = str(best_result["Model"])
    best_pipeline = trained_pipelines[best_model_name]

    return best_pipeline, best_model_name, results


def save_feature_importance(best_pipeline: Pipeline) -> None:
    """Save feature importance if the selected model supports it."""
    model = best_pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        return

    preprocessor = best_pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    readable_names = [
        name.replace("categorical__", "").replace("numeric__", "").replace("Region_", "Region: ").replace("Category_", "Category: ")
        for name in feature_names
    ]

    importance_df = pd.DataFrame(
        {
            "Feature": readable_names,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=importance_df.head(10),
        x="Importance",
        y="Feature",
        hue="Feature",
        palette="crest",
        legend=False,
    )
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "feature_importance.png", dpi=150)
    plt.close()


def save_to_mysql(df: pd.DataFrame) -> None:
    """
    Save cleaned sales data into MySQL.

    Set MYSQL_URL before running:
    mysql+pymysql://username:password@localhost:3306/retail_sales_db
    """
    mysql_url = os.getenv("MYSQL_URL")
    if not mysql_url:
        print("MYSQL_URL not found. Skipping MySQL load.")
        return

    try:
        from sqlalchemy import create_engine
    except ImportError:
        print("SQLAlchemy is not installed. Run: pip install SQLAlchemy PyMySQL")
        return

    sql_df = df.rename(
        columns={
            "Order ID": "order_id",
            "Order Date": "order_date",
            "Region": "region",
            "Category": "category",
            "Sub-Category": "sub_category",
            "Product Name": "product_name",
            "Sales": "sales",
            "Profit": "profit",
            "Quantity": "quantity",
            "Order_Year": "order_year",
            "Order_Month": "order_month",
        }
    )

    engine = create_engine(mysql_url)
    sql_df.to_sql("retail_sales", con=engine, if_exists="replace", index=False)
    print("Cleaned data saved to MySQL table: retail_sales")


def train(load_mysql: bool = False) -> None:
    """Run the full training workflow."""
    raw_df = load_data()
    cleaned_df = clean_data(raw_df)

    generate_eda_charts(cleaned_df)

    best_pipeline, best_model_name, results = evaluate_models(cleaned_df)
    save_feature_importance(best_pipeline)

    artifact = {
        "model": best_pipeline,
        "model_name": best_model_name,
        "features": FEATURES,
        "metrics": results,
        "regions": sorted(cleaned_df["Region"].unique().tolist()),
        "categories": sorted(cleaned_df["Category"].unique().tolist()),
        "trained_on": datetime.now().strftime("%Y-%m-%d"),
    }

    joblib.dump(artifact, MODEL_PATH)
    pd.DataFrame(results).to_csv(METRICS_PATH, index=False)

    print("\nModel comparison:")
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nBest model: {best_model_name}")
    print(f"Saved model artifact: {MODEL_PATH}")
    print(f"Saved EDA graphs: {GRAPHS_DIR}")

    if load_mysql:
        save_to_mysql(cleaned_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train retail sales prediction models.")
    parser.add_argument(
        "--load-mysql",
        action="store_true",
        help="Load cleaned data into MySQL using the MYSQL_URL environment variable.",
    )
    args = parser.parse_args()

    train(load_mysql=args.load_mysql)
