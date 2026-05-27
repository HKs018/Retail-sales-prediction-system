"""
Flask web app for predicting retail sales.

Run with:
python app.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
DATA_PATH = BASE_DIR / "sales_data.csv"

app = Flask(__name__)


def load_model_artifact() -> dict | None:
    """Load the saved model package created by train_model.py."""
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


def load_dashboard_summary() -> dict:
    """Create simple business insight numbers for the homepage."""
    if not DATA_PATH.exists():
        return {
            "total_sales": 0,
            "total_profit": 0,
            "average_order_value": 0,
            "best_category": "N/A",
            "best_region": "N/A",
        }

    df = pd.read_csv(DATA_PATH)
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df["Profit"] = pd.to_numeric(df["Profit"], errors="coerce")
    df = df.dropna(subset=["Sales", "Profit"])

    best_category = df.groupby("Category")["Sales"].sum().idxmax()
    best_region = df.groupby("Region")["Sales"].sum().idxmax()

    return {
        "total_sales": round(df["Sales"].sum(), 2),
        "total_profit": round(df["Profit"].sum(), 2),
        "average_order_value": round(df["Sales"].mean(), 2),
        "best_category": best_category,
        "best_region": best_region,
    }


@app.route("/")
def index():
    """Display homepage, dashboard insights, graphs, and prediction form."""
    artifact = load_model_artifact()
    summary = load_dashboard_summary()

    regions = artifact["regions"] if artifact else ["Central", "East", "South", "West"]
    categories = artifact["categories"] if artifact else ["Furniture", "Office Supplies", "Technology"]
    metrics = artifact["metrics"] if artifact else []
    model_name = artifact["model_name"] if artifact else "Model not trained"

    graphs = [
        "graphs/monthly_sales_trend.png",
        "graphs/sales_by_region.png",
        "graphs/sales_by_category.png",
        "graphs/profit_analysis.png",
        "graphs/correlation_heatmap.png",
        "graphs/feature_importance.png",
    ]

    return render_template(
        "index.html",
        regions=regions,
        categories=categories,
        summary=summary,
        metrics=metrics,
        model_name=model_name,
        graphs=graphs,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Predict sales from form values."""
    artifact = load_model_artifact()

    if artifact is None:
        return render_template(
            "result.html",
            error="Model file not found. Run python train_model.py first.",
        )

    try:
        region = request.form["region"]
        category = request.form["category"]
        quantity = float(request.form["quantity"])
        profit = float(request.form["profit"])

        input_df = pd.DataFrame(
            [
                {
                    "Region": region,
                    "Category": category,
                    "Quantity": quantity,
                    "Profit": profit,
                    "Order_Month": datetime.now().month,
                }
            ]
        )

        prediction = artifact["model"].predict(input_df)[0]
        prediction = max(float(prediction), 0)

        return render_template(
            "result.html",
            prediction=round(prediction, 2),
            region=region,
            category=category,
            quantity=quantity,
            profit=profit,
            model_name=artifact["model_name"],
        )

    except ValueError:
        return render_template(
            "result.html",
            error="Please enter valid numeric values for quantity and profit.",
        )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
