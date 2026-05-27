-- Retail Sales Prediction System SQL queries
-- Database: MySQL

CREATE DATABASE IF NOT EXISTS retail_sales_db;
USE retail_sales_db;

CREATE TABLE IF NOT EXISTS retail_sales (
    order_id VARCHAR(20),
    order_date DATE,
    region VARCHAR(50),
    category VARCHAR(100),
    sub_category VARCHAR(100),
    product_name VARCHAR(150),
    sales DECIMAL(10, 2),
    profit DECIMAL(10, 2),
    quantity INT,
    order_year INT,
    order_month INT
);

-- 1. Top-selling categories
SELECT
    category,
    ROUND(SUM(sales), 2) AS total_sales
FROM retail_sales
GROUP BY category
ORDER BY total_sales DESC;

-- 2. Monthly revenue
SELECT
    DATE_FORMAT(order_date, '%Y-%m') AS sales_month,
    ROUND(SUM(sales), 2) AS monthly_revenue
FROM retail_sales
GROUP BY sales_month
ORDER BY sales_month;

-- 3. Region-wise sales
SELECT
    region,
    ROUND(SUM(sales), 2) AS total_sales
FROM retail_sales
GROUP BY region
ORDER BY total_sales DESC;

-- 4. Average profit by category
SELECT
    category,
    ROUND(AVG(profit), 2) AS average_profit
FROM retail_sales
GROUP BY category
ORDER BY average_profit DESC;

-- 5. Top 5 products by sales
SELECT
    product_name,
    category,
    ROUND(SUM(sales), 2) AS total_sales
FROM retail_sales
GROUP BY product_name, category
ORDER BY total_sales DESC
LIMIT 5;
