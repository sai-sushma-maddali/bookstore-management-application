# Bookstore Analytics & Recommendation Dashboard

A Streamlit-based analytics dashboard for a bookstore database (MySQL).  
It provides customer insights, review insights, demographics exploration, and sales performance analytics — plus a simple, explainable book recommendation feature driven by purchase history.

---

## Tech Stack
- **Frontend / App**: Streamlit
- **Database**: MySQL
- **Data Handling**: Pandas
- **Visualizations**: Matplotlib, Seaborn, Squarify (Treemap)

---

## Key Features

### 1) Book Recommendation
Given a **Customer ID**, the app:
1. Identifies the customer’s **most frequently purchased genre**.
2. Displays the customer’s **previously purchased books** within that genre.
3. Recommends **top-selling books** from the same genre that the customer **has not purchased yet**.

This provides a transparent recommendation workflow (not a black box).

---

### 2) Customer Insights
- **Churn risk identification**: Lists customers whose last purchase was more than *N days* ago.
- Customer-level metrics:
  - Recency (days since last order)
  - Monetary value (total spend)
  - Average Order Value (AOV)
  - Customer Lifetime Value (CLV)
  - Average days between purchases
  - Tenure (days between first and latest order)

---

### 3) Reviews Insights
- Sentiment analysis by year
- Top reviewers
- Top reviewed books
- Genre sentiment distribution (best/worst ordering)

---

### 4) Customer Demographics
- Gender distribution (pie chart)
- Preferred language distribution (treemap)

---

### 5) Sales Performance
- Genre-wise sales treemap by year
- Top & bottom-selling books (horizontal bar chart with value labels)

---

### 6) Sales Insights
- Books with declining sales
- Revenue contribution by publisher

---

## Recommendation Logic (SQL Summary)

The recommendation is based on three SQL views:

### View 1: `customer_book_details`
Maps each customer to the books they purchased and the associated genre.

### View 2: `customer_frequent_genre`
Finds the **most frequently purchased genre** per customer using aggregation and window ranking.

### View 3: `best_sellers_by_genre`
Computes best-selling books per genre based on total quantity sold.

### Recommendation Query
For a given customer:
- Find favorite genre from `customer_frequent_genre`
- Find already purchased `book_id`s from `customer_book_details`
- Recommend books from `best_sellers_by_genre` in the same genre excluding purchased `book_id`s

This yields **genre-personalized** recommendations driven by historical behavior.

