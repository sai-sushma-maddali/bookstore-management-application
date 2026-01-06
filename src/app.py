
import streamlit as st
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
import seaborn as sns
import squarify


# -------------------------
# DB helpers
# -------------------------
def get_connection(host, port, user, password, database):
    return mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        autocommit=True
    )


def run_select_query(conn, sql, params=None):
   
    if params is None:
        return pd.read_sql(sql, conn)
    return pd.read_sql(sql, conn, params=params)


def run_multi_statement(conn, sql_script):
    cur = conn.cursor()
    try:
        for _ in cur.execute(sql_script, multi=True):
            pass
    finally:
        cur.close()


# -------------------------
# SQL: Views
# -------------------------
VIEWS_SQL = """
CREATE OR REPLACE VIEW customer_book_details AS
SELECT o.customer_id, od.book_id, b.title, g.genre_title
FROM orders o
JOIN order_details od ON o.order_id = od.order_id
JOIN books b ON od.book_id = b.book_id
JOIN genre g ON b.genre_id = g.genre_id;

CREATE OR REPLACE VIEW customer_frequent_genre AS
WITH GenreFrequency AS (
    SELECT o.customer_id, g.genre_title, COUNT(*) AS genre_count
    FROM orders o
    JOIN order_details od ON o.order_id = od.order_id
    JOIN books b ON od.book_id = b.book_id
    JOIN genre g ON b.genre_id = g.genre_id
    GROUP BY o.customer_id, g.genre_title
)
SELECT customer_id, genre_title, genre_count
FROM (
    SELECT customer_id, genre_title, genre_count,
           RANK() OVER (PARTITION BY customer_id ORDER BY genre_count DESC) AS rnk
    FROM GenreFrequency
) RankedGenres
WHERE rnk = 1;

CREATE OR REPLACE VIEW best_sellers_by_genre AS
WITH book_sales_per_book AS (
    SELECT od.book_id, b.genre_id, SUM(od.quantity) AS total_book_sales
    FROM order_details od
    JOIN books b ON od.book_id = b.book_id
    GROUP BY od.book_id, b.genre_id
),
max_sales_per_genre AS (
    SELECT genre_id, MAX(total_book_sales) AS max_sales_values_per_genre
    FROM book_sales_per_book
    GROUP BY genre_id
)
SELECT DISTINCT bspb.genre_id, g.genre_title, b.title,
       bspb.book_id, bspb.total_book_sales
FROM book_sales_per_book bspb
JOIN max_sales_per_genre mspg
  ON bspb.genre_id = mspg.genre_id
 AND bspb.total_book_sales = mspg.max_sales_values_per_genre
JOIN books b ON bspb.book_id = b.book_id
JOIN genre g ON bspb.genre_id = g.genre_id;
"""


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Bookstore Analytics", layout="wide")
st.title("Bookstore Analytics ")

with st.sidebar:
    st.header("Database Connection")

    host = st.text_input("Host", value="localhost")
    port = st.number_input("Port", min_value=1, max_value=65535, value=3306, step=1)
    user = st.text_input("User", value="root")
    password = st.text_input("Password", value="", type="password")
    database = st.text_input("Database", value="dynamicdata")

    connect_btn = st.button("Connect")


# -------------------------
# Connect to DB
# -------------------------
conn = None
connected = False

if connect_btn:
    try:
        conn = get_connection(host, int(port), user, password, database)
        connected = True
        st.sidebar.success("Connected.")
    except Exception as e:
        st.sidebar.error(f"Connection failed: {e}")

# If user didn't click connect, still try to connect once so app is usable.
if conn is None:
    try:
        conn = get_connection(host, int(port), user, password, database)
        connected = True
    except Exception:
        connected = False

if not connected:
    st.warning("Please enter valid DB credentials in the sidebar (then click Connect).")
    st.stop()



# -------------------------
# Tabs
# -------------------------
tabs = st.tabs([
    "Book Recommendation",
    "Customer Insights",
    "Reviews Insights",
    "Customer Demographics",
    "Sales: Performance",
    "Sales: Insights",
])


# -------------------------
# 1) Book Recommendations
# -------------------------
with tabs[0]:
    # st.subheader("Recommendations")

    customer_id = st.number_input("Enter Customer ID", min_value=1, value=100000, step=1)
    run_btn = st.button("Recommend Books")

    if run_btn:
        try:
            # Step 1: most frequent genre
            q1 = """
                SELECT genre_title
                FROM customer_frequent_genre
                WHERE customer_id = %s
                LIMIT 1;
            """
            fav = run_select_query(conn, q1, params=(int(customer_id),))
            preferred_genre = fav.iloc[0]["genre_title"]
            st.subheader("Customer’s Preferred Genre")
            st.dataframe(fav, use_container_width=True)
            

            if fav.empty:
                st.info("No frequent genre found. Customer may have no purchases.")
            else:
                fav_genre = str(fav.iloc[0]["genre_title"])

                # Step 2: books purchased in that genre
                q2 = """
                    SELECT *
                    FROM customer_book_details
                    WHERE customer_id = %s
                    AND genre_title = (
                        SELECT genre_title
                        FROM customer_frequent_genre
                        WHERE customer_id = %s
                        LIMIT 1
                    );
                """
                purchased = run_select_query(conn, q2, params=(int(customer_id), int(customer_id)))
                # st.subheader("Previously Purchased Books in This Genre")
                # st.dataframe(purchased, use_container_width=True)
                with st.expander("**Previously Purchased Books in This Genre**"):
                    st.dataframe(purchased, use_container_width=True)
                # Step 3 
                q3_original = """
                    WITH already_purchased AS (
                        SELECT *
                        FROM customer_book_details
                        WHERE customer_id = %s
                        AND genre_title = (
                            SELECT genre_title
                            FROM customer_frequent_genre
                            WHERE customer_id = %s
                            LIMIT 1
                        )
                    )
                    SELECT DISTINCT(best_sellers_by_genre.genre_title) AS genre_title,
                        best_sellers_by_genre.title,
                        best_sellers_by_genre.book_id
                    FROM best_sellers_by_genre
                    LEFT JOIN already_purchased
                    ON best_sellers_by_genre.genre_title = already_purchased.genre_title
                    WHERE best_sellers_by_genre.genre_title = already_purchased.genre_title;
                """
                original_out = run_select_query(conn, q3_original, params=(int(customer_id), int(customer_id)))
       

                # Step 3 
                q3_recommend = """
                    WITH purchased AS (
                        SELECT DISTINCT book_id
                        FROM customer_book_details
                        WHERE customer_id = %s
                        AND genre_title = %s
                    )
                    SELECT
                        bsg.genre_title,
                        bsg.title,
                        bsg.book_id
                    FROM best_sellers_by_genre bsg
                    WHERE bsg.genre_title = %s
                    AND bsg.book_id NOT IN (SELECT book_id FROM purchased)
                    ORDER BY bsg.total_book_sales DESC, bsg.title;
                """
                recs = run_select_query(conn, q3_recommend, params=(int(customer_id), fav_genre, fav_genre))
                st.subheader("Recommended Bestsellers in Customer's  Favorite Genre")
                if recs.empty:
                    st.info("No recommendations found (customer may already own the best-seller(s) for this genre).")
                else:
                    st.dataframe(recs, use_container_width=True)

        except Exception as e:
            st.error(f"Recommendation flow failed: {e}")

    # try:
    #     conn.close()
    # except Exception:
    #     pass

# -------------------------
# 2) Customer Insights (churn + per-customer metrics)
# -------------------------
with tabs[1]:
    
    c1, c2 = st.columns([1, 2])

    with c1:
        st.markdown("#### Churn Risk")
        churn_days = st.number_input("Days since last order", min_value=1, value=90, step=10)
        churn_btn = st.button("Get churn-risk customers")

        st.divider()
        st.markdown("#### Customer Metrics")
        cust_id = st.number_input("Customer ID", min_value=1, value=100000, step=1, key="cust_metrics")
        metric = st.selectbox(
            "Metric",
            [
                "Recency",
                "Monetary",
                "Average Order Value",
                "Customer Lifetime Value",
                "Avg days between purchases",
                "Tenure",
            ]
        )
        metric_btn = st.button("Run metric")

    with c2:
        if churn_btn:
            try:
                q = f"""
                    WITH customer_last_order_date AS (
                        SELECT customer_id, MAX(order_date) AS last_order_date
                        FROM orders
                        GROUP BY customer_id
                    )
                    SELECT customer_id, last_order_date,
                           DATEDIFF(CURDATE(), last_order_date) AS days_since_order
                    FROM customer_last_order_date
                    WHERE DATEDIFF(CURDATE(), last_order_date) > {int(churn_days)}
                    ORDER BY days_since_order DESC;
                """
                df = run_select_query(conn, q)
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(e)

        if metric_btn:
            try:
                if metric == "Recency":
                    q = """
                        WITH customer_last_order_date AS (
                            SELECT customer_id, MAX(order_date) AS last_order_date
                            FROM orders
                            GROUP BY customer_id
                        )
                        SELECT customer_id, last_order_date,
                               DATEDIFF(CURDATE(), last_order_date) AS days_since_order
                        FROM customer_last_order_date
                        WHERE customer_id = %s
                        ORDER BY DATEDIFF(CURDATE(), last_order_date) ASC;
                    """
                    df = run_select_query(conn, q, params=(int(cust_id),))

                elif metric == "Monetary":
                    q = """
                        SELECT o.customer_id, c.first_name, c.last_name,
                               ROUND(SUM(o.total_amount),2) AS total_spend
                        FROM orders o
                        JOIN customer c ON o.customer_id = c.customer_id
                        WHERE o.customer_id = %s
                        GROUP BY o.customer_id, c.first_name, c.last_name
                        ORDER BY SUM(o.total_amount) DESC;
                    """
                    df = run_select_query(conn, q, params=(int(cust_id),))

                elif metric == "Average Order Value":
                    q = """
                        SELECT * FROM customer_avg_value
                        WHERE customer_id = %s;
                    """
                    df = run_select_query(conn, q, params=(int(cust_id),))

                elif metric == "Customer Lifetime Value":
                    q = """
                        SELECT cav.customer_id, cav.first_name, cav.last_name,
                               ROUND(cav.avg_order_value * cto.total_orders,2) AS life_time_value
                        FROM customer_avg_value cav
                        LEFT JOIN customer_total_orders cto
                          ON cav.customer_id = cto.customer_id
                        WHERE cav.customer_id = %s;
                    """
                    df = run_select_query(conn, q, params=(int(cust_id),))

                elif metric == "Avg days between purchases":
                    q = """
                        WITH OrderIntervals AS (
                            SELECT customer_id, order_date,
                                   LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date) AS previous_order_date
                            FROM orders
                        )
                        SELECT customer_id,
                               ROUND(AVG(DATEDIFF(order_date, previous_order_date)),2) AS avg_days_between_purchases
                        FROM OrderIntervals
                        WHERE previous_order_date IS NOT NULL AND customer_id = %s
                        GROUP BY customer_id
                        ORDER BY avg_days_between_purchases ASC;
                    """
                    df = run_select_query(conn, q, params=(int(cust_id),))

                else:  # Tenure
                    q = """
                        SELECT customer_id,
                               MIN(order_date) AS first_order_date,
                               MAX(order_date) AS latest_order_date,
                               DATEDIFF(MAX(order_date), MIN(order_date)) AS tenure
                        FROM orders
                        WHERE customer_id = %s
                        GROUP BY customer_id
                        ORDER BY tenure ASC;
                    """
                    df = run_select_query(conn, q, params=(int(cust_id),))

                st.dataframe(df, use_container_width=True)

            except Exception as e:
                st.error(e)


# -------------------------
# 3) Reviews Insights (top reviewers, genres, best books, sentiment proc)
# -------------------------
with tabs[2]:

    year = st.selectbox("Year", [2022, 2023, 2024], index=0)
    if st.button("Run Sentiment Analysis"):
        try:
            cur = conn.cursor()
            cur.callproc("sentiment_by_year", [int(year)])
            rows = []
            for result in cur.stored_results():
                rows = result.fetchall()
            cur.close()

            df = pd.DataFrame(rows, columns=["review_classification", "total_reviews"])
            st.dataframe(df, use_container_width=True)

            if not df.empty:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(df["review_classification"].astype(str), df["total_reviews"].astype(float))
                ax.set_title(f"Sentiment by Year ({year})")
                ax.set_xlabel("Classification")
                ax.set_ylabel("Total Reviews")
                ax.tick_params(axis="x", rotation=30)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Stored procedure call failed: {e}")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Top reviewers"):
            try:
                q = """
                    SELECT r.customer_id, c.first_name, c.last_name,
                           COUNT(*) AS number_of_reviews_given
                    FROM reviews r
                    JOIN customer c ON r.customer_id = c.customer_id
                    GROUP BY r.customer_id, c.first_name, c.last_name
                    ORDER BY number_of_reviews_given DESC;
                """
                df = run_select_query(conn, q)
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(e)

    with c2:
        if st.button("Top 10 reviewed books"):
            try:
                q = """
                    SELECT book_id, title, review_classification
                    FROM reviews_per_book
                    LIMIT 10;
                """
                df = run_select_query(conn, q)
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(e)

    st.divider()

    c3, c4 = st.columns(2)
    with c3:
        if st.button("Genres by sentiment (best ordering)"):
            try:
                q = """
                    SELECT g.genre_title, rs.review_classification, COUNT(*) AS review_count
                    FROM review_sentiment rs
                    JOIN books b ON rs.book_id = b.book_id
                    JOIN genre g ON b.genre_id = g.genre_id
                    GROUP BY rs.review_classification, g.genre_title
                    ORDER BY rs.review_classification DESC, review_count DESC;
                """
                df = run_select_query(conn, q)
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(e)

    with c4:
        if st.button("Genres by sentiment (worst ordering)"):
            try:
                q = """
                    SELECT g.genre_title, rs.review_classification, COUNT(*) AS review_count
                    FROM review_sentiment rs
                    JOIN books b ON rs.book_id = b.book_id
                    JOIN genre g ON b.genre_id = g.genre_id
                    GROUP BY rs.review_classification, g.genre_title
                    ORDER BY rs.review_classification ASC, review_count DESC;
                """
                df = run_select_query(conn, q)
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(e)


# -------------------------
# 4) Customer Demographics (SELECT * FROM customer)
# -------------------------
with tabs[3]:

    if st.button("Load demographics"):
        try:
            df = run_select_query(conn, "SELECT * FROM customer;")
            st.dataframe(df, use_container_width=True)

            if not df.empty:
                # Gender pie (grouped)
                if "gender" in df.columns:
                    g = df.copy()
                    g["gender_grouped"] = g["gender"].apply(lambda x: x if x in ["Male", "Female"] else "Others")
                    counts = g["gender_grouped"].value_counts()

                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
                    ax.set_title("Gender Distribution")
                    st.pyplot(fig)

                # Preferred language treemap
                if "preferred_language" in df.columns:
                    counts = df["preferred_language"].value_counts()
                    pct = (counts / counts.sum()) * 100
                    labels = [f"{lang}\n{p:.1f}%" for lang, p in zip(counts.index, pct)]

                    fig, ax = plt.subplots(figsize=(9, 5))
                    squarify.plot(sizes=counts.values, label=labels, alpha=0.85, ax=ax)
                    ax.set_title("Preferred Language Distribution (%)")
                    ax.axis("off")
                    st.pyplot(fig)

        except Exception as e:
            st.error(e)



# -------------------------
# 5) Sales: Performance
# -------------------------
with tabs[4]:
    st.subheader("Performance Overview")

    year = st.selectbox("Year", [2022, 2023, 2024], index=0, key="perf_year")
    show_treemap = st.checkbox("Genre-wise sales treemap", value=True)
    show_top_bottom = st.checkbox("Top & bottom selling books", value=True)

    if st.button("Run performance queries"):
        if show_treemap:
            try:
                q = f"""
                    SELECT g.genre_title, SUM(od.quantity) AS total_books_sold
                    FROM order_details od
                    JOIN books b ON od.book_id = b.book_id
                    JOIN genre g ON b.genre_id = g.genre_id
                    JOIN orders o ON od.order_id = o.order_id
                    WHERE YEAR(o.order_date) = {int(year)}
                    GROUP BY g.genre_id
                    ORDER BY total_books_sold DESC;
                """
                df = run_select_query(conn, q)
                # st.dataframe(df, use_container_width=True)

                if not df.empty:
                    fig, ax = plt.subplots(figsize=(25, 10))
                    squarify.plot(
                        sizes=df["total_books_sold"],
                        label=df["genre_title"],
                        alpha=0.85,
                        ax=ax
                    )
                    for text in ax.texts:
                        text.set_fontsize(12) 
                    ax.set_title(f"Genre-wise Book Sales ({year})")
                    ax.axis("off")
                    st.pyplot(fig)
            except Exception as e:
                st.error(e)

        if show_top_bottom:
            try:
                q = f"""
                    SELECT b.title, SUM(od.quantity) AS total_book_sales
                    FROM order_details od
                    JOIN books b ON od.book_id = b.book_id
                    JOIN orders o ON od.order_id = o.order_id
                    WHERE YEAR(o.order_date) = {int(year)}
                    GROUP BY b.book_id
                    ORDER BY total_book_sales DESC;
                """
                df = run_select_query(conn, q)

                if df.empty:
                    st.info("No rows returned.")
                else:
                    if len(df) >= 20:
                        df = pd.concat([df.head(10), df.tail(10)], ignore_index=True)

                    # st.dataframe(df, use_container_width=True)

                    fig, ax = plt.subplots(figsize=(12, 7))

                    # 1) Sort so chart reads top-to-bottom
                    df_plot = df.sort_values("total_book_sales", ascending=False).copy()

                    # 2) Shorten long titles for readability (keep full title elsewhere if needed)
                    max_len = 35
                    df_plot["title_short"] = df_plot["title"].astype(str).apply(
                        lambda x: x if len(x) <= max_len else x[:max_len] + "…"
                    )

                    # 3) Horizontal bars (much better for long labels)
                    sns.barplot(
                        data=df_plot,
                        y="title_short",
                        x="total_book_sales",
                        ax=ax
                    )

                    ax.set_title(f"Top & Bottom Selling Books ({year})", pad=12)
                    ax.set_xlabel("Total Books Sold")
                    ax.set_ylabel("")  # remove y-axis label (titles already show)
                    ax.grid(axis="x", linestyle="--", alpha=0.4)

                    # 4) Add value labels at the end of each bar
                    for p in ax.patches:
                        width = p.get_width()
                        ax.text(
                            width + (0.02 * df_plot["total_book_sales"].max()),
                            p.get_y() + p.get_height() / 2,
                            f"{int(width)}",
                            va="center"
                        )

                    fig.tight_layout()
                    st.pyplot(fig)

                    # Optional: show mapping of shortened titles to full titles
                    with st.expander("See full book titles"):
                        st.dataframe(df_plot[["title", "total_book_sales"]].sort_values("total_book_sales", ascending=False),
                                    use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(e)


# -------------------------
# 6) Sales: Insights
# -------------------------
with tabs[5]:
    st.subheader("Sales Insights")

    insight = st.selectbox(
        "Insight query",
        ["Books with Declining Sales", "Revenue Contribution by Publisher"]
    )

    if st.button("Run insight query"):
        try:
            if insight == "Poorly Performing Books":
                q = """
                    SELECT
                        b.book_id, b.title,
                        COUNT(r.review_id) AS total_reviews,
                        AVG(r.rating) AS avg_rating,
                        SUM(CASE WHEN r.rating <= 2 THEN 1 ELSE 0 END) AS negative_review_count
                    FROM books b
                    LEFT JOIN reviews r ON b.book_id = r.book_id
                    GROUP BY b.book_id, b.title
                    HAVING avg_rating < 3 OR negative_review_count > 3
                    ORDER BY negative_review_count DESC, avg_rating ASC;
                """
                df = run_select_query(conn, q)

            elif insight == "Books with Declining Sales":
                q = """
                    WITH MonthlyBookSales AS (
                        SELECT od.book_id, DATE_FORMAT(o.order_date, '%Y-%m-01') AS Month,
                               SUM(od.quantity) AS Monthly_Sales
                        FROM order_details od
                        JOIN orders o ON od.order_id = o.order_id
                        GROUP BY od.book_id, Month
                    ),
                    SalesTrend AS (
                        SELECT book_id, Month, Monthly_Sales,
                               LAG(Monthly_Sales) OVER (PARTITION BY book_id ORDER BY Month) AS Prev_Month_Sales
                        FROM MonthlyBookSales
                    ),
                    DecliningSales AS (
                        SELECT book_id, Month, Monthly_Sales, Prev_Month_Sales,
                               ROUND(((Monthly_Sales - Prev_Month_Sales) / Prev_Month_Sales) * 100, 2) AS Sales_Change
                        FROM SalesTrend
                        WHERE Prev_Month_Sales IS NOT NULL
                    )
                    SELECT b.book_id, b.title, d.Month, d.Monthly_Sales,
                           d.Prev_Month_Sales, d.Sales_Change
                    FROM DecliningSales d
                    JOIN books b ON d.book_id = b.book_id
                    WHERE d.Sales_Change < -10
                    ORDER BY d.Sales_Change ASC
                    LIMIT 10;
                """
                df = run_select_query(conn, q)

            else:
                q = """
                    SELECT p.publisher_name, SUM(o.total_amount) AS Publisher_Revenue
                    FROM publisher p
                    JOIN books b ON p.publisher_id = b.publisher_id
                    JOIN order_details od ON b.book_id = od.book_id
                    JOIN orders o ON od.order_id = o.order_id
                    GROUP BY p.publisher_name
                    ORDER BY Publisher_Revenue DESC;
                """
                df = run_select_query(conn, q)

            st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error(e)

# Close connection at the very end
try:
    conn.close()
except Exception:
    pass
