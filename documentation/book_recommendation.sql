-- what all books did each customer order

CREATE VIEW customer_book_details AS
SELECT orders.customer_id, order_details.book_id, books.title, genre.genre_title
FROM orders
JOIN order_details
ON orders.order_id = order_details.order_id
JOIN books
ON order_details.book_id = books.book_id
JOIN genre
ON books.genre_id = genre.genre_id;



-- Most frequently ordered genre by each customer
CREATE VIEW customer_frequent_genre AS
WITH GenreFrequency AS (
    SELECT orders.customer_id, genre.genre_title, COUNT(*) AS genre_count
    FROM orders
    JOIN order_details ON orders.order_id = order_details.order_id
    JOIN books ON order_details.book_id = books.book_id
    JOIN genre ON books.genre_id = genre.genre_id
    GROUP BY orders.customer_id, genre.genre_title
)
SELECT customer_id, genre_title, genre_count
FROM (
    SELECT customer_id, genre_title, genre_count,
           RANK() OVER (PARTITION BY customer_id ORDER BY genre_count DESC) AS `rank`
    FROM GenreFrequency
) RankedGenres
WHERE `rank` = 1;

SELECT * FROM customer_frequent_genre;

-- best sellers by genre
CREATE VIEW best_sellers_by_genre AS
WITH book_sales_per_book AS (
    SELECT order_details.book_id, books.genre_id, SUM(quantity) AS total_book_sales
    FROM order_details
    JOIN books ON order_details.book_id = books.book_id
    GROUP BY order_details.book_id, books.genre_id
),
max_sales_per_genre AS (
    SELECT genre_id, MAX(total_book_sales) AS max_sales_values_per_genre
    FROM book_sales_per_book
    GROUP BY genre_id
)
SELECT DISTINCT book_sales_per_book.genre_id, genre.genre_title, books.title, 
       book_sales_per_book.book_id, book_sales_per_book.total_book_sales
FROM book_sales_per_book
JOIN max_sales_per_genre 
ON book_sales_per_book.genre_id = max_sales_per_genre.genre_id 
AND book_sales_per_book.total_book_sales = max_sales_per_genre.max_sales_values_per_genre
JOIN books ON book_sales_per_book.book_id = books.book_id
JOIN genre ON book_sales_per_book.genre_id = genre.genre_id;

SELECT * FROM customer_book_details;
SELECT * FROM customer_frequent_genre;
SELECT * FROM best_sellers_by_genre;

-- Recommendation

-- 1). What is the most frequent genre by each customer

SELECT genre_title FROM customer_frequent_genre
WHERE customer_id = 100000
LIMIT 1;

-- 2). What all books did he already read in the most frequently ordered genre

SELECT * FROM customer_book_details
WHERE customer_id = 100000 AND genre_title = (SELECT genre_title FROM customer_frequent_genre
	WHERE customer_id = 100000
	LIMIT 1);
    
-- 3). what are the books he did read from the top selling books of that genre

WITH already_purchased AS (
    SELECT * FROM customer_book_details
    WHERE customer_id = 100000 AND genre_title = (
        SELECT genre_title FROM customer_frequent_genre
        WHERE customer_id = 100000
        LIMIT 1
    )
)
SELECT DISTINCT(best_sellers_by_genre.genre_title), best_sellers_by_genre.title, best_sellers_by_genre.book_id 
FROM best_sellers_by_genre
LEFT JOIN already_purchased
ON best_sellers_by_genre.genre_title = already_purchased.genre_title
WHERE best_sellers_by_genre.genre_title = already_purchased.genre_title ;
