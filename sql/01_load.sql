-- ---------------------------------------------------------------------------
-- Load the CSV export into the typed schema. Run from the repository root.
--
-- Columns are named and cast explicitly rather than taken from read_csv_auto's
-- inference: the zip prefixes are zero-padded strings in the export and would
-- otherwise arrive as text in one table and integers in another, which breaks
-- the joins silently rather than loudly.
-- ---------------------------------------------------------------------------

INSERT INTO customers
SELECT customer_id, customer_unique_id,
       TRY_CAST(customer_zip_code_prefix AS INTEGER),
       lower(trim(customer_city)), upper(trim(customer_state))
FROM read_csv_auto('data/raw/olist_customers_dataset.csv', header=true);

INSERT INTO sellers
SELECT seller_id,
       TRY_CAST(seller_zip_code_prefix AS INTEGER),
       lower(trim(seller_city)), upper(trim(seller_state))
FROM read_csv_auto('data/raw/olist_sellers_dataset.csv', header=true);

INSERT INTO products
SELECT product_id, product_category_name,
       TRY_CAST(product_name_lenght AS INTEGER),
       TRY_CAST(product_description_lenght AS INTEGER),
       TRY_CAST(product_photos_qty AS INTEGER),
       TRY_CAST(product_weight_g AS DOUBLE),
       TRY_CAST(product_length_cm AS DOUBLE),
       TRY_CAST(product_height_cm AS DOUBLE),
       TRY_CAST(product_width_cm AS DOUBLE)
FROM read_csv_auto('data/raw/olist_products_dataset.csv', header=true);

INSERT INTO category_translation
SELECT product_category_name, product_category_name_english
FROM read_csv_auto('data/raw/product_category_name_translation.csv', header=true);

INSERT INTO orders
SELECT order_id, customer_id, order_status,
       TRY_CAST(order_purchase_timestamp     AS TIMESTAMP),
       TRY_CAST(order_approved_at            AS TIMESTAMP),
       TRY_CAST(order_delivered_carrier_date AS TIMESTAMP),
       TRY_CAST(order_delivered_customer_date AS TIMESTAMP),
       TRY_CAST(order_estimated_delivery_date AS TIMESTAMP)
FROM read_csv_auto('data/raw/olist_orders_dataset.csv', header=true);

-- Two products referenced by order_items are absent from the products export.
-- The FK would reject the whole load, so the gap is filled with stub rows and
-- then reported by the data-quality suite as a real referential-integrity miss.
INSERT INTO products (product_id)
SELECT DISTINCT i.product_id
FROM read_csv_auto('data/raw/olist_order_items_dataset.csv', header=true) i
LEFT JOIN products p ON p.product_id = i.product_id
WHERE p.product_id IS NULL;

INSERT INTO order_items
SELECT order_id, TRY_CAST(order_item_id AS INTEGER), product_id, seller_id,
       TRY_CAST(shipping_limit_date AS TIMESTAMP),
       TRY_CAST(price AS DOUBLE), TRY_CAST(freight_value AS DOUBLE)
FROM read_csv_auto('data/raw/olist_order_items_dataset.csv', header=true);

INSERT INTO order_payments
SELECT order_id, TRY_CAST(payment_sequential AS INTEGER), payment_type,
       TRY_CAST(payment_installments AS INTEGER), TRY_CAST(payment_value AS DOUBLE)
FROM read_csv_auto('data/raw/olist_order_payments_dataset.csv', header=true);

INSERT INTO order_reviews
SELECT review_id, order_id, TRY_CAST(review_score AS INTEGER),
       review_comment_title, review_comment_message,
       TRY_CAST(review_creation_date AS TIMESTAMP),
       TRY_CAST(review_answer_timestamp AS TIMESTAMP)
FROM read_csv_auto('data/raw/olist_order_reviews_dataset.csv', header=true);

INSERT INTO geolocation
SELECT TRY_CAST(geolocation_zip_code_prefix AS INTEGER),
       TRY_CAST(geolocation_lat AS DOUBLE), TRY_CAST(geolocation_lng AS DOUBLE),
       lower(trim(geolocation_city)), upper(trim(geolocation_state))
FROM read_csv_auto('data/raw/olist_geolocation_dataset.csv', header=true);

-- Median centroid per zip prefix, inside a Brazil bounding box. The raw table
-- contains points in the Atlantic and one in the northern hemisphere; a mean
-- would carry them into the answer, a median will not.
INSERT INTO zip_centroid
SELECT zip_prefix,
       median(lat) AS lat,
       median(lng) AS lng,
       count(*)    AS n_points
FROM geolocation
WHERE lat BETWEEN -34.0 AND 5.5
  AND lng BETWEEN -74.0 AND -34.0
GROUP BY zip_prefix;
