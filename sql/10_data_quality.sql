-- ---------------------------------------------------------------------------
-- Data-quality suite.
--
-- Run before anything is modelled, and re-run whenever the source changes.
-- Each row is one assertion with a count of failures, so the output is a
-- report rather than a pass/fail: on a real export almost nothing is perfectly
-- clean, and the useful question is which defects are large enough to change a
-- decision. Severity encodes that judgement:
--
--   error  -- would corrupt the analysis; must be handled before modelling
--   warn   -- must be handled or explicitly excluded, and said out loud
--   info   -- known shape of the data, recorded so it is not rediscovered
-- ---------------------------------------------------------------------------

CREATE OR REPLACE TABLE dq_checks AS
WITH raw AS (

-- ---- referential integrity ------------------------------------------------
SELECT 'orders.customer_id_orphans' AS check_name, 'orders' AS scope, 'error' AS severity,
       (SELECT count(*) FROM orders o LEFT JOIN customers c USING (customer_id)
        WHERE c.customer_id IS NULL) AS n_fail,
       (SELECT count(*) FROM orders) AS n_total,
       'orders referencing a customer row that does not exist' AS note
UNION ALL SELECT 'order_items.product_id_orphans', 'order_items', 'error',
       (SELECT count(*) FROM order_items i LEFT JOIN products p USING (product_id)
        WHERE p.product_id IS NULL), (SELECT count(*) FROM order_items),
       'line items whose product is missing from the product export'
UNION ALL SELECT 'products.stub_rows_created', 'products', 'warn',
       (SELECT count(*) FROM products WHERE product_category_name IS NULL AND weight_g IS NULL),
       (SELECT count(*) FROM products),
       'product rows with no attributes at all, incl. stubs added at load time'
UNION ALL SELECT 'orders.no_line_items', 'orders', 'warn',
       (SELECT count(*) FROM orders o LEFT JOIN order_items i USING (order_id)
        WHERE i.order_id IS NULL), (SELECT count(*) FROM orders),
       'orders with no line items; cannot be priced or attributed to a seller'
UNION ALL SELECT 'orders.no_payment_record', 'orders', 'warn',
       (SELECT count(*) FROM orders o LEFT JOIN order_payments p USING (order_id)
        WHERE p.order_id IS NULL), (SELECT count(*) FROM orders),
       'orders with no payment row'

-- ---- uniqueness -----------------------------------------------------------
UNION ALL SELECT 'reviews.duplicate_review_id', 'order_reviews', 'warn',
       (SELECT count(*) FROM order_reviews) - (SELECT count(DISTINCT review_id) FROM order_reviews),
       (SELECT count(*) FROM order_reviews),
       'review_id is not unique in the export; it cannot be used as a key'
UNION ALL SELECT 'reviews.orders_with_multiple_reviews', 'order_reviews', 'warn',
       (SELECT count(*) FROM (SELECT order_id FROM order_reviews
                              GROUP BY order_id HAVING count(*) > 1)),
       (SELECT count(DISTINCT order_id) FROM order_reviews),
       'orders carrying more than one review; must be aggregated, not joined'
UNION ALL SELECT 'customers.unique_id_reuse', 'customers', 'info',
       (SELECT count(*) FROM customers) - (SELECT count(DISTINCT customer_unique_id) FROM customers),
       (SELECT count(*) FROM customers),
       'customer_id is per-order; customer_unique_id is the person. Repeat buyers'

-- ---- completeness ---------------------------------------------------------
UNION ALL SELECT 'orders.delivered_without_delivery_date', 'orders', 'error',
       (SELECT count(*) FROM orders WHERE order_status = 'delivered' AND delivered_at IS NULL),
       (SELECT count(*) FROM orders WHERE order_status = 'delivered'),
       'status says delivered but no delivery timestamp; excluded from the label'
UNION ALL SELECT 'orders.missing_approved_at', 'orders', 'warn',
       (SELECT count(*) FROM orders WHERE approved_at IS NULL AND order_status <> 'canceled'),
       (SELECT count(*) FROM orders WHERE order_status <> 'canceled'),
       'non-cancelled orders never approved'
UNION ALL SELECT 'orders.missing_carrier_handoff', 'orders', 'info',
       (SELECT count(*) FROM orders WHERE carrier_handoff_at IS NULL AND order_status = 'delivered'),
       (SELECT count(*) FROM orders WHERE order_status = 'delivered'),
       'delivered orders with no carrier handoff timestamp'
UNION ALL SELECT 'products.missing_category', 'products', 'warn',
       (SELECT count(*) FROM products WHERE product_category_name IS NULL),
       (SELECT count(*) FROM products), 'products with no category'
UNION ALL SELECT 'products.missing_weight_or_dims', 'products', 'warn',
       (SELECT count(*) FROM products
        WHERE weight_g IS NULL OR length_cm IS NULL OR height_cm IS NULL OR width_cm IS NULL),
       (SELECT count(*) FROM products), 'products with no weight or dimensions'
UNION ALL SELECT 'categories.untranslated', 'products', 'info',
       (SELECT count(*) FROM (SELECT DISTINCT p.product_category_name FROM products p
        LEFT JOIN category_translation t USING (product_category_name)
        WHERE p.product_category_name IS NOT NULL AND t.product_category_name IS NULL)),
       (SELECT count(DISTINCT product_category_name) FROM products),
       'categories with no English translation'

-- ---- temporal consistency -------------------------------------------------
UNION ALL SELECT 'orders.approved_before_purchase', 'orders', 'error',
       (SELECT count(*) FROM orders WHERE approved_at < purchased_at),
       (SELECT count(*) FROM orders WHERE approved_at IS NOT NULL),
       'approval timestamp earlier than the purchase timestamp'
UNION ALL SELECT 'orders.handoff_before_approval', 'orders', 'warn',
       (SELECT count(*) FROM orders WHERE carrier_handoff_at < approved_at),
       (SELECT count(*) FROM orders WHERE carrier_handoff_at IS NOT NULL AND approved_at IS NOT NULL),
       'handed to the carrier before the payment was approved'
UNION ALL SELECT 'orders.delivered_before_handoff', 'orders', 'warn',
       (SELECT count(*) FROM orders WHERE delivered_at < carrier_handoff_at),
       (SELECT count(*) FROM orders WHERE delivered_at IS NOT NULL AND carrier_handoff_at IS NOT NULL),
       'delivered to the customer before leaving the seller'
UNION ALL SELECT 'orders.promise_before_purchase', 'orders', 'error',
       (SELECT count(*) FROM orders WHERE estimated_delivery_at < purchased_at),
       (SELECT count(*) FROM orders WHERE estimated_delivery_at IS NOT NULL),
       'promised delivery date earlier than the order date'
UNION ALL SELECT 'reviews.created_before_purchase', 'order_reviews', 'warn',
       (SELECT count(*) FROM order_reviews r JOIN orders o USING (order_id)
        WHERE r.review_created_at < o.purchased_at),
       (SELECT count(*) FROM order_reviews r JOIN orders o USING (order_id)),
       'review created before the order it reviews'

-- ---- value ranges ---------------------------------------------------------
UNION ALL SELECT 'reviews.score_out_of_range', 'order_reviews', 'error',
       (SELECT count(*) FROM order_reviews WHERE review_score NOT BETWEEN 1 AND 5),
       (SELECT count(*) FROM order_reviews), 'review score outside 1-5'
UNION ALL SELECT 'order_items.nonpositive_price', 'order_items', 'warn',
       (SELECT count(*) FROM order_items WHERE price <= 0),
       (SELECT count(*) FROM order_items), 'line items priced at or below zero'
UNION ALL SELECT 'order_items.zero_freight', 'order_items', 'info',
       (SELECT count(*) FROM order_items WHERE freight_value = 0),
       (SELECT count(*) FROM order_items), 'free-shipping line items'
UNION ALL SELECT 'payments.zero_value_non_voucher', 'order_payments', 'warn',
       (SELECT count(*) FROM order_payments WHERE payment_value = 0 AND payment_type <> 'voucher'),
       (SELECT count(*) FROM order_payments), 'non-voucher payment rows of zero value'
UNION ALL SELECT 'products.implausible_weight', 'products', 'warn',
       (SELECT count(*) FROM products WHERE weight_g <= 0 OR weight_g > 40000),
       (SELECT count(*) FROM products WHERE weight_g IS NOT NULL),
       'weight non-positive or above 40 kg'

-- ---- geography ------------------------------------------------------------
UNION ALL SELECT 'geolocation.outside_brazil_bbox', 'geolocation', 'warn',
       (SELECT count(*) FROM geolocation
        WHERE lat NOT BETWEEN -34.0 AND 5.5 OR lng NOT BETWEEN -74.0 AND -34.0),
       (SELECT count(*) FROM geolocation),
       'coordinates outside Brazil; dropped when building zip centroids'
UNION ALL SELECT 'customers.zip_without_centroid', 'customers', 'warn',
       (SELECT count(*) FROM customers c LEFT JOIN zip_centroid z
        ON z.zip_prefix = c.customer_zip_prefix WHERE z.zip_prefix IS NULL),
       (SELECT count(*) FROM customers),
       'customer zip prefixes with no geolocation match; distance is unknown'
UNION ALL SELECT 'sellers.zip_without_centroid', 'sellers', 'warn',
       (SELECT count(*) FROM sellers s LEFT JOIN zip_centroid z
        ON z.zip_prefix = s.seller_zip_prefix WHERE z.zip_prefix IS NULL),
       (SELECT count(*) FROM sellers),
       'seller zip prefixes with no geolocation match'
UNION ALL SELECT 'geolocation.state_disagrees_with_customer', 'geolocation', 'info',
       (SELECT count(*) FROM customers c JOIN geolocation g
        ON g.zip_prefix = c.customer_zip_prefix AND g.state <> c.customer_state),
       (SELECT count(*) FROM customers c JOIN geolocation g
        ON g.zip_prefix = c.customer_zip_prefix),
       'zip prefixes that straddle a state border in the geolocation table'
)
SELECT check_name, scope, severity, n_fail, n_total,
       CASE WHEN n_total > 0 THEN round(100.0 * n_fail / n_total, 3) ELSE NULL END AS fail_pct,
       note
FROM raw
ORDER BY CASE severity WHEN 'error' THEN 0 WHEN 'warn' THEN 1 ELSE 2 END, n_fail DESC;
