-- ---------------------------------------------------------------------------
-- order_fact: one row per delivered order, with the label and two clearly
-- separated feature sets.
--
-- The separation is the whole point of this file.
--
--   at_*    known when the customer clicks buy. These are the only columns the
--           predictive model may use, because a model that decides whether to
--           intervene on an order has to decide it at that moment.
--   post_*  observed later -- when the seller handed the parcel over, how long
--           it sat, what the customer scored the order. Never fed to the model.
--           They exist for the root-cause decomposition, which is a description
--           of what happened, not a prediction of what will.
--
-- Most public notebooks on this dataset quietly use order_delivered_carrier_date
-- as a feature and report ROC-AUC around 0.95. That number is unreachable: at
-- prediction time the parcel has not been handed over yet. scripts/03_model.py
-- fits the leaky variant on purpose and reports both, so the size of the
-- self-deception is measured rather than asserted.
--
-- The seller and route history columns are the subtle case. A seller's past
-- late rate is legitimate signal, but only the part of it that was *observable*
-- by the time this order was placed: a previous order counts only once it has
-- been delivered, not once it has been ordered. That is an as-of join on
-- delivery date, not a group-by on seller, and the difference is worth roughly
-- a point of AUC in the optimistic direction if you get it wrong.
-- ---------------------------------------------------------------------------

-- Population and label -------------------------------------------------------
CREATE OR REPLACE VIEW v_order_label AS
SELECT
    o.order_id,
    o.customer_id,
    o.purchased_at,
    o.approved_at,
    o.carrier_handoff_at,
    o.delivered_at,
    o.estimated_delivery_at,
    -- The promise is a calendar date, so lateness is judged on calendar days.
    -- Comparing raw timestamps would mark an order delivered at 09:00 on the
    -- promised day as late, which is not how the promise was made.
    CASE WHEN o.delivered_at::DATE > o.estimated_delivery_at::DATE THEN 1 ELSE 0 END AS is_late,
    date_diff('day', o.estimated_delivery_at::DATE, o.delivered_at::DATE) AS days_vs_promise
FROM orders o
WHERE o.order_status = 'delivered'
  AND o.delivered_at IS NOT NULL
  AND o.estimated_delivery_at IS NOT NULL
  AND o.purchased_at IS NOT NULL;

-- Basket shape ---------------------------------------------------------------
CREATE OR REPLACE VIEW v_items_agg AS
SELECT
    i.order_id,
    count(*)                              AS n_items,
    count(DISTINCT i.seller_id)           AS n_sellers,
    count(DISTINCT i.product_id)          AS n_products,
    count(DISTINCT p.product_category_name) AS n_categories,
    sum(i.price)                          AS total_price,
    sum(i.freight_value)                  AS total_freight,
    max(i.price)                          AS max_item_price,
    sum(coalesce(p.weight_g, 0))          AS total_weight_g,
    max(p.length_cm * p.height_cm * p.width_cm) AS max_volume_cm3,
    avg(p.photos_qty)                     AS avg_photos,
    max(CASE WHEN p.weight_g IS NULL OR p.length_cm IS NULL THEN 1 ELSE 0 END) AS any_missing_dims,
    min(i.shipping_limit_date)            AS first_shipping_limit
FROM order_items i
LEFT JOIN products p USING (product_id)
GROUP BY i.order_id;

-- The line the order is really "about": the most expensive item. Ties broken
-- by product_id so the choice is reproducible run to run.
CREATE OR REPLACE VIEW v_primary_line AS
SELECT order_id, seller_id AS primary_seller_id, product_id AS primary_product_id,
       price AS primary_price
FROM (
    SELECT i.*, row_number() OVER (PARTITION BY i.order_id
                                   ORDER BY i.price DESC, i.product_id) AS rn
    FROM order_items i
) WHERE rn = 1;

-- Payment: the largest payment row decides the recorded method.
CREATE OR REPLACE VIEW v_payment AS
SELECT order_id,
       max(n_rows)                AS n_payment_rows,
       max(total_value)           AS payment_value,
       arg_max(payment_type, payment_value)   AS payment_type,
       arg_max(payment_installments, payment_value) AS payment_installments
FROM (
    SELECT order_id, payment_type, payment_installments, payment_value,
           count(*)     OVER (PARTITION BY order_id) AS n_rows,
           sum(payment_value) OVER (PARTITION BY order_id) AS total_value
    FROM order_payments
) GROUP BY order_id;

-- Review outcome. Orders with several reviews are averaged rather than joined,
-- which would otherwise duplicate the order row -- see the DQ check.
CREATE OR REPLACE VIEW v_review AS
SELECT order_id, avg(review_score::DOUBLE) AS review_score, count(*) AS n_reviews
FROM order_reviews GROUP BY order_id;

-- ---------------------------------------------------------------------------
-- Observable history: as-of aggregates.
--
-- One event per (seller, delivery date) so the cumulative sums have no ties,
-- then an ASOF join that takes the state of that cumulative series strictly
-- before the day the order was placed.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_events AS
SELECT l.order_id, pl.primary_seller_id AS seller_id, c.customer_state,
       s.seller_state, l.delivered_at::DATE AS ev_date, l.is_late
FROM v_order_label l
JOIN v_primary_line pl USING (order_id)
JOIN customers c ON c.customer_id = l.customer_id
LEFT JOIN sellers s ON s.seller_id = pl.primary_seller_id;

CREATE OR REPLACE VIEW v_seller_cum AS
SELECT seller_id, ev_date,
       sum(n)      OVER (PARTITION BY seller_id ORDER BY ev_date ROWS UNBOUNDED PRECEDING) AS cum_n,
       sum(n_late) OVER (PARTITION BY seller_id ORDER BY ev_date ROWS UNBOUNDED PRECEDING) AS cum_late
FROM (SELECT seller_id, ev_date, count(*) AS n, sum(is_late) AS n_late
      FROM v_events GROUP BY 1, 2);

CREATE OR REPLACE VIEW v_route_cum AS
SELECT seller_state, customer_state, ev_date,
       sum(n)      OVER (PARTITION BY seller_state, customer_state ORDER BY ev_date ROWS UNBOUNDED PRECEDING) AS cum_n,
       sum(n_late) OVER (PARTITION BY seller_state, customer_state ORDER BY ev_date ROWS UNBOUNDED PRECEDING) AS cum_late
FROM (SELECT seller_state, customer_state, ev_date, count(*) AS n, sum(is_late) AS n_late
      FROM v_events GROUP BY 1, 2, 3);

-- The smoothing prior is itself an as-of quantity: the late rate over every
-- order delivered so far. Using the final overall rate would leak the future
-- into every early row, by a small amount, in the helpful direction.
CREATE OR REPLACE VIEW v_global_cum AS
SELECT ev_date,
       sum(n)      OVER (ORDER BY ev_date ROWS UNBOUNDED PRECEDING) AS cum_n,
       sum(n_late) OVER (ORDER BY ev_date ROWS UNBOUNDED PRECEDING) AS cum_late
FROM (SELECT ev_date, count(*) AS n, sum(is_late) AS n_late FROM v_events GROUP BY 1);

-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE order_fact AS
WITH base AS (
    SELECT l.*, pl.primary_seller_id, pl.primary_product_id,
           c.customer_state, c.customer_zip_prefix, c.customer_unique_id,
           s.seller_state, s.seller_zip_prefix,
           pr.product_category_name AS primary_category,
           ia.*  EXCLUDE (order_id),
           pay.* EXCLUDE (order_id),
           rv.review_score, rv.n_reviews
    FROM v_order_label l
    JOIN v_items_agg ia USING (order_id)
    JOIN v_primary_line pl USING (order_id)
    JOIN customers c ON c.customer_id = l.customer_id
    LEFT JOIN sellers  s  ON s.seller_id = pl.primary_seller_id
    LEFT JOIN products pr ON pr.product_id = pl.primary_product_id
    LEFT JOIN v_payment pay USING (order_id)
    LEFT JOIN v_review  rv  USING (order_id)
),
geo AS (
    SELECT b.*, zc.lat AS cust_lat, zc.lng AS cust_lng,
           zs.lat AS sell_lat, zs.lng AS sell_lng
    FROM base b
    LEFT JOIN zip_centroid zc ON zc.zip_prefix = b.customer_zip_prefix
    LEFT JOIN zip_centroid zs ON zs.zip_prefix = b.seller_zip_prefix
),
with_seller_hist AS (
    SELECT g.*, sc.cum_n AS seller_prior_n, sc.cum_late AS seller_prior_late
    FROM geo g
    ASOF LEFT JOIN v_seller_cum sc
      ON g.primary_seller_id = sc.seller_id AND g.purchased_at::DATE > sc.ev_date
),
with_route_hist AS (
    SELECT h.*, rc.cum_n AS route_prior_n, rc.cum_late AS route_prior_late
    FROM with_seller_hist h
    ASOF LEFT JOIN v_route_cum rc
      ON h.seller_state = rc.seller_state AND h.customer_state = rc.customer_state
     AND h.purchased_at::DATE > rc.ev_date
),
with_global AS (
    SELECT r.*, gc.cum_n AS global_prior_n, gc.cum_late AS global_prior_late
    FROM with_route_hist r
    ASOF LEFT JOIN v_global_cum gc ON r.purchased_at::DATE > gc.ev_date
)
SELECT
    order_id, customer_unique_id, primary_seller_id,
    purchased_at, delivered_at, estimated_delivery_at,
    is_late, days_vs_promise,

    -- ---- known at purchase --------------------------------------------------
    date_diff('day', purchased_at::DATE, estimated_delivery_at::DATE) AS at_promise_days,
    date_diff('day', purchased_at::DATE, first_shipping_limit::DATE)  AS at_seller_deadline_days,
    extract('hour'  FROM purchased_at)                                AS at_purchase_hour,
    extract('dow'   FROM purchased_at)                                AS at_purchase_dow,
    extract('month' FROM purchased_at)                                AS at_purchase_month,
    CASE WHEN extract('dow' FROM purchased_at) IN (0, 6) THEN 1 ELSE 0 END AS at_is_weekend,
    date_diff('day', (SELECT min(purchased_at)::DATE FROM v_order_label), purchased_at::DATE)
                                                                      AS at_days_since_start,
    n_items       AS at_n_items,
    n_sellers     AS at_n_sellers,
    n_products    AS at_n_products,
    n_categories  AS at_n_categories,
    total_price   AS at_total_price,
    total_freight AS at_total_freight,
    max_item_price AS at_max_item_price,
    CASE WHEN total_price + total_freight > 0
         THEN total_freight / (total_price + total_freight) END       AS at_freight_ratio,
    total_weight_g   AS at_total_weight_g,
    max_volume_cm3   AS at_max_volume_cm3,
    avg_photos       AS at_avg_photos,
    any_missing_dims AS at_any_missing_dims,
    customer_state   AS at_customer_state,
    seller_state     AS at_seller_state,
    CASE WHEN customer_state = seller_state THEN 1 ELSE 0 END         AS at_same_state,
    primary_category AS at_primary_category,
    payment_type     AS at_payment_type,
    payment_installments AS at_payment_installments,
    payment_value    AS at_payment_value,
    n_payment_rows   AS at_n_payment_rows,

    -- Great-circle distance between the seller's and the customer's zip
    -- centroid. Null when either prefix has no geolocation row; left as null
    -- rather than imputed to zero, which would read as "same building".
    CASE WHEN cust_lat IS NOT NULL AND sell_lat IS NOT NULL THEN
        2 * 6371.0 * asin(sqrt(
            pow(sin(radians(cust_lat - sell_lat) / 2), 2) +
            cos(radians(sell_lat)) * cos(radians(cust_lat)) *
            pow(sin(radians(cust_lng - sell_lng) / 2), 2)))
    END AS at_distance_km,

    -- ---- observable history (as-of, leakage-free) ---------------------------
    coalesce(seller_prior_n, 0) AS at_seller_prior_n,
    coalesce(route_prior_n, 0)  AS at_route_prior_n,
    -- The prior is itself smoothed toward 0.08 with a weight of 5 orders.
    -- Without that inner smoothing the prior is a raw ratio, and in the first
    -- weeks of the data -- when a handful of orders have been delivered and
    -- none of them late -- it evaluates to exactly 0. Every cold-start seller
    -- then inherits "has never been late", which is the strongest possible
    -- claim to make from no evidence at all, and it points the wrong way.
    (coalesce(seller_prior_late, 0)
       + 20 * (coalesce(global_prior_late, 0) + 5 * 0.08) / (coalesce(global_prior_n, 0) + 5))
        / (coalesce(seller_prior_n, 0) + 20)                          AS at_seller_prior_late_rate,
    (coalesce(route_prior_late, 0)
       + 50 * (coalesce(global_prior_late, 0) + 5 * 0.08) / (coalesce(global_prior_n, 0) + 5))
        / (coalesce(route_prior_n, 0) + 50)                           AS at_route_prior_late_rate,

    -- ---- observed after the fact: diagnosis only, never a feature -----------
    date_diff('hour', purchased_at, approved_at)                      AS post_approval_lag_h,
    date_diff('hour', approved_at, carrier_handoff_at) / 24.0         AS post_handoff_lag_d,
    date_diff('hour', carrier_handoff_at, delivered_at) / 24.0        AS post_transit_d,
    date_diff('hour', purchased_at, delivered_at) / 24.0              AS post_total_d,
    CASE WHEN carrier_handoff_at > first_shipping_limit THEN 1 ELSE 0 END AS post_seller_missed_deadline,
    review_score  AS post_review_score,
    n_reviews     AS post_n_reviews
FROM with_global;
