-- ---------------------------------------------------------------------------
-- Cohort and segment analysis.
--
-- Every rate here carries a Wilson score interval. That is not decoration: the
-- segments differ in size by three orders of magnitude, and a naive ranking of
-- late rate puts a seller with 3 orders and 2 misses at the top of the table
-- every single time. The interval is what stops the report recommending action
-- against noise. Wilson rather than the normal approximation because the rates
-- are near 0.07 and several segments are small, where the normal interval
-- misbehaves and can even reach below zero.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE MACRO wilson_low(k, n) AS
    CASE WHEN n = 0 THEN NULL ELSE
        ((k::DOUBLE / n) + 1.96 * 1.96 / (2 * n)
         - 1.96 * sqrt(((k::DOUBLE / n) * (1 - (k::DOUBLE / n)) + 1.96 * 1.96 / (4 * n)) / n))
        / (1 + 1.96 * 1.96 / n) END;

CREATE OR REPLACE MACRO wilson_high(k, n) AS
    CASE WHEN n = 0 THEN NULL ELSE
        ((k::DOUBLE / n) + 1.96 * 1.96 / (2 * n)
         + 1.96 * sqrt(((k::DOUBLE / n) * (1 - (k::DOUBLE / n)) + 1.96 * 1.96 / (4 * n)) / n))
        / (1 + 1.96 * 1.96 / n) END;

-- Monthly cohort: is the problem getting better or worse, and is the promise
-- itself moving? Olist pads the estimate heavily, so a falling late rate can
-- mean an improving network or simply a longer promise.
CREATE OR REPLACE VIEW v_cohort_month AS
SELECT date_trunc('month', purchased_at)::DATE AS cohort_month,
       count(*)                                AS n_orders,
       sum(is_late)                            AS n_late,
       round(avg(is_late), 4)                  AS late_rate,
       round(wilson_low(sum(is_late), count(*)), 4)  AS late_rate_lo,
       round(wilson_high(sum(is_late), count(*)), 4) AS late_rate_hi,
       round(avg(at_promise_days), 2)          AS avg_promise_days,
       round(avg(post_total_d), 2)             AS avg_actual_days,
       round(avg(at_promise_days) - avg(post_total_d), 2) AS avg_slack_days,
       round(avg(post_review_score), 3)        AS avg_review
FROM order_fact GROUP BY 1 ORDER BY 1;

-- Destination state. Volume is concentrated in SP, so a portfolio-level late
-- rate is essentially the SP late rate; the tail states are where it hurts.
CREATE OR REPLACE VIEW v_segment_state AS
SELECT at_customer_state AS customer_state,
       count(*) AS n_orders, sum(is_late) AS n_late,
       round(avg(is_late), 4) AS late_rate,
       round(wilson_low(sum(is_late), count(*)), 4)  AS late_rate_lo,
       round(wilson_high(sum(is_late), count(*)), 4) AS late_rate_hi,
       round(avg(at_distance_km)) AS avg_distance_km,
       round(avg(at_promise_days), 1) AS avg_promise_days,
       round(avg(post_total_d), 1) AS avg_actual_days,
       round(avg(post_review_score), 3) AS avg_review
FROM order_fact GROUP BY 1 HAVING count(*) >= 100 ORDER BY late_rate DESC;

-- Distance bands, cut on round numbers rather than quantiles so the boundaries
-- mean something operationally.
CREATE OR REPLACE VIEW v_segment_distance AS
SELECT CASE WHEN at_distance_km IS NULL THEN 'unknown'
            WHEN at_distance_km <   50 THEN 'a. <50km'
            WHEN at_distance_km <  200 THEN 'b. 50-200km'
            WHEN at_distance_km <  500 THEN 'c. 200-500km'
            WHEN at_distance_km < 1000 THEN 'd. 500-1000km'
            WHEN at_distance_km < 2000 THEN 'e. 1000-2000km'
            ELSE 'f. 2000km+' END AS distance_band,
       count(*) AS n_orders, sum(is_late) AS n_late,
       round(avg(is_late), 4) AS late_rate,
       round(wilson_low(sum(is_late), count(*)), 4)  AS late_rate_lo,
       round(wilson_high(sum(is_late), count(*)), 4) AS late_rate_hi,
       round(avg(at_promise_days), 1) AS avg_promise_days,
       round(avg(post_total_d), 1) AS avg_actual_days
FROM order_fact GROUP BY 1 ORDER BY 1;

-- Category, translated. Heavy and bulky categories are the interesting rows.
CREATE OR REPLACE VIEW v_segment_category AS
SELECT coalesce(t.product_category_name_english, f.at_primary_category, 'unknown') AS category,
       count(*) AS n_orders, sum(f.is_late) AS n_late,
       round(avg(f.is_late), 4) AS late_rate,
       round(wilson_low(sum(f.is_late), count(*)), 4)  AS late_rate_lo,
       round(wilson_high(sum(f.is_late), count(*)), 4) AS late_rate_hi,
       round(avg(f.at_total_weight_g)) AS avg_weight_g,
       round(avg(f.at_freight_ratio), 4) AS avg_freight_ratio
FROM order_fact f
LEFT JOIN category_translation t ON t.product_category_name = f.at_primary_category
GROUP BY 1 HAVING count(*) >= 300 ORDER BY late_rate DESC;

-- Seller portfolio, quadranted on volume and reliability. The action implied by
-- each quadrant is different, which is the point of segmenting at all.
CREATE OR REPLACE VIEW v_segment_seller AS
WITH s AS (
    SELECT primary_seller_id AS seller_id, at_seller_state AS seller_state,
           count(*) AS n_orders, sum(is_late) AS n_late,
           avg(is_late) AS late_rate,
           wilson_low(sum(is_late), count(*))  AS late_rate_lo,
           wilson_high(sum(is_late), count(*)) AS late_rate_hi,
           avg(post_handoff_lag_d) AS avg_handoff_lag_d,
           avg(post_seller_missed_deadline) AS missed_deadline_rate,
           avg(post_review_score) AS avg_review,
           sum(at_total_price) AS gmv
    FROM order_fact GROUP BY 1, 2
), g AS (SELECT avg(is_late) AS base FROM order_fact)
SELECT s.*,
       CASE WHEN n_orders >= 50 AND late_rate_lo > (SELECT base FROM g) THEN 'high volume, reliably late'
            WHEN n_orders >= 50 AND late_rate_hi < (SELECT base FROM g) THEN 'high volume, reliably good'
            WHEN n_orders >= 50 THEN 'high volume, indistinguishable from base'
            WHEN late_rate_lo > (SELECT base FROM g) THEN 'low volume, evidence of a problem'
            ELSE 'low volume, too few orders to judge' END AS seller_segment
FROM s ORDER BY n_orders DESC;

-- Repeat purchasing. customer_unique_id is the person; customer_id is per
-- order, and confusing the two is the classic first mistake on this dataset --
-- it makes every customer look like a one-time buyer.
CREATE OR REPLACE VIEW v_customer_repeat AS
WITH c AS (
    SELECT customer_unique_id, count(*) AS n_orders,
           sum(is_late) AS n_late, sum(at_total_price) AS lifetime_value,
           min(purchased_at) AS first_order, max(purchased_at) AS last_order,
           avg(post_review_score) AS avg_review
    FROM order_fact GROUP BY 1
)
SELECT CASE WHEN n_orders = 1 THEN '1 order' WHEN n_orders = 2 THEN '2 orders'
            ELSE '3+ orders' END AS repeat_band,
       count(*) AS n_customers, sum(n_orders) AS n_orders,
       round(sum(n_late)::DOUBLE / sum(n_orders), 4) AS late_rate,
       round(avg(lifetime_value), 2) AS avg_lifetime_value,
       round(avg(avg_review), 3) AS avg_review
FROM c GROUP BY 1 ORDER BY 1;
