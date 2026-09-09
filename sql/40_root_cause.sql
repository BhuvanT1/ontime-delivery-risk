-- ---------------------------------------------------------------------------
-- Root-cause analysis.
--
-- Everything in this file is ASSOCIATIONAL. The data is observational: nothing
-- was randomised, orders were not assigned to sellers or routes by anyone's
-- experiment, and any of these relationships can be produced by a common cause
-- rather than by the mechanism it suggests. So the language throughout is
-- "associated with", and where a confound is obvious it is adjusted for and
-- the adjustment is shown rather than hidden.
--
-- The one adjustment worth naming: state-level late rates are confounded by
-- distance almost completely. Northern states are both far from the São Paulo
-- seller base and served by thinner logistics. Direct standardisation
-- reweights every state onto the national distance mix, which answers a
-- narrower and much more useful question -- how does this state perform
-- against orders of comparable distance? -- without claiming to have
-- identified a cause.
-- ---------------------------------------------------------------------------

-- Where the time actually goes, for late and on-time orders side by side.
CREATE OR REPLACE VIEW v_stage_timing AS
SELECT CASE WHEN is_late = 1 THEN 'late' ELSE 'on time' END AS outcome,
       count(*) AS n_orders,
       round(median(post_approval_lag_h), 2)  AS med_approval_lag_h,
       round(median(post_handoff_lag_d), 2)   AS med_handoff_lag_d,
       round(median(post_transit_d), 2)       AS med_transit_d,
       round(median(post_total_d), 2)         AS med_total_d,
       round(median(at_promise_days), 2)      AS med_promise_days,
       round(avg(post_seller_missed_deadline), 4) AS missed_seller_deadline_rate
FROM order_fact GROUP BY 1;

-- The same split within a distance band, so the comparison is between orders
-- that had a comparable job to do.
CREATE OR REPLACE VIEW v_stage_timing_by_distance AS
SELECT CASE WHEN at_distance_km IS NULL THEN 'unknown'
            WHEN at_distance_km <  200 THEN 'a. <200km'
            WHEN at_distance_km <  500 THEN 'b. 200-500km'
            WHEN at_distance_km < 1000 THEN 'c. 500-1000km'
            ELSE 'd. 1000km+' END AS distance_band,
       CASE WHEN is_late = 1 THEN 'late' ELSE 'on time' END AS outcome,
       count(*) AS n_orders,
       round(median(post_handoff_lag_d), 2) AS med_handoff_lag_d,
       round(median(post_transit_d), 2)     AS med_transit_d,
       round(avg(post_seller_missed_deadline), 4) AS missed_seller_deadline_rate
FROM order_fact GROUP BY 1, 2 ORDER BY 1, 2;

-- Which stage the excess sits in. For each late order, the delay against the
-- on-time median of its own distance band is split between the seller-side
-- stage (order to carrier handoff) and the carrier-side stage (handoff to
-- door). Shares are of the total excess, so they sum to roughly 100% and can
-- be read as "where the lost days are", not as a causal decomposition.
CREATE OR REPLACE VIEW v_delay_attribution AS
WITH banded AS (
    SELECT *, CASE WHEN at_distance_km IS NULL THEN 'unknown'
                   WHEN at_distance_km <  200 THEN 'a. <200km'
                   WHEN at_distance_km <  500 THEN 'b. 200-500km'
                   WHEN at_distance_km < 1000 THEN 'c. 500-1000km'
                   ELSE 'd. 1000km+' END AS distance_band
    FROM order_fact
),
norms AS (
    SELECT distance_band,
           median(post_handoff_lag_d) FILTER (WHERE is_late = 0) AS norm_handoff,
           median(post_transit_d)     FILTER (WHERE is_late = 0) AS norm_transit
    FROM banded GROUP BY 1
)
SELECT b.distance_band,
       count(*) AS n_late_orders,
       round(sum(greatest(b.post_handoff_lag_d - n.norm_handoff, 0)), 0) AS excess_seller_days,
       round(sum(greatest(b.post_transit_d     - n.norm_transit, 0)), 0) AS excess_carrier_days,
       round(100.0 * sum(greatest(b.post_handoff_lag_d - n.norm_handoff, 0))
             / nullif(sum(greatest(b.post_handoff_lag_d - n.norm_handoff, 0))
                    + sum(greatest(b.post_transit_d - n.norm_transit, 0)), 0), 1) AS pct_seller_side,
       round(100.0 * sum(greatest(b.post_transit_d - n.norm_transit, 0))
             / nullif(sum(greatest(b.post_handoff_lag_d - n.norm_handoff, 0))
                    + sum(greatest(b.post_transit_d - n.norm_transit, 0)), 0), 1) AS pct_carrier_side
FROM banded b JOIN norms n USING (distance_band)
WHERE b.is_late = 1 AND b.post_handoff_lag_d IS NOT NULL AND b.post_transit_d IS NOT NULL
GROUP BY 1 ORDER BY 1;

-- Are some destinations worse than their geography explains?
--
-- Direct standardisation is the textbook first answer and it is the wrong tool
-- here. It reweights every state onto the national distance mix, which needs a
-- late rate for every state in every distance band -- and most states sell into
-- only one or two. Filling the empty cells with the national rate makes the
-- method quietly assume each state would be exactly average on the routes it
-- does not serve, and since the empty cells then carry most of the weight,
-- every state gets dragged back toward the national average. Alagoas, the
-- worst state in the raw table at 21% late, comes out near the middle of the
-- pack purely as an artefact of that substitution.
--
-- Indirect standardisation asks the answerable question instead: given this
-- state's actual mix of distances and time periods, how many late orders would
-- the national rates predict, and how many were there? The ratio is an SMR --
-- above 1 means worse than comparable orders elsewhere. It uses only strata the
-- state genuinely has, so nothing is imputed.
--
-- Strata are distance band x half-year: the network changed materially over the
-- two years covered, and a state whose volume arrived late in the period would
-- otherwise be compared against rates that no longer applied to it.
-- The interval is Byar's approximation to the exact Poisson interval, which
-- behaves at the small observed counts several states have.
CREATE OR REPLACE VIEW v_state_smr AS
WITH banded AS (
    SELECT at_customer_state AS customer_state, is_late,
           CASE WHEN at_distance_km IS NULL THEN 'unknown'
                WHEN at_distance_km <  200 THEN 'a. <200km'
                WHEN at_distance_km <  500 THEN 'b. 200-500km'
                WHEN at_distance_km < 1000 THEN 'c. 500-1000km'
                ELSE 'd. 1000km+' END AS distance_band,
           extract('year' FROM purchased_at) * 2
             + CASE WHEN extract('month' FROM purchased_at) > 6 THEN 1 ELSE 0 END AS half_year
    FROM order_fact
),
national AS (
    SELECT distance_band, half_year, avg(is_late) AS nat_rate, count(*) AS nat_n
    FROM banded GROUP BY 1, 2
),
by_state AS (
    SELECT b.customer_state,
           count(*)                        AS n_orders,
           sum(b.is_late)                  AS observed,
           sum(n.nat_rate)                 AS expected
    FROM banded b JOIN national n USING (distance_band, half_year)
    GROUP BY 1
)
SELECT customer_state, n_orders, observed::INTEGER AS observed,
       round(expected, 1) AS expected,
       round(observed / nullif(expected, 0), 3) AS smr,
       round(observed * pow(1 - 1.0 / (9 * observed) - 1.96 / (3 * sqrt(observed)), 3)
             / nullif(expected, 0), 3) AS smr_lo,
       round((observed + 1) * pow(1 - 1.0 / (9 * (observed + 1)) + 1.96 / (3 * sqrt(observed + 1)), 3)
             / nullif(expected, 0), 3) AS smr_hi,
       round(observed::DOUBLE / n_orders, 4) AS crude_rate,
       -- Excess late orders over what comparable routes predict. SMR ranks by
       -- how bad a state is; this ranks by how much of the problem it is, and
       -- the two orderings are not the same -- which is the point of showing
       -- both to anyone deciding where to spend effort.
       round(observed - expected, 0) AS excess_late,
       CASE WHEN observed * pow(1 - 1.0 / (9 * observed) - 1.96 / (3 * sqrt(observed)), 3)
                 / nullif(expected, 0) > 1 THEN 'worse than comparable orders'
            WHEN (observed + 1) * pow(1 - 1.0 / (9 * (observed + 1)) + 1.96 / (3 * sqrt(observed + 1)), 3)
                 / nullif(expected, 0) < 1 THEN 'better than comparable orders'
            ELSE 'not distinguishable' END AS verdict
FROM by_state
WHERE n_orders >= 100 AND observed > 0
ORDER BY smr DESC;

-- How bad is late? A one-day miss and a three-week miss are not the same
-- event, and a binary label hides that entirely.
CREATE OR REPLACE VIEW v_late_severity AS
SELECT CASE WHEN days_vs_promise <= 0 THEN 'on time'
            WHEN days_vs_promise <=  2 THEN 'a. 1-2 days late'
            WHEN days_vs_promise <=  7 THEN 'b. 3-7 days late'
            WHEN days_vs_promise <= 14 THEN 'c. 8-14 days late'
            ELSE 'd. 15+ days late' END AS severity,
       count(*) AS n_orders,
       round(100.0 * count(*) / (SELECT count(*) FROM order_fact), 2) AS pct_of_orders,
       round(avg(post_review_score), 3) AS avg_review,
       round(avg(CASE WHEN post_review_score <= 2 THEN 1.0 ELSE 0.0 END), 4) AS pct_1_or_2_star
FROM order_fact GROUP BY 1 ORDER BY 1;

-- Association between lateness and the review score. Reported with an interval
-- and labelled as an association: customers who received a late order are not
-- a randomised group, and both the delay and the low score can follow from a
-- third thing (a difficult route, a poor seller) rather than from each other.
CREATE OR REPLACE VIEW v_late_review_assoc AS
SELECT CASE WHEN is_late = 1 THEN 'late' ELSE 'on time' END AS outcome,
       count(*) FILTER (WHERE post_review_score IS NOT NULL) AS n_reviewed,
       round(avg(post_review_score), 3) AS avg_review,
       round(avg(CASE WHEN post_review_score <= 2 THEN 1.0 ELSE 0.0 END), 4) AS rate_1_or_2_star,
       round(wilson_low(count(*) FILTER (WHERE post_review_score <= 2),
                        count(*) FILTER (WHERE post_review_score IS NOT NULL)), 4) AS rate_lo,
       round(wilson_high(count(*) FILTER (WHERE post_review_score <= 2),
                         count(*) FILTER (WHERE post_review_score IS NOT NULL)), 4) AS rate_hi
FROM order_fact GROUP BY 1;
