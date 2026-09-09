-- ---------------------------------------------------------------------------
-- Olist e-commerce warehouse: typed DDL.
--
-- The CSV export is untyped and the obvious shortcut is read_csv_auto straight
-- into a view. Declaring the schema by hand instead is deliberate: it fixes the
-- types once, names the keys, and -- most usefully -- makes the constraints the
-- data *fails* explicit rather than silent. Every relationship below that is
-- documented but not enforced is a data-quality finding, checked in
-- 10_data_quality.sql.
-- ---------------------------------------------------------------------------

DROP TABLE IF EXISTS order_reviews;
DROP TABLE IF EXISTS order_payments;
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS sellers;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS category_translation;
DROP TABLE IF EXISTS geolocation;
DROP TABLE IF EXISTS zip_centroid;

CREATE TABLE customers (
    customer_id             VARCHAR PRIMARY KEY,   -- one per ORDER, not per person
    customer_unique_id      VARCHAR NOT NULL,      -- the actual person
    customer_zip_prefix     INTEGER,
    customer_city           VARCHAR,
    customer_state          VARCHAR
);

CREATE TABLE sellers (
    seller_id               VARCHAR PRIMARY KEY,
    seller_zip_prefix       INTEGER,
    seller_city             VARCHAR,
    seller_state            VARCHAR
);

CREATE TABLE products (
    product_id              VARCHAR PRIMARY KEY,
    product_category_name   VARCHAR,
    name_length             INTEGER,
    description_length      INTEGER,
    photos_qty              INTEGER,
    weight_g                DOUBLE,
    length_cm               DOUBLE,
    height_cm               DOUBLE,
    width_cm                DOUBLE
);

CREATE TABLE category_translation (
    product_category_name           VARCHAR PRIMARY KEY,
    product_category_name_english   VARCHAR
);

CREATE TABLE orders (
    order_id                VARCHAR PRIMARY KEY,
    customer_id             VARCHAR NOT NULL REFERENCES customers(customer_id),
    order_status            VARCHAR NOT NULL,
    purchased_at            TIMESTAMP,
    approved_at             TIMESTAMP,
    carrier_handoff_at      TIMESTAMP,   -- seller -> logistics partner
    delivered_at            TIMESTAMP,   -- arrival with the customer
    estimated_delivery_at   TIMESTAMP    -- the promise shown at checkout
);

CREATE TABLE order_items (
    order_id                VARCHAR NOT NULL REFERENCES orders(order_id),
    order_item_id           INTEGER NOT NULL,      -- line number within the order
    product_id              VARCHAR NOT NULL REFERENCES products(product_id),
    seller_id               VARCHAR NOT NULL REFERENCES sellers(seller_id),
    shipping_limit_date     TIMESTAMP,             -- seller's contractual handoff deadline
    price                   DOUBLE,
    freight_value           DOUBLE,
    PRIMARY KEY (order_id, order_item_id)
);

CREATE TABLE order_payments (
    order_id                VARCHAR NOT NULL REFERENCES orders(order_id),
    payment_sequential      INTEGER NOT NULL,
    payment_type            VARCHAR,
    payment_installments    INTEGER,
    payment_value           DOUBLE,
    PRIMARY KEY (order_id, payment_sequential)
);

-- No primary key on purpose. review_id is duplicated in the source export and
-- a handful of orders carry more than one review; asserting uniqueness here
-- would reject the load instead of surfacing the problem. Quantified in
-- 10_data_quality.sql and handled explicitly when the fact table is built.
CREATE TABLE order_reviews (
    review_id               VARCHAR,
    order_id                VARCHAR,
    review_score            INTEGER,
    review_comment_title    VARCHAR,
    review_comment_message  VARCHAR,
    review_created_at       TIMESTAMP,
    review_answered_at      TIMESTAMP
);

CREATE TABLE geolocation (
    zip_prefix              INTEGER,
    lat                     DOUBLE,
    lng                     DOUBLE,
    city                    VARCHAR,
    state                   VARCHAR
);

-- One row per zip prefix. The raw geolocation table holds ~1M points, several
-- thousand per prefix, some of them far outside Brazil. The centroid is taken
-- as a MEDIAN rather than a mean so those outliers cannot drag a prefix into
-- the ocean, and the bounding box drops the few that are plainly wrong.
CREATE TABLE zip_centroid (
    zip_prefix              INTEGER PRIMARY KEY,
    lat                     DOUBLE,
    lng                     DOUBLE,
    n_points                INTEGER
);
