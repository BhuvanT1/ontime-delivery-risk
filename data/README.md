# Data

This project uses the public **Brazilian E-Commerce Public Dataset by Olist**
(~100k orders, September 2016 – August 2018, CC BY-NC-SA 4.0).

Nine CSVs are expected in `data/raw/`:

```
olist_customers_dataset.csv          olist_order_reviews_dataset.csv
olist_geolocation_dataset.csv        olist_orders_dataset.csv
olist_order_items_dataset.csv        olist_products_dataset.csv
olist_order_payments_dataset.csv     olist_sellers_dataset.csv
product_category_name_translation.csv
```

Get them from the Kaggle dataset page (`olistbr/brazilian-ecommerce`), or with
the Kaggle CLI:

```bash
kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw --unzip
```

The files are gitignored — about 125 MB in total, dominated by the 61 MB
geolocation table.

## A note on the copy used here

The reviews table in the copy this repository was developed against has 100,000
rows with 827 duplicated `review_id` values and 555 orders carrying more than
one review. Other published copies have around 99,224 rows. Nothing downstream
assumes a row count: `sql/10_data_quality.sql` measures these properties and
reports them, `sql/20_order_fact.sql` aggregates reviews per order rather than
joining them, and the `order_fact` row count is asserted to equal the order
count by `tests/test_sql.py`. If your copy differs, the DQ report will say so.
