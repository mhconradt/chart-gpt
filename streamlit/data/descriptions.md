"Here is a sample of the `WEB_SALES` dataset: {_md}. Other tables in the schema are: {sorted(samples)}. Generate descriptions of each column in the `WEB_SALES` dataset, particularly types, i.e. categorical, date, etc. conventions for the data, i.e. storing country codes vs. country names."

Based on the data provided, here are the descriptions of each column in
the `WEB_SALES` dataset:

- `WS_SOLD_DATE_SK`: This column represents the sold date as a surrogate key.
  It is of a numeric type.

- `WS_SOLD_TIME_SK`: This column represents the sold time as a surrogate key.
  It is of a numeric type.

- `WS_SHIP_DATE_SK`: This column represents the ship date as a surrogate key.
  It is of a numeric type.

- `WS_ITEM_SK`: This column represents the item as a surrogate key. It is of a
  numeric type.

- `WS_BILL_CUSTOMER_SK`: This column represents the billed customer as a
  surrogate key. It is of a numeric type.

- `WS_BILL_CDEMO_SK`: This column represents the demographic information of the
  billed customer as a surrogate key. It is of a numeric type.

- `WS_BILL_HDEMO_SK`: This column represents the household demographic
  information of the billed customer as a surrogate key. It is of a numeric
  type.

- `WS_BILL_ADDR_SK`: This column represents the billed address as a surrogate
  key. It is of a numeric type.

- `WS_SHIP_CUSTOMER_SK`: This column represents the shipped customer as a
  surrogate key. It is of a numeric type.

- `WS_SHIP_CDEMO_SK`: This column represents the demographic information of the
  shipped customer as a surrogate key. It is of a numeric type.

- `WS_SHIP_HDEMO_SK`: This column represents the household demographic
  information of the shipped customer as a surrogate key. It is of a numeric
  type.

- `WS_SHIP_ADDR_SK`: This column represents the shipped address as a surrogate
  key. It is of a numeric type.

- `WS_WEB_PAGE_SK`: This column represents the web page as a surrogate key. It
  is of a numeric type.

- `WS_WEB_SITE_SK`: This column represents the web site as a surrogate key. It
  is of a numeric type.

- `WS_SHIP_MODE_SK`: This column represents the shipping mode as a surrogate
  key. It is of a numeric type.

- `WS_WAREHOUSE_SK`: This column represents the warehouse as a surrogate key.
  It is of a numeric type.

- `WS_PROMO_SK`: This column represents the promotion as a surrogate key. It is
  of a numeric type.

- `WS_ORDER_NUMBER`: This column represents the order number. It is of a
  numeric type.

- `WS_QUANTITY`: This column represents the quantity ordered. It is of a
  numeric type.

- `WS_WHOLESALE_COST`: This column represents the wholesale cost of the item.
  It is of a numeric type.

- `WS_LIST_PRICE`: This column represents the listed price of the item. It is
  of a numeric type.

- `WS_SALES_PRICE`: This column represents the sales price of the item. It is
  of a numeric type.

- `WS_EXT_DISCOUNT_AMT`: This column represents the extended discount amount of
  the item. It is of a numeric type.

- `WS_EXT_SALES_PRICE`: This column represents the extended sales price of the
  item. It is of a numeric type.

- `WS_EXT_WHOLESALE_COST`: This column represents the extended wholesale cost
  of the item. It is of a numeric type.

- `WS_EXT_LIST_PRICE`: This column represents the extended list price of the
  item. It is of a numeric type.

- `WS_EXT_TAX`: This column represents the extended tax amount. It is of a
  numeric type.

- `WS_COUPON_AMT`: This column represents the coupon amount. It is of a numeric
  type.

- `WS_EXT_SHIP_COST`: This column represents the extended shipping cost. It is
  of a numeric type.

- `WS_NET_PAID`: This column represents the net amount paid. It is of a numeric
  type.

- `WS_NET_PAID_INC_TAX`: This column represents the net amount paid including
  tax. It is of a numeric type.

- `WS_NET_PAID_INC_SHIP`: This column represents the net amount paid including
  shipping cost. It is of a numeric type.

- `WS_NET_PAID_INC_SHIP_TAX`: This column represents the net amount paid
  including both shipping cost and tax. It is of a numeric type.

- `WS_NET_PROFIT`: This column represents the net profit. It is of a numeric
  type.

Note: Based on the column names, it appears that the data follows a surrogate
key convention for different entities such as customers, items, addresses, web
pages, etc. This means that the actual details for these entities may be stored
in other tables and are referenced in this dataset using surrogate keys.