-- BigQuery SQL to extract interaction data from GA4 events table
-- Table: rhomberg-377305.analytics_330861185.events_20251114
-- Output: sample_interactions.csv format

WITH parsed_events AS (
  SELECT
    -- Customer identification
    COALESCE(user_id, user_pseudo_id) AS customer_id,
    
    -- Product identification (from items array or event parameters)
    items[SAFE_OFFSET(0)].item_id AS product_id,
    
    -- Event mapping to interaction types
    CASE 
      WHEN event_name = 'view_item' THEN 'view'
      WHEN event_name = 'select_item' THEN 'click'
      WHEN event_name = 'add_to_cart' THEN 'add_to_cart'
      WHEN event_name = 'add_to_wishlist' THEN 'add_to_wishlist'
      WHEN event_name IN ('purchase', 'in_app_purchase') THEN 'purchase'
      ELSE event_name
    END AS interaction_type,
    
    -- Timestamp
    TIMESTAMP_MICROS(event_timestamp) AS timestamp,
    
    -- Device information
    CASE
      WHEN device.category = 'mobile' THEN 'mobile'
      WHEN device.category = 'tablet' THEN 'tablet'
      WHEN device.category = 'desktop' THEN 'desktop'
      ELSE 'mobile'
    END AS device_type,
    
    -- Traffic source
    COALESCE(
      collected_traffic_source.manual_source,
      traffic_source.source,
      'direct'
    ) AS referrer_source,
    
    -- Additional context
    geo.country,
    platform,
    event_value_in_usd
    
  FROM 
    `rhomberg-377305.analytics_330861185.events_20251114`
  
  WHERE
    -- Only events we care about
    event_name IN (
      'view_item',
      'select_item', 
      'add_to_cart',
      'add_to_wishlist',
      'purchase',
      'in_app_purchase'
    )
    
    -- Must have product ID
    AND ARRAY_LENGTH(items) > 0
    AND items[SAFE_OFFSET(0)].item_id IS NOT NULL
    
    -- Must have user identifier
    AND (user_id IS NOT NULL OR user_pseudo_id IS NOT NULL)
)

SELECT
  -- Generate interaction_id
  CONCAT('INT', LPAD(CAST(ROW_NUMBER() OVER (ORDER BY timestamp) AS STRING), 6, '0')) AS interaction_id,
  
  customer_id,
  product_id,
  interaction_type,
  timestamp,
  device_type,
  
  -- Map traffic sources to simplified categories
  CASE
    WHEN LOWER(referrer_source) LIKE '%google%' OR LOWER(referrer_source) LIKE '%search%' THEN 'google'
    WHEN LOWER(referrer_source) LIKE '%facebook%' OR LOWER(referrer_source) LIKE '%instagram%' THEN 'social'
    WHEN LOWER(referrer_source) LIKE '%email%' THEN 'email'
    WHEN referrer_source = 'direct' THEN 'direct'
    ELSE 'organic'
  END AS referrer_source

FROM 
  parsed_events

-- Filter out any null values
WHERE 
  customer_id IS NOT NULL
  AND product_id IS NOT NULL
  AND interaction_type IS NOT NULL

ORDER BY 
  timestamp DESC;
