-- Weighted co-occurrence matrix: purchase + add_to_cart + view_item
WITH weighted_events AS (
  SELECT 
    user_pseudo_id,
    TIMESTAMP_TRUNC(TIMESTAMP_MICROS(event_timestamp), HOUR) as session_window,
    item.item_id as product_id,
    -- Assign weights by event importance
    CASE 
      WHEN event_name = 'purchase' THEN 10.0        -- Highest weight
      WHEN event_name = 'add_to_cart' THEN 5.0     -- Medium weight  
      WHEN event_name = 'view_item' THEN 1.0       -- Lower weight
      ELSE 1.0
    END as event_weight
  FROM `rhomberg-377305.analytics_330861185.events_*`,
    UNNEST(items) as item
  WHERE _TABLE_SUFFIX BETWEEN '20251114' AND '20251205'
    AND event_name IN ('purchase', 'add_to_cart', 'view_item')
    AND item.item_id IS NOT NULL
  GROUP BY 1, 2, 3, 4
)
SELECT 
  a.product_id as product_a,
  b.product_id as product_b,
  -- Sum weighted co-occurrences
  SUM(LEAST(a.event_weight, b.event_weight)) as weighted_co_occurrence,
  COUNT(*) as raw_count,
  -- Breakdown by event type for analysis
  SUM(CASE WHEN LEAST(a.event_weight, b.event_weight) = 10 THEN 1 ELSE 0 END) as purchase_pairs,
  SUM(CASE WHEN LEAST(a.event_weight, b.event_weight) = 5 THEN 1 ELSE 0 END) as cart_pairs,
  SUM(CASE WHEN LEAST(a.event_weight, b.event_weight) = 1 THEN 1 ELSE 0 END) as view_pairs
FROM weighted_events a
JOIN weighted_events b 
  ON a.user_pseudo_id = b.user_pseudo_id
  AND a.session_window = b.session_window
  AND a.product_id < b.product_id
GROUP BY 1, 2
HAVING weighted_co_occurrence >= 5  -- Higher threshold due to bigger weights
ORDER BY weighted_co_occurrence DESC
LIMIT 100000
