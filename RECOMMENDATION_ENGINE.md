# 🎯 Jewelry Recommendation Engine
## Combining Visual Similarity + Customer Behavior + Demographics

---

## 📋 Overview

This recommendation engine combines **three powerful signals** to provide personalized jewelry recommendations:

1. **Visual Similarity** - Using your trained image embedding model (MobileNetV2)
2. **Customer Behavior** - Purchase history, browsing patterns, interactions
3. **Customer Demographics** - Age, gender, location, preferences

### Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT SOURCES                            │
├──────────────┬──────────────────────┬──────────────────────┤
│   Product    │   Customer Orders    │  Customer Profile    │
│   Images     │   & Interactions     │   & Demographics     │
└──────┬───────┴──────────┬───────────┴──────────┬───────────┘
       │                  │                      │
       ▼                  ▼                      ▼
┌─────────────┐   ┌──────────────┐      ┌──────────────┐
│  Image      │   │ Collaborative│      │ Demographic  │
│  Embeddings │   │  Filtering   │      │  Filtering   │
│ (MobileNet) │   │              │      │              │
└──────┬──────┘   └──────┬───────┘      └──────┬───────┘
       │                 │                      │
       └─────────────────┴──────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Hybrid Recommender  │
              │  (Weighted Ensemble) │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Top-N Personalized   │
              │   Recommendations    │
              └──────────────────────┘
```

---

## 📊 Required Data & Schema

### 1. **Product Catalog** (`products.csv`)

Core product information with visual and textual features.

| Column | Type | Required | Description | Example |
|--------|------|----------|-------------|---------|
| `product_id` | int/string | ✅ | Unique product identifier | `111350` |
| `title` | string | ✅ | Product name | `Rhomberg Partnerring Silber` |
| `description` | text | ⚠️ | Detailed description | `Zeitloser Partnerring...` |
| `category` | string | ✅ | Product category | `rings`, `necklaces`, `earrings` |
| `subcategory` | string | ⚠️ | Subcategory | `Partnerring`, `Solitaire` |
| `brand` | string | ✅ | Brand name | `Rhomberg` |
| `collection` | string | ⚠️ | Collection name | `Classic`, `Modern` |
| `price` | float | ✅ | Current price in EUR | `34.00` |
| `sale_price` | float | ❌ | Sale price if on sale | `29.00` |
| `image_url` | string | ✅ | Primary image URL | `https://api.rhomberg.net/...` |
| `additional_images` | string | ❌ | Comma-separated URLs | `url1,url2,url3` |
| `material` | string | ✅ | Primary material | `Silber`, `Gold`, `Platin` |
| `color` | string | ⚠️ | Primary color | `weiss`, `gelb`, `rose` |
| `gender` | string | ⚠️ | Target gender | `unisex`, `female`, `male` |
| `age_group` | string | ❌ | Target age | `adult`, `teen` |
| `stone_type` | string | ❌ | Gemstone type | `diamond`, `ruby`, `none` |
| `weight` | float | ❌ | Weight in grams | `1.5` |
| `size` | string | ❌ | Size/dimensions | `3mm`, `50cm` |
| `stock_status` | string | ✅ | Availability | `in_stock`, `out_of_stock` |
| `created_date` | date | ⚠️ | Product added date | `2024-01-15` |

**Legend**: ✅ Required | ⚠️ Highly Recommended | ❌ Optional

---

### 2. **Customer Profile** (`customers.csv`)

Demographic and preference information about customers.

| Column | Type | Required | Description | Example |
|--------|------|----------|-------------|---------|
| `customer_id` | int/string | ✅ | Unique customer identifier | `C12345` |
| `email` | string | ⚠️ | Email (hashed for privacy) | `hash_abc123...` |
| `registration_date` | date | ✅ | Account creation date | `2023-06-15` |
| `gender` | string | ⚠️ | Self-identified gender | `female`, `male`, `other`, `unknown` |
| `age` | int | ⚠️ | Age in years | `32` |
| `age_group` | string | ⚠️ | Age bracket | `18-25`, `26-35`, `36-45`, `46+` |
| `location_country` | string | ✅ | Country code | `DE`, `AT`, `CH` |
| `location_state` | string | ❌ | State/region | `Bayern`, `Wien` |
| `location_city` | string | ❌ | City | `Munich`, `Vienna` |
| `postal_code` | string | ❌ | Postal code | `80331` |
| `preferred_language` | string | ⚠️ | Language preference | `de`, `en` |
| `income_bracket` | string | ❌ | Income range | `low`, `medium`, `high`, `luxury` |
| `customer_segment` | string | ⚠️ | Customer type | `first-time`, `regular`, `vip`, `inactive` |
| `lifetime_value` | float | ⚠️ | Total spent (EUR) | `1250.50` |
| `total_orders` | int | ⚠️ | Number of orders | `8` |
| `avg_order_value` | float | ⚠️ | Average order in EUR | `156.31` |
| `preferred_categories` | string | ❌ | Comma-separated | `rings,necklaces` |
| `preferred_materials` | string | ❌ | Comma-separated | `gold,diamond` |
| `preferred_price_range` | string | ❌ | Price preference | `budget`, `mid`, `premium`, `luxury` |
| `marketing_consent` | boolean | ⚠️ | Email consent | `true`, `false` |
| `last_active_date` | date | ✅ | Last interaction | `2024-11-08` |

---

### 3. **Order History** (`orders.csv`)

Transactional data showing customer purchases.

| Column | Type | Required | Description | Example |
|--------|------|----------|-------------|---------|
| `order_id` | int/string | ✅ | Unique order identifier | `ORD123456` |
| `customer_id` | int/string | ✅ | Customer who placed order | `C12345` |
| `order_date` | datetime | ✅ | Order timestamp | `2024-10-15 14:32:00` |
| `order_status` | string | ✅ | Order state | `completed`, `pending`, `cancelled`, `returned` |
| `total_amount` | float | ✅ | Total order value (EUR) | `125.00` |
| `discount_amount` | float | ❌ | Discount applied | `12.50` |
| `payment_method` | string | ❌ | Payment type | `credit_card`, `paypal`, `bank_transfer` |
| `shipping_method` | string | ❌ | Delivery method | `standard`, `express` |
| `delivery_date` | date | ❌ | Actual delivery date | `2024-10-18` |
| `num_items` | int | ⚠️ | Items in order | `2` |
| `is_gift` | boolean | ❌ | Gift order flag | `true`, `false` |
| `occasion` | string | ❌ | Purchase reason | `birthday`, `anniversary`, `wedding`, `christmas` |

---

### 4. **Order Items** (`order_items.csv`)

Line items showing which products were purchased in each order.

| Column | Type | Required | Description | Example |
|--------|------|----------|-------------|---------|
| `order_item_id` | int/string | ✅ | Unique line item ID | `OI789012` |
| `order_id` | int/string | ✅ | Parent order | `ORD123456` |
| `product_id` | int/string | ✅ | Product purchased | `111350` |
| `quantity` | int | ✅ | Number of units | `1` |
| `unit_price` | float | ✅ | Price per unit (EUR) | `34.00` |
| `discount_applied` | float | ❌ | Discount on item | `3.40` |
| `final_price` | float | ⚠️ | Actual paid price | `30.60` |

---

### 5. **Customer Interactions** (`interactions.csv`)

Browsing behavior, clicks, views, wishlist, cart abandonment.

| Column | Type | Required | Description | Example |
|--------|------|----------|-------------|---------|
| `interaction_id` | int/string | ✅ | Unique interaction ID | `INT456789` |
| `customer_id` | int/string | ✅ | Customer performing action | `C12345` |
| `product_id` | int/string | ✅ | Product interacted with | `111350` |
| `interaction_type` | string | ✅ | Type of interaction | `view`, `click`, `add_to_cart`, `remove_from_cart`, `add_to_wishlist`, `purchase` |
| `timestamp` | datetime | ✅ | When interaction occurred | `2024-11-05 10:23:15` |
| `session_id` | string | ⚠️ | Browser session | `sess_abc123` |
| `interaction_duration` | int | ❌ | Seconds spent viewing | `45` |
| `device_type` | string | ❌ | Device used | `mobile`, `desktop`, `tablet` |
| `referrer_source` | string | ❌ | Traffic source | `google`, `instagram`, `email`, `direct` |

**Interaction types priority** (for recommendation weight):
- `purchase` → Highest signal (explicit preference)
- `add_to_wishlist` → Strong signal
- `add_to_cart` → Medium-strong signal
- `view` (>30s) → Medium signal
- `click` → Weak signal

---

### 6. **Product Embeddings** (`product_embeddings.npy` or `.csv`)

Pre-computed image embeddings from your trained model (optional but recommended for speed).

| Column | Type | Required | Description | Example |
|--------|------|----------|-------------|---------|
| `product_id` | int/string | ✅ | Product identifier | `111350` |
| `embedding_vector` | array/float[] | ✅ | 1280-dim image embedding | `[0.123, -0.456, ...]` |
| `embedding_model` | string | ⚠️ | Model version | `mobilenet_v2_collection_v1` |
| `created_date` | datetime | ❌ | Embedding generation date | `2024-11-01 12:00:00` |

**Format options**:
- **NumPy format** (recommended): `product_embeddings.npy` + `product_ids.txt`
- **CSV format**: product_id, emb_0, emb_1, ..., emb_1279
- **HDF5 format**: Efficient for large catalogs (>10k products)

---

## 🔧 Recommendation Strategies

### Strategy 1: **Content-Based Filtering (Visual Similarity)**
*"Customers who viewed this product also like visually similar items"*

**How it works**:
1. Extract image embeddings for all products using your trained MobileNetV2 model
2. When user views/purchases product X, compute cosine similarity with all other products
3. Recommend top-N most visually similar products

**Use cases**:
- New customers (no purchase history)
- "Similar items" carousel on product pages
- Style consistency recommendations

**Pros**: Works immediately, no cold start problem
**Cons**: Doesn't learn user preferences, limited diversity

---

### Strategy 2: **Collaborative Filtering (User-Item)**
*"Customers who bought X also bought Y"*

**How it works**:
1. Build user-item interaction matrix (rows=customers, cols=products)
2. Use matrix factorization (e.g., SVD, ALS) or nearest neighbors
3. Find similar customers → recommend items they purchased

**Use cases**:
- Homepage "Recommended for you"
- Email campaigns
- Cross-sell during checkout

**Pros**: Discovers non-obvious patterns, learns implicit preferences
**Cons**: Cold start for new users/products, requires sufficient data

---

### Strategy 3: **Demographic Filtering**
*"Popular items for women aged 25-35 in Germany"*

**How it works**:
1. Segment customers by demographics (age, gender, location)
2. Compute popularity scores per segment
3. Recommend trending items for user's segment

**Use cases**:
- New customer onboarding
- Seasonal campaigns
- Gift recommendations

**Pros**: Simple, explainable, works with limited data
**Cons**: Can reinforce stereotypes, less personalized

---

### Strategy 4: **Hybrid Ensemble (RECOMMENDED)**
*"Combining all signals for maximum relevance"*

**How it works**:
```python
final_score = (
    0.40 * visual_similarity_score +      # Image embeddings
    0.35 * collaborative_filtering_score + # User-item interactions
    0.15 * demographic_score +             # Segment popularity
    0.10 * recency_boost                   # Recent views/purchases
)
```

**Weight tuning**: Adjust based on A/B testing and business metrics (CTR, conversion, revenue)

---

## 🚀 Implementation Roadmap

### Phase 1: Data Preparation (Week 1)
- [ ] Export customer data to required schemas
- [ ] Validate data quality (missing values, duplicates)
- [ ] Generate image embeddings for all products
- [ ] Create train/test split for evaluation

### Phase 2: Baseline Models (Week 2)
- [ ] Implement content-based filtering (visual similarity)
- [ ] Implement demographic filtering
- [ ] Evaluate metrics: Precision@K, Recall@K, NDCG

### Phase 3: Advanced Models (Week 3-4)
- [ ] Implement collaborative filtering (ALS or neural CF)
- [ ] Build hybrid ensemble
- [ ] Add business rules (stock, price range, exclusions)

### Phase 4: Deployment (Week 5)
- [ ] Create API endpoint for real-time recommendations
- [ ] Batch compute recommendations for email campaigns
- [ ] A/B testing framework

---

## 📈 Evaluation Metrics

### Offline Metrics (Historical Data)
- **Precision@K**: % of recommended items that user actually purchased
- **Recall@K**: % of user's future purchases that were recommended
- **NDCG@K**: Normalized Discounted Cumulative Gain (ranking quality)
- **Coverage**: % of catalog that gets recommended
- **Diversity**: Average dissimilarity between recommended items

### Online Metrics (A/B Testing)
- **Click-Through Rate (CTR)**: % of recommendations clicked
- **Conversion Rate**: % of recommendations purchased
- **Revenue per Recommendation**: Average $ generated
- **Average Order Value (AOV)**: Impact on basket size

---

## 🛠 Technical Stack Recommendations

### Option A: Quick Start (Python + scikit-learn)
```python
# Visual similarity: cosine similarity on embeddings
# Collaborative: sklearn.neighbors.NearestNeighbors
# Hybrid: weighted ensemble
```
**Pros**: Fast to implement, no infrastructure
**Cons**: Limited scalability, manual updates

### Option B: Production Scale (Python + Specialized Libraries)
```python
# Collaborative: implicit library (ALS)
# Neural CF: PyTorch + custom architecture
# Vector search: FAISS or Annoy for fast nearest neighbors
# API: FastAPI + Redis caching
```
**Pros**: Scales to millions, real-time inference
**Cons**: More complex setup

### Option C: Enterprise (Cloud Services)
- **AWS Personalize**: Managed recommendation service
- **Google Recommendations AI**: Retail-focused
- **Azure Personalizer**: Reinforcement learning

**Pros**: Minimal maintenance, auto-scaling
**Cons**: Vendor lock-in, higher cost, less customization

---

## 📝 Example Use Cases

### 1. Product Page: "You May Also Like"
```
Input: User viewing product #111350 (silver partner ring)
Strategy: Visual similarity (70%) + Demographic (30%)
Output: Top 8 similar rings in same price range for user's gender/age
```

### 2. Homepage: "Recommended For You"
```
Input: Logged-in customer C12345
Strategy: Collaborative filtering (60%) + Recent interactions (40%)
Output: Top 12 personalized picks based on purchase history + browsing
```

### 3. Email Campaign: "New Arrivals You'll Love"
```
Input: Customer segment (VIP female customers 30-40)
Strategy: Demographic (50%) + Visual similarity to past purchases (50%)
Output: 6 new products matching their style + popularity in segment
```

### 4. Cart Abandonment: "Complete Your Look"
```
Input: Products in abandoned cart
Strategy: Visual similarity (80%) + Cross-sell rules (20%)
Output: Matching/complementary items (earrings to match necklace)
```

---

## 🔐 Privacy & Ethics Considerations

### Data Privacy (GDPR Compliance)
- ✅ Hash/encrypt customer identifiers
- ✅ Allow customers to opt-out of personalization
- ✅ Provide data deletion on request (right to be forgotten)
- ✅ Transparent about data usage in privacy policy

### Algorithmic Fairness
- ⚠️ Monitor for gender/age bias in recommendations
- ⚠️ Ensure price diversity (don't only show expensive items to high-value customers)
- ⚠️ Avoid filter bubbles (add serendipity/exploration)

### Business Rules
- Don't recommend out-of-stock items
- Respect customer budget (filter by past AOV ± margin)
- Seasonal appropriateness (no Christmas items in July)
- Brand preferences (if customer never buys brand X, deprioritize)

---

## 📚 Next Steps

1. **Data Audit**: Check what customer data you currently have access to
2. **Priority Use Case**: Start with 1-2 high-impact scenarios (e.g., product page + homepage)
3. **Baseline Implementation**: Visual similarity recommender (leverages your existing model)
4. **Iterate**: Add collaborative filtering once you have interaction data pipeline
5. **Measure Impact**: A/B test against random/popularity baseline

---

## 🤝 Questions to Answer Before Building

**Data Availability**:
- [ ] How many customers do you have? (Need >1000 for collaborative filtering)
- [ ] How many orders per month? (Need activity for fresh signals)
- [ ] Do you track browsing behavior (views, clicks)? (Critical for interactions data)
- [ ] Can you access historical order data? (Minimum 6 months recommended)

**Business Goals**:
- [ ] Primary metric: CTR, conversion, revenue, or engagement?
- [ ] Diversity vs relevance trade-off? (Show similar items vs surprising finds)
- [ ] Real-time vs batch? (API calls vs daily email campaigns)

**Technical Constraints**:
- [ ] Latency requirements? (< 100ms for web, < 1s for email)
- [ ] Infrastructure? (On-premise vs cloud)
- [ ] Team expertise? (ML engineers available?)

---

## 📬 Contact & Support

For implementation help:
1. Review your existing customer database schema
2. Map columns to the schemas above
3. Identify gaps (missing data to collect)
4. Start with visual similarity baseline (uses your existing model!)

**Estimated timeline**: 4-6 weeks from data collection to production MVP

---

*Last updated: November 2025*
