# Customer Style Matching with Image Similarity
## Quick Reference Guide

---

## 🎯 Core Concept

**Match customer taste by learning their visual preferences from interaction history.**

Each customer → Style Profile (1280-dim vector) → Similar Products

---

## 📂 What You Have Now

### **1. Core Implementation** (`code/customer_style_profiler.py`)
- `CustomerStyleProfiler` class
- Builds style profiles from weighted interactions
- Recommends products based on cosine similarity
- Finds customers with similar taste

### **2. Comprehensive Guide** (`STYLE_RECOMMENDATION_GUIDE.md`)
- Step-by-step notebook format
- Shows how to integrate with your trained model
- Includes visualization and analysis code
- Production deployment tips

### **3. Quick Demo** (`code/demo_style_recommendations.py`)
- Runnable example with sample data
- Shows recommendations in action
- No GPU required (uses simulated embeddings)

### **4. Sample Data** (`data/sample_*.csv`)
- 20 customers with demographics
- 40 orders with purchase history
- 120 interactions (views, clicks, purchases)
- Ready for testing

---

## 🚀 Quick Start

### **Run the Demo (5 minutes)**

```bash
cd /project
python code/demo_style_recommendations.py
```

This shows:
- Customer style profiles
- Personalized recommendations
- Similar customers by taste
- Style consistency metrics

### **Integrate with Your Model (1 hour)**

1. **Extract embeddings** for all products:
```python
from customer_style_profiler import CustomerStyleProfiler
import torch
from torchvision import models

# Load your trained model
model = load_your_mobilenet_model()

# Extract embeddings
product_embeddings = {}
for product_id in product_catalog:
    image = load_image(product_id)
    embedding = model.extract_features(image)  # 1280-dim
    product_embeddings[product_id] = embedding
```

2. **Build customer profiles**:
```python
profiler = CustomerStyleProfiler(product_embeddings)
profiles = profiler.build_all_profiles(interactions_df)
```

3. **Generate recommendations**:
```python
recommendations = profiler.recommend_by_style(
    customer_id='C10001',
    top_n=10,
    exclude_products=already_purchased
)
```

---

## 💡 How It Works

### **Input: Customer Interactions**
```
Customer C10001:
- Viewed: Silver ring (#111350) 
- Viewed: Gold necklace (#111899)
- Purchased: Silver ring (#111350)
```

### **Processing: Weighted Average**
```python
style_profile = (
    10.0 * embedding[111350] +  # Purchase (high weight)
     1.0 * embedding[111899] +  # View (low weight)
     1.0 * embedding[111350]    # View
) / 12.0
```

### **Output: Similar Products**
```
Recommendations:
1. Silver bracelet (0.89 similarity)
2. Platinum ring (0.85 similarity)
3. Silver earrings (0.82 similarity)
```

---

## 🔑 Key Parameters to Tune

### **Interaction Weights**
```python
INTERACTION_WEIGHTS = {
    'purchase': 10.0,        # Strongest signal
    'add_to_wishlist': 5.0,  # Strong intent
    'add_to_cart': 3.0,      # Medium intent
    'view': 1.0,             # Weak signal
    'click': 0.5             # Weakest signal
}
```

**Recommendation**: Start with these defaults, adjust based on A/B testing.

### **Recency Decay**
```python
recency_half_life_days = 30  # Weight halves every 30 days
```

**Recommendation**: 
- 30 days for fast fashion/trendy items
- 90 days for classic/timeless jewelry

### **Minimum Interactions**
```python
min_interactions = 3  # Need at least 3 interactions to build profile
```

**Recommendation**:
- 3 for initial testing (more customers)
- 5-10 for production (higher quality profiles)

### **Similarity Threshold**
```python
min_similarity = 0.5  # Only recommend items with >50% style match
```

**Recommendation**:
- 0.3-0.5 for discovery (more diverse)
- 0.6-0.8 for precision (very similar items)

---

## 📊 Expected Performance

### **Coverage**
- ~60-70% of customers will have profiles (need min interactions)
- Cold start: use demographic filtering for new customers

### **Accuracy** (based on similar systems)
- **Precision@10**: 15-25% (1-2 of top 10 will be purchased)
- **CTR improvement**: 20-40% vs random
- **Conversion lift**: 10-20% vs popularity baseline

### **Computational Cost**
- Profile building: ~0.1ms per customer (once daily)
- Recommendation: ~1-5ms per customer (real-time)
- Scales to 100k+ products with FAISS indexing

---

## 🎨 Use Cases Ranked by Impact

### **1. Product Page: "Similar Items"** ⭐⭐⭐⭐⭐
```
Input: Current product + customer style profile
Output: Visually similar items matching their taste
Impact: High CTR, low implementation effort
```

### **2. Homepage: "Recommended For You"** ⭐⭐⭐⭐⭐
```
Input: Customer style profile
Output: Top-N best matches from catalog
Impact: High conversion, requires login
```

### **3. Email: "New Arrivals You'll Love"** ⭐⭐⭐⭐
```
Input: New products + customer style profiles
Output: Personalized newsletter
Impact: Medium, good for engagement
```

### **4. Cart: "Complete Your Look"** ⭐⭐⭐
```
Input: Items in cart + style profile
Output: Complementary products
Impact: Medium, increases AOV
```

### **5. Post-Purchase: "Based on Your Order"** ⭐⭐⭐
```
Input: Just-purchased items
Output: Similar products for next visit
Impact: Low immediate, good for retention
```

---

## 🔧 Advanced Features

### **Multi-Cluster Style Profiles**
Some customers have multiple distinct styles:
```python
from sklearn.cluster import KMeans

# Cluster customer's purchases into style groups
kmeans = KMeans(n_clusters=3)
style_centers = kmeans.fit(customer_purchase_embeddings)

# Recommend from each cluster
for center in style_centers:
    recommendations = find_similar(center, all_products)
```

**When to use**: VIP customers with >20 purchases

### **Style Evolution Tracking**
Monitor how taste changes over time:
```python
# Build profiles for different time windows
profile_last_30_days = build_profile(recent_interactions)
profile_6_months_ago = build_profile(old_interactions)

# Measure shift
style_shift = cosine_distance(profile_recent, profile_old)
```

**When to use**: Personalization research, trend analysis

### **Explainable Recommendations**
Show WHY an item was recommended:
```python
# Find most similar item customer previously liked
best_match = max(
    customer_purchases,
    key=lambda p: similarity(p, recommended_item)
)

explanation = f"Because you liked {best_match.name}"
```

**When to use**: Increase trust, improve UX

---

## ⚠️ Common Pitfalls

### **1. Filter Bubbles**
**Problem**: Only showing similar items → boring, repetitive
**Solution**: Add 10-20% random/popular items for exploration

### **2. Cold Start**
**Problem**: New customers have no interactions
**Solution**: Fallback to demographic or popularity-based recommendations

### **3. Outlier Products**
**Problem**: Customer bought gift (not their taste) → profile skewed
**Solution**: 
- Weight recent interactions higher
- Filter out gift purchases (if tagged)
- Use median instead of mean for aggregation

### **4. Stale Profiles**
**Problem**: Customer's taste changed, profile outdated
**Solution**: Rebuild profiles nightly or weekly

### **5. Same Product Recommended**
**Problem**: Algorithm suggests items customer already owns
**Solution**: Always exclude purchased products from recommendations

---

## 📈 Measuring Success

### **Offline Metrics** (Historical Data)
```python
# Precision: % of recommendations that user clicked/purchased
precision = clicked_recommendations / total_recommendations

# Recall: % of user's purchases that were recommended
recall = recommended_and_purchased / total_purchases

# Coverage: % of catalog that gets recommended
coverage = unique_recommended_products / total_catalog_size
```

### **Online Metrics** (A/B Test)
```
Control Group: Random recommendations
Treatment Group: Style-based recommendations

Measure:
- CTR (click-through rate)
- Conversion rate
- Revenue per user
- Average order value
```

**Target**: 20-30% CTR improvement is realistic

---

## 🎯 Implementation Checklist

- [ ] Extract embeddings for all products (use trained MobileNetV2)
- [ ] Export customer interaction data (views, carts, purchases)
- [ ] Build initial style profiles (batch job)
- [ ] Test recommendations for sample customers
- [ ] Implement recommendation API endpoint
- [ ] Add fallback logic (cold start, errors)
- [ ] Set up nightly profile updates
- [ ] Deploy to staging environment
- [ ] Run A/B test (2 weeks minimum)
- [ ] Analyze metrics and iterate
- [ ] Roll out to production

**Estimated Timeline**: 2-3 weeks from start to production

---

## 🤝 Questions & Next Steps

**Ready to implement?**

1. Start with the demo: `python code/demo_style_recommendations.py`
2. Extract real embeddings from your trained model
3. Test on sample customers
4. Deploy to product pages (lowest risk, high impact)
5. Measure and iterate

**Need help?**
- See `RECOMMENDATION_ENGINE.md` for full system architecture
- See `STYLE_RECOMMENDATION_GUIDE.md` for detailed code examples
- Review `code/customer_style_profiler.py` for API documentation

---

*Built on top of your MobileNetV2 image similarity model!*
