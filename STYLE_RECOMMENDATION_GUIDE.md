# Style-Based Recommendation System
## Using Image Embeddings to Match Customer Taste

This notebook demonstrates how to build personalized recommendations by matching customer visual preferences with product image similarity.

## 📋 Overview

**Core Idea**: Each customer has a unique "visual taste" that can be learned from their interaction history. By analyzing which products they view, add to cart, and purchase, we can create a **style profile** (embedding vector) that represents their aesthetic preferences.

### How It Works

```
Customer Interactions → Weighted Image Embeddings → Style Profile → Similar Products
```

1. **Extract**: Get embeddings for all products customer interacted with
2. **Weight**: More weight for purchases > wishlists > views
3. **Aggregate**: Compute weighted average → customer's style vector
4. **Recommend**: Find products with embeddings similar to style vector

---

## 🔧 Setup

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import sys

# Add code directory to path
sys.path.append('/project/code')
from customer_style_profiler import CustomerStyleProfiler

# Load your trained model
from torchvision import models, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

---

## 📊 Load Sample Data

```python
# Load customer data
customers = pd.read_csv('/project/data/sample_customers.csv')
orders = pd.read_csv('/project/data/sample_orders.csv')
order_items = pd.read_csv('/project/data/sample_order_items.csv')
interactions = pd.read_csv('/project/data/sample_interactions.csv')

# Load product catalog
products = pd.read_csv('/project/data/feed_a62656-2_de.csv', sep='\t', low_memory=False)

print(f"Customers: {len(customers)}")
print(f"Orders: {len(orders)}")
print(f"Interactions: {len(interactions)}")
print(f"Products: {len(products)}")
```

---

## 🎨 Step 1: Extract Image Embeddings for All Products

Use your trained MobileNetV2 model to generate embeddings.

```python
# Load model (same architecture as collection predictor)
class ImageEmbeddingExtractor(nn.Module):
    """Extract 1280-dim embeddings from MobileNetV2."""
    
    def __init__(self):
        super().__init__()
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.features = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten to (batch, 1280)
        return x

# Initialize model
embedding_model = ImageEmbeddingExtractor().to(device)
embedding_model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("✓ Model ready for embedding extraction")
```

```python
def extract_embedding(image_path):
    """Extract embedding for a single image."""
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embedding = embedding_model(img_tensor)
        
        return embedding.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Extract embeddings for sample products
# (In production, do this for entire catalog and cache results)

product_embeddings = {}
image_dir = Path('/project/data/collection_images')

# Get unique products from interactions
unique_products = interactions['product_id'].unique()

print(f"\\nExtracting embeddings for {len(unique_products)} products...")

for product_id in unique_products:
    image_path = image_dir / f"{product_id}.jpg"
    
    if image_path.exists():
        embedding = extract_embedding(str(image_path))
        if embedding is not None:
            product_embeddings[str(product_id)] = embedding

print(f"✓ Extracted {len(product_embeddings)} embeddings")
print(f"  Embedding dimension: {list(product_embeddings.values())[0].shape[0]}")
```

---

## 👤 Step 2: Build Customer Style Profiles

```python
# Initialize the profiler
profiler = CustomerStyleProfiler(
    product_embeddings=product_embeddings,
    recency_half_life_days=30  # Recent interactions weighted higher
)

# Build profiles for all customers
customer_profiles = profiler.build_all_profiles(
    interactions_df=interactions,
    min_interactions=3  # Need at least 3 interactions
)

print(f"\\nBuilt style profiles for {len(customer_profiles)} customers")
print(f"Customers with profiles: {list(customer_profiles.keys())[:5]}...")
```

---

## 🎯 Step 3: Generate Style-Based Recommendations

```python
# Example: Recommend for customer C10001
customer_id = 'C10001'

# Get products they already interacted with
interacted_products = interactions[
    interactions['customer_id'] == customer_id
]['product_id'].unique().tolist()

print(f"\\nCustomer {customer_id} previously interacted with:")
print(f"  Products: {interacted_products}")

# Get style-based recommendations
recommendations = profiler.recommend_by_style(
    customer_id=customer_id,
    top_n=10,
    exclude_products=[str(p) for p in interacted_products],
    min_similarity=0.5  # Only show reasonably similar items
)

print(f"\\nTop 10 Style-Based Recommendations:")
for i, (product_id, similarity) in enumerate(recommendations, 1):
    # Get product details
    product_info = products[products['id'] == int(product_id)]
    if not product_info.empty:
        title = product_info.iloc[0]['title']
        price = product_info.iloc[0]['price']
        print(f"  {i}. [{product_id}] {title[:50]}... (€{price}) - {similarity:.3f}")
    else:
        print(f"  {i}. Product {product_id} - Similarity: {similarity:.3f}")
```

---

## 📊 Step 4: Analyze Customer Style Diversity

```python
# Find customers with similar taste
similar_customers = profiler.find_style_neighbors(
    customer_id='C10001',
    top_n=5
)

print(f"\\nCustomers with similar style to C10001:")
for customer_id, similarity in similar_customers:
    customer_info = customers[customers['customer_id'] == customer_id]
    if not customer_info.empty:
        age = customer_info.iloc[0]['age']
        gender = customer_info.iloc[0]['gender']
        segment = customer_info.iloc[0]['customer_segment']
        print(f"  {customer_id}: {similarity:.3f} - {gender}, {age}y, {segment}")
```

---

## 🔍 Step 5: Visualize Style Profiles (Optional)

```python
# Compare style profiles using t-SNE or PCA
from sklearn.decomposition import PCA

# Collect all style profiles
profile_matrix = np.array([
    customer_profiles[cid] for cid in customer_profiles.keys()
])

# Reduce to 2D for visualization
pca = PCA(n_components=2)
profiles_2d = pca.fit_transform(profile_matrix)

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(profiles_2d[:, 0], profiles_2d[:, 1], alpha=0.6, s=100)

for i, customer_id in enumerate(customer_profiles.keys()):
    plt.annotate(customer_id, (profiles_2d[i, 0], profiles_2d[i, 1]), 
                 fontsize=8, alpha=0.7)

plt.xlabel('Style Dimension 1')
plt.ylabel('Style Dimension 2')
plt.title('Customer Style Profiles (PCA Projection)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\\nVariance explained by 2 components: {pca.explained_variance_ratio_.sum():.2%}")
```

---

## 🚀 Production Integration

### Save Embeddings for Fast Lookup

```python
# Save product embeddings to disk
embeddings_file = Path('/project/models/product_embeddings.npz')
np.savez_compressed(
    embeddings_file,
    product_ids=list(product_embeddings.keys()),
    embeddings=np.array(list(product_embeddings.values()))
)

print(f"✓ Saved {len(product_embeddings)} embeddings to {embeddings_file}")
```

### Save Customer Profiles

```python
# Save customer style profiles
profiles_file = Path('/project/models/customer_style_profiles.npz')
np.savez_compressed(
    profiles_file,
    customer_ids=list(customer_profiles.keys()),
    profiles=np.array(list(customer_profiles.values()))
)

print(f"✓ Saved {len(customer_profiles)} customer profiles to {profiles_file}")
```

---

## 💡 Advanced: Hybrid Recommendation Score

Combine style matching with other signals:

```python
def hybrid_recommendation_score(customer_id, product_id, 
                                style_weight=0.5,
                                category_weight=0.2,
                                price_weight=0.15,
                                popularity_weight=0.15):
    """
    Compute hybrid score combining multiple signals.
    
    Returns: float between 0 and 1
    """
    # 1. Style similarity
    style_score = 0.0
    if customer_id in customer_profiles and product_id in product_embeddings:
        profile = customer_profiles[customer_id]
        embedding = product_embeddings[product_id]
        style_score = np.dot(profile, embedding)
    
    # 2. Category preference (has customer bought this category before?)
    category_score = 0.5  # Placeholder - implement based on order history
    
    # 3. Price affinity (does price match customer's typical range?)
    price_score = 0.5  # Placeholder - implement based on AOV
    
    # 4. Popularity (how many others purchased this?)
    popularity_score = 0.5  # Placeholder - implement based on order frequency
    
    # Weighted combination
    final_score = (
        style_weight * style_score +
        category_weight * category_score +
        price_weight * price_score +
        popularity_weight * popularity_score
    )
    
    return final_score

# Example usage
score = hybrid_recommendation_score('C10001', '111350')
print(f"Hybrid recommendation score: {score:.3f}")
```

---

## 🎯 Key Insights

### Why This Works:

1. **Visual Consistency**: People tend to have consistent aesthetic preferences (modern vs vintage, minimalist vs ornate)
2. **Implicit Signals**: Even views/clicks reveal taste without purchases
3. **Temporal Evolution**: Recent interactions weighted higher (tastes change)
4. **Cross-Category**: Style transfers across jewelry types (rings + necklaces)

### Limitations:

1. **Cold Start**: New customers need interactions first
2. **Exploration**: May create filter bubbles (only similar items)
3. **Context**: Doesn't account for occasions (daily wear vs gifts)

### Solutions:

- **Cold Start**: Use demographic filtering initially
- **Exploration**: Add random 10-20% recommendations for diversity
- **Context**: Segment by purchase occasion when available

---

## 📈 Next Steps

1. **Extract embeddings** for full product catalog (~1,500 products)
2. **Update profiles** nightly based on new interactions
3. **A/B test** style-based vs random/popularity recommendations
4. **Monitor** click-through rate (CTR) and conversion improvements
5. **Tune weights** in hybrid scoring based on business metrics

---

*This approach leverages your existing image similarity model for personalized recommendations!*
