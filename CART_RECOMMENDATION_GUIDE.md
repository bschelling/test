# Cart Recommendation System Guide

## Overview

This cart recommendation system provides intelligent product suggestions based on items currently in a customer's shopping cart. It combines **visual similarity** (style matching) with **collaborative filtering** (co-purchase patterns) to deliver relevant recommendations.

## How Customer Interactions Are Used

The system leverages customer interaction data in two key ways:

### 1. Co-Purchase Patterns (Collaborative Filtering)

**Data Source**: `data/sample_interactions.csv`

**Process**:
- Analyzes historical purchase events to identify products frequently bought together
- Builds a symmetric co-purchase matrix counting pair-wise product associations
- Example: If customers often buy Ring X and Necklace Y together, this creates a strong co-purchase link

**Implementation**:
- Only completed `purchase` events are used (not views or add-to-cart actions)
- Products purchased by the same customer in the same session are linked
- Scores are normalized to 0-1 range for consistent weighting

**Weight in Final Score**: 30% (`CO_PURCHASE_WEIGHT = 0.3`)

### 2. Visual Similarity (Content-Based Filtering)

**Data Source**: Product images from `data/rhomberg_final/images/`

**Process**:
- Extracts visual embeddings using a fine-tuned MobileNetV2 model
- Calculates average style embedding for all items in cart
- Finds products with similar visual characteristics (metal type, design style, etc.)

**Weight in Final Score**: 70% (`VISUAL_SIMILARITY_WEIGHT = 0.7`)

## Recommendation Algorithm

### Step-by-Step Process

1. **Extract Cart Embeddings**
   - Get visual embeddings for each product in cart
   - Calculate average embedding (represents overall cart style)

2. **Calculate Visual Similarity**
   - Compare cart's average embedding against all products
   - Use cosine similarity or euclidean distance
   - Generates visual similarity score (0-1) for each product

3. **Add Co-Purchase Boost**
   - For each cart item, lookup co-purchased products
   - Aggregate co-purchase counts across all cart items
   - Normalize to 0-1 range

4. **Combine Scores**
   ```
   Final Score = (0.7 × Visual Similarity) + (0.3 × Co-Purchase Score)
   ```

5. **Apply Filters & Boosts**
   - **Exclude cart items**: Remove products already in cart
   - **Same collection boost**: 1.5× multiplier for matching collections
   - **Price range filter**: Penalize products outside price tolerance (0.7× multiplier)

6. **Rank & Return**
   - Sort by final score (descending)
   - Return top K recommendations (default: 10)

## Configuration Options

### Weighting Parameters

```python
VISUAL_SIMILARITY_WEIGHT = 0.7   # Weight for visual/style matching
CO_PURCHASE_WEIGHT = 0.3          # Weight for co-purchase patterns
```

### Filtering & Boosting

```python
EXCLUDE_CART_ITEMS = True         # Remove cart products from recommendations
SAME_COLLECTION_BOOST = 1.5       # Boost factor for same collection
PRICE_RANGE_TOLERANCE = 0.5       # ±50% price range (0 = disabled)
```

### Model Settings

```python
SIMILARITY_METHOD = 'cosine'      # 'cosine' or 'euclidean'
TOP_K = 10                        # Number of recommendations to return
IMG_SIZE = 224                    # Image input size for model
EMBEDDING_DIM = 1280              # MobileNetV2 feature dimension
```

## What's Currently NOT Used

The interaction data contains additional information that could be leveraged:

- ❌ **Individual customer history**: No personalization based on specific customer's past purchases
- ❌ **View/click patterns**: Only purchase events are considered
- ❌ **Add-to-cart events**: Could indicate strong intent even without purchase
- ❌ **Time-based trends**: No seasonality or trending product analysis
- ❌ **Product popularity**: View counts and overall popularity not factored in

## Usage Examples

### Basic Usage

```python
# Automatic cart selection (from interaction data)
CUSTOM_CART_PRODUCT_IDS = None
recommendations = recommend_for_cart(sample_cart, top_k=10)
```

### Custom Product IDs

```python
# Specify exact products for testing
CUSTOM_CART_PRODUCT_IDS = ['12345', '67890', '11111']
recommendations = recommend_for_cart(CUSTOM_CART_PRODUCT_IDS, top_k=10)
```

### Output Format

The recommendation function returns a DataFrame with:

| Column | Description |
|--------|-------------|
| `product_id` | Unique product identifier |
| `final_score` | Combined recommendation score (0-1) |
| `visual_similarity` | Visual matching score (0-1) |
| `co_purchase_score` | Co-purchase pattern score (0-1) |
| `product_name` | Product name/description (if available) |
| `price` | Product price (if available) |
| `collection` | Product collection (if available) |
| `category` | Product category (if available) |

## Data Requirements

### Required Files

1. **Product Catalog**: `data/feed_a62656-2_de.csv`
   - Tab-delimited file with product metadata
   - Must contain `artikel_id` or `id` column

2. **Interaction Data**: `data/sample_interactions.csv`
   - Customer interaction events
   - Required columns: `customer_id`, `product_id`, `interaction_type`, `timestamp`
   - Interaction types: `view`, `add_to_cart`, `purchase`

3. **Product Images**: `data/rhomberg_final/images/`
   - JPG images named by product ID (e.g., `12345.jpg`)
   - Used for visual embedding extraction

4. **Pre-computed Embeddings** (optional): `models/hybrid_product_embeddings.npz`
   - Speeds up recommendations by avoiding re-extraction
   - Automatically created after first run

## Performance Optimization

### Embedding Cache

- Embeddings are cached in memory during runtime
- Saved to disk for future sessions
- Only re-extracted if:
  - New products added
  - Model fine-tuned
  - Cache file deleted

### Fine-Tuning Benefits

Running the fine-tuning step (Cell 10) improves recommendations by:
- Adapting MobileNetV2 to jewelry-specific features
- Learning domain-specific visual patterns
- Typically improves accuracy by 15-20%
- Only needs to run once (model saved to disk)

## Troubleshooting

### No Recommendations Generated

**Problem**: Empty recommendation DataFrame

**Possible Causes**:
- No valid embeddings for cart items
- Missing product images
- All products filtered out

**Solution**: Check that cart product IDs have corresponding images

### Low Co-Purchase Scores

**Problem**: All co-purchase scores are 0

**Possible Causes**:
- Insufficient interaction data
- No purchase events in data
- Cart products never co-purchased

**Solution**: Visual similarity will still work; consider collecting more interaction data

### Out of Memory Errors

**Problem**: GPU/CPU memory exhausted

**Possible Causes**:
- Too many products being processed
- Large batch sizes during fine-tuning

**Solution**: 
- Reduce `IMG_SIZE` to 128 or 160
- Process products in smaller batches
- Use CPU instead of GPU for inference

## Future Enhancements

Potential improvements to consider:

1. **Personalization**: Incorporate individual customer purchase history
2. **Multi-signal interactions**: Use view and add-to-cart events
3. **Popularity boost**: Factor in overall product popularity
4. **Diversity**: Ensure recommendations span multiple categories
5. **Real-time learning**: Update co-purchase matrix incrementally
6. **A/B testing**: Framework for comparing recommendation strategies
7. **Explainability**: Show why each product was recommended

## Technical Architecture

```
Input: Cart Product IDs
    ↓
[Visual Analysis]          [Behavioral Analysis]
    ↓                            ↓
Extract embeddings      Query co-purchase matrix
Calculate similarity    Aggregate co-purchases
    ↓                            ↓
    └──────── Combine ──────────┘
                ↓
        [Filtering & Ranking]
                ↓
        Apply business rules
        Sort by final score
                ↓
        Output: Top K Products
```

## Notebook Structure

| Cell | Section | Purpose |
|------|---------|---------|
| 1 | Configuration | Set hyperparameters and paths |
| 2 | Dependencies | Import libraries |
| 3 | Load Data | Read products, interactions, embeddings |
| 4 | Image Model | Initialize MobileNetV2 for embeddings |
| 5 | Fine-tuning | Optional: Adapt model to jewelry |
| 6 | Extract Embeddings | Process all product images |
| 7 | Co-Purchase Matrix | Build collaborative filtering data |
| 8 | Recommendation Engine | Main algorithm implementation |
| 9 | Test Recommendations | Generate sample recommendations |
| 10 | Visualization | Display results with images |

---

**Last Updated**: November 30, 2025  
**Version**: 1.0  
**Notebook**: `cart_recommendations.ipynb`
