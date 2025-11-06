# Jewelry Image Similarity Model

This branch contains an image similarity model using Siamese networks and triplet loss.

## What's Different from Classification?

### Classification Model (main branch)
- **Goal**: Assign each image to one of 8 categories
- **Output**: Class label (ring, necklace, etc.)
- **Method**: Transfer learning with softmax classifier
- **Use case**: "What type of jewelry is this?"

### Similarity Model (this branch)
- **Goal**: Learn visual similarity between images
- **Output**: 128-dimensional embedding vector
- **Method**: Siamese network with triplet loss
- **Use case**: "Find jewelry that looks like this"

## How It Works

### 1. Triplet Learning
For each training sample, we use three images:
- **Anchor**: Reference image (e.g., a silver ring)
- **Positive**: Different image of same category (another silver ring)
- **Negative**: Image from different category (a necklace)

### 2. Triplet Loss
The model learns to:
- Make anchor-positive distance **small** (similar items close together)
- Make anchor-negative distance **large** (dissimilar items far apart)
- Maintain a margin between them

Formula: `L = max(d(anchor, positive) - d(anchor, negative) + margin, 0)`

### 3. Embeddings
Each image is converted to a 128-D vector where:
- Similar images have close embeddings (small Euclidean distance)
- Dissimilar images have far embeddings (large distance)

### 4. Similarity Search
To find similar items:
1. Get embedding for query image
2. Compute distances to all other embeddings
3. Return k-nearest neighbors

## Key Advantages

✅ **Open-set recognition**: Can add new items without retraining  
✅ **Fine-grained similarity**: Captures visual details beyond categories  
✅ **Flexible**: Works for recommendation, search, clustering  
✅ **No fixed classes**: Not limited to training categories  

## Files

- `jewelry_similarity_notebook.ipynb` - Main training notebook
- `jewelry_classification_notebook.ipynb` - Original classification model
- `models/best_similarity_model.pth` - Trained similarity model
- `models/jewelry_embeddings.pkl` - Pre-computed embeddings for all images

## Quick Start

```bash
# Switch to similarity branch
git checkout image-similarity

# Open notebook
jupyter lab jewelry_similarity_notebook.ipynb

# Or run all cells programmatically
jupyter nbconvert --to notebook --execute jewelry_similarity_notebook.ipynb
```

## Performance Metrics

We evaluate using **Precision@K**:
- How many of the top-K retrieved items are from the same category?

Expected results (after 30 epochs):
- Precision@1: ~0.85-0.95 (85-95% of closest matches are correct)
- Precision@5: ~0.75-0.90
- Precision@10: ~0.70-0.85

## Use Cases

### E-commerce
- "Customers who viewed this also viewed..."
- Visual search: Upload photo, find similar products
- Duplicate product detection

### Organization
- Auto-cluster similar items
- Group related products
- Find outliers/misclassified items

### Recommendation
- Content-based filtering
- Style-based suggestions
- "More like this" features

## Technical Details

- **Architecture**: MobileNetV2 backbone + embedding head
- **Embedding dim**: 128
- **Loss**: Triplet loss with margin=1.0
- **Optimizer**: Adam (lr=0.0001)
- **Training**: 30 epochs on triplet dataset
- **Batch size**: 16 triplets per batch

## Comparison with Other Approaches

| Method | Pros | Cons |
|--------|------|------|
| **Classification** | Simple, fast inference | Fixed classes, no similarity measure |
| **Siamese (this)** | Flexible, captures similarity | Requires triplet mining |
| **Autoencoders** | Unsupervised | May not preserve semantic similarity |
| **CLIP** | Zero-shot, text queries | Large model, needs more compute |

## Next Steps

1. **Hard triplet mining**: Focus on difficult examples
2. **Online triplet mining**: Generate triplets during training
3. **Multi-scale features**: Combine features from multiple layers
4. **Attention mechanisms**: Focus on important image regions
5. **Metric learning**: Try other losses (N-pair, ArcFace)

## References

- [FaceNet: A Unified Embedding for Face Recognition](https://arxiv.org/abs/1503.03832)
- [Deep Metric Learning](https://arxiv.org/abs/1706.07567)
- [Triplet Loss and Online Triplet Mining](https://omoindrot.github.io/triplet-loss)
