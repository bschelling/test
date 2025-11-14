"""
Quick Demo: Customer Style-Based Recommendations

This script demonstrates how to match customer taste using image embeddings.
Run this to see the concept in action with sample data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append('/project/code')
from customer_style_profiler import CustomerStyleProfiler

def main():
    print("=" * 70)
    print("CUSTOMER STYLE-BASED RECOMMENDATION DEMO")
    print("=" * 70)
    
    # Step 1: Load interaction data
    print("\n📊 Loading sample data...")
    interactions = pd.read_csv('/project/data/sample_interactions.csv')
    customers = pd.read_csv('/project/data/sample_customers.csv')
    
    print(f"  Interactions: {len(interactions):,}")
    print(f"  Customers: {len(customers)}")
    print(f"  Unique products: {interactions['product_id'].nunique()}")
    
    # Step 2: Simulate product embeddings
    # (In production, these would be real embeddings from your MobileNetV2 model)
    print("\n🎨 Creating simulated product embeddings...")
    print("  (In production: extract from trained image model)")
    
    np.random.seed(42)
    unique_products = interactions['product_id'].unique()
    
    # Create embeddings with structure:
    # - Product 111350 & 111899 are similar (both silver rings)
    # - Product 564522 is different (watch/higher price)
    product_embeddings = {}
    
    for product_id in unique_products:
        if str(product_id) in ['111350', '111899']:
            # Similar style (silver rings)
            base_vector = np.array([1.0, 0.5, 0.2] * 427)[:1280]  # Repeated pattern
            noise = np.random.randn(1280) * 0.1
            embedding = base_vector + noise
        else:
            # Different style
            base_vector = np.array([0.2, 0.8, 1.0] * 427)[:1280]
            noise = np.random.randn(1280) * 0.1
            embedding = base_vector + noise
        
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        product_embeddings[str(product_id)] = embedding
    
    print(f"  ✓ Created {len(product_embeddings)} embeddings (1280-dim each)")
    
    # Step 3: Build customer style profiles
    print("\n👤 Building customer style profiles...")
    profiler = CustomerStyleProfiler(
        product_embeddings=product_embeddings,
        recency_half_life_days=30
    )
    
    customer_profiles = profiler.build_all_profiles(
        interactions_df=interactions,
        min_interactions=3
    )
    
    print(f"  ✓ Built profiles for {len(customer_profiles)} customers")
    
    # Step 4: Demo recommendations for a customer
    print("\n" + "=" * 70)
    print("RECOMMENDATION EXAMPLE: Customer C10001")
    print("=" * 70)
    
    customer_id = 'C10001'
    
    # Show customer's history
    customer_interactions = interactions[interactions['customer_id'] == customer_id]
    customer_info = customers[customers['customer_id'] == customer_id].iloc[0]
    
    print(f"\nCustomer Profile:")
    print(f"  Age: {customer_info['age']} ({customer_info['age_group']})")
    print(f"  Gender: {customer_info['gender']}")
    print(f"  Segment: {customer_info['customer_segment']}")
    print(f"  Lifetime Value: €{customer_info['lifetime_value']:.2f}")
    print(f"  Total Orders: {customer_info['total_orders']}")
    
    print(f"\nInteraction History:")
    interaction_summary = customer_interactions.groupby(['product_id', 'interaction_type']).size()
    for (product, interaction), count in interaction_summary.items():
        print(f"  Product {product}: {interaction} ({count}x)")
    
    # Get products they already interacted with
    interacted_products = customer_interactions['product_id'].unique().astype(str).tolist()
    
    # Generate recommendations
    print(f"\n🎯 Top 5 Style-Based Recommendations:")
    print(f"   (Excluding already purchased/viewed products)")
    
    recommendations = profiler.recommend_by_style(
        customer_id=customer_id,
        top_n=5,
        exclude_products=interacted_products
    )
    
    if recommendations:
        for i, (product_id, similarity) in enumerate(recommendations, 1):
            print(f"\n  {i}. Product {product_id}")
            print(f"     Style similarity: {similarity:.4f} ({similarity*100:.1f}%)")
            print(f"     Why: Matches your preference for similar designs")
    else:
        print("  (No recommendations - insufficient interaction data)")
    
    # Step 5: Find similar customers
    print("\n" + "=" * 70)
    print("CUSTOMERS WITH SIMILAR TASTE")
    print("=" * 70)
    
    similar_customers = profiler.find_style_neighbors(customer_id, top_n=3)
    
    if similar_customers:
        print(f"\nTop 3 customers with similar style to {customer_id}:")
        for i, (other_id, similarity) in enumerate(similar_customers, 1):
            other_info = customers[customers['customer_id'] == other_id]
            if not other_info.empty:
                other = other_info.iloc[0]
                print(f"\n  {i}. {other_id} - Similarity: {similarity:.4f}")
                print(f"     Age: {other['age']}, Gender: {other['gender']}")
                print(f"     Segment: {other['customer_segment']}")
                print(f"     → Could use for collaborative filtering!")
    
    # Step 6: Summary statistics
    print("\n" + "=" * 70)
    print("SYSTEM STATISTICS")
    print("=" * 70)
    
    print(f"\nCustomer Profiles: {len(customer_profiles)}")
    print(f"Product Embeddings: {len(product_embeddings)}")
    print(f"Total Interactions: {len(interactions):,}")
    
    print(f"\nInteraction Type Breakdown:")
    for interaction_type, count in interactions['interaction_type'].value_counts().items():
        print(f"  {interaction_type}: {count}")
    
    # Calculate average similarity within customer purchases
    print(f"\nStyle Consistency:")
    consistencies = []
    for cid in list(customer_profiles.keys())[:5]:  # Sample 5 customers
        customer_products = interactions[
            interactions['customer_id'] == cid
        ]['product_id'].unique()
        
        if len(customer_products) >= 2:
            embeddings = [product_embeddings[str(p)] for p in customer_products 
                         if str(p) in product_embeddings]
            if len(embeddings) >= 2:
                # Average pairwise similarity
                similarities = []
                for i, emb1 in enumerate(embeddings):
                    for emb2 in embeddings[i+1:]:
                        sim = np.dot(emb1, emb2)
                        similarities.append(sim)
                avg_sim = np.mean(similarities)
                consistencies.append(avg_sim)
                print(f"  {cid}: {avg_sim:.3f} (higher = more consistent taste)")
    
    if consistencies:
        print(f"\n  Average style consistency: {np.mean(consistencies):.3f}")
    
    print("\n" + "=" * 70)
    print("✓ DEMO COMPLETE")
    print("=" * 70)
    print("\nNext Steps:")
    print("  1. Extract real embeddings from your trained MobileNetV2 model")
    print("  2. Run on full product catalog (~1,500 products)")
    print("  3. Update profiles daily as new interactions arrive")
    print("  4. A/B test against baseline (random/popularity)")
    print("  5. Measure CTR and conversion improvements")
    print("\nSee STYLE_RECOMMENDATION_GUIDE.md for implementation details.")

if __name__ == "__main__":
    main()
