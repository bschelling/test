"""
Customer Style Profiler
Builds visual taste profiles for customers based on interaction history
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple

class CustomerStyleProfiler:
    """Build and manage customer style profiles from image embeddings."""
    
    # Interaction strength weights
    INTERACTION_WEIGHTS = {
        'purchase': 10.0,
        'add_to_wishlist': 5.0,
        'add_to_cart': 3.0,
        'view': 1.0,
        'click': 0.5
    }
    
    def __init__(self, 
                 product_embeddings: Dict[str, np.ndarray],
                 recency_half_life_days: int = 30):
        """
        Initialize the profiler.
        
        Args:
            product_embeddings: Dict mapping product_id -> embedding vector
            recency_half_life_days: Days for exponential decay (default 30)
        """
        self.product_embeddings = product_embeddings
        self.recency_half_life = recency_half_life_days
        self.customer_profiles = {}
        
    def compute_recency_weight(self, interaction_date: datetime) -> float:
        """
        Compute exponential decay weight based on how old the interaction is.
        
        Args:
            interaction_date: When the interaction happened
            
        Returns:
            Weight between 0 and 1 (1 = today, 0.5 = half_life days ago)
        """
        days_ago = (datetime.now() - interaction_date).days
        decay_rate = np.log(2) / self.recency_half_life
        return np.exp(-decay_rate * days_ago)
    
    def build_style_profile(self,
                           customer_id: str,
                           interactions_df: pd.DataFrame,
                           min_interactions: int = 3) -> np.ndarray:
        """
        Build a style profile for a single customer.
        
        Args:
            customer_id: Customer identifier
            interactions_df: DataFrame with columns: 
                           [customer_id, product_id, interaction_type, timestamp]
            min_interactions: Minimum interactions needed (default 3)
            
        Returns:
            Style profile vector (same dimension as product embeddings)
            Returns None if insufficient data
        """
        # Filter to this customer's interactions
        customer_interactions = interactions_df[
            interactions_df['customer_id'] == customer_id
        ].copy()
        
        if len(customer_interactions) < min_interactions:
            return None
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(customer_interactions['timestamp']):
            customer_interactions['timestamp'] = pd.to_datetime(
                customer_interactions['timestamp']
            )
        
        weighted_embeddings = []
        total_weight = 0.0
        
        for _, interaction in customer_interactions.iterrows():
            product_id = str(interaction['product_id'])
            
            # Skip if product embedding not found
            if product_id not in self.product_embeddings:
                continue
            
            # Get embedding
            embedding = self.product_embeddings[product_id]
            
            # Compute weight
            interaction_weight = self.INTERACTION_WEIGHTS.get(
                interaction['interaction_type'], 1.0
            )
            recency_weight = self.compute_recency_weight(interaction['timestamp'])
            combined_weight = interaction_weight * recency_weight
            
            # Add weighted embedding
            weighted_embeddings.append(combined_weight * embedding)
            total_weight += combined_weight
        
        if total_weight == 0:
            return None
        
        # Compute weighted average
        style_profile = np.sum(weighted_embeddings, axis=0) / total_weight
        
        # Normalize to unit vector
        style_profile = style_profile / np.linalg.norm(style_profile)
        
        # Cache the profile
        self.customer_profiles[customer_id] = style_profile
        
        return style_profile
    
    def build_all_profiles(self,
                          interactions_df: pd.DataFrame,
                          min_interactions: int = 3) -> Dict[str, np.ndarray]:
        """
        Build style profiles for all customers.
        
        Args:
            interactions_df: DataFrame with interaction history
            min_interactions: Minimum interactions per customer
            
        Returns:
            Dictionary mapping customer_id -> style_profile
        """
        customer_ids = interactions_df['customer_id'].unique()
        
        profiles = {}
        for customer_id in customer_ids:
            profile = self.build_style_profile(
                customer_id, 
                interactions_df, 
                min_interactions
            )
            if profile is not None:
                profiles[customer_id] = profile
        
        self.customer_profiles = profiles
        return profiles
    
    def recommend_by_style(self,
                          customer_id: str,
                          top_n: int = 10,
                          exclude_products: List[str] = None,
                          min_similarity: float = 0.0) -> List[Tuple[str, float]]:
        """
        Recommend products based on customer's style profile.
        
        Args:
            customer_id: Customer to recommend for
            top_n: Number of recommendations
            exclude_products: Product IDs to exclude (already purchased)
            min_similarity: Minimum similarity threshold (0-1)
            
        Returns:
            List of (product_id, similarity_score) tuples
        """
        if customer_id not in self.customer_profiles:
            return []
        
        style_profile = self.customer_profiles[customer_id]
        exclude_products = set(exclude_products or [])
        
        # Compute similarity to all products
        recommendations = []
        for product_id, embedding in self.product_embeddings.items():
            if product_id in exclude_products:
                continue
            
            # Cosine similarity
            similarity = np.dot(style_profile, embedding) / (
                np.linalg.norm(style_profile) * np.linalg.norm(embedding)
            )
            
            if similarity >= min_similarity:
                recommendations.append((product_id, float(similarity)))
        
        # Sort by similarity and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_n]
    
    def get_style_diversity_score(self, customer_id: str) -> float:
        """
        Measure how diverse a customer's taste is (0 = very consistent, 1 = very diverse).
        
        Args:
            customer_id: Customer to analyze
            
        Returns:
            Diversity score (higher = more diverse taste)
        """
        if customer_id not in self.customer_profiles:
            return None
        
        # Get all embeddings for products this customer interacted with
        # This requires access to interaction data - simplified here
        # In practice, you'd compute variance of embeddings relative to style profile
        
        # Placeholder implementation
        return 0.5
    
    def find_style_neighbors(self,
                            customer_id: str,
                            top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Find customers with similar style profiles.
        
        Args:
            customer_id: Reference customer
            top_n: Number of similar customers to return
            
        Returns:
            List of (customer_id, similarity) tuples
        """
        if customer_id not in self.customer_profiles:
            return []
        
        reference_profile = self.customer_profiles[customer_id]
        
        similarities = []
        for other_id, other_profile in self.customer_profiles.items():
            if other_id == customer_id:
                continue
            
            similarity = np.dot(reference_profile, other_profile)
            similarities.append((other_id, float(similarity)))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]


# Example usage
if __name__ == "__main__":
    # Mock data for demonstration
    print("Customer Style Profiler - Example Usage")
    print("=" * 60)
    
    # Simulate product embeddings (1280-dim from MobileNetV2)
    np.random.seed(42)
    product_ids = ['111350', '111899', '564522', '100001', '100002']
    product_embeddings = {
        pid: np.random.randn(1280) / 10  # Small random embeddings
        for pid in product_ids
    }
    
    # Normalize embeddings
    for pid in product_embeddings:
        product_embeddings[pid] /= np.linalg.norm(product_embeddings[pid])
    
    # Simulate interaction data
    interactions_data = {
        'customer_id': ['C10001', 'C10001', 'C10001', 'C10002', 'C10002'],
        'product_id': ['111350', '111899', '111350', '564522', '111899'],
        'interaction_type': ['view', 'view', 'purchase', 'purchase', 'add_to_wishlist'],
        'timestamp': [
            '2024-11-10 10:00:00',
            '2024-11-10 10:05:00',
            '2024-11-10 10:15:00',
            '2024-11-12 14:30:00',
            '2024-11-12 14:35:00'
        ]
    }
    interactions_df = pd.DataFrame(interactions_data)
    
    # Initialize profiler
    profiler = CustomerStyleProfiler(
        product_embeddings=product_embeddings,
        recency_half_life_days=30
    )
    
    # Build profiles
    profiles = profiler.build_all_profiles(interactions_df, min_interactions=2)
    
    print(f"\nBuilt {len(profiles)} customer style profiles")
    print(f"Customers: {list(profiles.keys())}")
    
    # Get recommendations for C10001
    recommendations = profiler.recommend_by_style(
        customer_id='C10001',
        top_n=3,
        exclude_products=['111350']  # Already purchased
    )
    
    print(f"\nRecommendations for C10001:")
    for product_id, score in recommendations:
        print(f"  Product {product_id}: {score:.4f} similarity")
    
    # Find similar customers
    similar_customers = profiler.find_style_neighbors('C10001', top_n=2)
    print(f"\nCustomers with similar taste to C10001:")
    for customer_id, score in similar_customers:
        print(f"  {customer_id}: {score:.4f} similarity")
