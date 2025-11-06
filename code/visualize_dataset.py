"""
Visualize sample images from the generated jewelry dataset.
"""

import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import random

def visualize_dataset_samples(data_dir='data/raw_jewelry', samples_per_category=3):
    """
    Display sample images from each category.
    """
    data_path = Path(data_dir)
    categories = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    
    num_categories = len(categories)
    fig, axes = plt.subplots(num_categories, samples_per_category, 
                            figsize=(15, 2.5 * num_categories))
    
    if num_categories == 1:
        axes = axes.reshape(1, -1)
    
    for i, category in enumerate(categories):
        category_path = data_path / category
        images = list(category_path.glob('*.jpg'))
        
        # Select random samples
        sample_images = random.sample(images, min(samples_per_category, len(images)))
        
        for j, img_path in enumerate(sample_images):
            img = Image.open(img_path)
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_title(f"{category.upper()}\n({len(images)} images)", 
                                    fontweight='bold', fontsize=12)
            else:
                axes[i, j].set_title(f"Sample {j+1}", fontsize=10)
    
    plt.suptitle('Jewelry Dataset Preview', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('data/dataset_preview.png', dpi=150, bbox_inches='tight')
    print("✅ Dataset preview saved to: data/dataset_preview.png")
    plt.show()


if __name__ == '__main__':
    visualize_dataset_samples()
