"""
Download sample jewelry images for testing the classifier.
Uses free image sources with proper licensing.
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import time

def download_image(url, filepath):
    """Download an image from a URL."""
    try:
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
    return False

def download_sample_jewelry_images(output_dir='../data/raw_jewelry'):
    """
    Download sample jewelry images from Unsplash (free to use).
    """
    output_path = Path(output_dir)
    
    # Sample jewelry images from Unsplash
    # Unsplash provides free high-quality images
    # Format: https://images.unsplash.com/photo-{id}?w=800&q=80
    
    jewelry_samples = {
        'rings': [
            'https://images.unsplash.com/photo-1605100804763-247f67b3557e?w=800&q=80',  # Diamond ring
            'https://images.unsplash.com/photo-1515562141207-7a88fb7ce338?w=800&q=80',  # Wedding rings
            'https://images.unsplash.com/photo-1611591437281-460bfbe1220a?w=800&q=80',  # Gold ring
            'https://images.unsplash.com/photo-1590497920084-e6e8e97e46ba?w=800&q=80',  # Engagement ring
            'https://images.unsplash.com/photo-1603561596112-0a132b757442?w=800&q=80',  # Silver ring
        ],
        'necklaces': [
            'https://images.unsplash.com/photo-1599643478518-a784e5dc4c8f?w=800&q=80',  # Pearl necklace
            'https://images.unsplash.com/photo-1535632066927-ab7c9ab60908?w=800&q=80',  # Gold necklace
            'https://images.unsplash.com/photo-1506630448388-4e683c67ddb0?w=800&q=80',  # Diamond necklace
            'https://images.unsplash.com/photo-1611591437281-460bfbe1220a?w=800&q=80',  # Chain necklace
            'https://images.unsplash.com/photo-1602751584552-8ba73aad10e1?w=800&q=80',  # Pendant necklace
        ],
        'earrings': [
            'https://images.unsplash.com/photo-1535556116002-6281ff3e9f04?w=800&q=80',  # Gold earrings
            'https://images.unsplash.com/photo-1596944924616-7b38e7cfac36?w=800&q=80',  # Diamond earrings
            'https://images.unsplash.com/photo-1617038260897-41a1f14a8ca0?w=800&q=80',  # Pearl earrings
            'https://images.unsplash.com/photo-1595934840260-4784c50a2bdd?w=800&q=80',  # Hoop earrings
            'https://images.unsplash.com/photo-1573408301185-9146fe634ad0?w=800&q=80',  # Stud earrings
        ],
        'bracelets': [
            'https://images.unsplash.com/photo-1611591437281-460bfbe1220a?w=800&q=80',  # Gold bracelet
            'https://images.unsplash.com/photo-1602751584552-8ba73aad10e1?w=800&q=80',  # Silver bracelet
            'https://images.unsplash.com/photo-1573408301185-9146fe634ad0?w=800&q=80',  # Charm bracelet
            'https://images.unsplash.com/photo-1588444650700-dc1c2f3e2729?w=800&q=80',  # Bangle
            'https://images.unsplash.com/photo-1589674781759-c0c0b0e5e3c0?w=800&q=80',  # Pearl bracelet
        ],
        'watches': [
            'https://images.unsplash.com/photo-1523170335258-f5ed11844a49?w=800&q=80',  # Luxury watch
            'https://images.unsplash.com/photo-1524805444758-089113d48a6d?w=800&q=80',  # Silver watch
            'https://images.unsplash.com/photo-1533139143976-37a6043d2e29?w=800&q=80',  # Gold watch
            'https://images.unsplash.com/photo-1587836374228-4c4bad90d3d0?w=800&q=80',  # Sport watch
            'https://images.unsplash.com/photo-1522312346375-d1a52e2b99b3?w=800&q=80',  # Classic watch
        ],
    }
    
    print("Downloading sample jewelry images...")
    print("Source: Unsplash (https://unsplash.com - Free to use)")
    print()
    
    total_downloaded = 0
    
    for category, urls in jewelry_samples.items():
        category_path = output_path / category
        category_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nDownloading {category}...")
        for idx, url in enumerate(tqdm(urls, desc=category)):
            filepath = category_path / f"{category}_{idx+1}.jpg"
            
            if filepath.exists():
                print(f"  {filepath.name} already exists, skipping...")
                continue
            
            if download_image(url, filepath):
                total_downloaded += 1
                time.sleep(0.5)  # Be nice to the server
            else:
                print(f"  Failed to download {filepath.name}")
    
    print(f"\n✅ Downloaded {total_downloaded} images successfully!")
    print(f"📁 Images saved to: {output_path}")
    
    return output_path


def create_test_images_subset(source_dir='../data/raw_jewelry', 
                               test_dir='../data/test_images',
                               images_per_class=2):
    """
    Create a small test set from the downloaded images.
    """
    import shutil
    
    source_path = Path(source_dir)
    test_path = Path(test_dir)
    test_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreating test images subset...")
    
    for category_dir in source_path.iterdir():
        if category_dir.is_dir():
            images = list(category_dir.glob('*.jpg'))[:images_per_class]
            
            for img in images:
                shutil.copy2(img, test_path / f"{category_dir.name}_{img.name}")
    
    print(f"✅ Test images created in: {test_path}")
    return test_path


if __name__ == '__main__':
    # Download sample images
    output_dir = download_sample_jewelry_images()
    
    # Create a test subset
    test_dir = create_test_images_subset()
    
    print("\n" + "="*60)
    print("Setup complete! You can now:")
    print("1. View the downloaded images in:", output_dir)
    print("2. Use test images from:", test_dir)
    print("3. Run the notebook to organize and train on this data")
    print("="*60)
