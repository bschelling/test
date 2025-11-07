"""
Process Rhomberg jewelry CSV and download images via HTTP
- On spark: Direct access to /mnt/img/jpeg/detailbilder/{size}/{id}.jpg
- Remote: Download from image_link URLs (faster than SSH!)
"""

import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
import socket

# Configuration
CSV_PATH = Path("/project/data/jewlery.csv")
OUTPUT_DIR = Path("/project/data/rhomberg_final")
IMAGES_DIR = OUTPUT_DIR / "images"

# Settings
MAX_IMAGES_PER_CATEGORY = 200  # Set to None for ALL images
SPARK_SHARE_PATH = "/mnt/img/jpeg/detailbilder"
PREFERRED_SIZE = "200"  # Use 200px for faster downloads

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
IMAGES_DIR.mkdir(exist_ok=True, parents=True)

def is_on_spark():
    """Check if running on spark machine"""
    return Path(SPARK_SHARE_PATH).exists()

def clean_category(product_type):
    """Extract main category"""
    if pd.isna(product_type):
        return 'unknown'
    
    parts = str(product_type).split('>')
    if len(parts) > 0:
        main = parts[0].strip().lower()
        # Map German to English
        mapping = {
            'fingerringe': 'rings',
            'ohrschmuck': 'earrings',
            'halsschmuck': 'necklaces',
            'armschmuck': 'bracelets',
            'anhänger': 'pendants',
            'piercing': 'piercing',
            'fußketten': 'anklets'
        }
        for de, en in mapping.items():
            if de in main:
                return en
        return main.split()[0] if main else 'unknown'
    return 'unknown'

def extract_material(material_str):
    """Extract primary material"""
    if pd.isna(material_str):
        return 'unknown'
    
    material = str(material_str).lower()
    
    if 'platin' in material:
        return 'platinum'
    elif 'gold' in material:
        return 'gold'
    elif 'silber' in material or 'silver' in material:
        return 'silver'
    elif 'edelstahl' in material or 'stainless' in material:
        return 'stainless_steel'
    elif 'titan' in material:
        return 'titan'
    else:
        return material.split()[0] if material.split() else 'unknown'

def download_image_http(url, dest_path):
    """Download image via HTTP"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        return False

def copy_local_image(product_id, dest_path):
    """Copy from local spark share"""
    try:
        # Try different sizes
        for size in [PREFERRED_SIZE, "360", "155", "100"]:
            source_path = Path(SPARK_SHARE_PATH) / size / f"{product_id}.jpg"
            if source_path.exists():
                import shutil
                shutil.copy2(source_path, dest_path)
                return True
        return False
    except Exception as e:
        return False

def main():
    print("=" * 70)
    print("RHOMBERG JEWELRY DATA PROCESSING")
    print("=" * 70)
    
    # Check environment
    on_spark = is_on_spark()
    print(f"\n{'🎯 Running ON spark machine' if on_spark else '🌐 Running REMOTELY'}")
    print(f"{'   → Using local image files' if on_spark else '   → Downloading images via HTTP'}")
    
    # Read CSV
    print(f"\nReading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, sep='\t', on_bad_lines='skip')
    print(f"Total products: {len(df):,}")
    
    # Process data
    print("\nProcessing metadata...")
    df['category'] = df['product_type'].apply(clean_category)
    df['material_clean'] = df['material'].apply(extract_material)
    df['gender_clean'] = df['gender'].fillna('unisex')
    df['price_clean'] = df['price'].str.replace(' EUR', '').str.replace(',', '.').astype(float, errors='ignore')
    
    def get_price_range(price):
        try:
            p = float(price)
            if p < 50: return 'budget'
            elif p < 100: return 'mid_range'
            elif p < 300: return 'premium'
            else: return 'luxury'
        except:
            return 'unknown'
    
    df['price_range'] = df['price_clean'].apply(get_price_range)
    
    # Filter products with images
    df_with_images = df[df['image_link'].notna()].copy()
    
    # Sample if needed
    if MAX_IMAGES_PER_CATEGORY:
        print(f"\n📊 Sampling up to {MAX_IMAGES_PER_CATEGORY} images per category...")
        sampled_dfs = []
        for category in df_with_images['category'].unique():
            cat_df = df_with_images[df_with_images['category'] == category]
            n = min(MAX_IMAGES_PER_CATEGORY, len(cat_df))
            sampled = cat_df.sample(n=n, random_state=42)
            sampled_dfs.append(sampled)
            print(f"  {category:15s}: {n:4d} products")
        df_to_process = pd.concat(sampled_dfs, ignore_index=True)
    else:
        df_to_process = df_with_images
    
    print(f"\nTotal to process: {len(df_to_process):,} products")
    
    # Process images
    print("\n" + "=" * 70)
    print("PROCESSING IMAGES")
    print("=" * 70)
    
    results = []
    found = 0
    not_found = 0
    skipped = 0
    
    for idx, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="Processing"):
        category = row['category']
        product_id = row['id']
        
        # Create destination
        category_dir = IMAGES_DIR / category
        category_dir.mkdir(exist_ok=True)
        dest_filename = f"{category}_{product_id}.jpg"
        dest_path = category_dir / dest_filename
        
        # Skip if exists
        if dest_path.exists():
            skipped += 1
            found += 1
            results.append({
                'filename': dest_filename,
                'filepath': str(dest_path),
                'product_id': product_id,
                'title': row['title'],
                'category': category,
                'gender': row['gender_clean'],
                'material': row['material_clean'],
                'price': row['price_clean'],
                'price_range': row['price_range'],
                'source': 'cached'
            })
            continue
        
        # Get image
        success = False
        if on_spark:
            # Copy from local share
            success = copy_local_image(product_id, dest_path)
        else:
            # Download via HTTP
            success = download_image_http(row['image_link'], dest_path)
        
        if success:
            found += 1
            results.append({
                'filename': dest_filename,
                'filepath': str(dest_path),
                'product_id': product_id,
                'title': row['title'],
                'category': category,
                'gender': row['gender_clean'],
                'material': row['material_clean'],
                'price': row['price_clean'],
                'price_range': row['price_range'],
                'source': 'local' if on_spark else 'http'
            })
        else:
            not_found += 1
    
    # Save metadata
    if results:
        results_df = pd.DataFrame(results)
        metadata_path = OUTPUT_DIR / 'jewelry_metadata.csv'
        results_df.to_csv(metadata_path, index=False)
        
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"✓ Successfully processed: {found:,} images")
        if skipped > 0:
            print(f"⚡ Skipped (cached): {skipped:,} images")
        print(f"✗ Not found: {not_found:,} images")
        print(f"\n✓ Metadata: {metadata_path}")
        print(f"✓ Images: {IMAGES_DIR}")
        
        print("\n" + "=" * 70)
        print("DATASET SUMMARY")
        print("=" * 70)
        print(f"\nCategories ({results_df['category'].nunique()}):")
        print(results_df['category'].value_counts())
        print(f"\nGenders ({results_df['gender'].nunique()}):")
        print(results_df['gender'].value_counts())
        print(f"\nMaterials ({results_df['material'].nunique()}):")
        print(results_df['material'].value_counts().head(10))
        print(f"\n✓ Dataset ready for training!")
    else:
        print("\n✗ No images processed")

if __name__ == "__main__":
    main()
