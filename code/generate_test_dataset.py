"""
Create a synthetic jewelry dataset for testing the classification model.
Generates simple images with text labels and shapes to simulate different jewelry types.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import random

def create_jewelry_image(category, img_size=(400, 400), seed=None):
    """
    Create a synthetic jewelry image with visual characteristics for each category.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Create image with random background
    bg_color = tuple(np.random.randint(200, 255, 3).tolist())
    img = Image.new('RGB', img_size, bg_color)
    draw = ImageDraw.Draw(img)
    
    # Define category-specific visual features
    if category == 'rings':
        # Draw circular ring
        center_x, center_y = img_size[0] // 2, img_size[1] // 2
        outer_radius = random.randint(80, 120)
        inner_radius = outer_radius - 20
        
        # Ring body
        draw.ellipse([center_x - outer_radius, center_y - outer_radius,
                     center_x + outer_radius, center_y + outer_radius],
                    fill=(random.randint(180, 220), random.randint(150, 190), random.randint(50, 100)))
        draw.ellipse([center_x - inner_radius, center_y - inner_radius,
                     center_x + inner_radius, center_y + inner_radius],
                    fill=bg_color)
        
        # Gem on top
        gem_size = 15
        draw.ellipse([center_x - gem_size, center_y - outer_radius - gem_size,
                     center_x + gem_size, center_y - outer_radius + gem_size],
                    fill=(random.randint(100, 255), random.randint(100, 255), random.randint(200, 255)))
    
    elif category == 'necklaces':
        # Draw chain/necklace
        y_start = 50
        y_end = img_size[1] - 50
        num_beads = random.randint(8, 12)
        
        # Chain
        for i in range(num_beads):
            x = img_size[0] // 2 + int(60 * np.sin(i * np.pi / (num_beads - 1)))
            y = y_start + (y_end - y_start) * i // (num_beads - 1)
            bead_size = random.randint(15, 25)
            draw.ellipse([x - bead_size, y - bead_size, x + bead_size, y + bead_size],
                        fill=(random.randint(150, 200), random.randint(140, 190), random.randint(60, 110)))
        
        # Pendant at bottom
        pendant_x = img_size[0] // 2
        pendant_y = y_end
        draw.ellipse([pendant_x - 30, pendant_y - 30, pendant_x + 30, pendant_y + 30],
                    fill=(random.randint(50, 150), random.randint(100, 200), random.randint(150, 255)))
    
    elif category == 'earrings':
        # Draw two earrings
        left_x = img_size[0] // 3
        right_x = 2 * img_size[0] // 3
        y_pos = img_size[1] // 2
        
        for x in [left_x, right_x]:
            # Hook
            draw.line([(x, y_pos - 40), (x, y_pos)], 
                     fill=(random.randint(150, 200), random.randint(150, 200), random.randint(100, 150)), 
                     width=3)
            # Dangle
            dangle_size = random.randint(20, 35)
            draw.ellipse([x - dangle_size, y_pos, x + dangle_size, y_pos + dangle_size * 2],
                        fill=(random.randint(100, 200), random.randint(100, 200), random.randint(150, 255)))
    
    elif category == 'bracelets':
        # Draw bracelet as elongated circle
        center_x, center_y = img_size[0] // 2, img_size[1] // 2
        width = random.randint(140, 180)
        height = random.randint(100, 130)
        thickness = 20
        
        # Outer
        draw.ellipse([center_x - width, center_y - height,
                     center_x + width, center_y + height],
                    fill=(random.randint(180, 220), random.randint(160, 200), random.randint(70, 120)))
        # Inner
        draw.ellipse([center_x - width + thickness, center_y - height + thickness,
                     center_x + width - thickness, center_y + height - thickness],
                    fill=bg_color)
        
        # Add beads
        num_beads = random.randint(6, 10)
        for i in range(num_beads):
            angle = 2 * np.pi * i / num_beads
            bead_x = center_x + int((width - thickness // 2) * np.cos(angle))
            bead_y = center_y + int((height - thickness // 2) * np.sin(angle))
            bead_size = 8
            draw.ellipse([bead_x - bead_size, bead_y - bead_size,
                         bead_x + bead_size, bead_y + bead_size],
                        fill=(random.randint(100, 255), random.randint(100, 255), random.randint(150, 255)))
    
    elif category == 'pendants':
        # Draw pendant on chain
        center_x = img_size[0] // 2
        
        # Chain
        draw.line([(center_x, 50), (center_x, 200)],
                 fill=(random.randint(150, 200), random.randint(150, 200), random.randint(100, 150)),
                 width=2)
        
        # Pendant shape
        pendant_y = 220
        shape_type = random.choice(['diamond', 'heart', 'circle'])
        
        if shape_type == 'circle':
            draw.ellipse([center_x - 40, pendant_y - 40, center_x + 40, pendant_y + 40],
                        fill=(random.randint(100, 200), random.randint(100, 200), random.randint(150, 255)))
        elif shape_type == 'diamond':
            points = [(center_x, pendant_y - 50), (center_x + 40, pendant_y),
                     (center_x, pendant_y + 50), (center_x - 40, pendant_y)]
            draw.polygon(points, fill=(random.randint(100, 200), random.randint(100, 200), random.randint(150, 255)))
        else:  # heart
            draw.ellipse([center_x - 35, pendant_y - 20, center_x, pendant_y + 10],
                        fill=(random.randint(150, 255), random.randint(50, 100), random.randint(50, 100)))
            draw.ellipse([center_x, pendant_y - 20, center_x + 35, pendant_y + 10],
                        fill=(random.randint(150, 255), random.randint(50, 100), random.randint(50, 100)))
    
    elif category == 'brooches':
        # Draw decorative brooch
        center_x, center_y = img_size[0] // 2, img_size[1] // 2
        
        # Base
        draw.ellipse([center_x - 60, center_y - 40, center_x + 60, center_y + 40],
                    fill=(random.randint(150, 200), random.randint(140, 190), random.randint(60, 110)))
        
        # Decorative gems
        num_gems = random.randint(5, 8)
        for i in range(num_gems):
            angle = 2 * np.pi * i / num_gems
            gem_x = center_x + int(35 * np.cos(angle))
            gem_y = center_y + int(25 * np.sin(angle))
            gem_size = random.randint(8, 15)
            draw.ellipse([gem_x - gem_size, gem_y - gem_size,
                         gem_x + gem_size, gem_y + gem_size],
                        fill=(random.randint(100, 255), random.randint(100, 255), random.randint(150, 255)))
    
    elif category == 'anklets':
        # Similar to bracelet but thinner
        center_x, center_y = img_size[0] // 2, img_size[1] // 2
        width = random.randint(120, 160)
        height = random.randint(80, 110)
        
        # Chain
        num_links = 20
        for i in range(num_links):
            angle = 2 * np.pi * i / num_links
            x1 = center_x + int(width * np.cos(angle))
            y1 = center_y + int(height * np.sin(angle))
            x2 = center_x + int(width * np.cos(angle + np.pi / num_links))
            y2 = center_y + int(height * np.sin(angle + np.pi / num_links))
            draw.line([(x1, y1), (x2, y2)],
                     fill=(random.randint(180, 220), random.randint(180, 220), random.randint(150, 200)),
                     width=3)
        
        # Charm
        charm_x = center_x + width
        charm_y = center_y
        draw.rectangle([charm_x - 15, charm_y - 15, charm_x + 15, charm_y + 15],
                      fill=(random.randint(100, 200), random.randint(100, 200), random.randint(150, 255)))
    
    elif category == 'watches':
        # Draw watch
        center_x, center_y = img_size[0] // 2, img_size[1] // 2
        
        # Watch face
        face_size = random.randint(80, 100)
        draw.ellipse([center_x - face_size, center_y - face_size,
                     center_x + face_size, center_y + face_size],
                    fill=(random.randint(200, 240), random.randint(200, 240), random.randint(200, 240)))
        draw.ellipse([center_x - face_size + 5, center_y - face_size + 5,
                     center_x + face_size - 5, center_y + face_size - 5],
                    outline=(50, 50, 50), width=2)
        
        # Watch hands
        draw.line([(center_x, center_y), (center_x + 30, center_y - 40)],
                 fill=(30, 30, 30), width=3)
        draw.line([(center_x, center_y), (center_x + 50, center_y)],
                 fill=(30, 30, 30), width=2)
        
        # Strap
        strap_width = 30
        draw.rectangle([center_x - strap_width // 2, center_y - face_size - 80,
                       center_x + strap_width // 2, center_y - face_size],
                      fill=(random.randint(50, 100), random.randint(50, 100), random.randint(50, 100)))
        draw.rectangle([center_x - strap_width // 2, center_y + face_size,
                       center_x + strap_width // 2, center_y + face_size + 80],
                      fill=(random.randint(50, 100), random.randint(50, 100), random.randint(50, 100)))
    
    # Add some noise for realism
    img_array = np.array(img)
    noise = np.random.randint(-10, 10, img_array.shape, dtype=np.int16)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    
    return img


def generate_dataset(output_dir='data/raw_jewelry', images_per_category=50):
    """
    Generate a complete synthetic jewelry dataset.
    """
    output_path = Path(output_dir)
    
    categories = ['rings', 'necklaces', 'earrings', 'bracelets', 
                  'pendants', 'brooches', 'anklets', 'watches']
    
    print(f"Generating synthetic jewelry dataset...")
    print(f"Categories: {categories}")
    print(f"Images per category: {images_per_category}")
    print(f"Output directory: {output_path}")
    print()
    
    total_images = 0
    
    for category in categories:
        category_path = output_path / category
        category_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating {category}...")
        for i in range(images_per_category):
            img = create_jewelry_image(category, seed=i)
            img_path = category_path / f"{category}_{i+1:03d}.jpg"
            img.save(img_path, quality=85)
            total_images += 1
        
        print(f"  ✓ Created {images_per_category} images in {category_path}")
    
    print(f"\n{'='*60}")
    print(f"✅ Dataset generation complete!")
    print(f"📊 Total images: {total_images}")
    print(f"📁 Location: {output_path.absolute()}")
    print(f"{'='*60}")
    
    # Print summary
    print(f"\nDataset structure:")
    for category in categories:
        category_path = output_path / category
        num_files = len(list(category_path.glob('*.jpg')))
        print(f"  {category:12s}: {num_files:3d} images")
    
    print(f"\n🎯 You can now run the notebook to:")
    print(f"   1. Organize data into train/val/test splits")
    print(f"   2. Train the jewelry classifier")
    print(f"   3. Evaluate and make predictions")
    
    return output_path


if __name__ == '__main__':
    # Generate dataset with 50 images per category (400 total images)
    dataset_path = generate_dataset(
        output_dir='data/raw_jewelry',
        images_per_category=50
    )
