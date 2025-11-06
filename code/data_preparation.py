"""
Data preparation utilities for jewelry classification.
"""

import os
import shutil
import random
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm


def organize_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Organize images into train/val/test splits.
    
    Args:
        source_dir: Directory with subdirectories for each class
        output_dir: Output directory for organized dataset
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1"
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_path / split).mkdir(parents=True, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(class_dirs)} classes")
    
    for class_dir in tqdm(class_dirs, desc="Processing classes"):
        class_name = class_dir.name
        
        # Create class directories in each split
        for split in ['train', 'val', 'test']:
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
        
        # Get all images in this class
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(class_dir.glob(ext)))
        
        # Shuffle images
        random.shuffle(image_files)
        
        # Split data
        n_total = len(image_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        # Copy files
        for img_file in train_files:
            shutil.copy2(img_file, output_path / 'train' / class_name / img_file.name)
        
        for img_file in val_files:
            shutil.copy2(img_file, output_path / 'val' / class_name / img_file.name)
        
        for img_file in test_files:
            shutil.copy2(img_file, output_path / 'test' / class_name / img_file.name)
        
        print(f"{class_name}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    print(f"\nDataset organized in {output_dir}")


def check_and_clean_images(data_dir, min_size=(50, 50), remove_corrupted=True):
    """
    Check for corrupted images and optionally remove them.
    
    Args:
        data_dir: Directory containing images
        min_size: Minimum acceptable image size (width, height)
        remove_corrupted: Whether to remove corrupted images
    """
    data_path = Path(data_dir)
    corrupted_images = []
    small_images = []
    
    print("Checking images...")
    
    for img_file in tqdm(list(data_path.rglob('*.jpg')) + 
                         list(data_path.rglob('*.jpeg')) + 
                         list(data_path.rglob('*.png')) +
                         list(data_path.rglob('*.JPG')) +
                         list(data_path.rglob('*.JPEG')) +
                         list(data_path.rglob('*.PNG'))):
        try:
            # Try to open image
            img = Image.open(img_file)
            img.verify()
            
            # Check size
            img = Image.open(img_file)  # Need to reopen after verify
            if img.size[0] < min_size[0] or img.size[1] < min_size[1]:
                small_images.append(img_file)
            
        except Exception as e:
            corrupted_images.append((img_file, str(e)))
    
    print(f"\nFound {len(corrupted_images)} corrupted images")
    print(f"Found {len(small_images)} images smaller than {min_size}")
    
    if remove_corrupted and corrupted_images:
        print("\nRemoving corrupted images...")
        for img_file, error in corrupted_images:
            print(f"Removing {img_file}: {error}")
            os.remove(img_file)
    
    if remove_corrupted and small_images:
        print("\nRemoving small images...")
        for img_file in small_images:
            print(f"Removing {img_file}")
            os.remove(img_file)
    
    return corrupted_images, small_images


def analyze_dataset(data_dir):
    """
    Analyze dataset statistics.
    
    Args:
        data_dir: Directory containing subdirectories for each class
    """
    data_path = Path(data_dir)
    
    # Count images per class
    class_counts = {}
    image_sizes = []
    aspect_ratios = []
    
    class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    for class_dir in tqdm(class_dirs, desc="Analyzing classes"):
        class_name = class_dir.name
        
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(class_dir.glob(ext)))
        
        class_counts[class_name] = len(image_files)
        
        # Sample some images for size analysis
        sample_size = min(100, len(image_files))
        for img_file in random.sample(image_files, sample_size):
            try:
                img = Image.open(img_file)
                image_sizes.append(img.size)
                aspect_ratios.append(img.size[0] / img.size[1])
            except:
                pass
    
    print("\n=== Dataset Analysis ===")
    print(f"\nTotal classes: {len(class_counts)}")
    print(f"Total images: {sum(class_counts.values())}")
    
    print("\nImages per class:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {count}")
    
    if image_sizes:
        widths = [size[0] for size in image_sizes]
        heights = [size[1] for size in image_sizes]
        
        print("\nImage size statistics:")
        print(f"  Width: min={min(widths)}, max={max(widths)}, avg={np.mean(widths):.1f}")
        print(f"  Height: min={min(heights)}, max={max(heights)}, avg={np.mean(heights):.1f}")
        print(f"  Aspect ratio: min={min(aspect_ratios):.2f}, max={max(aspect_ratios):.2f}, avg={np.mean(aspect_ratios):.2f}")
    
    return class_counts


def augment_minority_classes(data_dir, target_count=None, output_dir=None):
    """
    Augment images for minority classes to balance the dataset.
    
    Args:
        data_dir: Directory containing subdirectories for each class
        target_count: Target number of images per class (default: max class count)
        output_dir: Output directory (default: same as input)
    """
    from PIL import Image
    from torchvision import transforms
    import torchvision.transforms.functional as TF
    
    data_path = Path(data_dir)
    output_path = Path(output_dir) if output_dir else data_path
    
    # Count images per class
    class_counts = {}
    class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    for class_dir in class_dirs:
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(class_dir.glob(ext)))
        class_counts[class_dir.name] = len(image_files)
    
    if target_count is None:
        target_count = max(class_counts.values())
    
    print(f"Target count per class: {target_count}")
    
    # Create augmentation transforms
    augmentation_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.2, 0.2),
            scale=(0.8, 1.2),
            shear=15
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    ])
    
    for class_dir in tqdm(class_dirs, desc="Augmenting classes"):
        class_name = class_dir.name
        current_count = class_counts[class_name]
        
        if current_count >= target_count:
            print(f"\n{class_name}: Already has {current_count} images, skipping")
            continue
        
        needed = target_count - current_count
        print(f"\n{class_name}: Generating {needed} additional images")
        
        # Get all images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(class_dir.glob(ext)))
        
        # Create output directory
        output_class_dir = output_path / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate augmented images
        generated = 0
        while generated < needed:
            # Pick a random image
            img_file = random.choice(image_files)
            img = Image.open(img_file).convert('RGB')
            
            # Apply augmentation
            aug_img = augmentation_transforms(img)
            
            # Save with unique name
            output_file = output_class_dir / f"aug_{generated:04d}_{img_file.name}"
            aug_img.save(output_file)
            
            generated += 1
        
        print(f"{class_name}: Generated {generated} images")


def create_sample_dataset_structure(output_dir):
    """
    Create a sample directory structure for jewelry dataset.
    
    Args:
        output_dir: Directory where sample structure will be created
    """
    output_path = Path(output_dir)
    
    # Common jewelry categories
    categories = [
        'rings',
        'necklaces',
        'earrings',
        'bracelets',
        'pendants',
        'brooches',
        'anklets',
        'watches'
    ]
    
    for category in categories:
        (output_path / category).mkdir(parents=True, exist_ok=True)
    
    print(f"Created sample dataset structure with {len(categories)} categories:")
    for category in categories:
        print(f"  - {category}")
    
    print(f"\nPlace your images in the respective category folders under {output_dir}")
