# Quick Start Guide for Jewelry Classification

## ✅ Dataset Ready!

Your synthetic jewelry dataset has been generated with:
- **8 categories**: rings, necklaces, earrings, bracelets, pendants, brooches, anklets, watches
- **400 total images**: 50 images per category
- **Location**: `data/raw_jewelry/`

## 🚀 Next Steps

### Option 1: Run the Full Notebook
Open `jewelry_classification_notebook.ipynb` and run all cells sequentially.

### Option 2: Quick Test with Code Below

Run this code to quickly organize data and start training:

```python
import sys
sys.path.append('code')

from data_preparation import organize_dataset
from jewelry_classifier import JewelryClassifier, create_data_loaders

# 1. Organize dataset into train/val/test splits
print("Organizing dataset...")
organize_dataset(
    source_dir='data/raw_jewelry',
    output_dir='data/organized_jewelry',
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

# 2. Create data loaders
print("\nCreating data loaders...")
train_loader, val_loader, test_loader, class_names = create_data_loaders(
    data_dir='data/organized_jewelry',
    img_size=(224, 224),
    batch_size=16,  # Smaller batch for testing
    num_workers=2,
    augment=True
)

print(f"\n✅ Ready to train!")
print(f"Classes: {class_names}")
print(f"Training samples: {len(train_loader.dataset)}")
print(f"Validation samples: {len(val_loader.dataset)}")
print(f"Test samples: {len(test_loader.dataset)}")

# 3. Build and train model (quick test with 5 epochs)
print("\n🔨 Building model...")
classifier = JewelryClassifier(
    img_size=(224, 224),
    num_classes=len(class_names),
    model_type='mobilenet'  # Faster than efficientnet
)

model = classifier.build_model(len(class_names))

print("\n🎯 Training model (quick test - 5 epochs)...")
history = classifier.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=5,  # Quick test
    save_best=True,
    model_path='models/jewelry_quick_test.pth'
)

# 4. Evaluate
print("\n📊 Evaluating model...")
test_metrics, predictions, true_labels = classifier.evaluate(test_loader)

for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\n✅ Quick test complete!")
print("For better results, increase epochs to 30+ and use 'efficientnet' model.")
```

## 📊 Visualize Your Dataset

```python
from code.visualize_dataset import visualize_dataset_samples
visualize_dataset_samples()
```

## 🎯 Expected Results

With this synthetic dataset and quick training:
- **Training time**: ~5-10 minutes (5 epochs on CPU)
- **Expected accuracy**: ~60-80% (will improve with more epochs)
- **For production**: Train for 30+ epochs with data augmentation

## 📝 Dataset Statistics

```
Category      Images
---------     ------
rings         50
necklaces     50
earrings      50
bracelets     50
pendants      50
brooches      50
anklets       50
watches       50
---------     ------
TOTAL         400
```

After organizing (70/15/15 split):
- Training: 280 images (35 per class)
- Validation: 60 images (7-8 per class)
- Test: 60 images (7-8 per class)

## 💡 Tips for Better Results

1. **Increase epochs**: Use 30-50 epochs for better accuracy
2. **Try different models**: 
   - `mobilenet` - Fast, lower accuracy
   - `efficientnet` - Balanced (recommended)
   - `resnet50` - Slower, higher accuracy
   - `vit` - State-of-the-art, requires more data
3. **Fine-tuning**: After initial training, unfreeze layers and train more
4. **Real images**: Replace synthetic images with real jewelry photos for production use

## 🔄 Regenerate Dataset

If you want more images or different variety:

```bash
python code/generate_test_dataset.py
```

Edit the script to change `images_per_category` (default: 50).
