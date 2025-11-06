# Jewelry Image Classification Project

This project provides a complete solution for classifying jewelry images using deep learning and transfer learning techniques with PyTorch.

## 📋 Overview

This system can classify jewelry items into different categories such as:
- Rings
- Necklaces
- Earrings
- Bracelets
- Pendants
- Brooches
- Anklets
- Watches

## 🚀 Features

- **Transfer Learning**: Uses pre-trained models (EfficientNet, ResNet50, MobileNet, Vision Transformer)
- **Data Augmentation**: Automatic augmentation to improve model generalization
- **Data Preparation Tools**: Utilities for organizing, cleaning, and analyzing datasets
- **Comprehensive Training Pipeline**: Including automatic early stopping and learning rate scheduling
- **Evaluation Tools**: Confusion matrices, classification reports, and visualizations
- **Easy Prediction Interface**: Simple API for making predictions on new images

## 📁 Project Structure

```
project/
├── code/
│   ├── jewelry_classifier.py      # Main classifier class
│   └── data_preparation.py        # Data preprocessing utilities
├── data/
│   ├── raw_jewelry/              # Raw images organized by class
│   ├── organized_jewelry/        # Train/val/test split
│   └── scratch/                  # Temporary files
├── models/                        # Saved models and training logs
├── jewelry_classification_notebook.ipynb  # Main training notebook
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🛠️ Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Verify installation:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__} installed'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📊 Dataset Preparation

### Option 1: Use Your Own Dataset

1. Organize your images in folders by category:
```
data/raw_jewelry/
├── rings/
│   ├── ring1.jpg
│   ├── ring2.jpg
│   └── ...
├── necklaces/
│   ├── necklace1.jpg
│   └── ...
└── earrings/
    └── ...
```

2. Or use the data preparation script:
```python
from code.data_preparation import create_sample_dataset_structure
create_sample_dataset_structure('data/raw_jewelry')
```

### Option 2: Download a Public Dataset

You can use datasets from:
- Kaggle: Search for "jewelry dataset"
- Google Images: Use bulk download tools
- Custom web scraping: Ensure you have proper rights

## 🎓 Training the Model

### Quick Start (Jupyter Notebook)

1. Open `jewelry_classification_notebook.ipynb`
2. Run cells sequentially to:
   - Prepare data
   - Build model
   - Train classifier
   - Evaluate performance
   - Make predictions

### Using Python Scripts

```python
from code.jewelry_classifier import JewelryClassifier, create_data_loaders

# Create data loaders
train_loader, val_loader, test_loader, class_names = create_data_loaders(
    data_dir='data/organized_jewelry',
    img_size=(224, 224),
    batch_size=32
)

# Build classifier
classifier = JewelryClassifier(
    img_size=(224, 224),
    model_type='efficientnet'
)
model = classifier.build_model(num_classes=len(class_names))

# Train
history = classifier.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=30
)

# Save model
classifier.save_model('models/jewelry_classifier.pth')
```

## 🔍 Making Predictions

### Single Image Prediction

```python
import torch
from PIL import Image
from torchvision import transforms

# Load model
classifier = JewelryClassifier()
classifier.load_model('models/jewelry_classifier_final.pth')

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess image
img = Image.open('path/to/image.jpg').convert('RGB')
img_tensor = transform(img).unsqueeze(0)

# Predict
classifier.model.eval()
with torch.no_grad():
    outputs = classifier.model(img_tensor.to(classifier.device))
    probs = torch.softmax(outputs, dim=1)
    predicted_class = class_names[outputs.argmax(1).item()]
    confidence = probs[0].max().item() * 100

print(f"Predicted: {predicted_class} ({confidence:.2f}%)")
```

## 📈 Model Performance Tips

1. **More Data is Better**: Aim for at least 100-200 images per class
2. **Balanced Classes**: Try to have similar numbers of images for each class
3. **High Quality Images**: Clear, well-lit images work best
4. **Data Augmentation**: Use augmentation for smaller datasets
5. **Fine-tuning**: Unfreeze base model layers for better performance

## 🔧 Customization

### Change Base Model

```python
classifier = JewelryClassifier(
    model_type='vit'  # or 'efficientnet', 'resnet50', 'mobilenet'
)
```

### Adjust Image Size

```python
classifier = JewelryClassifier(
    img_size=(299, 299)  # Larger images may improve accuracy
)
```

### Modify Data Augmentation

Edit the `create_data_loaders` function in `jewelry_classifier.py`:
```python
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(40),  # Increase rotation
    # ... other transforms
])
```

## 📊 Evaluation Metrics

The model provides several metrics:
- **Accuracy**: Overall classification accuracy
- **Top-3 Accuracy**: Percentage where true class is in top 3 predictions
- **Confusion Matrix**: Shows which classes are confused with each other
- **Classification Report**: Precision, recall, and F1-score per class

## 🚀 Deployment

### Option 1: Flask Web App

```python
from flask import Flask, request, jsonify
import torch
from PIL import Image

app = Flask(__name__)
classifier = JewelryClassifier()
classifier.load_model('models/jewelry_classifier_final.pth')

@app.route('/predict', methods=['POST'])
def predict():
    # Handle image upload and prediction
    pass
```

### Option 2: TorchServe

```bash
# Archive model for TorchServe
torch-model-archiver --model-name jewelry_classifier \
    --version 1.0 \
    --model-file code/jewelry_classifier.py \
    --serialized-file models/jewelry_classifier_final.pth \
    --handler image_classifier

# Start TorchServe
torchserve --start --model-store model_store --models jewelry_classifier.mar
```

### Option 3: Mobile (PyTorch Mobile)

```python
# Convert to TorchScript
model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("jewelry_classifier_mobile.pt")
```

## 📝 Common Issues

### Out of Memory
- Reduce batch size
- Use a smaller base model (MobileNet)
- Reduce image size
- Use mixed precision training

### Low Accuracy
- Collect more training data
- Increase training epochs
- Try fine-tuning
- Check data quality

### Overfitting
- Increase data augmentation
- Add more dropout
- Use regularization
- Collect more diverse data

## 🤝 Contributing

Feel free to:
- Add new features
- Improve documentation
- Report bugs
- Suggest enhancements

## 📄 License

This project is provided as-is for educational and commercial use.

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- timm library for pre-trained models
- Pre-trained model providers (ImageNet)
- Open source community

## 📧 Support

For questions or issues:
1. Check the Jupyter notebook examples
2. Review the code documentation
3. Search for similar issues in the codebase

---

**Happy Classifying! 💍📸**

