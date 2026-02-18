# Off-Road Semantic Segmentation Hackathon

This repository contains a deep learning solution for semantic segmentation of off-road terrain images. The model uses DINOv2 as a backbone with a custom ConvNeXt-style segmentation head to classify 10 different terrain classes.

## 🎯 Project Overview

The goal of this project is to perform semantic segmentation on off-road images, identifying and classifying different terrain types including:
- Background
- Trees
- Lush Bushes
- Dry Grass
- Dry Bushes
- Ground Clutter
- Logs
- Rocks
- Landscape
- Sky

## 🏗️ Architecture

- **Backbone**: DINOv2 (Vision Transformer) - Small variant (`dinov2_vits14`)
- **Segmentation Head**: Custom ConvNeXt-style decoder
- **Input Size**: 480x270 (resized from 960x540)
- **Number of Classes**: 10

## 📁 Project Structure

```
Hackathon/
├── README.md                          # This file
├── Train.py                           # Training script
├── Test.py                            # Testing/Inference script
├── requirements.txt                   # Python dependencies
├── Offroad_Segmentation_Training_Dataset/
│   ├── train/
│   │   ├── Color_Images/              # Training images
│   │   └── Segmentation/              # Training masks
│   └── val/
│       ├── Color_Images/              # Validation images
│       └── Segmentation/              # Validation masks
└── Offroad_Segmentation_testImages/
    ├── Color_Images/                  # Test images
    └── Segmentation/                  # Test masks (for evaluation)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended) or CPU
- PyTorch 1.9+

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd Hackathon
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training

Train the model using the provided training script:

```bash
python Train.py
```

**Training Parameters:**
- Batch size: 2
- Learning rate: 1e-4
- Optimizer: SGD with momentum (0.9)
- Epochs: 10
- Image size: 480x270

**Outputs:**
- `segmentation_head.pth`: Trained model weights
- `train_stats/`: Directory containing:
  - Training curves (loss, IoU, Dice, accuracy)
  - `evaluation_metrics.txt`: Detailed metrics per epoch

### Testing/Inference

Run inference on test images:

```bash
python Test.py --model_path segmentation_head.pth --data_dir Offroad_Segmentation_testImages --output_dir predictions
```

**Arguments:**
- `--model_path`: Path to trained model weights (default: `segmentation_head.pth`)
- `--data_dir`: Path to test dataset directory (default: `Offroad_Segmentation_testImages`)
- `--output_dir`: Directory to save predictions (default: `./predictions`)
- `--batch_size`: Batch size for inference (default: 2)
- `--num_samples`: Number of comparison visualizations to save (default: 5)

**Outputs:**
- `predictions/masks/`: Raw prediction masks (class IDs 0-9)
- `predictions/masks_color/`: Colored RGB visualization masks
- `predictions/comparisons/`: Side-by-side comparison images
- `predictions/evaluation_metrics.txt`: Evaluation metrics
- `predictions/per_class_metrics.png`: Per-class IoU bar chart

## 📊 Evaluation Metrics

The model is evaluated using:
- **Mean IoU (Intersection over Union)**: Primary metric
- **Dice Coefficient**: F1 score for segmentation
- **Pixel Accuracy**: Overall classification accuracy
- **Per-Class IoU**: Individual class performance

## 🔧 Model Details

### Segmentation Head Architecture

The custom segmentation head consists of:
1. **Stem**: 7x7 convolution + GELU activation
2. **Block**: Depthwise separable convolution (7x7) + pointwise convolution (1x1) + GELU
3. **Classifier**: 1x1 convolution to output class logits

### Training Process

1. DINOv2 backbone extracts patch tokens from input images
2. Segmentation head processes tokens to generate class predictions
3. Predictions are upsampled to original image size using bilinear interpolation
4. Cross-entropy loss is computed and backpropagated

## 📈 Results

After training, you can expect:
- Training curves showing loss, IoU, Dice, and accuracy over epochs
- Validation metrics for model selection
- Per-class performance analysis

## 🛠️ Customization

### Changing Hyperparameters

Edit `Train.py` to modify:
- Batch size
- Learning rate
- Number of epochs
- Image dimensions
- Optimizer settings

### Using Different Backbone

Modify the `BACKBONE_SIZE` variable in `Train.py`:
- `"small"`: dinov2_vits14 (default)
- `"base"`: dinov2_vitb14_reg
- `"large"`: dinov2_vitl14_reg
- `"giant"`: dinov2_vitg14_reg

## 📝 Dataset Format

The dataset should follow this structure:
```
dataset/
├── Color_Images/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── Segmentation/
    ├── image1.png  # Same filename as corresponding image
    ├── image2.png
    └── ...
```

**Mask Format:**
- Masks use specific pixel values that map to classes:
  - 0 → Background
  - 100 → Trees
  - 200 → Lush Bushes
  - 300 → Dry Grass
  - 500 → Dry Bushes
  - 550 → Ground Clutter
  - 700 → Logs
  - 800 → Rocks
  - 7100 → Landscape
  - 10000 → Sky

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce batch size in `Train.py` or `Test.py`
- Use a smaller backbone variant
- Reduce image dimensions

### Model Not Found
- Ensure `segmentation_head.pth` exists in the same directory as `Test.py`
- Or specify the correct path using `--model_path` argument

### Dataset Not Found
- Verify dataset paths in the scripts
- Ensure `Color_Images` and `Segmentation` folders exist in your dataset directory

## 📄 License

This project is part of a hackathon submission. Please refer to the hackathon guidelines for usage terms.

## 👥 Authors

Hackathon Submission

## 🙏 Acknowledgments

- DINOv2 by Meta AI Research
- PyTorch team for the deep learning framework
