"""
Off-Road Semantic Segmentation Testing/Inference Script
Evaluates a trained segmentation model on test images and saves predictions
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import argparse
from tqdm import tqdm

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')


# ============================================================================
# Configuration
# ============================================================================

# Mapping from raw pixel values to class IDs
VALUE_MAP = {
    0: 0,        # background
    100: 1,      # Trees
    200: 2,      # Lush Bushes
    300: 3,      # Dry Grass
    500: 4,      # Dry Bushes
    550: 5,      # Ground Clutter
    700: 6,      # Logs
    800: 7,      # Rocks
    7100: 8,     # Landscape
    10000: 9     # Sky
}

# Class names for visualization
CLASS_NAMES = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

N_CLASSES = len(VALUE_MAP)

# Color palette for visualization (10 distinct colors)
COLOR_PALETTE = np.array([
    [0, 0, 0],        # Background - black
    [34, 139, 34],    # Trees - forest green
    [0, 255, 0],      # Lush Bushes - lime
    [210, 180, 140],  # Dry Grass - tan
    [139, 90, 43],    # Dry Bushes - brown
    [128, 128, 0],    # Ground Clutter - olive
    [139, 69, 19],    # Logs - saddle brown
    [128, 128, 128],  # Rocks - gray
    [160, 82, 45],    # Landscape - sienna
    [135, 206, 235],  # Sky - sky blue
], dtype=np.uint8)


# ============================================================================
# Utility Functions
# ============================================================================

def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in VALUE_MAP.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


def mask_to_color(mask):
    """Convert a class mask to a colored RGB image."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(N_CLASSES):
        color_mask[mask == class_id] = COLOR_PALETTE[class_id]
    return color_mask


# ============================================================================
# Dataset
# ============================================================================

class MaskDataset(Dataset):
    """Dataset for loading images and segmentation masks."""
    
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.mask_transform = mask_transform
        self.data_ids = sorted(os.listdir(self.image_dir)) if os.path.exists(self.image_dir) else []

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = Image.open(img_path).convert("RGB")
        
        # Handle case where masks might not exist (pure inference)
        if os.path.exists(mask_path):
            mask = Image.open(mask_path)
            mask = convert_mask(mask)
        else:
            # Create dummy mask if not available
            mask = Image.new('L', image.size, 0)

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask) * 255

        return image, mask, data_id


# ============================================================================
# Model: Segmentation Head (ConvNeXt-style)
# ============================================================================

class SegmentationHeadConvNeXt(nn.Module):
    """ConvNeXt-style segmentation head for DINOv2 features."""
    
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7, padding=3),
            nn.GELU()
        )

        self.block = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.GELU(),
        )

        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block(x)
        return self.classifier(x)


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=10):
    """Compute IoU for each class and return mean IoU and per-class IoU."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    iou_per_class = []
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).cpu().numpy())

    return np.nanmean(iou_per_class), iou_per_class


def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    """Compute Dice coefficient per class and return mean and per-class Dice."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    dice_per_class = []
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        dice_score = (2. * intersection + smooth) / (pred_inds.sum().float() + target_inds.sum().float() + smooth)
        dice_per_class.append(dice_score.cpu().numpy())

    return np.mean(dice_per_class), dice_per_class


def compute_pixel_accuracy(pred, target):
    """Compute pixel accuracy."""
    pred_classes = torch.argmax(pred, dim=1)
    return (pred_classes == target).float().mean().cpu().numpy()


# ============================================================================
# Visualization Functions
# ============================================================================

def save_prediction_comparison(img_tensor, gt_mask, pred_mask, output_path, data_id):
    """Save a side-by-side comparison of input, ground truth, and prediction."""
    # Denormalize image
    img = img_tensor.cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = img * std + mean
    img = np.clip(img, 0, 1)

    # Convert masks to color
    gt_color = mask_to_color(gt_mask.cpu().numpy().astype(np.uint8))
    pred_color = mask_to_color(pred_mask.cpu().numpy().astype(np.uint8))

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img)
    axes[0].set_title('Input Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(gt_color)
    axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(pred_color)
    axes[2].set_title('Prediction', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    plt.suptitle(f'Sample: {data_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_metrics_summary(results, output_dir):
    """Save metrics summary to a text file and create bar chart."""
    os.makedirs(output_dir, exist_ok=True)

    # Save text summary
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(filepath, 'w') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Mean IoU:          {results['mean_iou']:.4f}\n")
        f.write(f"Mean Dice:          {results['mean_dice']:.4f}\n")
        f.write(f"Mean Pixel Accuracy: {results['mean_pixel_acc']:.4f}\n")
        f.write("=" * 60 + "\n\n")

        f.write("Per-Class IoU:\n")
        f.write("-" * 50 + "\n")
        for i, (name, iou) in enumerate(zip(CLASS_NAMES, results['class_iou'])):
            iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
            f.write(f"  {name:<20}: {iou_str}\n")

    print(f"\nSaved evaluation metrics to {filepath}")

    # Create bar chart for per-class IoU
    fig, ax = plt.subplots(figsize=(12, 6))

    valid_iou = [iou if not np.isnan(iou) else 0 for iou in results['class_iou']]
    colors = [COLOR_PALETTE[i] / 255 for i in range(N_CLASSES)]
    ax.bar(range(N_CLASSES), valid_iou, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.set_ylabel('IoU', fontsize=12, fontweight='bold')
    ax.set_title(f'Per-Class IoU (Mean: {results["mean_iou"]:.4f})', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.axhline(y=results['mean_iou'], color='red', linestyle='--', linewidth=2, label='Mean')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved per-class metrics chart to '{output_dir}/per_class_metrics.png'")


# ============================================================================
# Main Testing Function
# ============================================================================

def main():
    """Main testing/inference function."""
    # Get script directory for default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description='Off-Road Semantic Segmentation Testing Script')
    parser.add_argument('--model_path', type=str, default=os.path.join(script_dir, 'segmentation_head.pth'),
                        help='Path to trained model weights')
    parser.add_argument('--data_dir', type=str, default=os.path.join(script_dir, 'Offroad_Segmentation_testImages'),
                        help='Path to test dataset directory')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                        help='Directory to save prediction visualizations')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for inference')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of comparison visualizations to save')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("=" * 80)

    # Image dimensions (must match training)
    w = int(((960 / 2) // 14) * 14)  # 480
    h = int(((540 / 2) // 14) * 14)  # 270

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
    ])

    # Create dataset
    print(f"Loading dataset from {args.data_dir}...")
    valset = MaskDataset(data_dir=args.data_dir, transform=transform, mask_transform=mask_transform)
    
    if len(valset) == 0:
        print(f"ERROR: No images found in {args.data_dir}/Color_Images")
        return
    
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Loaded {len(valset)} samples")
    print("=" * 80)

    # Load DINOv2 backbone
    print("Loading DINOv2 backbone...")
    BACKBONE_SIZE = "small"
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14_reg",
        "large": "vitl14_reg",
        "giant": "vitg14_reg",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"

    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    backbone_model.eval()
    backbone_model.to(device)
    print("Backbone loaded successfully!")

    # Get embedding dimension
    sample_img, _, _ = valset[0]
    sample_img = sample_img.unsqueeze(0).to(device)
    with torch.no_grad():
        output = backbone_model.forward_features(sample_img)["x_norm_patchtokens"]
    n_embedding = output.shape[2]
    print(f"Embedding dimension: {n_embedding}")

    # Load classifier
    print(f"Loading model from {args.model_path}...")
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model file not found at {args.model_path}")
        print("Please train the model first using Train.py")
        return
    
    classifier = SegmentationHeadConvNeXt(
        in_channels=n_embedding,
        out_channels=N_CLASSES,
        tokenW=w // 14,
        tokenH=h // 14
    )
    classifier.load_state_dict(torch.load(args.model_path, map_location=device))
    classifier = classifier.to(device)
    classifier.eval()
    print("Model loaded successfully!")
    print("=" * 80)

    # Create subdirectories for outputs
    masks_dir = os.path.join(args.output_dir, 'masks')
    masks_color_dir = os.path.join(args.output_dir, 'masks_color')
    comparisons_dir = os.path.join(args.output_dir, 'comparisons')
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(masks_color_dir, exist_ok=True)
    os.makedirs(comparisons_dir, exist_ok=True)

    # Run evaluation and save predictions
    print(f"\nRunning inference on {len(valset)} images...")

    iou_scores = []
    dice_scores = []
    pixel_accuracies = []
    all_class_iou = []
    all_class_dice = []
    sample_count = 0
    has_ground_truth = os.path.exists(os.path.join(args.data_dir, 'Segmentation'))

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Processing", unit="batch")
        for batch_idx, (imgs, labels, data_ids) in enumerate(pbar):
            imgs, labels = imgs.to(device), labels.to(device)

            # Forward pass
            output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
            logits = classifier(output.to(device))
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

            labels_squeezed = labels.squeeze(dim=1).long()
            predicted_masks = torch.argmax(outputs, dim=1)

            # Calculate metrics if ground truth is available
            if has_ground_truth:
                iou, class_iou = compute_iou(outputs, labels_squeezed, num_classes=N_CLASSES)
                dice, class_dice = compute_dice(outputs, labels_squeezed, num_classes=N_CLASSES)
                pixel_acc = compute_pixel_accuracy(outputs, labels_squeezed)

                iou_scores.append(iou)
                dice_scores.append(dice)
                pixel_accuracies.append(pixel_acc)
                all_class_iou.append(class_iou)
                all_class_dice.append(class_dice)

            # Save predictions for every image
            for i in range(imgs.shape[0]):
                data_id = data_ids[i]
                base_name = os.path.splitext(data_id)[0]

                # Save raw prediction mask (class IDs 0-9)
                pred_mask = predicted_masks[i].cpu().numpy().astype(np.uint8)
                pred_img = Image.fromarray(pred_mask)
                pred_img.save(os.path.join(masks_dir, f'{base_name}_pred.png'))

                # Save colored prediction mask (RGB visualization)
                pred_color = mask_to_color(pred_mask)
                cv2.imwrite(os.path.join(masks_color_dir, f'{base_name}_pred_color.png'),
                            cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))

                # Save comparison visualization for first N samples (if ground truth available)
                if has_ground_truth and sample_count < args.num_samples:
                    save_prediction_comparison(
                        imgs[i], labels_squeezed[i], predicted_masks[i],
                        os.path.join(comparisons_dir, f'sample_{sample_count}_comparison.png'),
                        data_id
                    )

                sample_count += 1

            # Update progress bar
            if has_ground_truth:
                pbar.set_postfix(iou=f"{iou:.3f}")
            else:
                pbar.set_postfix(processed=f"{sample_count}/{len(valset)}")

    # Aggregate and save results
    if has_ground_truth:
        mean_iou = np.nanmean(iou_scores)
        mean_dice = np.nanmean(dice_scores)
        mean_pixel_acc = np.mean(pixel_accuracies)

        # Average per-class metrics
        avg_class_iou = np.nanmean(all_class_iou, axis=0)
        avg_class_dice = np.nanmean(all_class_dice, axis=0)

        results = {
            'mean_iou': mean_iou,
            'mean_dice': mean_dice,
            'mean_pixel_acc': mean_pixel_acc,
            'class_iou': avg_class_iou
        }

        # Print results
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print(f"Mean IoU:          {mean_iou:.4f}")
        print(f"Mean Dice:          {mean_dice:.4f}")
        print(f"Mean Pixel Accuracy: {mean_pixel_acc:.4f}")
        print("=" * 80)

        # Save all results
        save_metrics_summary(results, args.output_dir)
    else:
        print("\nGround truth masks not found. Skipping evaluation metrics.")
        print("Predictions have been saved successfully.")

    print(f"\nPrediction complete! Processed {len(valset)} images.")
    print(f"\nOutputs saved to {args.output_dir}/")
    print(f"  - masks/           : Raw prediction masks (class IDs 0-9)")
    print(f"  - masks_color/     : Colored prediction masks (RGB)")
    if has_ground_truth:
        print(f"  - comparisons/     : Side-by-side comparison images ({args.num_samples} samples)")
        print(f"  - evaluation_metrics.txt")
        print(f"  - per_class_metrics.png")


if __name__ == "__main__":
    main()
