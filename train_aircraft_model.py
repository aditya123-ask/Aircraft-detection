"""
Train YOLOv8 model for aircraft-only detection.
Uses the downloaded Open Images airplane dataset.
"""
import os
import shutil
import random
from pathlib import Path
import yaml

# Configuration
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
MODELS_DIR = BASE_DIR / "models"

# Training parameters
TRAIN_SPLIT = 0.8  # 80% train, 20% validation
EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 16

def prepare_dataset():
    """Split dataset into train and validation sets."""
    print("Preparing dataset...")
    
    # Get all image files
    image_files = sorted(IMAGES_DIR.glob("*.jpg"))
    print(f"Found {len(image_files)} images")
    
    if len(image_files) == 0:
        print("No images found!")
        return False
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(image_files)
    
    split_idx = int(len(image_files) * TRAIN_SPLIT)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Train: {len(train_files)}, Validation: {len(val_files)}")
    
    # Create directories
    (TRAIN_DIR / "images").mkdir(parents=True, exist_ok=True)
    (TRAIN_DIR / "labels").mkdir(parents=True, exist_ok=True)
    (VAL_DIR / "images").mkdir(parents=True, exist_ok=True)
    (VAL_DIR / "labels").mkdir(parents=True, exist_ok=True)
    
    # Copy train files
    for img_file in train_files:
        # Copy image
        shutil.copy2(img_file, TRAIN_DIR / "images" / img_file.name)
        # Copy label
        label_file = LABELS_DIR / f"{img_file.stem}.txt"
        if label_file.exists():
            shutil.copy2(label_file, TRAIN_DIR / "labels" / label_file.name)
    
    # Copy validation files
    for img_file in val_files:
        # Copy image
        shutil.copy2(img_file, VAL_DIR / "images" / img_file.name)
        # Copy label
        label_file = LABELS_DIR / f"{img_file.stem}.txt"
        if label_file.exists():
            shutil.copy2(label_file, VAL_DIR / "labels" / label_file.name)
    
    print("Dataset preparation completed!")
    return True

def create_data_yaml():
    """Create data configuration file for YOLO training."""
    data_config = {
        'path': str(DATASET_DIR.resolve()),
        'train': str((TRAIN_DIR / "images").resolve()),
        'val': str((VAL_DIR / "images").resolve()),
        'nc': 1,  # Number of classes (only aircraft)
        'names': ['aircraft']  # Class names
    }
    
    yaml_path = BASE_DIR / "aircraft_data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"Created data config: {yaml_path}")
    return yaml_path

def train_model(data_yaml):
    """Train YOLOv8 model."""
    from ultralytics import YOLO
    
    print("\nStarting training...")
    print(f"Epochs: {EPOCHS}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    
    # Load a pre-trained model (transfer learning)
    # Using yolov8n (nano) for faster training, can use yolov8s or yolov8m for better accuracy
    model = YOLO('yolov8n.pt')
    
    # Train the model
    results = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        name='aircraft_detection',
        project=str(MODELS_DIR / 'training_results'),
        exist_ok=True,
        patience=20,  # Early stopping patience
        save=True,
        device='cpu'  # Change to 0 if you have GPU
    )
    
    return results

def copy_best_model():
    """Copy the best trained model to models folder."""
    training_dir = MODELS_DIR / 'training_results' / 'aircraft_detection'
    best_model = training_dir / 'weights' / 'best.pt'
    
    if best_model.exists():
        target_path = MODELS_DIR / 'aircraft_custom.pt'
        shutil.copy2(best_model, target_path)
        print(f"\nBest model saved to: {target_path}")
        return target_path
    else:
        print("Best model not found!")
        return None

def cleanup_temp_files():
    """Remove temporary training files to save space."""
    print("\nCleaning up temporary files...")
    
    # Remove training results (keep only best model)
    training_dir = MODELS_DIR / 'training_results'
    if training_dir.exists():
        shutil.rmtree(training_dir)
        print(f"Removed: {training_dir}")
    
    # Remove train/val split folders (keep original dataset)
    if TRAIN_DIR.exists():
        shutil.rmtree(TRAIN_DIR)
        print(f"Removed: {TRAIN_DIR}")
    
    if VAL_DIR.exists():
        shutil.rmtree(VAL_DIR)
        print(f"Removed: {VAL_DIR}")
    
    print("Cleanup completed!")

def main():
    print("=" * 60)
    print("Aircraft Detection Model Training")
    print("=" * 60)
    
    # Step 1: Prepare dataset
    if not prepare_dataset():
        return
    
    # Step 2: Create data config
    data_yaml = create_data_yaml()
    
    # Step 3: Train model
    try:
        results = train_model(data_yaml)
        
        # Step 4: Copy best model
        best_model_path = copy_best_model()
        
        if best_model_path:
            print("\n" + "=" * 60)
            print("Training completed successfully!")
            print("=" * 60)
            print(f"Model saved: {best_model_path}")
            print("\nYou can now use this model in the Streamlit app.")
            print("Select 'aircraft_custom.pt' from the model dropdown.")
        
        # Step 5: Cleanup
        cleanup_temp_files()
        
    except Exception as e:
        print(f"\nError during training: {e}")
        print("\nMake sure you have ultralytics installed:")
        print("pip install ultralytics")

if __name__ == "__main__":
    main()
