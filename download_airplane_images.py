"""
Script to download airplane images from Open Images Dataset.
Filters validation-annotations-bbox.csv for airplane class (/m/0cmf2)
and downloads only those images with YOLO format annotations.
"""
import csv
import os
import urllib.request
import shutil
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent
CSV_FILE = BASE_DIR / "models" / "validation-annotations-bbox.csv"
IMAGE_LIST_FILE = BASE_DIR / "models" / "airplane_image_list.txt"
DOWNLOAD_FOLDER = Path("D:/dependences/open_images_airplanes")
DATASET_IMAGES = BASE_DIR / "dataset" / "images"
DATASET_LABELS = BASE_DIR / "dataset" / "labels"

# Airplane class ID in Open Images
AIRPLANE_CLASS = "/m/0cmf2"

def extract_airplane_annotations():
    """Extract airplane annotations from CSV and create image list."""
    print("Reading annotations CSV...")
    
    airplane_images = set()
    annotations = []
    
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['LabelName'] == AIRPLANE_CLASS:
                image_id = row['ImageID']
                airplane_images.add(image_id)
                annotations.append({
                    'image_id': image_id,
                    'xmin': float(row['XMin']),
                    'xmax': float(row['XMax']),
                    'ymin': float(row['YMin']),
                    'ymax': float(row['YMax'])
                })
    
    print(f"Found {len(airplane_images)} unique airplane images")
    print(f"Total airplane annotations: {len(annotations)}")
    
    # Create image list file for downloader
    with open(IMAGE_LIST_FILE, 'w') as f:
        for img_id in sorted(airplane_images):
            f.write(f"validation/{img_id}\n")
    
    print(f"Created image list: {IMAGE_LIST_FILE}")
    return airplane_images, annotations

def download_images():
    """Download images using Open Images downloader."""
    print("\nDownloading Open Images downloader...")
    
    downloader_url = "https://raw.githubusercontent.com/openimages/dataset/master/downloader.py"
    downloader_path = BASE_DIR / "downloader.py"
    
    try:
        urllib.request.urlretrieve(downloader_url, downloader_path)
        print(f"Downloaded: {downloader_path}")
    except Exception as e:
        print(f"Error downloading downloader.py: {e}")
        return False
    
    # Create download folder
    DOWNLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading images to: {DOWNLOAD_FOLDER}")
    print("This may take a while...")
    
    # Run downloader
    import subprocess
    cmd = [
        "python", str(downloader_path),
        str(IMAGE_LIST_FILE),
        f"--download_folder={DOWNLOAD_FOLDER}",
        "--num_processes=5"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Download completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during download: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False

def convert_to_yolo_format(annotations, airplane_images):
    """Convert Open Images annotations to YOLO format."""
    print("\nConverting annotations to YOLO format...")
    
    # Create dataset folders
    DATASET_IMAGES.mkdir(parents=True, exist_ok=True)
    DATASET_LABELS.mkdir(parents=True, exist_ok=True)
    
    # Group annotations by image
    image_annotations = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)
    
    # Process each image
    converted_count = 0
    for img_id in airplane_images:
        # Source image path
        src_image = DOWNLOAD_FOLDER / f"{img_id}.jpg"
        
        if not src_image.exists():
            print(f"Warning: Image not found: {src_image}")
            continue
        
        # Copy image to dataset
        dst_image = DATASET_IMAGES / f"{img_id}.jpg"
        shutil.copy2(src_image, dst_image)
        
        # Create YOLO annotation file
        label_file = DATASET_LABELS / f"{img_id}.txt"
        
        with open(label_file, 'w') as f:
            for ann in image_annotations.get(img_id, []):
                # Convert to YOLO format (normalized)
                x_center = (ann['xmin'] + ann['xmax']) / 2
                y_center = (ann['ymin'] + ann['ymax']) / 2
                width = ann['xmax'] - ann['xmin']
                height = ann['ymax'] - ann['ymin']
                
                # Class 0 = airplane
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        converted_count += 1
    
    print(f"Converted {converted_count} images to YOLO format")
    print(f"Images saved to: {DATASET_IMAGES}")
    print(f"Labels saved to: {DATASET_LABELS}")

def cleanup():
    """Remove temporary files to save space."""
    print("\nCleaning up temporary files...")
    
    # Remove downloaded images from temp folder (already copied to dataset)
    if DOWNLOAD_FOLDER.exists():
        shutil.rmtree(DOWNLOAD_FOLDER)
        print(f"Removed: {DOWNLOAD_FOLDER}")
    
    # Remove downloader script
    downloader_path = BASE_DIR / "downloader.py"
    if downloader_path.exists():
        downloader_path.unlink()
        print(f"Removed: {downloader_path}")
    
    # Remove image list file
    if IMAGE_LIST_FILE.exists():
        IMAGE_LIST_FILE.unlink()
        print(f"Removed: {IMAGE_LIST_FILE}")
    
    print("Cleanup completed!")

def main():
    print("=" * 60)
    print("Open Images Airplane Dataset Downloader")
    print("=" * 60)
    
    # Step 1: Extract airplane annotations
    airplane_images, annotations = extract_airplane_annotations()
    
    if not airplane_images:
        print("No airplane images found!")
        return
    
    # Step 2: Download images
    success = download_images()
    
    if not success:
        print("Download failed. Please check your internet connection.")
        return
    
    # Step 3: Convert to YOLO format
    convert_to_yolo_format(annotations, airplane_images)
    
    # Step 4: Cleanup
    cleanup()
    
    print("\n" + "=" * 60)
    print("Dataset preparation completed!")
    print("=" * 60)
    print(f"Images: {DATASET_IMAGES}")
    print(f"Labels: {DATASET_LABELS}")
    print("\nYou can now train your YOLO model using these images.")

if __name__ == "__main__":
    main()
