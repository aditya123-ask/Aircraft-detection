"""
Script to download a better aircraft detection model for satellite imagery.
"""
import os
import urllib.request
import sys

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Option 1: YOLOv8n (nano) - faster but less accurate
# Option 2: YOLOv8s (small) - balanced
# Option 3: YOLOv8m (medium) - more accurate but slower

# For satellite aircraft detection, we need a custom model.
# Here are some options:

MODEL_OPTIONS = {
    "1": {
        "name": "YOLOv8s (COCO - Current)",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt",
        "filename": "yolov8s.pt",
        "description": "Standard COCO model - NOT recommended for satellite imagery"
    },
    "2": {
        "name": "YOLOv8n (COCO - Nano)",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
        "filename": "yolov8n.pt",
        "description": "Smaller, faster - but still not for satellite imagery"
    },
    "3": {
        "name": "YOLOv8m (COCO - Medium)",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt",
        "filename": "yolov8m.pt",
        "description": "More accurate COCO model - but still not for satellite imagery"
    }
}

def download_file(url, filepath):
    """Download a file with progress."""
    print(f"Downloading from: {url}")
    print(f"Saving to: {filepath}")
    
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        print(f"\rProgress: {percent:.1f}%", end="")
    
    urllib.request.urlretrieve(url, filepath, reporthook=report_progress)
    print("\nDownload complete!")

def main():
    print("=" * 60)
    print("Aircraft Detection Model Downloader")
    print("=" * 60)
    print("\nIMPORTANT: For satellite aircraft detection,")
    print("you need a model trained on aerial/satellite imagery.")
    print("Standard COCO models will NOT work well.\n")
    
    print("Available models:")
    for key, model in MODEL_OPTIONS.items():
        print(f"\n{key}. {model['name']}")
        print(f"   Description: {model['description']}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION:")
    print("=" * 60)
    print("\nFor satellite aircraft detection, download a custom model from:")
    print("\n1. Roboflow Universe:")
    print("   https://universe.roboflow.com/")
    print("   Search: 'aircraft satellite' or 'airplane aerial'")
    print("\n2. Hugging Face:")
    print("   https://huggingface.co/models")
    print("   Search: 'yolo aircraft satellite'")
    print("\n3. Train your own using:")
    print("   - DOTA dataset")
    print("   - xView dataset")
    print("   - Airbus Aircraft dataset")
    print("\n" + "=" * 60)
    
    choice = input("\nEnter your choice (1-3) or 'q' to quit: ").strip()
    
    if choice.lower() == 'q':
        print("Exiting...")
        return
    
    if choice not in MODEL_OPTIONS:
        print("Invalid choice!")
        return
    
    model = MODEL_OPTIONS[choice]
    filepath = os.path.join(MODELS_DIR, model['filename'])
    
    if os.path.exists(filepath):
        overwrite = input(f"{model['filename']} already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Skipping download.")
            return
    
    try:
        download_file(model['url'], filepath)
        print(f"\nModel saved to: {filepath}")
        print(f"\nTo use this model in the app:")
        print(f"1. Select 'Custom path' in the YOLOv8 Weights dropdown")
        print(f"2. Enter: {filepath}")
    except Exception as e:
        print(f"\nError downloading model: {e}")
        print("\nPlease download manually from:")
        print(model['url'])

if __name__ == "__main__":
    main()
