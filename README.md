# AI Airspace Surveillance & Aircraft Detection System

## Overview
This Streamlit application integrates Google Maps with YOLOv8-based aircraft detection to create an airspace surveillance-style interface. Users can search locations, select detection regions, run inference, and visualize bounding boxes, counts, coordinates, proximity, and heatmaps.

## Setup
1. Install Python 3.10+
2. Install dependencies:

```
pip install -r requirements.txt
```

3. Add your Google Maps API key in the app sidebar when running.

## Run
```
streamlit run app.py
```

## Google Maps API Key
Enable the following in Google Cloud Console:
- Maps Static API
- Geocoding API

Paste the API key in the app sidebar. The map image will load automatically.

## Model Weights
The app uses YOLOv8 weights. Provide a model path in the sidebar:
- Default: `models/yolov8s.pt`
- Model cache directory is set to `D:\dependences` to keep large downloads off the C drive.

## Training Custom Models
1. Place your aircraft dataset in `dataset/`
2. Train with Ultralytics:

```
yolo task=detect mode=train model=yolov8s.pt data=dataset/data.yaml epochs=100 imgsz=640
```

3. Update the app sidebar to point to the trained weights.

## Outputs
Processed images are saved into `outputs/` with timestamped filenames.

## Future Improvements
- Real-time multi-source aircraft feeds and tracking history
- Advanced military zone monitoring workflows
- Multi-model ensembling for higher satellite detection accuracy
- Automated map region scheduling and alerting

## Detection Process

1. **Select Region on Map**  
   ![Select Region](images/select_region.png)

2. **Run Detection**  
   ![Detection Results](images/detection_results.png)

3. **Aircraft Count Displayed**  
   ![Aircraft Count](images/aircraft_count.png)
