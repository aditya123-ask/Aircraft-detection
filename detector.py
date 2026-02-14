from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from utils import ensure_ultralytics_home


class AircraftDetector:
    def __init__(self, weights_path: str, ultralytics_home: str):
        ensure_ultralytics_home(ultralytics_home)
        self.model = YOLO(weights_path)

    def preprocess_image(self, image_bgr: np.ndarray) -> np.ndarray:
        """Enhance image for better aircraft detection in satellite imagery."""
        # Convert to LAB color space for better contrast
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge back
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Slight sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened

    def detect(self, image_bgr: np.ndarray, conf: float, debug: bool = False) -> List[Dict[str, object]]:
        # Preprocess image for better detection
        processed_image = self.preprocess_image(image_bgr)
        
        # Run prediction with optimized parameters for aerial imagery
        results = self.model.predict(
            source=processed_image, 
            conf=conf, 
            verbose=False,
            iou=0.3,  # Lower IoU threshold to detect overlapping aircraft
            augment=True,  # Enable test-time augmentation
            agnostic_nms=True  # Class-agnostic NMS
        )
        
        if not results:
            return []
        result = results[0]
        detections = []
        names = self.model.names
        all_detections = []  # For debugging
        
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = names.get(class_id, str(class_id))
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            confidence = float(box.conf[0])
            
            # Store all detections for debugging
            all_detections.append({
                "label": label,
                "confidence": confidence,
                "bbox": (float(x1), float(y1), float(x2), float(y2))
            })
            
            # Accept both "airplane" and "aircraft" classes if available
            # Reject obvious false positives like "clock", "sports ball", etc.
            false_positive_labels = ["clock", "sports ball", "frisbee", "mouse", "remote", "cell phone", 
                                     "traffic light", "stop sign", "fire hydrant", "parking meter", "bench"]
            if label in false_positive_labels:
                continue
            
            # For satellite imagery, we mainly want "airplane" class
            # Other classes like "truck", "bus" are usually false positives in this context
            if label not in ["airplane", "aircraft"]:
                continue
            
            # Calculate area and filter out detections that are too small or too large
            width = x2 - x1
            height = y2 - y1
            area = width * height
            img_area = image_bgr.shape[0] * image_bgr.shape[1]
            area_ratio = area / img_area
            
            # Filter out detections that are too small (< 0.05% of image) or too large (> 50% of image)
            if area_ratio < 0.0005 or area_ratio > 0.5:
                continue
            
            # Filter out detections with extreme aspect ratios (not aircraft-like)
            aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
            if aspect_ratio > 5:  # Aircraft shouldn't be too elongated
                continue
            
            if area_ratio < 0.005:  # Lower threshold for small aircraft
                size_class = "Small"
            elif area_ratio < 0.03:
                size_class = "Medium"
            else:
                size_class = "Large"
            detections.append(
                {
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "confidence": confidence,
                    "label": "Aircraft",
                    "size_class": size_class,
                    "original_label": label,
                }
            )
        
        if debug:
            return detections, all_detections
        return detections

    def draw_detections(
        self, image_bgr: np.ndarray, detections: List[Dict[str, object]], show_all: bool = False
    ) -> np.ndarray:
        output = image_bgr.copy()
        height, width = output.shape[:2]
        
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            confidence = det["confidence"]
            
            # Get label - use original label if showing all objects
            if show_all and "original_label" in det:
                label_text = det["original_label"]
            elif "label" in det:
                label_text = det["label"]
            else:
                label_text = "Aircraft"
            
            label = f"{label_text} {confidence:.2f}"
            
            # Use different colors for different object types
            if label_text == "airplane" or label_text == "Aircraft":
                color = (0, 255, 255)  # Cyan for aircraft
            elif label_text in ["truck", "bus"]:
                color = (255, 165, 0)  # Orange for trucks/buses
            else:
                color = (255, 0, 255)  # Magenta for other objects
            
            # Draw rectangle
            cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw label background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(
                output, 
                (int(x1), max(0, int(y1) - text_size[1] - 5)), 
                (int(x1) + text_size[0], int(y1)), 
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                output,
                label,
                (int(x1), max(0, int(y1) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # Black text
                1,
                cv2.LINE_AA,
            )
            
            # Draw detection number
            cv2.putText(
                output,
                str(idx + 1),
                (int(x1), int(y2) + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        
        # Add detection count
        count_text = f"Detections: {len(detections)}"
        cv2.putText(
            output,
            count_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        
        return output

    def heatmap_overlay(
        self, image_bgr: np.ndarray, detections: List[Dict[str, object]]
    ) -> np.ndarray:
        if not detections:
            return image_bgr.copy()
        height, width = image_bgr.shape[:2]
        heat = np.zeros((height, width), dtype=np.float32)
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(heat, (cx, cy), 25, 1.0, -1)
        heat = cv2.GaussianBlur(heat, (0, 0), 15)
        heat = np.clip(heat / (heat.max() + 1e-6), 0, 1)
        heat_color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(image_bgr, 0.7, heat_color, 0.3, 0)
