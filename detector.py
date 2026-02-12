from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from utils import ensure_ultralytics_home


class AircraftDetector:
    def __init__(self, weights_path: str, ultralytics_home: str):
        ensure_ultralytics_home(ultralytics_home)
        self.model = YOLO(weights_path)

    def detect(self, image_bgr: np.ndarray, conf: float) -> List[Dict[str, object]]:
        results = self.model.predict(source=image_bgr, conf=conf, verbose=False)
        if not results:
            return []
        result = results[0]
        detections = []
        names = self.model.names
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = names.get(class_id, str(class_id))
            if label != "airplane":
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            confidence = float(box.conf[0])
            area_ratio = ((x2 - x1) * (y2 - y1)) / (image_bgr.shape[0] * image_bgr.shape[1])
            if area_ratio < 0.01:
                size_class = "Small"
            elif area_ratio < 0.05:
                size_class = "Medium"
            else:
                size_class = "Large"
            detections.append(
                {
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "confidence": confidence,
                    "label": "Aircraft",
                    "size_class": size_class,
                }
            )
        return detections

    def draw_detections(
        self, image_bgr: np.ndarray, detections: List[Dict[str, object]]
    ) -> np.ndarray:
        output = image_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            confidence = det["confidence"]
            label = f"Aircraft Accuracy {confidence:.2f}"
            cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            cv2.putText(
                output,
                label,
                (int(x1), max(0, int(y1) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
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
