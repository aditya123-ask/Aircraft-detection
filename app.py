import math
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import cv2
import folium
import numpy as np
import streamlit as st
from folium.plugins import Draw
from PIL import Image
from streamlit_folium import st_folium

from detector import AircraftDetector
from map_interface import crop_by_bbox, crop_by_center, get_static_map_image
from utils import (
    fetch_opensky_states,
    geocode_location,
    geocode_location_arcgis,
    map_pixel_to_latlon,
    meters_per_pixel,
    pairwise_distances_meters,
)


APP_TITLE = "AI Airspace Surveillance & Aircraft Detection System"
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def set_theme() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .main { background-color: #0b0f1a; color: #e6eefb; }
        .stButton>button { background-color: #1e2a44; color: #e6eefb; border-radius: 6px; }
        .stTextInput>div>div>input { background-color: #101827; color: #e6eefb; }
        .stSelectbox>div>div>div { background-color: #101827; color: #e6eefb; }
        .stNumberInput>div>div>input { background-color: #101827; color: #e6eefb; }
        .stMarkdown, .stMetric, .stDataFrame { color: #e6eefb; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(image_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def parse_latlon(text: str):
    try:
        parts = [p.strip() for p in str(text).split(",")]
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
    except Exception:
        return None
    return None


def circle_to_bbox(center_lat: float, center_lon: float, radius_m: float) -> Tuple[float, float, float, float]:
    if radius_m <= 0:
        return center_lat, center_lat, center_lon, center_lon
    lat_delta = radius_m / 111320
    lon_delta = radius_m / (111320 * math.cos(math.radians(center_lat)))
    lat_min = center_lat - lat_delta
    lat_max = center_lat + lat_delta
    lon_min = center_lon - lon_delta
    lon_max = center_lon + lon_delta
    return lat_min, lat_max, lon_min, lon_max


def bbox_from_drawing(
    drawing: dict,
    fallback_center_lat: float,
    fallback_center_lon: float,
) -> Tuple[float, float, float, float] | None:
    if not drawing:
        return None
    geometry = drawing.get("geometry") or {}
    properties = drawing.get("properties") or {}
    if geometry.get("type") == "Polygon":
        coords = geometry.get("coordinates") or []
        if coords:
            ring = coords[0]
            lats = [point[1] for point in ring]
            lons = [point[0] for point in ring]
            if lats and lons:
                lat_min, lat_max = min(lats), max(lats)
                lon_min, lon_max = min(lons), max(lons)
                return lat_min, lat_max, lon_min, lon_max
    elif geometry.get("type") == "Point" and "radius" in properties:
        radius = float(properties["radius"])
        point = geometry.get("coordinates") or [fallback_center_lon, fallback_center_lat]
        lat_center = float(point[1])
        lon_center = float(point[0])
        return circle_to_bbox(lat_center, lon_center, radius)
    return None


def format_distance(value_m: float) -> str:
    if value_m >= 1000:
        return f"{value_m / 1000:.2f} km"
    return f"{value_m:.1f} m"


def detection_alert(count: int, near_airport: bool, label: str) -> str:
    if count == 0:
        return "No aircraft detected in selected zone."
    if near_airport:
        return f"ALERT: {count} aircraft detected near {label}"
    return f"ALERT: {count} aircraft detected in selected zone."


def run_detection_flow(
    map_image: Image.Image,
    crop_box: Tuple[int, int, int, int],
    center_lat: float,
    center_lon: float,
    zoom: int,
    map_width: int,
    map_height: int,
    detector: AircraftDetector,
    conf_threshold: float,
) -> Tuple[Image.Image, List[dict], List[Tuple[float, float]]]:
    image_bgr = pil_to_bgr(map_image)
    detections = detector.detect(image_bgr, conf=conf_threshold)
    if not detections and conf_threshold > 0.08:
        detections = detector.detect(image_bgr, conf=0.08)
    if detections:
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            lat, lon = map_pixel_to_latlon(
                cx + crop_box[0],
                cy + crop_box[1],
                center_lat,
                center_lon,
                zoom,
                map_width,
                map_height,
            )
            det["latlon"] = (lat, lon)
            det["center_px"] = (cx, cy)
    drawn = detector.draw_detections(image_bgr, detections)
    drawn_pil = bgr_to_pil(drawn)
    centers = [det["center_px"] for det in detections if "center_px" in det]
    return drawn_pil, detections, centers


def main() -> None:
    set_theme()
    st.title(APP_TITLE)

    with st.sidebar:
        st.header("Mission Controls")
        provider = st.selectbox("Map/Geocode Provider", ["Google", "ArcGIS"], index=1)
        if provider == "Google":
            try:
                default_api = st.secrets["google_maps_api_key"]
            except Exception:
                default_api = os.environ.get("GOOGLE_MAPS_API_KEY", "")
        else:
            try:
                default_api = st.secrets["arcgis_api_key"]
            except Exception:
                default_api = os.environ.get("ARC_GIS_API_KEY", "")
        api_label = "Google API Key" if provider == "Google" else "ArcGIS API Key"
        api_key = st.text_input(api_label, type="password", value=default_api)
        api_missing = not api_key
        location = st.text_input("Search Location", value="Los Angeles International Airport")
        zoom = st.slider("Zoom Level", min_value=3, max_value=20, value=17)
        maptype = st.selectbox("Map View", ["satellite", "hybrid"])
        map_width = st.number_input("Map Width (px)", min_value=300, max_value=640, value=640)
        map_height = st.number_input("Map Height (px)", min_value=300, max_value=640, value=640)
        crop_mode = st.selectbox("Detection Area", ["Draw On Map", "Center Crop", "Enter Coordinates"])
        crop_width = st.slider("Crop Width (px)", min_value=128, max_value=int(map_width), value=512)
        crop_height = st.slider("Crop Height (px)", min_value=128, max_value=int(map_height), value=512)
        manual_lat = st.number_input("Fallback Latitude", value=33.9416, format="%.6f")
        manual_lon = st.number_input("Fallback Longitude", value=-118.4085, format="%.6f")
        models_dir = BASE_DIR / "models"
        model_files = sorted(models_dir.glob("*.pt"))
        model_options = [str(path) for path in model_files]
        model_options.append("Custom path")
        default_model = str(models_dir / "yolov8s.pt")
        model_choice = st.selectbox("YOLOv8 Weights", model_options, index=0)
        if model_choice == "Custom path":
            weights_path = st.text_input("Custom Weights Path", value=default_model)
        else:
            weights_path = model_choice
        ultralytics_home = st.text_input(
            "Model Cache Folder",
            value="D:\\dependences",
        )
        detect_button = st.button("Detect Aircraft", disabled=api_missing)

    if api_missing:
        st.info("Add your API key to start.")
        return
    conf_threshold = 0.2

    if provider == "Google":
        geo = geocode_location(api_key, location)
    else:
        geo = geocode_location_arcgis(api_key, location)
    if not geo:
        parsed = parse_latlon(location)
        if parsed:
            center_lat, center_lon = parsed
            formatted = f"{center_lat:.6f}, {center_lon:.6f}"
            geo = (center_lat, center_lon, formatted)
        else:
            center_lat, center_lon = float(manual_lat), float(manual_lon)
            formatted = f"{center_lat:.6f}, {center_lon:.6f}"
            geo = (center_lat, center_lon, formatted)

    center_lat, center_lon, formatted = geo
    col_map, col_results = st.columns([1.1, 0.9])
    image_width = int(map_width)
    detection_crop = None
    crop_box = (0, 0, int(map_width), int(map_height))
    bbox_inputs = None
    if crop_mode == "Draw On Map":
        with col_map:
            st.subheader(f"{provider} Map Panel")
            base_tiles = "OpenStreetMap" if maptype == "hybrid" else None
            map_object = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles=base_tiles)
            if maptype == "satellite":
                folium.TileLayer(
                    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                    attr="Esri",
                    name="Esri World Imagery",
                    overlay=True,
                    control=False,
                ).add_to(map_object)
            draw = Draw(
                draw_options={
                    "polyline": False,
                    "polygon": False,
                    "rectangle": True,
                    "circle": True,
                    "marker": False,
                    "circlemarker": False,
                },
                edit_options={"edit": True, "remove": True},
            )
            draw.add_to(map_object)
            map_data = st_folium(map_object, width=image_width, height=int(map_height), key="map_draw")

        drawing = None
        if map_data:
            map_center = map_data.get("center")
            map_zoom = map_data.get("zoom")
            if map_center:
                center_lat = float(map_center["lat"])
                center_lon = float(map_center["lng"])
            if map_zoom is not None:
                zoom = int(map_zoom)

            drawing = map_data.get("last_active_drawing")
            if not drawing:
                all_drawings = map_data.get("all_drawings") or []
                if all_drawings:
                    drawing = all_drawings[-1]
            if drawing:
                st.session_state["last_drawing"] = drawing
        if not drawing:
            drawing = st.session_state.get("last_drawing")

        bbox_inputs = bbox_from_drawing(drawing, center_lat, center_lon)

        provider_for_image = "google" if provider == "Google" else "arcgis"
        map_image, map_url = get_static_map_image(
            api_key, center_lat, center_lon, zoom, int(map_width), int(map_height), maptype, provider_for_image
        )

        if map_image is None:
            st.error(f"Failed to fetch map image from {provider} maps.")
            st.code(map_url)
            return

        if bbox_inputs and not all(math.isfinite(value) for value in bbox_inputs):
            bbox_inputs = None
        if bbox_inputs:
            lat_min, lat_max, lon_min, lon_max = bbox_inputs
            detection_crop, crop_box = crop_by_bbox(
                map_image,
                center_lat,
                center_lon,
                zoom,
                map_image.size[0],
                map_image.size[1],
                lat_min,
                lat_max,
                lon_min,
                lon_max,
            )
        else:
            detection_crop, crop_box = crop_by_center(map_image, int(crop_width), int(crop_height))

        if detection_crop is None or min(detection_crop.size) == 0:
            st.error("Selected zone is empty. Adjust zoom or selection.")
            return
        with col_map:
            st.subheader("Selected Detection Zone")
            st.image(detection_crop, use_container_width=True)
    else:
        provider_for_image = "google" if provider == "Google" else "arcgis"
        map_image, map_url = get_static_map_image(
            api_key, center_lat, center_lon, zoom, int(map_width), int(map_height), maptype, provider_for_image
        )

        if map_image is None:
            st.error(f"Failed to fetch map image from {provider} maps.")
            st.code(map_url)
            return

        with col_map:
            st.subheader(f"{provider} Map Panel")
            st.image(map_image, use_container_width=True)
            st.caption(f"Location: {formatted}")

        if crop_mode == "Enter Coordinates":
            st.subheader("Detection Area Coordinates")
            lat_min = st.number_input("Lat Min", value=center_lat - 0.01, format="%.6f")
            lat_max = st.number_input("Lat Max", value=center_lat + 0.01, format="%.6f")
            lon_min = st.number_input("Lon Min", value=center_lon - 0.01, format="%.6f")
            lon_max = st.number_input("Lon Max", value=center_lon + 0.01, format="%.6f")
            bbox_inputs = (lat_min, lat_max, lon_min, lon_max)
            if not all(math.isfinite(value) for value in bbox_inputs):
                bbox_inputs = None
            detection_crop, crop_box = crop_by_bbox(
                map_image,
                center_lat,
                center_lon,
                zoom,
                map_image.size[0],
                map_image.size[1],
                lat_min,
                lat_max,
                lon_min,
                lon_max,
            )
        else:
            detection_crop, crop_box = crop_by_center(map_image, int(crop_width), int(crop_height))

        if detection_crop is None or min(detection_crop.size) == 0:
            st.error("Selected zone is empty. Adjust zoom or selection.")
            return
        with col_map:
            st.subheader("Selected Detection Zone")
            st.image(detection_crop, use_container_width=True)

    if detect_button:
        detector = AircraftDetector(weights_path, ultralytics_home)
        processed_image, detections, centers = run_detection_flow(
            detection_crop,
            crop_box,
            center_lat,
            center_lon,
            zoom,
            map_image.size[0],
            map_image.size[1],
            detector,
            conf_threshold,
        )
        heatmap_image = detector.heatmap_overlay(pil_to_bgr(detection_crop), detections)
        heatmap_pil = bgr_to_pil(heatmap_image)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"detected_{timestamp}.png"
        processed_image.save(output_path)

        with col_results:
            st.subheader("Detection Output")
            st.image(processed_image, use_container_width=True)
            st.subheader("Heatmap Overlay")
            st.image(heatmap_pil, use_container_width=True)
            st.markdown(f"**Saved Output:** {output_path}")

        count = len(detections)
        near_airport = "airport" in formatted.lower() or "airfield" in formatted.lower()
        st.subheader("Detection Summary")
        st.metric("Total Aircraft Detected", count)
        st.markdown(detection_alert(count, near_airport, formatted))

        if detections:
            st.subheader("Detection Log")
            for idx, det in enumerate(detections, start=1):
                lat, lon = det.get("latlon", (None, None))
                st.markdown(
                    f"{idx}. Aircraft | Accuracy {det['confidence']:.2f} | "
                    f"Size {det['size_class']} | Lat {lat:.6f} | Lon {lon:.6f}"
                )

            meters_px = meters_per_pixel(center_lat, zoom)
            distances = pairwise_distances_meters(centers, meters_px)
            if distances:
                closest = min(distances, key=lambda item: item[2])
                st.subheader("Proximity")
                st.markdown(
                    f"Closest aircraft pair: {closest[0] + 1} & {closest[1] + 1} "
                    f"at {format_distance(closest[2])}"
                )

        if bbox_inputs:
            opensky = fetch_opensky_states(bbox_inputs)
            if opensky:
                st.subheader("OpenSky Live Aircraft (Optional)")
                st.dataframe(opensky, use_container_width=True)


if __name__ == "__main__":
    main()
