import io
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image


def geocode_location(api_key: str, location: str) -> Optional[Tuple[float, float, str]]:
    if not api_key or not location:
        return None
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": location, "key": api_key}
    response = requests.get(url, params=params, timeout=20)
    if response.status_code != 200:
        return None
    data = response.json()
    if data.get("status") != "OK":
        return None
    result = data["results"][0]
    lat = result["geometry"]["location"]["lat"]
    lon = result["geometry"]["location"]["lng"]
    formatted = result.get("formatted_address", location)
    return lat, lon, formatted


def geocode_location_arcgis(api_key: str, location: str) -> Optional[Tuple[float, float, str]]:
    if not location:
        return None
    url = "https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates"
    params = {"SingleLine": location, "f": "json"}
    if api_key:
        params["token"] = api_key
    response = requests.get(url, params=params, timeout=20)
    if response.status_code != 200:
        return None
    data = response.json()
    candidates = data.get("candidates") or []
    if not candidates:
        return None
    top = candidates[0]
    loc = top.get("location") or {}
    lon = loc.get("x")
    lat = loc.get("y")
    if lat is None or lon is None:
        return None
    formatted = top.get("address") or location
    return float(lat), float(lon), formatted


def latlon_to_web_mercator(lat: float, lon: float) -> Tuple[float, float]:
    origin_shift = 6378137.0
    x = origin_shift * math.radians(lon)
    y = origin_shift * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def build_static_map_url(
    api_key: str,
    center_lat: float,
    center_lon: float,
    zoom: int,
    width: int,
    height: int,
    maptype: str,
) -> str:
    base = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{center_lat},{center_lon}",
        "zoom": str(zoom),
        "size": f"{width}x{height}",
        "maptype": maptype,
        "scale": "1",
        "key": api_key,
    }
    query = "&".join(f"{k}={requests.utils.quote(v)}" for k, v in params.items())
    return f"{base}?{query}"


def build_static_map_url_arcgis(
    api_key: str,
    center_lat: float,
    center_lon: float,
    zoom: int,
    width: int,
    height: int,
    maptype: str,
) -> str:
    service = "World_Imagery"
    if maptype == "hybrid":
        service = "World_Imagery"
    base = f"https://services.arcgisonline.com/ArcGIS/rest/services/{service}/MapServer/export"
    meters_px = meters_per_pixel(center_lat, zoom)
    center_x, center_y = latlon_to_web_mercator(center_lat, center_lon)
    half_width = (width / 2) * meters_px
    half_height = (height / 2) * meters_px
    bbox = f"{center_x - half_width},{center_y - half_height},{center_x + half_width},{center_y + half_height}"
    params = {
        "bbox": bbox,
        "size": f"{width},{height}",
        "format": "png",
        "f": "image",
        "bboxSR": "3857",
        "imageSR": "3857",
    }
    if api_key:
        params["token"] = api_key
    query = "&".join(f"{k}={requests.utils.quote(str(v))}" for k, v in params.items())
    return f"{base}?{query}"


def build_static_map_url_osm(
    center_lat: float,
    center_lon: float,
    zoom: int,
    width: int,
    height: int,
    maptype: str,
) -> str:
    base = "https://staticmap.openstreetmap.de/staticmap.php"
    params = {
        "center": f"{center_lat},{center_lon}",
        "zoom": str(zoom),
        "size": f"{width}x{height}",
        "maptype": "mapnik",
        "markers": f"{center_lat},{center_lon},lightblue1",
    }
    query = "&".join(f"{k}={requests.utils.quote(v)}" for k, v in params.items())
    return f"{base}?{query}"


def fetch_map_image(url: str) -> Image.Image:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def latlon_to_world_px(lat: float, lon: float, zoom: int) -> Tuple[float, float]:
    siny = math.sin(lat * math.pi / 180)
    siny = min(max(siny, -0.9999), 0.9999)
    x = 256 * (0.5 + lon / 360)
    y = 256 * (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi))
    scale = 2**zoom
    return x * scale, y * scale


def world_px_to_latlon(x: float, y: float, zoom: int) -> Tuple[float, float]:
    scale = 2**zoom
    x /= scale
    y /= scale
    lon = (x / 256 - 0.5) * 360
    n = math.pi - 2 * math.pi * (y / 256 - 0.5)
    lat = 180 / math.pi * math.atan(0.5 * (math.exp(n) - math.exp(-n)))
    return lat, lon


def map_pixel_to_latlon(
    px: float,
    py: float,
    center_lat: float,
    center_lon: float,
    zoom: int,
    width: int,
    height: int,
) -> Tuple[float, float]:
    center_world_x, center_world_y = latlon_to_world_px(center_lat, center_lon, zoom)
    top_left_world_x = center_world_x - width / 2
    top_left_world_y = center_world_y - height / 2
    world_x = top_left_world_x + px
    world_y = top_left_world_y + py
    return world_px_to_latlon(world_x, world_y, zoom)


def latlon_to_map_pixel(
    lat: float,
    lon: float,
    center_lat: float,
    center_lon: float,
    zoom: int,
    width: int,
    height: int,
) -> Tuple[float, float]:
    center_world_x, center_world_y = latlon_to_world_px(center_lat, center_lon, zoom)
    top_left_world_x = center_world_x - width / 2
    top_left_world_y = center_world_y - height / 2
    world_x, world_y = latlon_to_world_px(lat, lon, zoom)
    return world_x - top_left_world_x, world_y - top_left_world_y


def meters_per_pixel(lat: float, zoom: int) -> float:
    return 156543.03392 * math.cos(lat * math.pi / 180) / (2**zoom)


def pairwise_distances_meters(
    centers_px: List[Tuple[float, float]], meters_per_px: float
) -> List[Tuple[int, int, float]]:
    distances = []
    for i in range(len(centers_px)):
        for j in range(i + 1, len(centers_px)):
            dx = centers_px[i][0] - centers_px[j][0]
            dy = centers_px[i][1] - centers_px[j][1]
            dist = math.sqrt(dx * dx + dy * dy) * meters_per_px
            distances.append((i, j, dist))
    return distances


def fetch_opensky_states(
    bbox: Tuple[float, float, float, float]
) -> Optional[List[Dict[str, object]]]:
    lat_min, lat_max, lon_min, lon_max = bbox
    url = "https://opensky-network.org/api/states/all"
    params = {
        "lamin": lat_min,
        "lamax": lat_max,
        "lomin": lon_min,
        "lomax": lon_max,
    }
    response = requests.get(url, params=params, timeout=20)
    if response.status_code != 200:
        return None
    data = response.json()
    states = data.get("states") or []
    results = []
    for state in states:
        results.append(
            {
                "icao24": state[0],
                "callsign": (state[1] or "").strip(),
                "country": state[2],
                "longitude": state[5],
                "latitude": state[6],
                "velocity": state[9],
                "heading": state[10],
                "altitude": state[13],
            }
        )
    return results


def ensure_ultralytics_home(path: str) -> None:
    if path:
        os.environ["ULTRALYTICS_HOME"] = path
