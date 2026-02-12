from typing import Optional, Tuple

from PIL import Image

from utils import (
    build_static_map_url_arcgis,
    build_static_map_url,
    build_static_map_url_osm,
    fetch_map_image,
    latlon_to_map_pixel,
)


def get_static_map_image(
    api_key: str,
    center_lat: float,
    center_lon: float,
    zoom: int,
    width: int,
    height: int,
    maptype: str,
    provider: str = "google",
) -> Tuple[Optional[Image.Image], str]:
    if provider.lower() == "google":
        url = build_static_map_url(api_key, center_lat, center_lon, zoom, width, height, maptype)
    elif provider.lower() == "arcgis":
        url = build_static_map_url_arcgis(
            api_key, center_lat, center_lon, zoom, width, height, maptype
        )
    else:
        url = build_static_map_url_osm(center_lat, center_lon, zoom, width, height, maptype)
    try:
        image = fetch_map_image(url)
        return image, url
    except Exception:
        return None, url


def crop_by_center(
    image: Image.Image, crop_width: int, crop_height: int
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    width, height = image.size
    crop_width = min(crop_width, width)
    crop_height = min(crop_height, height)
    left = max(0, int((width - crop_width) / 2))
    top = max(0, int((height - crop_height) / 2))
    right = min(width, left + crop_width)
    bottom = min(height, top + crop_height)
    return image.crop((left, top, right, bottom)), (left, top, right, bottom)


def crop_by_bbox(
    image: Image.Image,
    center_lat: float,
    center_lon: float,
    zoom: int,
    width: int,
    height: int,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    px_left, py_top = latlon_to_map_pixel(
        lat_max, lon_min, center_lat, center_lon, zoom, width, height
    )
    px_right, py_bottom = latlon_to_map_pixel(
        lat_min, lon_max, center_lat, center_lon, zoom, width, height
    )
    left = max(0, min(int(px_left), width))
    right = max(0, min(int(px_right), width))
    top = max(0, min(int(py_top), height))
    bottom = max(0, min(int(py_bottom), height))
    if right <= left:
        right = min(width, left + 1)
    if bottom <= top:
        bottom = min(height, top + 1)
    return image.crop((left, top, right, bottom)), (left, top, right, bottom)
